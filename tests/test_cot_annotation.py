"""Offline checks for annotation geometry and the vision chat API contract."""

import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/datasets"))
from annotate_cot_from_raw import validate_complete
from annotate_language_cot import ChatClient, annotate_episode, validate_plan
from build_json_cot import build_episode, project_waypoints


PLAN = {
    "goal": "Lift the block", "global_plan": "Approach and lift the block",
    "subtasks": [
        {"id": 1, "name": "Approach", "start": 0, "end": 1, "planning": "Move toward the block"},
        {"id": 2, "name": "Lift", "start": 2, "end": 3, "planning": "Raise the block"},
    ],
}


class AnnotationTest(unittest.TestCase):
    def test_single_camera_calibration_reaches_episode_fields(self):
        raw = {"num_frames": 1, "proprios": {"direct": [[1, 0, 2, 0.9, 0, 0, 0]]},
               "actions": {"direct": [[0, 0, 0, 0, 0, 0, -1]]}}
        calibration = {"intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 50},
                       "image_size": [100, 100], "camera_from_world": [
                           [1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        cot = build_episode(raw, None, "libero", calibration, 2, False)
        self.assertEqual(cot["frames"][0]["Acting"]["gripper_pose"], {"cam0": [50, 50]})
        self.assertEqual(cot["frames"][0]["Acting"]["waypoint_px"], {"cam0": [[50, 50], [-1, -1]]})

    def test_camera_translation_and_moving_camera(self):
        def transform(x):
            return [[1, 0, 0, -x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        calibration = {"intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 50},
                       "image_size": [200, 100], "frames": [
                           {"camera_from_world": transform(1)},
                           {"camera_from_world": transform(2)},
                       ]}
        # EEF orientation must not be used as a homogeneous coordinate.
        poses = [[1, 0, 2, 1.7, 0, 0], [2, 0, 2, -0.2, 0, 0]]
        self.assertEqual(project_waypoints(poses, 0, 2, calibration), ([[50, 50], [100, 50]], [50, 50]))
        self.assertEqual(project_waypoints(poses, 1, 2, calibration), ([[50, 50], [-1, -1]], [50, 50]))

    def test_plan_rejects_gaps_overlaps_and_missing_reason(self):
        validate_plan(PLAN, 4)
        for field, value in [("start", 1), ("start", 3), ("end", 4), ("planning", "")]:
            invalid = copy.deepcopy(PLAN)
            invalid["subtasks"][1][field] = value
            with self.assertRaises(ValueError):
                validate_plan(invalid, 4)

    def test_real_http_request_and_language_writeback(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), payload))
                result = PLAN if len(requests) == 1 else {"subtask_reasoning": "The observed motion advances the current subtask."}
                body = json.dumps({"choices": [{"message": {"content": json.dumps(result)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "json/chunk-0000").mkdir(parents=True)
                (root / "videos").mkdir()
                video = root / "videos/episode.mp4"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
                self.assertTrue(writer.isOpened())
                for i in range(5):
                    writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
                writer.release()
                raw_path = root / "json/chunk-0000/episode_0000000.json"
                raw = {"num_frames": 4, "observations": {"base": [{"path": "videos/episode.mp4", "start": 1, "end": 5}]},
                       "proprios": {"direct": [[0, 0, 1 + i * 0.01, 0, 0, 0, 0] for i in range(4)]},
                       "actions": {"direct": [[0, 0, 0, 0, 0, 0, -1] for _ in range(4)]},
                       "instruction": {"general": [{"conversations": [{"from": "human", "value": "Lift the block"}]}]}}
                raw_path.write_text(json.dumps(raw))
                cot = build_episode(raw, None, "libero", None, 2, False)
                with patch.dict(os.environ, {"TEST_COT_API_KEY": "offline-test-key"}):
                    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "test-vision-model", "TEST_COT_API_KEY")
                    cot = annotate_episode(raw, raw_path, cot, "libero", client, 4)
                self.assertEqual(len(requests), 3)
                path, auth, request = requests[0]
                self.assertEqual(path, "/v1/chat/completions")
                self.assertEqual(auth, "Bearer offline-test-key")
                self.assertEqual(request["model"], "test-vision-model")
                images = [x for x in request["messages"][1]["content"] if x["type"] == "image_url"]
                self.assertEqual(len(images), 4)
                self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
                for t, frame in enumerate(cot["frames"]):
                    self.assertEqual(frame["Understanding"]["goal"], PLAN["goal"])
                    self.assertEqual(frame["Planning"]["subtask_name"], "Approach" if t < 2 else "Lift")
                    self.assertEqual(len(frame["Planning"]["plan"]), 2)
                    self.assertTrue(frame["Planning"]["subtask_reasoning"])
                # Missing geometry must not be reported as complete annotations.
                with self.assertRaises(ValueError):
                    validate_complete(cot, 4)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
