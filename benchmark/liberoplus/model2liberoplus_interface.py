#!/usr/bin/env python3

"""MiBot TCP policy adapter for the official LIBERO-Plus environment."""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import pickle
import socket
import struct
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


class ActionTCPClient:
    def __init__(self, host: str, port: int, timeout: float = 600.0) -> None:
        self.host = host
        self.port = int(port)
        self.sock = socket.create_connection((host, self.port), timeout=timeout)
        self.sock.settimeout(timeout)
        LOGGER.info("Connected to MiBot server at %s:%d", host, self.port)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _recv_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            packet = self.sock.recv(size - len(data))
            if not packet:
                raise ConnectionError("Server closed the socket while sending a response")
            data.extend(packet)
        return bytes(data)

    def predict_chunk(self, payload: Dict[str, Any]) -> np.ndarray:
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(raw)) + raw)
        response_size = struct.unpack(">I", self._recv_exact(4))[0]
        response = pickle.loads(self._recv_exact(response_size))
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"MiBot server inference failed: {response['error']}")
        if isinstance(response, dict) and "action_chunk" in response:
            response = response["action_chunk"]
        return np.asarray(response, dtype=np.float32)


def _quat2axisangle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / denominator).astype(
        np.float32
    )


def _image_to_pil(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


@dataclasses.dataclass
class LiberoPlusPolicyArgs:
    host: str = "127.0.0.1"
    port: int = 10086
    suite: str = "libero_spatial"
    action_chunk_size: int = 10
    replan_steps: int = 5
    seed: int = 42
    socket_timeout: float = 600.0


class MiBotLiberoPlusPolicy:
    def __init__(self, args: LiberoPlusPolicyArgs) -> None:
        if args.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if not 0 < args.replan_steps <= args.action_chunk_size:
            raise ValueError("replan_steps must be in [1, action_chunk_size]")
        self.args = args
        self.client = ActionTCPClient(
            args.host, args.port, timeout=args.socket_timeout
        )
        self._plan: Deque[np.ndarray] = collections.deque()
        self.request_count = 0
        self.last_chunk_shape: Optional[List[int]] = None
        self.first_action: Optional[List[float]] = None

    def close(self) -> None:
        self.client.close()

    def reset(self) -> None:
        self._plan.clear()
        self.request_count = 0
        self.last_chunk_shape = None
        self.first_action = None

    @staticmethod
    def model_state(observation: Dict[str, Any]) -> np.ndarray:
        state = np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
                _quat2axisangle(observation["robot0_eef_quat"]),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        if state.shape != (8,):
            raise ValueError(f"Expected LIBERO state shape (8,), got {state.shape}")
        return state

    @staticmethod
    def model_images(observation: Dict[str, Any]) -> List[Image.Image]:
        # LIBERO's camera arrays are rotated 180 degrees in the training export.
        base = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        return [_image_to_pil(base), _image_to_pil(wrist)]

    def _request_chunk(
        self, observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        normalized_instruction = str(instruction).capitalize()
        payload = {
            "instruction": normalized_instruction,
            "raw_instruction": normalized_instruction,
            "state": self.model_state(observation),
            "images": self.model_images(observation),
            "view_names": ["Base View", "Left-Wrist View"],
            "task_id": self.args.suite,
            "robot_type": self.args.suite,
            "action_chunk_size": self.args.action_chunk_size,
            "seed": self.args.seed,
            "cot": False,
            "use_cot": False,
            "prompt_mode": "no_cot",
            "assistant_prefill": "<cot></cot>",
        }
        chunk = self.client.predict_chunk(payload)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise RuntimeError(f"Expected action chunk [T, 7], got {chunk.shape}")
        if chunk.shape[0] < self.args.replan_steps:
            raise RuntimeError(
                f"Server returned {chunk.shape[0]} actions; "
                f"need at least {self.args.replan_steps}"
            )
        if not np.isfinite(chunk).all():
            raise RuntimeError("Server returned NaN or infinite actions")

        self.request_count += 1
        self.last_chunk_shape = list(chunk.shape)
        if self.first_action is None:
            self.first_action = chunk[0].astype(float).tolist()
        return chunk.astype(np.float32)

    def predict(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        if not self._plan:
            chunk = self._request_chunk(observation, instruction)
            self._plan.extend(chunk[: self.args.replan_steps])
        action = np.asarray(self._plan.popleft(), dtype=np.float32)
        if action.shape != (7,):
            raise RuntimeError(f"Expected executable action shape (7,), got {action.shape}")
        return action

