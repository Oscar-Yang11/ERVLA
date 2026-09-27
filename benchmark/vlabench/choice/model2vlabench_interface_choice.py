#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import dataclasses
import logging
import pickle
import socket
import struct
from typing import Any, Dict, Tuple, Deque, List, Optional

import numpy as np
from PIL import Image

from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler


logger = logging.getLogger(__name__)


class ChoiceTCPClient:
    """Thin TCP client for the legacy and HF choice servers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10086):
        self.host = host
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        logger.info("Connected to Choice server at %s:%d", self.host, self.port)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def _send_with_length_prefix(self, obj: Dict[str, Any]) -> None:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(data)) + data)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            pkt = self.sock.recv(n - len(buf))
            if not pkt:
                raise ConnectionError("Socket closed while receiving response.")
            buf.extend(pkt)
        return bytes(buf)

    def _recv_with_length_prefix(self):
        n = struct.unpack(">I", self._recv_exact(4))[0]
        return pickle.loads(self._recv_exact(n))

    def predict_chunk(
        self,
        *,
        instruction: str,
        state: np.ndarray,
        images: List[Image.Image],
        view_names: List[str],
        task_id: str,
        action_chunk_size: int,
        raw_instruction: Optional[str] = None,
    ) -> np.ndarray:
        payload = {
            "instruction": str(instruction),
            "raw_instruction": str(raw_instruction if raw_instruction is not None else instruction),
            "state": np.asarray(state, dtype=np.float32),
            "images": images,
            "view_names": list(view_names),
            "task_id": str(task_id),

            "cot": False,
            "use_cot": False,
            "prompt_mode": "no_cot",
            "instruction_with_mode": f"{instruction} /no_cot",
            "action_chunk_size": int(action_chunk_size),
            "input_conversations": [
                {"from": "human", "value": f"{instruction} /no_cot"},
                {"from": "gpt", "value": "<cot></cot>"},
            ],
            "assistant_prefill": "<cot></cot>",
        }

        self._send_with_length_prefix(payload)
        out = self._recv_with_length_prefix()

        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"Choice server inference failed: {out['error']}")

        # New robust protocol: server returns pure-python dict
        if isinstance(out, dict) and "action_chunk" in out:
            return np.asarray(out["action_chunk"], dtype=np.float32)

        # Backward compatibility with old server returning ndarray directly
        return np.asarray(out, dtype=np.float32)


def _to_uint8(img: np.ndarray) -> Image.Image:
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray image, got {type(img)}")
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image with C in 1/3/4, got shape={img.shape}")

    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0 + 0.5).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.shape[2] == 1:
        return Image.fromarray(img[:, :, 0], mode="L").convert("RGB")

    mode = "RGB" if img.shape[2] == 3 else "RGBA"
    return Image.fromarray(img, mode=mode).convert("RGB")


def _resize_image(img: Image.Image, image_size: int) -> Image.Image:
    if img.size == (image_size, image_size):
        return img
    return img.resize((image_size, image_size))


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _extract_raw_ee(obs: Dict[str, Any]):
    if "ee_state" not in obs:
        raise KeyError(f"obs missing 'ee_state'. Available keys: {list(obs.keys())}")

    ee = np.asarray(obs["ee_state"], dtype=np.float32).reshape(-1)
    if ee.size < 8:
        raise ValueError(f"obs['ee_state'] expected length>=8 (pos3+quat4+gripper), got {ee.size}")

    pos_world = ee[:3]
    quat = ee[3:7]
    raw_gripper = float(ee[7])

    euler_world = np.asarray(quaternion_to_euler(quat), dtype=np.float32).reshape(-1)[:3]
    euler_world = _wrap_to_pi(euler_world)

    return pos_world.astype(np.float32), euler_world.astype(np.float32), raw_gripper


def _extract_state_for_model(obs: Dict[str, Any], pos_offset: np.ndarray) -> np.ndarray:
    pos_world, euler_world, raw_gripper = _extract_raw_ee(obs)
    pos_robot = pos_world - pos_offset

    gripper_state = raw_gripper

    state_model = np.concatenate(
        [pos_robot, euler_world, np.asarray([gripper_state], dtype=np.float32)],
        axis=0,
    )
    return state_model.astype(np.float32)


def _extract_state_for_execution(obs: Dict[str, Any]) -> np.ndarray:
    pos_world, euler_world, raw_gripper = _extract_raw_ee(obs)

    gripper_state = raw_gripper

    state_exec = np.concatenate(
        [pos_world, euler_world, np.asarray([gripper_state], dtype=np.float32)],
        axis=0,
    )
    return state_exec.astype(np.float32)


@dataclasses.dataclass
class MibotChoicePolicyArgs:
    host: str = "127.0.0.1"
    port: int = 10086
    task_id: str = "vlabench_choice"

    action_chunk_size: int = 10
    replan_steps: int = 5
    image_size: int = 480

    pos_offset_x: float = 0.0
    pos_offset_y: float = -0.4
    pos_offset_z: float = 0.78

    gripper_threshold: float = 0.1
    gripper_open_value: float = 0.04


class MibotQwenChoiceVLABenchPolicy(Policy):
    def __init__(self, args: MibotChoicePolicyArgs):
        super().__init__(model=None)
        self.args = args

        if args.replan_steps > args.action_chunk_size:
            raise ValueError("replan_steps must be <= action_chunk_size")

        self.client = ChoiceTCPClient(args.host, args.port)
        self.plan_horizon = int(args.replan_steps)

        self._plan: Deque[Tuple[np.ndarray, np.ndarray, float]] = collections.deque(maxlen=self.plan_horizon)
        self.pos_offset = np.asarray(
            [args.pos_offset_x, args.pos_offset_y, args.pos_offset_z],
            dtype=np.float32,
        )

    @property
    def name(self) -> str:
        return "mibot_qwen_choice"

    def reset(self):
        self._plan.clear()

    def _build_payload(self, obs: Dict[str, Any]):
        second_img, _, front_img, wrist_img = obs["rgb"]

        images = [
            _resize_image(_to_uint8(front_img), self.args.image_size),
            _resize_image(_to_uint8(second_img), self.args.image_size),
            _resize_image(_to_uint8(wrist_img), self.args.image_size),
        ]
        view_names = ["Ego View", "Base View", "Left-Wrist View"]

        raw_instruction = str(obs["instruction"])
        instruction = (
            "The following observations are captured from multiple views.\n"
            "# Ego View\n<image>\n"
            "# Base View\n<image>\n"
            "# Left-Wrist View\n<image>\n"
            "Generate robot actions for the task:\n"
            f"{raw_instruction}"
        )

        state_model = _extract_state_for_model(obs, self.pos_offset)
        return instruction, raw_instruction, state_model, images, view_names

    def _request_action_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        instruction, raw_instruction, state_model, images, view_names = self._build_payload(obs)
        chunk = self.client.predict_chunk(
            instruction=instruction,
            raw_instruction=raw_instruction,
            state=state_model,
            images=images,
            view_names=view_names,
            task_id=self.args.task_id,
            action_chunk_size=self.args.action_chunk_size,
        )

        if chunk.ndim == 1:
            if chunk.shape[0] != 7:
                raise RuntimeError(f"Expected single action dim == 7, got shape={chunk.shape}")
            chunk = chunk[None, :]

        if chunk.ndim != 2:
            raise RuntimeError(f"Expected action chunk to be 2D [T, D], got shape={chunk.shape}")
        if chunk.shape[1] != 7:
            raise RuntimeError(f"Expected action dim == 7, got shape={chunk.shape}")
        if chunk.shape[0] < self.plan_horizon:
            raise RuntimeError(
                f"Server returned T={chunk.shape[0]} < required_plan_horizon={self.plan_horizon}"
            )
        return chunk.astype(np.float32)

    def predict(self, obs, **kwargs):
        if len(self._plan) == 0:
            chunk = self._request_action_chunk(obs)
            cur = _extract_state_for_execution(obs).copy()

            for t in range(self.plan_horizon):
                a = np.asarray(chunk[t], dtype=np.float32)

                cur[:6] = cur[:6] + a[:6]
                cur[3:6] = _wrap_to_pi(cur[3:6])

                g = float(a[6])
                self._plan.append((cur[:3].copy(), cur[3:6].copy(), g))

        pos_world, euler, gripper_scalar = self._plan.popleft()

        if gripper_scalar >= self.args.gripper_threshold:
            gripper_state = np.ones(2, dtype=np.float32) * self.args.gripper_open_value
        else:
            gripper_state = np.zeros(2, dtype=np.float32)

        return np.asarray(pos_world, dtype=np.float32), np.asarray(euler, dtype=np.float32), gripper_state
