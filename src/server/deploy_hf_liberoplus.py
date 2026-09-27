#!/usr/bin/env python3

"""Serve a local Hugging Face MiBot checkpoint for LIBERO-Plus evaluation."""

from __future__ import annotations

import argparse
import logging
import pickle
import socket
import struct
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoModel, AutoProcessor


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_CONFIG = REPO_ROOT / "configs/data/sources/libero.yaml"
LIBERO_ROBOT_TYPES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
}


def _recv_all(connection: socket.socket, size: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < size:
        packet = connection.recv(size - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def _send(connection: socket.socket, value: Any) -> None:
    raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack(">I", len(raw)) + raw)


def _flatten(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Action statistics cannot be empty")
    return array


def _load_action_stats(
    config_path: Path, source_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a mapping in action statistics config {config_path}, "
            f"got {type(config).__name__}"
        )
    # MiBot uses both Hydra-style ``sources.<name>`` files and source files
    # where the named dataset is the top-level mapping.
    sources = config.get("sources", config)
    source = sources.get(source_name) if isinstance(sources, dict) else None
    if not isinstance(source, dict):
        raise KeyError(f"Source {source_name!r} not found in {config_path}")
    try:
        mean = _flatten(source["mean"]["action_direct"])
        std = _flatten(source["std"]["action_direct"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing action_direct statistics in {config_path}") from exc
    if mean.shape != std.shape or mean.size < 7:
        raise ValueError(
            f"Invalid action statistics shapes: mean={mean.shape}, std={std.shape}"
        )
    if np.any(std[:7] <= 1e-8):
        raise ValueError(f"First seven action std values must be positive: {std[:7]}")
    return mean[:7], std[:7]


class HFLiberoPlusServer:
    def __init__(
        self,
        *,
        model_path: str,
        host: str,
        port: int,
        device: str,
        action_stats_config: str,
        action_stats_source: str,
        num_steps: int,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.host = host
        self.port = int(port)
        self.device = torch.device(device)
        self.num_steps = int(num_steps)
        self.request_count = 0

        LOGGER.info("Loading processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        processor_robot_types = set(self.processor.list_robot_types())
        missing_robot_types = sorted(LIBERO_ROBOT_TYPES - processor_robot_types)
        if missing_robot_types:
            raise ValueError(
                "HF processor is missing LIBERO robot types: "
                f"{missing_robot_types}; available={sorted(processor_robot_types)}"
            )

        LOGGER.info("Loading model from %s", model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.state_dim = int(getattr(self.model.config, "state_dim", 60))
        self.state_length = int(getattr(self.model.config, "state_length", 1))
        self.action_dim = int(getattr(self.model.config, "action_dim", 60))
        if self.state_dim < 8 or self.action_dim < 7:
            raise ValueError(
                f"Model dimensions are too small: state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}"
            )

        stats_path = Path(action_stats_config).expanduser().resolve()
        self.action_mean, self.action_std = _load_action_stats(
            stats_path, action_stats_source
        )
        LOGGER.info(
            "HF LIBERO-Plus server initialized: device=%s dtype=%s "
            "state=[%d,%d] action_dim=%d stats=%s:%s mean=%s std=%s",
            self.device,
            self.model_dtype,
            self.state_length,
            self.state_dim,
            self.action_dim,
            stats_path,
            action_stats_source,
            self.action_mean.tolist(),
            self.action_std.tolist(),
        )

    @staticmethod
    def _messages(images: List[Image.Image], instruction: str) -> List[Dict[str, Any]]:
        if len(images) != 2:
            raise ValueError(f"Expected Base and Left-Wrist images, got {len(images)}")
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": "The following observations are captured from multiple views.\n# Base View\n",
            },
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": images[1]},
            {
                "type": "text",
                "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
            },
        ]
        return [
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def _state(self, value: Any) -> torch.Tensor:
        raw = np.asarray(value, dtype=np.float32).reshape(-1)
        if raw.size != 8:
            raise ValueError(f"Expected eight-dimensional LIBERO state, got {raw.shape}")
        padded = np.zeros((self.state_length, self.state_dim), dtype=np.float32)
        padded[-1, :8] = raw
        return torch.from_numpy(padded)[None]

    def _action_mask(self, action_length: int) -> torch.Tensor:
        if action_length <= 0:
            raise ValueError("action_chunk_size must be positive")
        mask = torch.zeros(
            (1, action_length, self.action_dim), dtype=torch.float32
        )
        mask[:, :, :7] = 1.0
        return mask

    def _move_to_device(self, values: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                dtype = self.model_dtype if value.is_floating_point() else None
                moved[key] = value.to(self.device, dtype=dtype)
            else:
                moved[key] = value
        return moved

    @torch.no_grad()
    def predict(self, payload: Dict[str, Any]) -> np.ndarray:
        instruction = str(payload.get("raw_instruction", payload.get("instruction", "")))
        if not instruction:
            raise ValueError("Instruction is empty")
        images = payload.get("images")
        if not isinstance(images, list):
            raise ValueError("Payload field 'images' must be a list")
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("All payload images must be PIL images")

        robot_type = str(
            payload.get("robot_type", payload.get("task_id", "libero_spatial"))
        )
        if robot_type not in LIBERO_ROBOT_TYPES:
            raise ValueError(
                f"Unsupported LIBERO robot type {robot_type!r}; "
                f"available={sorted(LIBERO_ROBOT_TYPES)}"
            )
        action_length = int(payload.get("action_chunk_size", 10))
        state = self._state(payload.get("state"))
        action_mask = self._action_mask(action_length)

        model_inputs = self.processor.apply_chat_template(
            self._messages(images, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state,
            robot_type=robot_type,
        )
        model_inputs = dict(model_inputs)
        model_inputs["state"] = state
        model_inputs["action_mask"] = action_mask
        model_inputs["seed"] = int(payload.get("seed", 42))
        model_inputs["num_steps"] = self.num_steps
        model_inputs = self._move_to_device(model_inputs)

        output = self.model(**model_inputs)
        actions = getattr(output, "actions", None)
        if actions is None:
            raise RuntimeError(f"Model output has no actions field: {type(output)!r}")
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError(f"Expected model actions [1,T,D], got {tuple(actions.shape)}")
        if actions.shape[-1] < 7:
            raise RuntimeError(f"Model action dimension is too small: {actions.shape[-1]}")

        mean = torch.as_tensor(self.action_mean, device=actions.device).view(1, 1, 7)
        std = torch.as_tensor(self.action_std, device=actions.device).view(1, 1, 7)
        decoded = actions[:, :, :7].float() * std + mean
        chunk = decoded[0].cpu().numpy().astype(np.float32)
        if not np.isfinite(chunk).all():
            raise RuntimeError("Decoded action chunk contains NaN or infinity")

        self.request_count += 1
        LOGGER.info(
            "Inference request=%d robot_type=%s raw_shape=%s decoded_shape=%s "
            "first_action=%s",
            self.request_count,
            robot_type,
            tuple(actions.shape),
            tuple(chunk.shape),
            chunk[0].tolist(),
        )
        return chunk

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(16)
            LOGGER.info("HF LIBERO-Plus server listening on %s:%d", self.host, self.port)
            while True:
                connection, address = server.accept()
                LOGGER.info("Accepted connection from %s", address)
                try:
                    while True:
                        header = _recv_all(connection, 4)
                        if header is None:
                            break
                        payload_size = struct.unpack(">I", header)[0]
                        raw_payload = _recv_all(connection, payload_size)
                        if raw_payload is None:
                            break
                        try:
                            chunk = self.predict(pickle.loads(raw_payload))
                            response = {
                                "action_chunk": chunk.tolist(),
                                "shape": list(chunk.shape),
                                "dtype": str(chunk.dtype),
                            }
                        except Exception as exc:
                            LOGGER.error("Inference failed: %s", exc)
                            LOGGER.debug(traceback.format_exc())
                            response = {"error": f"{type(exc).__name__}: {exc}"}
                        _send(connection, response)
                finally:
                    connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-stats-config", default=str(DEFAULT_STATS_CONFIG))
    parser.add_argument("--action-stats-source", default="libero_all")
    parser.add_argument("--num-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    server = HFLiberoPlusServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        action_stats_config=args.action_stats_config,
        action_stats_source=args.action_stats_source,
        num_steps=args.num_steps,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
