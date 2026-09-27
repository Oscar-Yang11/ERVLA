#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Serve a local Hugging Face MiBoT checkpoint over the choice TCP protocol.

The VLABench choice evaluator sends raw images, instruction text, and a seven
dimensional state.  Hugging Face MiBoT checkpoints expect processor-produced
inputs, a padded state/action mask, and return an ActionGenerationOutput.  This
adapter keeps the evaluator-facing protocol stable while translating those
inputs and outputs on the server.
"""

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
DEFAULT_STATS_CONFIG = REPO_ROOT / "configs/data/sources/vlabench_choice.yaml"


def _recv_all(conn: socket.socket, length: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def _send_with_length_prefix(conn: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


def _flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Action statistics cannot be empty")
    return array.reshape(-1)


def _load_yaml_action_stats(path: Path, robot_type: str) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    source = (config or {}).get("sources", {}).get(robot_type)
    if not isinstance(source, dict):
        raise KeyError(f"Source {robot_type!r} not found in {path}")

    try:
        mean = _flatten_numeric(source["mean"]["action_direct"])
        std = _flatten_numeric(source["std"]["action_direct"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing action_direct mean/std in {path}") from exc

    if mean.shape != std.shape:
        raise ValueError(f"Action mean/std shape mismatch in {path}: {mean.shape} vs {std.shape}")
    if mean.size < 7 or np.any(std[:7] <= 1e-8):
        raise ValueError(f"Invalid first seven action std values in {path}: {std[:7]}")
    return mean[:7], std[:7]


def _processor_action_stats(processor: Any, robot_type: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    action_config = getattr(processor, "action_config", {})
    config = action_config.get(robot_type)
    if config is None:
        return None

    try:
        mean = _flatten_numeric(config["mean"].detach().cpu().numpy())
        std = _flatten_numeric(config["std"].detach().cpu().numpy())
    except (AttributeError, KeyError, TypeError) as exc:
        LOGGER.warning("Could not read action stats from processor: %s", exc)
        return None

    if mean.shape != std.shape or mean.size < 7:
        return None
    if np.any(std[:7] <= 1e-5):
        LOGGER.warning(
            "Processor action stats for %s look like placeholders; first std values=%s",
            robot_type,
            std[:7].tolist(),
        )
        return None
    return mean[:7], std[:7]


def _repeat_or_trim(values: np.ndarray, length: int) -> np.ndarray:
    if values.shape[0] == length:
        return values
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0)
    if values.shape[0] > length:
        return values[:length]
    tail = np.repeat(values[-1:], length - values.shape[0], axis=0)
    return np.concatenate([values, tail], axis=0)


class HFChoiceServer:
    def __init__(
        self,
        *,
        model_path: str,
        host: str,
        port: int,
        device: str = "cuda:0",
        robot_type: str = "vlabench_choice",
        num_steps: int = 5,
        action_stats_config: Optional[str] = str(DEFAULT_STATS_CONFIG),
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        self.host = host
        self.port = int(port)
        self.device = torch.device(device)
        self.robot_type = robot_type
        self.num_steps = int(num_steps)
        self.request_count = 0

        LOGGER.info("Loading HF processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        robot_types = list(self.processor.list_robot_types())
        if robot_type not in robot_types:
            raise ValueError(
                f"Robot type {robot_type!r} is not available in the HF processor. "
                f"Available: {robot_types}"
            )

        LOGGER.info("Loading HF model from %s", model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

        config = self.model.config
        self.state_dim = int(getattr(config, "state_dim", 60))
        self.state_length = int(getattr(config, "state_length", 1))
        self.action_dim = int(getattr(config, "action_dim", 60))

        processor_stats = _processor_action_stats(self.processor, robot_type)
        stats_source = "processor"
        stats = processor_stats
        if stats is None and action_stats_config:
            stats_path = Path(action_stats_config).expanduser()
            if stats_path.is_file():
                stats = _load_yaml_action_stats(stats_path, robot_type)
                stats_source = str(stats_path)
            else:
                raise FileNotFoundError(
                    f"Action stats config not found and HF processor stats are invalid: {stats_path}"
                )

        if stats is None:
            raise ValueError(
                "No valid action statistics found. Provide --action-stats-config or a HF processor "
                "with non-placeholder vlabench_choice statistics."
            )
        action_mean_7, action_std_7 = stats

        self.default_action_length = 10
        self.action_mean = torch.zeros(
            (1, self.default_action_length, self.action_dim), dtype=torch.float32
        )
        self.action_std = torch.full(
            (1, self.default_action_length, self.action_dim), 1e-6, dtype=torch.float32
        )
        copy_dim = min(7, self.action_dim)
        self.action_mean[:, :, :copy_dim] = torch.from_numpy(action_mean_7[:copy_dim]).view(1, 1, copy_dim)
        self.action_std[:, :, :copy_dim] = torch.from_numpy(action_std_7[:copy_dim]).view(1, 1, copy_dim)

        self.active_action_dim = copy_dim
        self.active_action_mask = torch.zeros_like(self.action_mean)
        self.active_action_mask[:, :, :copy_dim] = 1.0

        LOGGER.info(
            "HF server ready: device=%s dtype=%s state=[%d,%d] action=[%d,%d] "
            "robot_type=%s stats=%s mean7=%s std7=%s",
            self.device,
            self.model_dtype,
            self.state_length,
            self.state_dim,
            self.default_action_length,
            self.action_dim,
            robot_type,
            stats_source,
            action_mean_7.tolist(),
            action_std_7.tolist(),
        )

    @staticmethod
    def _build_messages(images: List[Image.Image], instruction: str) -> List[Dict[str, Any]]:
        if len(images) < 3:
            raise ValueError(f"Expected three camera images, got {len(images)}")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "\n# Base View\n"},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": images[2]},
            {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ]
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def _build_state(self, state_np: Any) -> torch.Tensor:
        state = np.asarray(state_np, dtype=np.float32).reshape(-1)
        padded = np.zeros((self.state_length, self.state_dim), dtype=np.float32)
        copy_dim = min(self.state_dim, state.size)
        padded[-1, :copy_dim] = state[:copy_dim]
        return torch.from_numpy(padded)[None]

    def _build_action_tensors(
        self, action_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = _repeat_or_trim(self.action_mean[0].numpy(), action_length)
        std = _repeat_or_trim(self.action_std[0].numpy(), action_length)
        mask = _repeat_or_trim(self.active_action_mask[0].numpy(), action_length)
        return (
            torch.from_numpy(mean)[None],
            torch.from_numpy(std)[None],
            torch.from_numpy(mask)[None],
        )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(self.device, dtype=self.model_dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    @torch.no_grad()
    def predict(self, payload: Dict[str, Any]) -> np.ndarray:
        instruction = str(payload.get("raw_instruction", payload.get("instruction", "")))
        images = payload.get("images")
        if not isinstance(images, list):
            raise ValueError("Payload field 'images' must be a list")

        requested_length = int(payload.get("action_chunk_size", self.default_action_length))
        if requested_length <= 0:
            raise ValueError(f"action_chunk_size must be positive, got {requested_length}")

        messages = self._build_messages(images, instruction)
        state = self._build_state(payload.get("state"))
        _, _, action_mask = self._build_action_tensors(requested_length)

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state,
            robot_type=self.robot_type,
        )
        model_inputs = dict(model_inputs)
        model_inputs["state"] = state
        model_inputs["action_mask"] = action_mask
        model_inputs["seed"] = int(payload.get("seed", 42))
        model_inputs["num_steps"] = self.num_steps
        model_inputs = self._move_batch_to_device(model_inputs)

        output = self.model(**model_inputs)
        actions = getattr(output, "actions", None)
        if actions is None:
            raise RuntimeError(f"HF model output has no 'actions' field: {type(output)!r}")
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError(f"Expected HF actions with shape [1,T,D], got {tuple(actions.shape)}")
        if actions.shape[-1] < self.active_action_dim:
            raise RuntimeError(
                f"HF action dimension {actions.shape[-1]} is smaller than {self.active_action_dim}"
            )

        action_mean, action_std, _ = self._build_action_tensors(actions.shape[1])
        action_mean = action_mean.to(actions.device)
        action_std = action_std.to(actions.device)
        decoded = actions.float() * action_std + action_mean
        action_chunk = decoded[0, :, : self.active_action_dim].cpu().numpy().astype(np.float32)

        self.request_count += 1
        if self.request_count == 1:
            LOGGER.info(
                "First HF inference: raw_shape=%s action_shape=%s first_action=%s",
                tuple(actions.shape),
                tuple(action_chunk.shape),
                action_chunk[0].tolist(),
            )
        return action_chunk

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(16)
            LOGGER.info("HF choice server listening on %s:%d", self.host, self.port)

            while True:
                conn, address = server_socket.accept()
                LOGGER.info("Accepted connection from %s", address)
                try:
                    while True:
                        header = _recv_all(conn, 4)
                        if header is None:
                            break
                        payload_size = struct.unpack(">I", header)[0]
                        raw_payload = _recv_all(conn, payload_size)
                        if raw_payload is None:
                            break
                        try:
                            payload = pickle.loads(raw_payload)
                            action_chunk = self.predict(payload)
                            response = {
                                "action_chunk": action_chunk.tolist(),
                                "shape": list(action_chunk.shape),
                                "dtype": str(action_chunk.dtype),
                            }
                        except Exception as exc:
                            LOGGER.error("Inference request failed: %s", exc)
                            LOGGER.debug(traceback.format_exc())
                            response = {"error": f"{type(exc).__name__}: {exc}"}
                        _send_with_length_prefix(conn, response)
                finally:
                    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-type", default="vlabench_choice")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--action-stats-config", default=str(DEFAULT_STATS_CONFIG))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    server = HFChoiceServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        robot_type=args.robot_type,
        num_steps=args.num_steps,
        action_stats_config=args.action_stats_config,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
