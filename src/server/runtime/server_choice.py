#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
server_choice.py

Thin TCP server wrapper around a loaded P2_Choice_KI model.
It:
  - receives request payload
  - calls process_choice_nocot.py to build a batch
  - runs model(batch, return_loss=False)
  - postprocesses output: [:, :7] + unnorm from config
  - returns continuous executable action chunk [T, 7]

Request payload:
{
    "instruction": str,
    "state": np.ndarray,
    "images": List[PIL.Image],
    "view_names": List[str],   # accepted but unused
    "task_id": str,            # accepted but unused
}

Response:
    np.ndarray [T, 7]
"""

import logging
import pickle
import socket
import struct
from typing import Any, Dict

import numpy as np
import torch

from .process_choice_nocot import ChoiceNoCotProcessor

logger = logging.getLogger(__name__)


def recv_with_length_prefix(conn):
    len_buf = conn.recv(4)
    if not len_buf or len(len_buf) < 4:
        raise ConnectionError("Failed to receive request length.")
    n = struct.unpack(">I", len_buf)[0]

    buf = b""
    while len(buf) < n:
        pkt = conn.recv(n - len(buf))
        if not pkt:
            raise ConnectionError("Socket closed while receiving request.")
        buf += pkt
    return pickle.loads(buf)


def send_with_length_prefix(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


class ChoiceTCPServer:
    def __init__(
        self,
        model,
        host: str,
        port: int,
        device: torch.device,
        model_name: str,
        action_mean,
        action_std,
    ):
        self.model = model
        self.host = host
        self.port = int(port)
        self.device = device

        state_shape = tuple(getattr(self.model, "state_shape", (1, 32)))
        action_shape = tuple(getattr(self.model, "action_shape", (10, 32)))
        knowledge_insulation = bool(getattr(self.model, "knowledge_insulation", True))

        self.processor = ChoiceNoCotProcessor(
            model_name=model_name,
            state_shape=state_shape,
            action_shape=action_shape,
            action_mean=action_mean,
            action_std=action_std,
        )
        self.knowledge_insulation = knowledge_insulation

        logger.info(
            "ChoiceTCPServer ready: host=%s port=%d state_shape=%s action_shape=%s KI=%s exec_action_dim=%d",
            self.host,
            self.port,
            state_shape,
            action_shape,
            self.knowledge_insulation,
            self.processor.exec_action_dim,
        )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def predict(self, payload: Dict[str, Any]) -> np.ndarray:
        instruction = str(payload["instruction"])
        images = payload["images"]
        state_np = np.asarray(payload["state"], dtype=np.float32)

        batch, meta = self.processor.build_batch(
            instruction=instruction,
            images=images,
            state_np=state_np,
            knowledge_insulation=self.knowledge_insulation,
        )
        batch = self._move_batch_to_device(batch)

        logger.info(
            "Infer request: input_len=%d base_len=%d state=%s",
            meta["full_len"], meta["base_len"], tuple(batch["state"].shape)
        )

        pred = self.model(batch, return_loss=False)

        if not torch.is_tensor(pred):
            raise RuntimeError(f"Expected tensor from model(..., return_loss=False), got {type(pred)}")
        if pred.ndim != 3:
            raise RuntimeError(f"Expected pred shape [B,T,D], got {tuple(pred.shape)}")

        pred_np = pred[0].detach().float().cpu().numpy()   # [T, D_model]

        # print("\n" + "=" * 120)
        # print("RAW MODEL OUTPUT DEBUG")
        # print("=" * 120)
        # print("raw pred shape:", pred_np.shape)   

        # print("\nraw pred[0, :16]:")
        # print(pred_np[0, :16])

        # print("\nraw pred[-1, :16]:")
        # print(pred_np[-1, :16])

        head = pred_np[:, :7]
        tail = pred_np[:, 7:]

        # print("\nhead shape (used for execution before unnorm):", head.shape)
        # print("tail shape (unused by current postprocess):", tail.shape)

        # print("\nhead first row:")
        # print(head[0])

        if tail.shape[1] > 0:
            print("\ntail first row:")
            print(tail[0])

            print("\ntail abs mean:", float(np.abs(tail).mean()))
            print("tail abs max :", float(np.abs(tail).max()))

            near_zero_ratio = float((np.abs(tail) < 1e-3).mean())
            print("tail near-zero ratio (|x| < 1e-3):", near_zero_ratio)

        action_chunk = self.processor.postprocess_action(pred_np)  # [T, exec_action_dim]

        print("\nPOSTPROCESSED ACTION CHUNK shape:", action_chunk.shape)
        print("postprocessed first row:", action_chunk[0])
        print("postprocessed last  row:", action_chunk[-1])
        print("=" * 120 + "\n")

        return action_chunk

    def serve_forever(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(16)
        logger.info("Choice server listening on %s:%d", self.host, self.port)

        try:
            while True:
                conn, addr = sock.accept()
                logger.info("Accepted connection from %s", addr)
                try:
                    while True:
                        payload = recv_with_length_prefix(conn)
                        out = self.predict(payload)
                        send_with_length_prefix(conn, {
                            "action_chunk": out.tolist(),
                            "shape": list(out.shape),
                            "dtype": str(out.dtype),
                        })
                except Exception as e:
                    logger.exception("Connection ended: %s", e)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
        finally:
            sock.close()