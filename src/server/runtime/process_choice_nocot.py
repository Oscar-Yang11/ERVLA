#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_choice_nocot.py

Build model batch for P2_Choice_KI inference in NO-COT mode only.

Input:
    instruction: str
    images: list[PIL.Image]
    state_np: np.ndarray, shape [S]

Output batch fields:
    input_ids, attention_mask, position_ids, pixel_values, image_grid_thw, ...
    state, action, action_mask
    action_segments
    action_vlm_condition_segments

Also provides postprocess:
    pred [T, D_model] -> take first 7 dims -> gaussian unnorm with config mean/std
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.transforms.load import VLMProcessor


class ChoiceNoCotProcessor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        state_shape=(1, 32),
        action_shape=(10, 32),
        action_mean=None,
        action_std=None,
    ):
        self.model_name = model_name
        self.state_length = int(state_shape[0])
        self.state_dim = int(state_shape[1])
        self.action_length = int(action_shape[0])
        self.action_dim = int(action_shape[1])

        self.vlm_processor = VLMProcessor(
            model=model_name,
            enable_choice=True,
            enable_l1=False,
        )

        if action_mean is None or action_std is None:
            raise ValueError("action_mean and action_std must be provided from model config.")

        self.action_mean = np.asarray(action_mean, dtype=np.float32).reshape(-1)
        self.action_std = np.asarray(action_std, dtype=np.float32).reshape(-1)

        if self.action_mean.shape[0] != self.action_std.shape[0]:
            raise ValueError(
                f"action_mean/action_std dim mismatch: "
                f"{self.action_mean.shape[0]} vs {self.action_std.shape[0]}"
            )
        self.exec_action_dim = int(self.action_mean.shape[0])

    def _pad_state(self, state_1d: np.ndarray) -> np.ndarray:
        """
        Incoming state from benchmark is usually flattened robot state:
            xyz + rpy + gripper  (often 7D)

        Training-side model state may be wider (e.g. 16 / 32 dims).
        We zero-pad to match model.state_shape and place current state
        in the last observation slot.
        """
        state_1d = np.asarray(state_1d, dtype=np.float32).reshape(-1)
        out = np.zeros((self.state_length, self.state_dim), dtype=np.float32)
        copy_dim = min(self.state_dim, state_1d.shape[0])
        out[-1, :copy_dim] = state_1d[:copy_dim]
        return out

    def build_conversations(self, instruction: str):
        conversations = [
            {"from": "human", "value": instruction + " /no_cot"},
            {"from": "gpt", "value": "<cot></cot>"},
        ]
        conversations, _ = self.vlm_processor.build_action_message(
            conversations,
            state_length=self.state_length,
            action_length=self.action_length,
        )
        return conversations

    def build_batch(
        self,
        instruction: str,
        images: List[Image.Image],
        state_np: np.ndarray,
        knowledge_insulation: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        conversations = self.build_conversations(instruction)

        messages = self.vlm_processor._build_messages(
            conversations,
            images=images,
            videos=[],
        )
        result = self.vlm_processor.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
        )

        position_ids = self.vlm_processor._build_position_ids(result)
        result["position_ids"] = position_ids

  
        base_conversations = conversations[:-2]
        base_messages = self.vlm_processor._build_messages(
            base_conversations,
            images=images,
            videos=[],
        )
        base_result = self.vlm_processor.processor.apply_chat_template(
            base_messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
        )
        base_len = int(base_result["input_ids"].shape[1])

        state_tensor = torch.from_numpy(self._pad_state(state_np))[None]  # [1, S_len, S_dim]
        action_tensor = torch.zeros((1, self.action_length, self.action_dim), dtype=torch.float32)

    
        action_mask = torch.zeros_like(action_tensor, dtype=torch.int32)
        action_mask[:, :, : self.exec_action_dim] = 1

        full_len = int(result["input_ids"].shape[1])

        batch = {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "position_ids": result["position_ids"],
            "pixel_values": result.get("pixel_values", None),
            "pixel_values_videos": result.get("pixel_values_videos", None),
            "image_grid_thw": result.get("image_grid_thw", None),
            "video_grid_thw": result.get("video_grid_thw", None),
            "state": state_tensor,
            "action": action_tensor,
            "action_mask": action_mask,
            "action_segments": torch.tensor([[0, full_len]], dtype=torch.int64),
            "action_vlm_condition_segments": torch.tensor([[0, base_len]], dtype=torch.int64),
        }

        meta = {
            "conversations": conversations,
            "base_len": base_len,
            "full_len": full_len,
            "knowledge_insulation": bool(knowledge_insulation),
        }
        return batch, meta

    def postprocess_action(self, pred_np: np.ndarray) -> np.ndarray:
        """
        pred_np: [T, D_model]
        Return:
            action_exec: [T, exec_action_dim]
                = take first exec_action_dim dims
                = gaussian unnorm using config mean/std
        """
        pred_np = np.asarray(pred_np, dtype=np.float32)
        if pred_np.ndim != 2:
            raise ValueError(f"Expected pred_np [T, D], got shape={pred_np.shape}")

        if pred_np.shape[1] < self.exec_action_dim:
            raise ValueError(
                f"Model output dim {pred_np.shape[1]} is smaller than exec_action_dim {self.exec_action_dim}"
            )

        action_exec = pred_np[:, : self.exec_action_dim]
        action_exec = action_exec * self.action_std[None, :] + self.action_mean[None, :]
        return action_exec