import fnmatch
import json
import os
import random
import re
import warnings
from copy import deepcopy
from typing import Sequence, Literal

import numpy as np
import torch

from decord import VideoReader
from mmengine import TRANSFORMS
from PIL import Image
from transformers import AutoProcessor

from src.data.transforms.utils import (
    get_rope_index_25,
    get_rope_index_3,
    get_value,
    IGNORE_INDEX,
    load_json,
    pil_to_base64,
    rotm2aa_batch,
)

##### P2_L1 & P2_Choice #####
SCORE_ID=151669 
STATE_ID=151670
ACTION_MAX_LENGTH=60


@TRANSFORMS.register_module()
class LoadData:
    def __init__(
        self,
        meta_info="",
        dataset_action_length=30,
        general_task_prob=0.0,
        subtask_task_prob=1.0,
        add_sides_prob=0.0,
        add_cot_prob=0.0,
    ):
        self.global_obs_length = int(os.environ.get("_obs_length", 1))
        self.global_obs_interval = int(os.environ.get("_obs_interval", 1))
        self.add_meta_to_prompt = os.environ.get("_add_meta_to_prompt", "False") == "True"

        self.meta_info = meta_info
        self.dataset_action_length = dataset_action_length
        self.offsets = np.arange(-self.global_obs_interval * (self.global_obs_length - 1), 1, self.global_obs_interval)

        # compute general / subtask probability
        self.task_name_list = ["general", "subtask"]
        self.general_task_prob = general_task_prob
        self.subtask_task_prob = subtask_task_prob
        self.task_prob_list = np.array([self.general_task_prob, self.subtask_task_prob])
        assert np.all(self.task_prob_list >= 0.0) and np.any(self.task_prob_list > 0.0)
        self.task_prob_list /= np.sum(self.task_prob_list)
        self.task_prob_list = np.cumsum(self.task_prob_list)

        # add sides prompt probability
        self.add_sides_prob = add_sides_prob

        # add-cot-to-prompt probability
        self.add_cot_prob = add_cot_prob

    def __call__(self, sample):
        if "frame_index" in sample:
            json_file = load_json(sample["file"])
            sample.update(json_file)
            sample["is_chunk"] = False
        else:
            sample["frame_index"] = 0
            sample["is_chunk"] = True

        prompt = self.get_prompt(sample)
        prompt_images = prompt.pop("images")

        all_media = []
        for prompt_image in prompt_images:
            if prompt_image in sample:
                tmp_media = [sample[prompt_image]]
            else:
                video_info = get_value(sample, prompt_image)
                if video_info is None:
                    raise ValueError(f"{prompt_image} is missing in {sample['file']}")
                tmp_media = self.get_frames(video_info=video_info, frame_index=sample["frame_index"])
            all_media.extend(tmp_media)

        flattened_media = []
        for view in all_media:
            for image in view:
                flattened_media.append(image)
    
        sample["text"] = prompt
        sample["action_mask"], sample["ideal_length"], sample["actual_length"] = self.get_action_mask(sample)
        sample["media"] = flattened_media

        return sample

    def get_prompt(self, sample):
        rand_value = np.random.random()
        for task_name, prob in zip(self.task_name_list, self.task_prob_list):
            if rand_value < prob:
                break

        prompt = random.choice(sample["instruction"][task_name])
        
        # add meta to prompt
        if self.add_meta_to_prompt:
            prompt["conversations"][0]["value"] = f"{self.meta_info}\n" + prompt["conversations"][0]["value"]

        # add sides prompt
        if np.random.random() < self.add_sides_prob and sample['meta'].get("sides", None) is not None:
            value = prompt["conversations"][0]["value"]
            if value.endswith("."):
                value = value[:-1]
            value += f" with the {sample['meta']['sides']} arm."
            prompt["conversations"][0]["value"] = value

        # add cot to prompt
        if np.random.random() < self.add_cot_prob and "cot" in sample["instruction"]:
            frame_index = sample["frame_index"]
            cot = sample["instruction"]["cot"]
            for action, end in zip(cot["actions"], cot["ends"]):
                if frame_index < end:
                    break
            prompt["conversations"][0]["value"] += " /cot"
            prompt["conversations"][1]["value"] = f"<cot>{action}</cot>"
        else:
            prompt["conversations"][0]["value"] += " /no_cot"
            prompt["conversations"][1]["value"] = "<cot></cot>"

        return prompt

    def get_action_mask(self, sample):
        ideal_length = self.dataset_action_length
        actual_length = min(ideal_length, sample["num_frames"] - sample["frame_index"])
        action_mask = np.zeros(ideal_length, dtype=np.int32)
        action_mask[:actual_length] = 1

        return action_mask, ideal_length, actual_length

    def get_frames(self, video_info, frame_index):
        frame_indices = np.clip(frame_index + self.offsets, min=0)
        all_obs = []
        for info in video_info:
            video = VideoReader(info["path"], num_threads=2)
            video.seek(0)
            all_obs.append(self.to_pil(video.get_batch(frame_indices.tolist()).asnumpy()))

        return all_obs

    def to_pil(self, video_frames):
        pil_frames = []
        for frame in video_frames:
            pil_img = Image.fromarray(frame)
            pil_frames.append(pil_img)
        return pil_frames


ROPE = {"Qwen/Qwen3-VL-4B-Instruct": get_rope_index_3, "Qwen/Qwen2.5-VL-3B-Instruct": get_rope_index_25}


@TRANSFORMS.register_module()
class VLMProcessor:
    def __init__(self, model="Qwen/Qwen3-VL-4B-Instruct", enable_l1=False, enable_choice=False):
        assert model in [
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
        ], "Model must in [Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen3-VL-4B-Instruct]"
        special_tokens = {}
        special_tokens["score"] = "<score>"
        special_tokens["state"] = "<state>"
        special_tokens.update({f"a_{i}": f"<a_{i}>" for i in range(ACTION_MAX_LENGTH)})
        self.processor = AutoProcessor.from_pretrained(model, use_fast=True, extra_special_tokens=special_tokens)
        assert self.processor.tokenizer.convert_tokens_to_ids("<score>") == SCORE_ID, "SCORE_ID is incorrect. Please check the model's tokenizer."
        assert self.processor.tokenizer.convert_tokens_to_ids("<state>") == STATE_ID, "SCORE_ID is incorrect. Please check the model's tokenizer."
        self.new_token_start_id = SCORE_ID
        self.get_rope_index = ROPE[model]
        
        self.enable_l1 = enable_l1
        self.enable_choice = enable_choice

    def __call__(self, sample):
        media = sample.get("media", [])
        images = [m for m in media if isinstance(m, Image.Image)]
        videos = [m for m in media if isinstance(m, list) and all(isinstance(frame, Image.Image) for frame in m)]

        conversations = sample["text"]["conversations"]
        action_condition_length = None

        action_target = None
        action_mask = None

        if (self.enable_choice or self.enable_l1) and ("action" in sample):
            # Check if the action is full without padding
            actual_length = sample["actual_length"]
            ideal_length = sample["ideal_length"]
            # if ideal_length != actual_length:
            #     raise ValueError("Ideal length must equal actual length in pre-training!")
            
            # Compute the length before adding 
            base_message = self._build_messages(conversations, images, videos)
            base_result = self.processor.apply_chat_template(base_message, tokenize=True, return_dict=True, do_resize=False, return_tensors="pt")
            action_condition_length = base_result["input_ids"].size(1)

            # Create action and state placeholder
            conversations, _ = self.build_action_message(conversations, sample["state"].shape[0], actual_length)
            action_target = sample["action"][:actual_length]
            action_mask = sample["action_mask"][:actual_length]

        messages = self._build_messages(conversations, images, videos)
        result = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, do_resize=False, return_tensors="pt")

        labels = self._build_label(result["input_ids"])
        result["labels"] = labels

        position_ids = self._build_position_ids(result)
        result["position_ids"] = position_ids
        
        if action_target is not None:
            result["action_target"] = torch.from_numpy(action_target)
            result["action_mask"] = torch.from_numpy(action_mask)
            result["action_actual_length"] = sample["actual_length"]
        
        if action_condition_length == None:
            action_condition_length = result["input_ids"].size(1)
        result["action_condition_length"] = action_condition_length

        vlm_result = {f"vlm_{k}": v for k, v in result.items()}

        sample.update(vlm_result)

        return sample

    def _build_messages(self, conversations, images=[], videos=[]):
        # Build media pools
        image_pool = [{"type": "image", "image": img} for img in images]
        video_pool = [{"type": "video", "video": vid} for vid in videos]

        messages = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"]

            if role == "user":
                content = []
                # Split text by <image> or <video> placeholders while keeping delimiters
                text_parts = re.split(r"(<image>|<video>)", text)

                for seg in text_parts:
                    if seg == "<image>":
                        if not image_pool:
                            raise ValueError("Number of <image> placeholders exceeds the number of provided images")
                        content.append(image_pool.pop(0))
                    elif seg == "<video>":
                        if not video_pool:
                            raise ValueError("Number of <video> placeholders exceeds the number of provided videos")
                        content.append(video_pool.pop(0))
                    elif seg:
                        content.append({"type": "text", "text": seg})

                messages.append({"role": role, "content": content})
            else:
                # Assistant messages contain only text
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})

        # Check for unused media files
        if image_pool:
            raise ValueError(f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)")
        if video_pool:
            raise ValueError(f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)")

        return messages

    def _build_label(self, input_ids):
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            # Detect "<im_start>assistant"
            # TODO: need to filter data containing "<im_start>"
            if input_ids_flat[pos] == 77091 and input_ids_flat[pos-1] == 151644:
                ans_start = pos + 2
                ans_end = ans_start
                # Detect "<im_end>"
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1

                # Found "<im_end>": a complete response
                if ans_end < L:
                    # ans_end + 2 will include "<im_end>\n"
                    labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                    pos = ans_end
                # Not found "<im_end>": an incomplete response
                else:
                    assert ans_end == L
                    labels[0, ans_start:] = input_ids[0, ans_start:]
                    pos = ans_end
            pos += 1
        
        index = input_ids >= self.new_token_start_id
        labels[index] = IGNORE_INDEX
        labels[:, 1:][index[:, :-1]] = IGNORE_INDEX

        return labels

    def _build_position_ids(self, result):
        if "image_grid_thw" in result:
            grid_thw = result.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in result:
            video_grid_thw = result.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.processor.image_processor.merge_size,
            result["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(torch.cat(video_grid_thw, dim=0) if video_grid_thw else None),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        return position_ids

    def build_action_message(self, conversations, state_length, action_length):
        # Construct the current state tokens
        state_tokens = "".join([f"<state>" for _ in range(state_length)])

        action_indices = list(range(action_length))

        action_tokens = "".join([f"<a_{i}>" for i in action_indices])

        if self.enable_choice:
            score_token = "<score>"
            action_response = f"{action_tokens}{score_token}"
        elif self.enable_l1:
            action_response = f"{action_tokens}"
        else:
            raise NotImplementedError

        # Create new conversation turns
        new_turns = [
            {
                "from": "human",
                "value": f"Robot state: {state_tokens}"
            },
            {
                "from": "gpt",
                "value": action_response
            }
        ]

        # Return the original conversations with new turns appended
        return conversations + new_turns, action_indices


@TRANSFORMS.register_module()
class GetAbsAction:
    def __init__(self, padding_action=None, data_flow=dict(), reversed_data_flow=dict()):
        self.padding_action = padding_action
        if padding_action is not None:
            self.padding_action = np.array(padding_action, dtype=np.float32)
            if self.padding_action.ndim == 1:
                self.padding_action = self.padding_action[None]

        self.data_flow = data_flow
        self.reversed_data_flow = reversed_data_flow

    def __call__(self, sample):
        for source, target in self.data_flow.items():
            data = get_value(sample, source)

            # action
            sample[target] = self._cal_action(deepcopy(data), sample["frame_index"], sample["actual_length"], sample["ideal_length"])

        return sample

    def _cal_action(self, data, frame_index, actual_length, ideal_length):
        offsets = np.arange(0, actual_length)
        target_frame_ids = frame_index + offsets

        action = np.array([data[i] for i in target_frame_ids], dtype=np.float32)

        # padding
        if actual_length < ideal_length:
            if self.padding_action is not None:
                padding = np.repeat(self.padding_action, ideal_length - actual_length, axis=0)
            else:
                padding = np.repeat(action[-1:], ideal_length - actual_length, axis=0)
            action = np.vstack((action, padding))

        return action


@TRANSFORMS.register_module()
class GetEEActionPos(GetAbsAction):
    def __init__(self, padding_action=None, data_flow=dict(), reversed_data_flow=dict(), ref_frame: Literal["base", "ee"] = "ee"):
        super().__init__(
            padding_action=padding_action,
            data_flow=data_flow,
            reversed_data_flow=reversed_data_flow,
        )
        self.ref_frame = ref_frame
    
    def __call__(self, sample):
        frame_index = sample["frame_index"]
        actual_length = sample["actual_length"]
        ideal_length = sample["ideal_length"]
        is_chunk = sample["is_chunk"]

        for target, source in self.reversed_data_flow.items():
            current_ee_rotm_key, current_ee_pos_key, action_ee_pos_key = source

            # current ee rotation matrix
            current_ee_rotm_list = get_value(sample, current_ee_rotm_key)
            if is_chunk:
                current_ee_rotm = current_ee_rotm_list[-1]
            else:
                current_ee_rotm = current_ee_rotm_list[frame_index]
            current_ee_rotm = np.array(current_ee_rotm, dtype=np.float32).reshape(3, 3)

            # current ee position
            current_ee_pos_list = get_value(sample, current_ee_pos_key)
            if is_chunk:
                current_ee_pos = current_ee_pos_list[-1]
            else:
                current_ee_pos = current_ee_pos_list[frame_index]
            current_ee_pos = np.array(current_ee_pos, dtype=np.float32)

            # action frame ids
            offsets = np.arange(0, actual_length)
            target_frame_ids = frame_index + offsets

            # action ee position
            action_ee_pos_list = get_value(sample, action_ee_pos_key)
            action_ee_pos_array = np.array([action_ee_pos_list[i] for i in target_frame_ids], dtype=np.float32)

            # compute accumulative relative action
            if self.ref_frame == "base":
                action = action_ee_pos_array - current_ee_pos
            elif self.ref_frame == "ee":
                action = current_ee_rotm.T @ (action_ee_pos_array - current_ee_pos).T
                action = action.T
            else:
                raise NotImplementedError

            # padding
            if actual_length < ideal_length:
                if self.padding_action is not None:
                    padding = np.repeat(self.padding_action, ideal_length - actual_length, axis=0)
                else:
                    padding = np.repeat(action[-1:], ideal_length - actual_length, axis=0)
                action = np.vstack((action, padding))

            sample[target] = action

        return sample


@TRANSFORMS.register_module()
class GetEEActionAA(GetAbsAction):
    def __init__(self, padding_action=None, data_flow=dict(), reversed_data_flow=dict(), ref_frame: Literal["base", "ee"] = "ee"):
        super().__init__(
            padding_action=padding_action,
            data_flow=data_flow,
            reversed_data_flow=reversed_data_flow,
        )
        self.ref_frame = ref_frame
    
    def __call__(self, sample):
        frame_index = sample["frame_index"]
        actual_length = sample["actual_length"]
        ideal_length = sample["ideal_length"]
        is_chunk = sample["is_chunk"]

        for target, source in self.reversed_data_flow.items():
            current_ee_rotm_key, action_ee_rotm_key = source

            # current ee rotation matrix
            current_ee_rotm_list = get_value(sample, current_ee_rotm_key)
            if is_chunk:
                current_ee_rotm = current_ee_rotm_list[-1]
            else:
                current_ee_rotm = current_ee_rotm_list[frame_index]
            current_ee_rotm = np.array(current_ee_rotm, dtype=np.float32).reshape(3, 3)

            # action frame ids
            offsets = np.arange(0, actual_length)
            target_frame_ids = frame_index + offsets

            # action ee rotation matrix
            action_ee_rotm_list = get_value(sample, action_ee_rotm_key)
            action_ee_rotm_array = np.array([action_ee_rotm_list[i] for i in target_frame_ids], dtype=np.float32)
            action_ee_rotm_array = action_ee_rotm_array.reshape(-1, 3, 3)

            # compute accumulative relative action (axis-angle)
            if self.ref_frame == "base":
                action_ee_rotms = action_ee_rotm_array @ current_ee_rotm.T
            elif self.ref_frame == "ee":
                action_ee_rotms = current_ee_rotm.T @ action_ee_rotm_array
            action = rotm2aa_batch(action_ee_rotms)

            # padding
            if actual_length < ideal_length:
                if self.padding_action is not None:
                    padding = np.repeat(self.padding_action, ideal_length - actual_length, axis=0)
                else:
                    padding = np.repeat(action[-1:], ideal_length - actual_length, axis=0)
                action = np.vstack((action, padding))

            sample[target] = action

        return sample


@TRANSFORMS.register_module()
class GetJointAction(GetAbsAction):
    def __call__(self, sample):
        frame_index = sample["frame_index"]
        actual_length = sample["actual_length"]
        ideal_length = sample["ideal_length"]
        is_chunk = sample["is_chunk"]

        for target, source in self.reversed_data_flow.items():
            current_joint_key, action_joint_key = source

            # current joint
            current_joint_list = get_value(sample, current_joint_key)
            if is_chunk:
                current_joint = current_joint_list[-1]
            else:
                current_joint = current_joint_list[frame_index]
            current_joint = np.array(current_joint, dtype=np.float32)

            # action frame ids
            offsets = np.arange(0, actual_length)
            target_frame_ids = frame_index + offsets

            # joint action
            action_joint_list = get_value(sample, action_joint_key)
            action_joint_array = np.array([action_joint_list[i] for i in target_frame_ids], dtype=np.float32)

            # compute accumulative relative action
            action = action_joint_array - current_joint

            # padding
            if actual_length < ideal_length:
                if self.padding_action is not None:
                    padding = np.repeat(self.padding_action, ideal_length - actual_length, axis=0)
                else:
                    padding = np.repeat(action[-1:], ideal_length - actual_length, axis=0)
                action = np.vstack((action, padding))

            sample[target] = action

        return sample


@TRANSFORMS.register_module()
class GetAbsState:
    def __init__(
        self,
        data_flow=dict(),
    ):
        self.obs_length = int(os.environ.get("_obs_length", 1))
        self.obs_interval = int(os.environ.get("_obs_interval", 1))

        self.offsets = np.arange(-self.obs_interval * (self.obs_length - 1), 1, self.obs_interval)

        self.data_flow = data_flow

    def __call__(self, sample):
        for source, target in self.data_flow.items():
            data = get_value(sample, source)
            if sample["is_chunk"]:
                x = self._cal_states(deepcopy(data), len(data) - 1)
            else:
                x = self._cal_states(deepcopy(data), sample["frame_index"])
            sample[target] = x
        return sample

    def _cal_states(self, data, frame_index):
        frame_indices = np.clip(frame_index + self.offsets, min=0)
        target_states = np.array([data[i] for i in frame_indices], dtype=np.float32)
        return target_states


@TRANSFORMS.register_module()
class GetRelState:
    def __init__(
        self,
        data_flow=dict(),
    ):
        self.obs_length = 2
        self.obs_interval = int(os.environ.get("_obs_interval", 1))

        self.offsets = np.arange(-self.obs_interval * (self.obs_length - 1), 1, self.obs_interval)

        self.data_flow = data_flow

    def __call__(self, sample):
        for source, target in self.data_flow.items():
            data = get_value(sample, source)
            if sample["is_chunk"]:
                x = self._cal_states(deepcopy(data), len(data) - 1)
            else:
                x = self._cal_states(deepcopy(data), sample["frame_index"])
            sample[target] = x
        return sample

    def _cal_states(self, data, frame_index):
        frame_indices = np.clip(frame_index + self.offsets, min=0)
        target_states = np.array([data[i] for i in frame_indices], dtype=np.float32)
        return np.abs(target_states[1:] - target_states[0:1]) * 30  # TODO: hardcode


@TRANSFORMS.register_module()
class GetAbsStateAA(GetAbsState):
    def _cal_states(self, data, frame_index):
        frame_indices = np.clip(frame_index + self.offsets, min=0)
        target_states = np.array([data[i] for i in frame_indices], dtype=np.float32).reshape(-1, 3, 3)
        target_states = rotm2aa_batch(target_states)
        return target_states


@TRANSFORMS.register_module()
class GetTaskStatusAndActionMask:
    def __init__(self, data_flow=dict()):
        self.data_flow = data_flow
        self.action_length = int(os.environ.get("_action_length", 1))

    def __call__(self, sample):
        ideal_length = sample["ideal_length"]

        for source, target in self.data_flow.items():
            traj_type = sample.get("trajectory_type", "success")
            action_mask = sample[source]
            new_action_mask = np.zeros((self.action_length, ), dtype=action_mask.dtype)
            task_status = np.zeros((self.action_length, 1), dtype=np.float32)

            if traj_type == "success":
                task_status[:ideal_length] = 1 - action_mask[:ideal_length, None]
                # new_action_mask[:ideal_length] = np.ones_like(action_mask[:ideal_length])
                new_action_mask[:ideal_length] = action_mask[:ideal_length]
            elif traj_type == "ongoing":
                task_status[:ideal_length] = np.zeros_like(action_mask[:ideal_length, None], dtype=np.float32)
                new_action_mask[:ideal_length] = action_mask[:ideal_length]
            elif traj_type == "invalid":
                task_status[:ideal_length] = np.zeros_like(action_mask[:ideal_length, None], dtype=np.float32) - 1
                new_action_mask[:ideal_length] = np.ones_like(action_mask[:ideal_length], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported trajectory_type: {traj_type}")

            sample[target[0]] = task_status
            sample[target[1]] = new_action_mask

        return sample


@TRANSFORMS.register_module()
class ComposeAction:
    def __init__(
        self,
        use_ideal_length=False,
        reversed_data_flow=dict(),
    ):
        self.use_ideal_length = use_ideal_length
        self.reversed_data_flow = reversed_data_flow

        action_length = int(os.environ.get("_action_length", 1))
        predefined_dim = -1

        components = list(reversed_data_flow.values())[-1]
        for item in components.values():
            if isinstance(item, (list, tuple)):
                if isinstance(item[1], (list, tuple)):
                    predefined_dim = max(predefined_dim, item[1][-1])
                else:
                    predefined_dim = max(predefined_dim, item[-1])
        assert predefined_dim > 0, "action dim must greater than 0."
        self.data_shape = (action_length, predefined_dim)

    def __call__(self, sample):
        for target, source in self.reversed_data_flow.items():
            if self.use_ideal_length:
                data_shape = (sample["ideal_length"], self.data_shape[1])
            else:
                data_shape = self.data_shape
            data = np.zeros(data_shape, dtype=np.float32)
            data_dim_mask = np.zeros((1, self.data_shape[-1]), dtype=np.int32)

            for component, indexs in source.items():
                if isinstance(indexs, (list, tuple)):
                    if isinstance(indexs[1], (list, tuple)):
                        s_start, s_end = indexs[0]
                        t_start, t_end = indexs[1]
                        if component in sample:
                            tmp_data = sample[component][:, s_start:s_end]
                            t_end = min(t_end, t_start + tmp_data.shape[-1])
                            data[:tmp_data.shape[0], t_start:t_end] = tmp_data
                            data_dim_mask[:, t_start:t_end] = 1
                    else:
                        start, end = indexs
                        if component in sample:
                            tmp_data = sample[component]
                            end = min(end, start + tmp_data.shape[-1])
                            data[:tmp_data.shape[0], start:end] = tmp_data
                            data_dim_mask[:, start:end] = 1
                elif component == "action_mask_key":
                    data_temporal_mask = sample[indexs]
            
            data_mask = data_temporal_mask[:, None] @ data_dim_mask

            sample[target] = data
            sample[f"{target}_mask"] = data_mask

        return sample


@TRANSFORMS.register_module()
class ComposeState:
    def __init__(
        self,
        reversed_data_flow=dict(),
    ):
        self.reversed_data_flow = reversed_data_flow

        obs_length = int(os.environ.get("_obs_length", 1))

        predefined_dim = -1
        components = list(reversed_data_flow.values())[-1]
        for item in components.values():
            if isinstance(item[1], (list, tuple)):
                predefined_dim = max(predefined_dim, item[1][-1])
            else:
                predefined_dim = max(predefined_dim, item[-1])
        assert predefined_dim > 0, "action dim must greater than 0."
        self.data_shape = (obs_length, predefined_dim)

    def __call__(self, sample):
        for target, source in self.reversed_data_flow.items():
            data = np.zeros(self.data_shape, dtype=np.float32)
            data_dim_mask = np.zeros((1, self.data_shape[-1]), dtype=np.int32)

            for component, indexs in source.items():
                if isinstance(indexs, (list, tuple)):
                    if isinstance(indexs[1], (list, tuple)):
                        # source: [[s_start, s_end], [t_start, t_end]]
                        # put source[s_start, s_end] at [t_start, t_end] of state
                        s_start, s_end = indexs[0]
                        t_start, t_end = indexs[1]
                        if component in sample:
                            tmp_data = sample[component][:, s_start:s_end]
                            t_end = min(t_end, t_start + tmp_data.shape[-1])
                            data[:, t_start:t_end] = tmp_data
                            data_dim_mask[:, t_start:t_end] = 1
                    else:
                        # source: [start, end]
                        # put source at [start, end] of state
                        start, end = indexs
                        if component in sample:
                            tmp_data = sample[component]
                            end = min(end, start + tmp_data.shape[-1])
                            data[:, start:end] = tmp_data
                            data_dim_mask[:, start:end] = 1

            # # Concatenate data with data_dim_mask
            # data_dim_mask = np.repeat(data_dim_mask, data.shape[0], axis=0)
            # data = np.concatenate((data, data_dim_mask), axis=1)

            sample[target] = data

        return sample


@TRANSFORMS.register_module()
class Collect:
    def __init__(self, patterns=None):
        if patterns is None:
            self.patterns = []
        elif isinstance(patterns, str):
            self.patterns = [patterns]
        else:
            self.patterns = list(patterns)

    def __call__(self, sample):
        results = {}
        for key in sample:
            for pattern in self.patterns:
                if fnmatch.fnmatch(key, pattern):
                    results[key] = sample[key]
                    break
        return results
