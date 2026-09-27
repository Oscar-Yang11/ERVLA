import json
import random
from typing import Literal

from PIL import Image

import numpy as np
from mmengine import TRANSFORMS
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)
import torchvision.transforms.functional as F

from src.data.transforms.utils import smart_resize
import cv2


@TRANSFORMS.register_module()
class Resize:
    def __init__(
        self,
        max_pixels=200704,
        factor=32,
        data_flow=None,
    ):
        self.factor = factor
        self.max_pixels = max_pixels

        # Data flow configuration - maps source keys to target keys in the sample dict
        self.data_flow = data_flow if data_flow is not None else {}

    def __call__(self, sample):
        """Process the sample dictionary by applying crop and resize to specified keys"""
        for source, target in self.data_flow.items():
            # if source not in sample:
            #     continue
            # Apply processing and store result in target key
            sample[target] = self._resize(sample[source])
        return sample

    def _resize(self, frames):
        frames = [smart_resize(frame, max_pixels=self.max_pixels, factor=self.factor) for frame in frames]
        return frames


@TRANSFORMS.register_module()
class RandomCrop:
    def __init__(
        self,
        crop_ratio,
        center_crop=False,
        data_flow=dict(),
    ):
        self.crop_ratio = crop_ratio
        self.center_crop = center_crop
        self.data_flow = data_flow
    
    def __call__(self, sample):
        for source, target in self.data_flow.items():
            frames = sample[source]
            cropped_frames = []
            for frame in frames:
                cropped_frames.append(self.crop(frame))
            sample[target] = cropped_frames
        return sample

    def crop(self, image):
        h, w = image.size[1], image.size[0]
        crop_h, crop_w = int(h * self.crop_ratio), int(w * self.crop_ratio)

        if self.center_crop:
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
        else:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)

        cropped = F.crop(image, top=top, left=left, height=crop_h, width=crop_w)
        assert cropped.size == (crop_w, crop_h)
        resized = F.resize(cropped, size=(h, w))
        assert resized.size == (w, h)
        return resized


@TRANSFORMS.register_module()
class UndistortFisheye:
    """Approximate fisheye undistortion without calibration.

    Maps fisheye image to a rectilinear projection using an equidistant model.
    ``strength`` interpolates between the original fisheye (0) and full
    rectilinear correction (1), so you can partially relieve distortion.
    """

    _remap_cache: dict = {}

    def __init__(
        self,
        fov_deg: float = 185.0,
        strength: float = 0.4,
        target_width: int = 792,
        height_offset: int = 50,
        data_flow: dict = dict(),
    ):
        self.fov_deg = fov_deg
        self.strength = strength
        self.target_width = target_width
        self.height_offset = height_offset
        self.data_flow = data_flow

        key = (fov_deg, strength, target_width, height_offset)
        if key not in UndistortFisheye._remap_cache:
            UndistortFisheye._remap_cache[key] = self._build_remap()
        self.src_x, self.src_y = UndistortFisheye._remap_cache[key]

    def __call__(self, sample):
        camera_type = sample.get("meta", {}).get("camera_type", None)
        for source, target in self.data_flow.items():
            if camera_type == "fisheye":
                frames = sample[source]
                frames = [self._undistort_fisheye(frame) for frame in frames]
                sample[target] = frames
        return sample

    def _build_remap(self):
        """Build remap tables for the fixed output size."""
        h, w = 720, self.target_width
        cx, cy = w / 2.0, h / 2.0
        radius = h / 2.0

        cy += self.height_offset

        fov_rad = np.deg2rad(self.fov_deg)
        f = radius / (fov_rad / 2.0)

        dst_x, dst_y = np.meshgrid(
            np.arange(w) - w / 2.0,
            np.arange(h) - h / 2.0,
        )

        r_dst = np.sqrt(dst_x**2 + dst_y**2)
        theta = np.arctan2(r_dst, f)
        r_src_full = f * theta
        r_src_none = r_dst
        r_src = r_src_none * (1.0 - self.strength) + r_src_full * self.strength

        r_dst_safe = np.where(r_dst == 0, 1, r_dst)
        scale = r_src / r_dst_safe
        scale = np.where(r_dst == 0, 1, scale)

        src_x = (dst_x * scale + cx).astype(np.float32)
        src_y = (dst_y * scale + cy).astype(np.float32)

        return src_x, src_y

    def _normalize_width(self, img: np.ndarray) -> np.ndarray:
        """Center-crop or pad to target_width. Height is always 720."""
        w = img.shape[1]
        tw = self.target_width

        if w > tw:
            left = (w - tw) // 2
            img = img[:, left:left + tw]
        elif w < tw:
            pad_left = (tw - w) // 2
            pad_right = tw - w - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)

        return img

    def _undistort_fisheye(self, frame):
        """Pad/crop to target size, apply cached remap. PIL in, PIL out."""
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        img = self._normalize_width(img)

        result = cv2.remap(
            img, self.src_x, self.src_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    

@TRANSFORMS.register_module()
class PhotoMetricDistortion:
    def __init__(
        self,
        brightness_delta=32.0,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=0.0,
        data_flow=dict(),
    ):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        self.data_flow = data_flow

    def __call__(self, sample):
        distortion_params = self._generate_distortion_params()
        apply_flags = self._generate_apply_flags(4)

        for source, target in self.data_flow.items():
            # if source not in sample:
            #     continue
            sample[target] = self._distort_video(sample[source], distortion_params, apply_flags)

        return sample

    def _generate_distortion_params(self):
        return {
            "brightness": 1 + random.uniform(-self.brightness_delta / 255.0, self.brightness_delta / 255.0),
            "contrast": random.uniform(*self.contrast_range),
            "saturation": random.uniform(*self.saturation_range),
            "hue": random.uniform(-self.hue_delta / 180.0, self.hue_delta / 180.0),
        }

    def _generate_apply_flags(self, num_params):
        return [random.randint(0, 1) == 0 for _ in range(num_params)]

    def _distort_video(self, video_frames, params, apply_flags):
        brightness = params["brightness"]
        contrast = params["contrast"]
        saturation = params["saturation"]
        hue = params["hue"]

        distorted_frames = []
        for frame in video_frames:
            # Apply transforms manually with consistent parameters
            if apply_flags[0]:
                frame = adjust_brightness(frame, brightness)
            if apply_flags[1]:
                frame = adjust_contrast(frame, contrast)
            if apply_flags[2]:
                frame = adjust_saturation(frame, saturation)
            if apply_flags[3]:
                frame = adjust_hue(frame, hue)
            distorted_frames.append(frame)

        return distorted_frames


@TRANSFORMS.register_module()
class Normalize:
    def __init__(self, mean=None, std=None, q01=None, q99=None, mode: Literal["gaussian", "quantile"]="gaussian", data_flow=dict()):
        self.mode = mode
        if self.mode == "gaussian":
            assert mean != None and std != None, "mean and std must not be None for guassian normalization."
            for key, value in mean.items():
                mean[key] = np.array(value, dtype=np.float32)
                assert mean[key].ndim == 2, f"{key}-mean should have exactly 2 dimensions"
            for key, value in std.items():
                std[key] = np.array(value, dtype=np.float32) + 1e-6
                assert std[key].ndim == 2, f"{key}-std should have exactly 2 dimensions"
            self.mean = mean
            self.std = std
            self.q01 = None
            self.q99 = None
        elif self.mode == "quantile":
            assert q01 != None and q99 != None, "q01 and q99 must not be None for quantile normalization."
            for key, value in q01.items():
                q01[key] = np.array(value, dtype=np.float32)
                assert q01[key].ndim == 2, f"{key}-q01 should have exactly 2 dimensions"
            self.q01 = q01
            for key, value in q99.items():
                q99[key] = np.array(value, dtype=np.float32) + 1e-6
                assert q99[key].ndim == 2, f"{key}-q99 should have exactly 2 dimensions"
            self.mean = None
            self.std = None
            self.q01 = q01
            self.q99 = q99
        else:
            raise ValueError("Invalid mode. Choose from 'gaussian' or 'quantile'.")

        self.data_flow = data_flow

    def __call__(self, sample):
        for source, target in self.data_flow.items():
            x = sample[source]

            if self.mode == "gaussian":
                tmp_length = x.shape[0]
                mean = self.mean[source][:tmp_length]
                std = self.std[source][:tmp_length]
                x = (x - mean) / std
            elif self.mode == "quantile":
                tmp_length = x.shape[0]
                q01 = self.q01[source][:tmp_length]
                q99 = self.q99[source][:tmp_length]
                x = (x - q01) / (q99 - q01)
                x = 2 * x - 1
            else:
                raise ValueError("Invalid mode. Choose from 'gaussian' or 'quantile'.")
            sample[target] = x

        return sample


@TRANSFORMS.register_module()
class BinDiscretize:
    def __init__(self, min=-1.0, max=1.0, num_bins=256, data_flow=dict()):
        self.min = min
        self.max = max
        self.num_bins = num_bins
        self.data_flow = data_flow

        self.bins = np.linspace(self.min, self.max, self.num_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
    
    def __call__(self, sample):
        for source, target in self.data_flow.items():
            data = sample[source]
            data = np.clip(data, a_min=float(self.min), a_max=float(self.max))
            tokenized_data = np.digitize(data, self.bins)  # [1, num_bins]
            sample[target] = tokenized_data[0].tolist()
        return sample

    def decode(self, tokenized_data):
        tokenized_data = np.clip(tokenized_data - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[tokenized_data]


@TRANSFORMS.register_module()
class InsertState2Prompt:
    def __init__(self, data_flow=dict(), insertion_trigger="Generate robot actions for the task:\n"):
        self.data_flow = data_flow
        self.insertion_trigger = insertion_trigger

    def __call__(self, sample):
        for source, target in self.data_flow.items():
            # Convert to string
            discretized_state = sample[source]
            discretized_state_str = " ".join(map(str, discretized_state))
            full_state_str = f"Robot state: {discretized_state_str}\n"

            # Insert into prompt
            text = json.loads(sample[target])
            prompt = text["conversations"][0]["value"]
            prompt = self.insert_before(prompt, full_state_str)
            text["conversations"][0]["value"] = prompt
            sample[target] = json.dumps(text)
        return sample
        
    def insert_before(self, source, inserting_str):
        splited_source = source.split(self.insertion_trigger)
        assert len(splited_source) == 2
        splited_source[0] += inserting_str
        return splited_source[0] + self.insertion_trigger + splited_source[1]

from src.data.transforms.utils import get_value

@TRANSFORMS.register_module()
class GetDeltaActionFromAbsEE:
    """
    Build delta EE action chunks from absolute EE actions.
    dims 0:6 use delta; dim 6 gripper keeps absolute value.
    """
    def __init__(
        self,
        padding_action=None,
        wrap_euler=True,
        euler_slice=(3, 6),
        delta_dim=6,
        reversed_data_flow=dict(),
    ):
        self.padding_action = padding_action
        if padding_action is not None:
            self.padding_action = np.array(padding_action, dtype=np.float32)
            if self.padding_action.ndim == 1:
                self.padding_action = self.padding_action[None]

        self.wrap_euler = wrap_euler
        self.euler_slice = tuple(euler_slice)
        self.delta_dim = int(delta_dim)
        self.reversed_data_flow = reversed_data_flow

    def __call__(self, sample):
        frame_index = sample["frame_index"]
        actual_length = sample["actual_length"]
        ideal_length = sample["ideal_length"]
        is_chunk = sample["is_chunk"]

        for target, source in self.reversed_data_flow.items():
            assert isinstance(source, (list, tuple)) and len(source) == 2, (
                f"{target} expects [state_key, action_key], but got {source}"
            )
            current_state_key, action_abs_key = source

            current_state_list = get_value(sample, current_state_key)
            if is_chunk:
                current_state = current_state_list[-1]
            else:
                current_state = current_state_list[frame_index]
            current_state = np.array(current_state, dtype=np.float32)

            action_abs_list = get_value(sample, action_abs_key)
            offsets = np.arange(0, actual_length)
            target_frame_ids = frame_index + offsets
            action_abs = np.array([action_abs_list[i] for i in target_frame_ids], dtype=np.float32)

            delta = np.array(action_abs, dtype=np.float32, copy=True)

            if self.delta_dim > 0:
                delta[0, :self.delta_dim] = action_abs[0, :self.delta_dim] - current_state[:self.delta_dim]
                if actual_length > 1:
                    delta[1:, :self.delta_dim] = action_abs[1:, :self.delta_dim] - action_abs[:-1, :self.delta_dim]

            if self.wrap_euler:
                s, e = self.euler_slice
                s = max(0, s)
                e = min(self.delta_dim, e)
                if s < e:
                    delta[:, s:e] = self._wrap_to_pi(delta[:, s:e])

            if actual_length < ideal_length:
                if self.padding_action is not None:
                    padding = np.repeat(self.padding_action, ideal_length - actual_length, axis=0)
                else:
                    padding = np.repeat(delta[-1:], ideal_length - actual_length, axis=0)
                delta = np.vstack((delta, padding))

            sample[target] = delta

        return sample

    @staticmethod
    def _wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi
