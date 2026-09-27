import math
import pickle
import socket
import struct

import torch

from PIL import Image
from transformers import AutoProcessor

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    image: Image, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    width, height = image.size
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return image.resize((w_bar, h_bar))


class Client:
    def __init__(self, host="localhost", port=50000):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("Client connected to server.")

    def _send_with_length_prefix(self, data):
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_with_length_prefix(self):
        len_data = self.client_socket.recv(4)
        if not len_data or len(len_data) < 4:
            raise ConnectionError("Failed to receive response length prefix.")
        data_len = struct.unpack(">I", len_data)[0]
        data = b""
        while len(data) < data_len:
            packet = self.client_socket.recv(data_len - len(data))
            if not packet:
                raise ConnectionError("Connection closed while receiving response.")
            data += packet
        return pickle.loads(data)

    def __call__(self, state, base_obs, left_wrist_obs, instruction):
        base_obs = smart_resize(base_obs)
        left_wrist_obs = smart_resize(left_wrist_obs)
        instruction = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThe base_obs view is <|vision_start|><|image_pad|><|vision_end|>\nThe wrist_left_obs view is <|vision_start|><|image_pad|><|vision_end|>\n\nGenerate actions for:\n{instruction}\n<|im_start|>assistant\n"
        task_id = "libero"
        state = torch.from_numpy(state)[None]
        data = self.processor(
            text=[instruction],
            images=[base_obs, left_wrist_obs],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        data["task_id"] = task_id
        data["state"] = state

        self._send_with_length_prefix(data)
        return self._recv_with_length_prefix()

    def close(self):
        self.client_socket.close()
        print("Client connection closed.")


if __name__ == "__main__":
    client = Client(host="localhost", port=50000)
    input_data = torch.rand((1, 30, 8), dtype=torch.float16)
    for _ in range(100):
        out = client(input_data)
