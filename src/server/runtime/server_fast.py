import os
import pickle
import socket
import struct
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoProcessor


class ServerFAST(mp.Process):
    def __init__(self, host, port, model, q01s, q99s, device, action_tokenizer_path=None):
        super(ServerFAST, self).__init__()
        self.host = host
        self.port = port
        self.model = model

        self.q01s = q01s
        self.q99s = q99s

        self.device = device

        tokenizer_path = action_tokenizer_path or os.environ.get(
            "FAST_ACTION_TOKENIZER", "models/FAST/pi_fast_tokenizer"
        )
        self.action_tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

    def _recv_all(self, conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, addr = server_socket.accept()

                try:
                    request_count = 0

                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            data_len_bytes = self._recv_all(conn, 4)
                            if not data_len_bytes:
                                break
                            data_len = struct.unpack(">I", data_len_bytes)[0]

                            data = self._recv_all(conn, data_len)
                            if not data:
                                break

                            tic = time.time()

                            input_data = pickle.loads(data)

                            data = {
                                key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
                                for key, value in input_data.items()
                            }

                            task_id = data.pop("task_id")

                            if task_id not in self.q01s:
                                task_id = list(self.q01s.keys())[0]

                            attention_mask = data.pop("attention_mask")
                            break_flag = False
                            forward_iter = 0
                            while (not break_flag) and (forward_iter < 10):
                                # Generate action tokens
                                action = self.model.generate(data)

                                # Convert to continuous actions
                                action_length, action_dim = self.q01s[task_id].shape
                                action = self.action_tokenizer.decode(
                                    action, time_horizon=action_length, action_dim=action_dim
                                )
                                if not np.all(action == 0.0):
                                    break_flag = True
                                forward_iter += 1

                            action = torch.tensor(action, device=self.device)
                            action = (action + 1) / 2
                            action = action * (self.q99s[task_id][None, ...] - self.q01s[task_id][None, ...] + 1e-6) + self.q01s[task_id][None, ...]
                            action = action[:, :, :8]

                            response = pickle.dumps(action.cpu())
                            conn.sendall(struct.pack(">I", len(response)) + response)

                            toc = time.time()
                            request_count += 1
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "avg_time": f"{(toc - tic) * 1000:.2f}ms",
                                }
                            )
                except Exception as e:
                    print(f"Error handling connection: {e}")
                finally:
                    conn.close()


if __name__ == "__main__":
    server = Server(host="localhost", port=10086, device="cpu")
    server.start()
    server.join()
