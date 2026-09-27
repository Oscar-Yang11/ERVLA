import pickle
import socket
import struct
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm


class Server(mp.Process):
    def __init__(self, host, port, model, means, stds, device):
        super(Server, self).__init__()
        self.host = host
        self.port = port
        self.model = model
        self.model.dit = torch.compile(self.model.dit, mode="reduce-overhead", fullgraph=True)

        self.means = means
        self.stds = stds

        self.action_masks = dict()
        for key, value in self.means.items():
            self.action_masks[key] = (self.stds[key] > 1e-5).int()

        self.device = device

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

                            if task_id not in self.means:
                                task_id = list(self.means.keys())[0]

                            if "action" not in data:
                                data["action"] = torch.zeros((data["input_ids"].shape[0], *self.means[task_id].shape), device=self.device, dtype=torch.bfloat16)
                            else:
                                data["action"] = (data["action"] - self.means[task_id]) / self.stds[task_id]

                            if "action_mask" not in data:
                                data["action_mask"] = self.action_masks[task_id][None]

                            action = self.model.generate(data)

                            action = action * self.stds[task_id] + self.means[task_id]

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
