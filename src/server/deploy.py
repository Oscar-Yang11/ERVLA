import argparse
import importlib
import os
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os.path import join as osp
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.multiprocessing as mp
from mmengine import Config

mp.set_start_method("spawn", force=True)


def load_cfg(path):
    return Config.fromfile(path)


def switch_commit(commit_id):
    result = subprocess.run(["git", "checkout", "./"], capture_output=True, text=True)
    result = subprocess.run(["git", "checkout", commit_id], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to switch to commit {commit_id}: {result.stderr}")


def format_state_dict(state_dict, prefix):
    new_state_dict = {}
    prefix_len = len(prefix)

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            new_state_dict[new_key] = value

    return new_state_dict


def build_model(cfg, model_path):
    MIMODEL = importlib.import_module("src.models").MIMODEL

    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)

    state_dict = torch.load(model_path, map_location="cpu")["module"]
    state_dict = format_state_dict(state_dict, "model.")

    model.eval()
    msg = model.load_state_dict(state_dict, assign=True)
    print(msg)

    return model

def extract_mean_std(cfg):
    means = {}
    stds = {}

    for source_name, source_config in cfg.data.params.train_datasets.sources.items():
        for transform in source_config.get("transforms", []):
            if transform.get("type") == "Normalize" and transform.get("mode", "gaussian") == "gaussian":
                mean_dict = transform.get("mean", {})
                std_dict = transform.get("std", {})
                means[source_name] = mean_dict
                stds[source_name] = std_dict

    return means, stds


def extract_dataset_length(cfg):
    dataset_action_lengths = {}
    for source_name, source_config in cfg.data.params.train_datasets.sources.items():
        for transform in source_config.get("transforms", []):
            if transform.get("type") == "LoadData":
                tmp_dataset_length = transform.get("dataset_action_length", 30)
                dataset_action_lengths[source_name] = tmp_dataset_length
    return dataset_action_lengths


def helper(args):
    cfg = load_cfg(osp(args.model, "config.py"))

    device = "cuda:0"
    model_path = osp(args.model, "last.ckpt/checkpoint", "mp_rank_00_model_states.pt")
    model = build_model(cfg, model_path).to(device)

    trans = list(cfg.data.params.train_datasets.sources.values())[0].transforms
    for t in trans:
        if t["type"] == "ComposeAction":
            composition = t["reversed_data_flow"]

    composition = list(composition.values())[-1]
    predefined_dim = -1
    for item in composition.values():
        if isinstance(item, (list, tuple)):
            if isinstance(item[1], (list, tuple)):
                predefined_dim = max(predefined_dim, item[1][-1])
            else:
                predefined_dim = max(predefined_dim, item[-1])

    means, stds = extract_mean_std(cfg)
    dataset_action_lengths = extract_dataset_length(cfg)

    for key, value in means.items():
        mean = torch.zeros((dataset_action_lengths[key], predefined_dim), device=device)
        for src, tgt in composition.items():
            src = src.replace('_gn', '')
            if isinstance(tgt, (list, tuple)):
                if isinstance(tgt[1], (list, tuple)):
                    s_start, s_end = tgt[0]
                    t_start, t_end = tgt[1]
                    if src in value:
                        mean[:dataset_action_lengths[key], t_start:t_end] = torch.tensor(value[src], device=device)[
                            :dataset_action_lengths[key], s_start:s_end
                        ]
                else:
                    start, end = tgt
                    if src in value:
                        tmp_data = torch.tensor(value[src], device=device)[:dataset_action_lengths[key]]
                        end = min(end, tmp_data.shape[1] + start)
                        mean[:dataset_action_lengths[key], start:end] = torch.tensor(value[src], device=device)[
                            :dataset_action_lengths[key]
                        ]
        means[key] = mean

    for key, value in stds.items():
        std = torch.ones((dataset_action_lengths[key], predefined_dim), device=device)
        for src, tgt in composition.items():
            src = src.replace('_gn', '')
            if isinstance(tgt, (list, tuple)):
                if isinstance(tgt[1], (list, tuple)):
                    s_start, s_end = tgt[0]
                    t_start, t_end = tgt[1]
                    if src in value:
                        std[:dataset_action_lengths[key], t_start:t_end] = torch.tensor(value[src], device=device)[
                            :dataset_action_lengths[key], s_start:s_end
                        ]
                else:
                    start, end = tgt
                    if src in value:
                        tmp_data = torch.tensor(value[src], device=device)[:dataset_action_lengths[key]]
                        end = min(end, tmp_data.shape[1] + start)
                        std[:dataset_action_lengths[key], start:end] = torch.tensor(value[src], device=device)[
                            :dataset_action_lengths[key]
                        ]
        stds[key] = std + 1e-6

    return model, (means, stds), device


def _build_config_handler(config_file: Path):
    class ConfigHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/healthz":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            if route == "/config.py" or (route.startswith("/configs/") and route.endswith(".py")):
                if not config_file.exists():
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(f"config not found: {config_file}".encode("utf-8"))
                    return

                content = config_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/x-python; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Config-Path", str(config_file))
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found")

        def log_message(self, format, *args):
            return

    return ConfigHandler


def start_config_http_server(model_dir: str, host: str, port: int) -> ThreadingHTTPServer:
    config_file = Path(model_dir) / "config.py"
    handler = _build_config_handler(config_file)
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, name="config-http", daemon=True)
    thread.start()
    print(f"Config HTTP server on http://{host}:{port} serving {config_file}")
    return httpd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--config-host", type=str, default=os.getenv("MODEL_CONFIG_HOST", "0.0.0.0"))
    parser.add_argument("--config-port", type=int, default=int(os.getenv("MODEL_CONFIG_PORT", "18086")))
    parser.add_argument("--disable-config-http", action="store_true", help="Disable config HTTP endpoint")
    return parser.parse_args()


if __name__ == "__main__":
    Server = importlib.import_module("src.server.runtime.server").Server

    args = parse_args()
    model, (means, stds), device = helper(args)

    config_httpd = None
    if not args.disable_config_http:
        config_httpd = start_config_http_server(args.model, args.config_host, args.config_port)

    try:
        server = Server(host=args.host, port=args.port, model=model, means=means, stds=stds, device=device)
        print(f"Starting inference server on {args.host}:{args.port}")
        server.start()
        server.join()
    except OSError as e:
        if e.errno == 98:
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Server interrupted")
    finally:
        if config_httpd is not None:
            config_httpd.shutdown()
            config_httpd.server_close()
