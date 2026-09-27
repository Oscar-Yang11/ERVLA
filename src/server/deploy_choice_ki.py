#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from os.path import join as osp

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.multiprocessing as mp
from mmengine import Config

import src.models
from src.models import MIMODEL
from src.models.VLA.P2_Choice_KI import P2_Choice_KI  # noqa: F401
from src.server.runtime.server_choice import ChoiceTCPServer

mp.set_start_method("spawn", force=True)


def load_cfg(path):
    return Config.fromfile(path)


def format_state_dict(state_dict, prefix):
    new_state_dict = {}
    prefix_len = len(prefix)
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[prefix_len:]] = value
    return new_state_dict


def build_model(cfg, ckpt_path):
    print("[DEBUG] src.models file:", src.models.__file__)
    print("[DEBUG] registry keys:", sorted(MIMODEL.module_dict.keys()))
    print("[DEBUG] cfg.model.params.model:", cfg.model.params.model)

    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)

    raw = torch.load(ckpt_path, map_location="cpu")
    if "module" not in raw:
        raise KeyError(f"Checkpoint does not contain key 'module': {ckpt_path}")

    state_dict = raw["module"]
    state_dict = format_state_dict(state_dict, "model.")

    model.eval()
    msg = model.load_state_dict(state_dict, assign=True)
    print(msg)
    return model


def _find_transform(source_cfg, ttype):
    for t in source_cfg.transforms:
        if t["type"] == ttype:
            return t
    return None


def _flatten_1xN(x):
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        return x[0]
    return x


def _extract_choice_stats_from_cfg(cfg):
    """
    Read action mean/std for vlabench_choice from config.py, consistent with training config.
    """
    source_name = "vlabench_choice"

    if source_name not in cfg.data.params.train_datasets.sources:
        raise ValueError(
            f"Source '{source_name}' not found in cfg.data.params.train_datasets.sources"
        )

    source_cfg = cfg.data.params.train_datasets.sources[source_name]

    # 优先读训练时实际使用的 Normalize
    norm_t = _find_transform(source_cfg, "Normalize")
    if norm_t is not None and norm_t.get("mode", None) == "gaussian":
        mean_cfg = norm_t["mean"]["action_direct"]
        std_cfg = norm_t["std"]["action_direct"]
    else:
        # fallback
        try:
            mean_cfg = cfg.sources.vlabench_choice.mean.action_direct
            std_cfg = cfg.sources.vlabench_choice.std.action_direct
        except Exception as e:
            raise ValueError(
                "Cannot find gaussian Normalize transform or "
                "cfg.sources.vlabench_choice.mean/std.action_direct"
            ) from e

    action_mean = _flatten_1xN(mean_cfg)
    action_std = _flatten_1xN(std_cfg)

    if len(action_mean) != len(action_std):
        raise ValueError(
            f"action mean/std dim mismatch: {len(action_mean)} vs {len(action_std)}"
        )

    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    for split_name in ["train_datasets", "val_datasets"]:
        split_cfg = getattr(cfg.data.params, split_name, None)
        if split_cfg is None:
            continue
        sources = getattr(split_cfg, "sources", None)
        if sources is None or source_name not in sources:
            continue
        split_source_cfg = sources[source_name]
        vp = _find_transform(split_source_cfg, "VLMProcessor")
        if vp is not None and "model" in vp:
            model_name = vp["model"]
            break

    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "model_name": model_name,
    }


def helper(args):
    cfg = load_cfg(osp(args.model_root, "config.py"))

    device = "cuda:0"
    model = build_model(cfg, args.ckpt_path).to(device)

    stats = _extract_choice_stats_from_cfg(cfg)
    return model, stats, device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_root",
        type=str,
        required=True,
        help="Experiment root dir that contains config.py",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to mp_rank_00_model_states.pt",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10086)
    return parser.parse_args()


def run_server(server):
    # 兼容不同 server 实现
    if hasattr(server, "start"):
        server.start()
        if hasattr(server, "join"):
            server.join()
        return

    if hasattr(server, "serve_forever"):
        server.serve_forever()
        return

    if hasattr(server, "run"):
        server.run()
        return

    raise AttributeError(
        "ChoiceTCPServer has no supported runner method. "
        f"Available public attrs: {[x for x in dir(server) if not x.startswith('_')]}"
    )


if __name__ == "__main__":
    args = parse_args()

    model, stats, device = helper(args)

    try:
        server = ChoiceTCPServer(
            host=args.host,
            port=args.port,
            model=model,
            device=device,
            model_name=stats["model_name"],
            action_mean=stats["action_mean"],
            action_std=stats["action_std"],
        )
        print(f"Starting choice KI server on {args.host}:{args.port}")
        run_server(server)

    except OSError as e:
        if e.errno == 98:
            print(f"Error: Port {args.port} is already in use.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Choice KI server interrupted")