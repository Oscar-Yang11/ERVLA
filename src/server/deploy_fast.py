import argparse
import subprocess
import sys

from os.path import join as osp

import torch
import torch.multiprocessing as mp
from mmengine import Config

from src.models import MIMODEL
from src.server.runtime.server_fast import ServerFAST

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
    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)

    state_dict = torch.load(model_path, map_location="cpu")["module"]
    state_dict = format_state_dict(state_dict, "model.")

    # load
    model.eval()
    msg = model.load_state_dict(state_dict, assign=True)
    print(msg)

    return model

def extract_quantile(cfg):
    q01s = {}
    q99s = {}
    
    for source_name, source_config in cfg.data.params.train_datasets.sources.items():        
        # Loop through transforms to find Normalize
        for transform in source_config.get('transforms', []):
            if transform.get('type') == 'Normalize' and transform.get('mode') == 'quantile':
                q01_dict = transform.get('q01', {})
                q99_dict = transform.get('q99', {})
                data_flow = transform.get('data_flow', {})

                new_q01_dict = {}
                new_q99_dict = {}
                for src, tgt in data_flow.items():
                    new_q01_dict[tgt] = q01_dict[src]
                    new_q99_dict[tgt] = q99_dict[src]
                
        q01s[source_name] = new_q01_dict
        q99s[source_name] = new_q99_dict

    return q01s, q99s

def helper(args):
    # load cfg
    cfg = load_cfg(osp(args.model, "config.py"))

    # switch commit
    # current_commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")[:12]
    # if current_commit_id != cfg.commit_id:
    #     switch_commit(cfg.commit_id)
    #     print("Commit ID does not match, please rerun.")
    #     exit(0)

    # load model
    device = "cuda:0"
    model_path = osp(args.model, "last.ckpt/checkpoint", "mp_rank_00_model_states.pt")
    model = build_model(cfg, model_path).to(device)

    # build denorm module
    trans = list(cfg.data.params.train_datasets.sources.values())[0].transforms
    for t in reversed(trans):
        # Finding the last ComposeAction in transforms
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
    action_length = cfg.data.params.action_length

    q01s, q99s = extract_quantile(cfg)

    for key, value in q01s.items():
        q01 = torch.zeros((action_length, predefined_dim), device=device)
        for src, tgt in composition.items():
            if isinstance(tgt, (list, tuple)):
                start, end = tgt
                if src in value:
                    q01[:, start:end] = torch.tensor(value[src], device=device)
        q01s[key] = q01

    for key, value in q99s.items():
        q99 = torch.zeros((action_length, predefined_dim), device=device)
        for src, tgt in composition.items():
            if isinstance(tgt, (list, tuple)):
                start, end = tgt
                if src in value:
                    q99[:, start:end] = torch.tensor(value[src], device=device)
        q99s[key] = q99 + 1e-6
    
    return model, (q01s, q99s), device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10086)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, (q01s, q99s), device = helper(args)
    
    try:
        server = ServerFAST(host=args.host, port=args.port, model=model, q01s=q01s, q99s=q99s, device=device)
        print(f"Starting server on {args.host}:{args.port}")
        server.start()
        server.join()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("Server interrupted")