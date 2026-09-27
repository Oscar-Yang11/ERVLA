import functools
import os
import subprocess

import torch

from lightning import seed_everything

from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy
from mmengine import Config
from omegaconf import OmegaConf
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers.utils import logging

from src.models import AUTO_WRAP_LAYERS

logger = logging.get_logger(__name__)


def helper(cfg):
    """Process the configuration."""
    # Resolve config
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    process_save_cfg(cfg)

    # Set random seed for reproducibility
    seed_everything(cfg.trainer.pop("seed", 42) + int(os.environ.get("RANK")), workers=True)

    # FSDP configuration
    strategy_helper(cfg)

    # Fill model
    cfg.model.params.optimizer = cfg.trainer.pop("optimizer")
    cfg.model.params.scheduler = cfg.trainer.pop("scheduler")

    return cfg


def fsdp_helper():
    params = {
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
        "auto_wrap_policy": functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=AUTO_WRAP_LAYERS,
        ),
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
        "cpu_offload": None,
        "use_orig_params": True,
        "limit_all_gathers": True,
        "sync_module_states": True,
        "forward_prefetch": True
    }

    strategy = FSDPStrategy(**params)

    return strategy


def strategy_helper(cfg):
    """Helper function for training configuration."""
    strategy_type = cfg.trainer.strategy.type
    if strategy_type == "deepspeed":
        strategy = DeepSpeedStrategy(**cfg.trainer.strategy.params)
    elif strategy_type == "fsdp":
        strategy = fsdp_helper()
    else:
        raise TypeError("Unsupprt strategy.")
    cfg.trainer.strategy = strategy
    return cfg


def get_git_commit_id():
    # """
    # Get the current Git commit ID.

    # Returns:
    #     str: The current Git commit ID.
    # """
    # try:
    #     commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    #     return commit_id[:12]
    # except subprocess.CalledProcessError:
    #     return None
    return os.getenv("GIT_COMMIT", "unknown")


def fill_num_nodes(cfg):
    """
    Fill the number of nodes in the configuration.

    Args:
        cfg (Config): The configuration object to modify.
    """
    if cfg.trainer.num_nodes > 0:
        cfg.trainer.num_nodes = cfg.trainer.num_nodes
        assert cfg.trainer.num_nodes <= int(
            os.environ.get("MLP_WORKER_NUM", 1)
        ), "Number of nodes exceeds available workers"
    else:
        cfg.trainer.num_nodes = int(os.environ.get("MLP_WORKER_NUM", 1))


def process_save_cfg(cfg):
    """
    Save the configuration to a file.

    Args:
        cfg (Config): The configuration object to save.
    """
    root_dir = os.path.join(
        cfg.trainer.default_root_dir, f"project_{cfg.trainer.project}", cfg.trainer.exp_name
    )
    cfg.trainer.default_root_dir = root_dir

    cfg.commit_id = get_git_commit_id()

    fill_num_nodes(cfg)
    fill_data_params(cfg)
    fill_model_params(cfg)

    os.environ["_max_steps"] = str(cfg.trainer.max_steps)

    os.makedirs(root_dir, exist_ok=True)
    cfg.dump(os.path.join(root_dir, "config.yaml"))
    cfg.dump(os.path.join(root_dir, "config.py"))
    cfg.dump(os.path.join("./assets", "config.py"))


def fill_data_params(cfg):
    """
    Fill the data paths in the configuration.
    """

    def expand_path(paths):
        if isinstance(paths, str):
            paths = [paths]
        expanded = []
        for path_str in paths:
            segments = path_str.split("*")
            base = segments[0]
            if len(segments) > 1:
                count = int(segments[1])
            else:
                count = 1
            expanded.extend([base] * count)
        return expanded


    for dataset, _ in cfg.data.params.train_datasets.sources.items():
        dataset_info = cfg.sources.get(dataset)
        cur_set = cfg.data.params.train_datasets.sources[dataset]
        cur_set.path = expand_path(dataset_info.train_path)
        if dataset_info.get("cot_path", None) is not None:
            cur_set.cot_path = dataset_info.cot_path
        if dataset_info.get("is_vlm_data", False):
            cur_set.transforms = dataset_info.transforms.training
        else:
            cur_set.transforms = dataset_info.transforms.training + cfg.data.params.train_datasets.transforms

    if "val_datasets" in cfg.data.params:
        for dataset, _ in cfg.data.params.val_datasets.sources.items():
            dataset_info = cfg.sources.get(dataset)
            cur_set = cfg.data.params.val_datasets.sources[dataset]
            cur_set.path = expand_path(dataset_info.val_path)
            if dataset_info.get("cot_path", None) is not None:
                cur_set.cot_path = dataset_info.cot_path
            if dataset_info.get("is_vlm_data", False):
                cur_set.transforms = dataset_info.transforms.validation
            else:
                cur_set.transforms = dataset_info.transforms.validation + cfg.data.params.val_datasets.transforms

    os.environ["_obs_length"] = str(cfg.data.params.obs_length)
    os.environ["_obs_interval"] = str(cfg.data.params.obs_interval)
    os.environ["_action_length"] = str(cfg.data.params.action_length)
    os.environ["_add_meta_to_prompt"] = str(cfg.data.params.add_meta_to_prompt)

    cfg.data.params.accumulate_grad_batches = cfg.trainer.get("accumulate_grad_batches", 1)

    cfg.pop("sources", None)


def fill_model_params(cfg):
    params = cfg.model.params.model
    for key, value in params.items():
        if value == "auto":
            params[key] = get_auto_value(cfg, key)
    del cfg.data.params.train_datasets.transforms
    if "val_datasets" in cfg.data.params:
        del cfg.data.params.val_datasets.transforms


def get_auto_value(cfg, key):
    if key == "state_shape":
        for trans in list(cfg.data.params.train_datasets.sources.values())[0].transforms:
            if trans.type == "ComposeState":
                assert (
                    len(list(trans.reversed_data_flow.values())) == 1
                ), "auto state shape does not support more than one flow in ComposeState"
                components = list(trans.reversed_data_flow.values())[0]
                max_dim = -1
                for _, value in components.items():
                    if isinstance(value, (list, tuple)):
                        if isinstance(value[1], (list, tuple)):
                            max_dim = max(max_dim, value[1][-1])
                        else:
                            max_dim = max(max_dim, value[-1])
        return (cfg.data.params.obs_length, max_dim)
    elif key == "action_shape":
        for trans in list(cfg.data.params.train_datasets.sources.values())[0].transforms:
            if trans.type == "ComposeAction":
                assert (
                    len(list(trans.reversed_data_flow.values())) == 1
                ), "auto action shape does not support more than one flow in ComposeAction"
                components = list(trans.reversed_data_flow.values())[0]
                max_dim = -1
                for _, value in components.items():
                    if isinstance(value, (list, tuple)):
                        if isinstance(value[1], (list, tuple)):
                            max_dim = max(max_dim, value[1][-1])
                        else:
                            max_dim = max(max_dim, value[-1])
        return (cfg.data.params.action_length, max_dim)
    else:
        raise ValueError(f"Unsupported key {key} for auto")
