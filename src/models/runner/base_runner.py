from importlib import import_module
import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from src.models import MIMODEL

logger = logging.get_logger(__name__)


@MIMODEL.register_module()
class BaseRunner(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the BaseRunner module.

        Args:
            params (Dict[str, Any]): A dictionary containing configurations for the model,
                                    optimizer, scheduler, and optionally pretrained weights.
        """
        super().__init__()
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")

    def configure_model(self) -> None:
        """
        Builds the model architecture and loads pretrained weights if provided.
        """
        self.model = MIMODEL.build(self._model)

        if self._pretrained:
            if os.path.isdir(self._pretrained):
                info = self._load_hf_checkpoint(self._pretrained)
            else:
                ckpt = torch.load(
                    self._pretrained,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
                info = self.load_state_dict(ckpt["module"], strict=False)
            logger.info(f"Loaded pretrained model from {self._pretrained}: {info}")

    def _load_hf_checkpoint(self, model_dir: str):
        """Load a sharded Hugging Face MiBoT directory into the training model.

        The released HF model stores keys without the Lightning ``model.``
        prefix.  The training runner uses the same module layout with that
        prefix, so only the prefix and the checkpoint container need adapting.
        Choice heads are intentionally absent from the base HF checkpoint and
        remain freshly initialized for choice post-training.
        """
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r") as file:
                weight_map = json.load(file)["weight_map"]
            shard_names = sorted(set(weight_map.values()))
        else:
            shard_names = ["model.safetensors"]

        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ImportError(
                "Loading an HF checkpoint requires the safetensors package"
            ) from error

        target_state = self.state_dict()
        state = {}
        shape_mismatches = []
        for shard_name in shard_names:
            shard_path = os.path.join(model_dir, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"HF checkpoint shard not found: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    target_key = key if key.startswith("model.") else f"model.{key}"
                    if target_key not in target_state:
                        continue
                    tensor = shard.get_tensor(key)
                    if tuple(tensor.shape) != tuple(target_state[target_key].shape):
                        shape_mismatches.append(
                            (target_key, tuple(tensor.shape), tuple(target_state[target_key].shape))
                        )
                        continue
                    state[target_key] = tensor

        if shape_mismatches:
            details = "; ".join(
                f"{name}: checkpoint={source}, model={target}"
                for name, source, target in shape_mismatches[:8]
            )
            raise RuntimeError(f"HF checkpoint does not match the configured model: {details}")

        info = self.load_state_dict(state, strict=False)
        allowed_missing = (
            "model.state_projector_choice.",
            "model.action_projector_choice.",
            "model.score_projector_choice.",
            # The training VLM replaces action/score token embeddings with
            # learned condition embeddings. They are intentionally absent
            # from the released HF inference checkpoint.
            "model.vlm.model.action_embed.",
            "model.vlm.model.score_embed.",
            # lm_head is tied to language_model.embed_tokens and is omitted
            # from sharded HF safetensors by Transformers.
            "model.vlm.lm_head.",
        )
        required_missing = [
            key for key in info.missing_keys if not key.startswith(allowed_missing)
        ]
        if required_missing:
            raise RuntimeError(
                "HF checkpoint is missing required training weights: "
                f"{required_missing[:20]}"
            )
        return info

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters())
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]]) -> Optimizer:
        """
        Builds an optimizer with weight decay applied conditionally based on parameter names.

        Args:
            cfg (Dict[str, Any]): Optimizer configuration dictionary, including 'type' and 'params'.
            parameters (Iterator[Tuple[str, torch.Tensor]]): Model parameters with their names.

        Returns:
            Optimizer: Constructed optimizer instance.
        """
        module_params = cfg.get("params", {})
        parameters = list(parameters)

        no_decay = [
            "bias",
            "norm",
            "Norm",
            "ln",
            "Ln",
            "rotary_emb",
            "adaln",
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": module_params.get("weight_decay", 0.1),
            },
            {
                "params": [p for n, p in parameters if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)

        return optimizer_class(optimizer_grouped_parameters, **module_params)

    @staticmethod
    def build_scheduler(cfg: Optional[Dict[str, Any]], optimizer: Optimizer) -> Optional[LRScheduler]:
        """
        Builds a learning rate scheduler from the given configuration.

        Args:
            cfg (Optional[Dict[str, Any]]): Scheduler configuration dictionary.
            optimizer (Optimizer): The optimizer to be used with the scheduler.

        Returns:
            Optional[LRScheduler]: Constructed learning rate scheduler, or None if no config is given.
        """
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing input tensors for the model.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the current training step.
        """
        self.log("train/token", batch["input_ids"].shape[1])
        self.model._global_step = self.trainer.global_step
        loss = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss.items():
            self.log(f"train/{loss_name}", loss_value)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Performs a single validation step.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing input tensors for the model.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader, useful when using multiple val datasets.

        Returns:
            torch.Tensor: Loss value for the current validation step.
        """
        loss = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss.items():
            self.log(f"val/{loss_name}", loss_value, sync_dist=True)
        return loss
