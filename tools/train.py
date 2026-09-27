import logging
import os

import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from mmengine import Config, DATASETS
from omegaconf import DictConfig
from src.models import MIMODEL

from src.utils.cfg_utils import helper
from src.utils.pl_console_logger import ConsoleLogger
from src.utils.gc_callback import GCCallback

warnings.filterwarnings("ignore")


def prepare(cfg: Dict[str, Any]) -> Tuple[Config, LightningDataModule, Any, List[Any]]:
    """Prepare the training environment."""
    ######################### Process config ###########################
    cfg: Config = helper(cfg)

    ########################## Build modules ###########################
    # dataset
    datamodule: LightningDataModule = DATASETS.build(cfg.data)

    # model
    model: LightningModule = MIMODEL.build(cfg.model)

    ########################## Build trainer ###########################
    # val_check_interval as int counts micro-batches; scale to optimizer steps
    # so it stays consistent with max_steps / save_interval under gradient accumulation.
    accumulate_grad_batches = cfg.trainer.get("accumulate_grad_batches", 1)
    val_check_interval = cfg.trainer.get("val_check_interval")
    if isinstance(val_check_interval, int) and accumulate_grad_batches > 1:
        cfg.trainer["val_check_interval"] = val_check_interval * accumulate_grad_batches

    # callbacks
    cfg.trainer["callbacks"] = [
        ModelSummary(max_depth=2),
        GCCallback(gc_interval=cfg.trainer.pop("gc_interval", 1000)),
        ModelCheckpoint(
            save_top_k=-1,
            save_last=True,
            every_n_train_steps=cfg.trainer.pop("save_interval", 10000),
            dirpath=cfg.trainer.default_root_dir,
            enable_version_counter=False,
        ),
    ]

    # logger
    logger: List[Any] = [
        WandbLogger(
                project=cfg.trainer.pop('project'),
                name=cfg.trainer.pop("exp_name"),
                entity="rfm",
                config=cfg,
            ),
        TensorBoardLogger(
            save_dir=os.getenv("TENSORBOARD_LOG_PATH", "./tensorboard_logs/"),
            name="",
            version="",
        ),
        ConsoleLogger(),
    ]

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
    resume_from_last: bool = cfg.trainer.pop("resume_from_last", False)
    if resume_from_last:
        last_ckpt = os.path.join(cfg.trainer.default_root_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
