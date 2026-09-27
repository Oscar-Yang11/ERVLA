import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule

from mmengine import Config, DATASETS
from torch.utils.data import DataLoader

from src.data.collate.qwen_2_5_vl_data_collate import CustomCollate
from src.data.datasets.json_dataset import JsonDataset
from src.data.datasets.stream_dataset import StreamDataset

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 9600))


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config):
        super().__init__()
        self.params: Config = params
        self.train_batch_size: int = params.train_datasets.get("batch_size", 16)
        if "val_datasets" in params:
            self.val_batch_size: int = params.val_datasets.get("batch_size", 16)
        self.loader_type: str = params.get("type", "stream")
        assert self.loader_type in ["stream", "json"], f"Only support [stream, json], but get {self.loader_type}"
        self._train_dataset: Optional[StreamDataset] = None
        self._pending_dataset_state: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        if self.loader_type != "stream" or self._train_dataset is None or self.trainer is None:
            return {}
        global_step = self.trainer.global_step
        accumulate = self.trainer.accumulate_grad_batches
        world_size = self.trainer.world_size
        num_samples = global_step * self.train_batch_size * world_size * accumulate
        return {"stream_dataset_state": self._train_dataset.state_dict(num_samples, from_beginning=True)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._pending_dataset_state = state.get("stream_dataset_state")

    def train_dataloader(self) -> DataLoader:
        if self.loader_type == "stream":
            self._train_dataset = StreamDataset(deepcopy(self.params), split="train", shuffle=True)
            if self._pending_dataset_state is not None:
                self._train_dataset.load_state_dict(self._pending_dataset_state)
                self._pending_dataset_state = None
            train_set = self._train_dataset
        else:
            train_set = JsonDataset(deepcopy(self.params), split="train")
        return DataLoader(
            train_set,
            batch_size=self.train_batch_size,
            shuffle=(self.loader_type != "stream"),
            num_workers=8,
            prefetch_factor=4,
            collate_fn=CustomCollate(max_length=MAX_LENGTH),
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[List[DataLoader]]:
        if "val_datasets" not in self.params:
            return []
        val_loaders: List[DataLoader] = []
        val_sources = self.params["val_datasets"]["sources"]
        assert isinstance(val_sources, dict), "val_datasets.sources must be a dict"

        for dataset_name, _ in val_sources.items():
            if self.loader_type == "stream":
                val_set = StreamDataset(deepcopy(self.params), split="val", val_set_name=dataset_name)
            else:
                val_set = JsonDataset(deepcopy(self.params), split="val", val_set_name=dataset_name)

            val_loader = DataLoader(
                val_set,
                batch_size=self.val_batch_size,
                num_workers=8,
                prefetch_factor=4,
                collate_fn=CustomCollate(max_length=MAX_LENGTH),
                persistent_workers=True,
                pin_memory=True,
            )
            val_loaders.append(val_loader)
        return val_loaders
