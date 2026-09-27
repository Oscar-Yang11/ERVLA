import os
import random
import time
from typing import Any, Dict, List, Optional

from mmengine.dataset import Compose

from streaming import Stream, StreamingDataset
from transformers.utils import logging

from src.data.datasets.json_dataset import compute_md5, JsonDataset, list_all_json_file, check_paths, compute_content_md5

logger = logging.get_logger(__name__)


class StreamDataset(StreamingDataset, JsonDataset):
    def __init__(
        self, params: Dict[str, Any], split: str = "train", shuffle: bool = False, val_set_name: Optional[str] = None
    ):
        """
        Initializes the JsonDataset for training or validation.

        Args:
            params (Dict[str, Any]): Configuration dictionary containing dataset parameters.
            split (str): Dataset split, e.g., 'train' or 'val'.
            val_set_name (Optional[str]): Name of the validation set (required if split is 'val').
        """
        self.params = params
        self.split = split
        self.val_set_name = val_set_name
        assert self.split == "train" or self.val_set_name is not None, "val_set_name must be provided for val set."

        self.batch_size = getattr(self.params, f"{self.split}_datasets").batch_size
        self.batching_method = "random"
        self.max_steps = params.get("max_steps", 1000)
        self.accumulate_grad_batches = params.get("accumulate_grad_batches", 1)

        if self.split == "train":
            self.max_samples = self.max_steps * self.batch_size * int(os.environ.get("WORLD_SIZE", 1)) * self.accumulate_grad_batches
        else:
            self.max_samples = None

        self.transforms = {}
        for source, source_params in getattr(self.params, f"{self.split}_datasets").sources.items():
            self.transforms[source] = Compose(source_params.transforms)

        self.load_data()

        self.stream_to_source = {}
        streams = self.get_streams()
        super().__init__(
            streams=streams,
            epoch_size=self.max_samples,
            batch_size=self.batch_size,
            batching_method=self.batching_method,
            shuffle=shuffle,
            allow_unsafe_types=True
        )
        self._skip_shard_lock()

    def _skip_shard_lock(self):
        """Skip filelock for local-only data (all on JuiceFS, no remote)."""
        from streaming.base.dataset import _ShardState
        for i in range(self.num_shards):
            self._shard_states[i] = _ShardState.LOCAL

    def prepare_shard(self, shard_id: int, blocking: bool = True) -> None:
        """No-op: all shards are local on JuiceFS, skip filelock."""
        pass

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a sample by index, applies transformations, and handles errors.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: Transformed sample.

        Raises:
            RuntimeError: If after max_attempts, a valid sample cannot be processed.
        """
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                sample = super().get_item(index)
                source_name = self.stream_to_source[self.streams[self.stream_per_shard[self.spanner[index][0]]]]
                return self.transforms[source_name](sample)
            except Exception as e:
                print(f"Error processing sample {sample.get('file', 'vlm data')} (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    index = random.randint(0, self.num_samples - 1)
        raise RuntimeError(f"After {max_attempts} attempts, still unable to process sample")

    def load_data(self) -> None:
        """
        Loads dataset configurations and samples for each source.
        For validation, keeps only the specified validation set.
        """
        data_params = getattr(self.params, f"{self.split}_datasets")
        for dataset, source in data_params["sources"].items():
            if self.val_set_name and dataset != self.val_set_name:
                data_params["sources"].pop(dataset)
            else:
                is_mds = check_paths(source["path"])
                if is_mds:
                    source["MDS"] = self.merge_index(source["path"], dataset)
                else:
                    source["MDS"] = self.list_samples(list_all_json_file(source["path"]), dataset)


    def get_streams(self) -> List[Stream]:
        """
        Retrieves streams by cfg.
        """
        streams = []
        data_params = getattr(self.params, f"{self.split}_datasets")["sources"]

        if self.max_samples is None:
            for source_name, params in data_params.items():
                streams.append(Stream(local=params.get("MDS")))
                self.stream_to_source[streams[-1]] = source_name
        else:
            for source_name, params in data_params.items():
                weight = params.get("weight", 1)
                streams.append(Stream(local=params.get("MDS"), proportion=weight))
                self.stream_to_source[streams[-1]] = source_name

        return streams

    def list_samples(self, files: List[str], dataset: str) -> str:
        """
        Lists all samples from the given JSON files and caches them using feather format.

        Args:
            files (List[str]): List of JSON file paths.
            dataset (str): Name of the dataset.

        Returns:
            pa.Table: Table containing samples.
        """
        if len(files) == 0:
            logger.info(f"No json file found for {dataset} {self.split} set.")
            return None
        rank = int(os.environ.get("RANK", "0"))
        current_md5 = compute_md5(files)
        cache_dir = os.path.join(os.environ.get("DATA_CACHE_ROOT", ".cache/data"), "unified", f"{current_md5}")
        done_file = os.path.join(cache_dir, ".done")

        if rank == 0:
            logger.info(f"{dataset} {self.split} set has {len(files)} files.")
            cached = self.check_cache(cache_dir, dataset, current_md5)
            if not cached:
                self.generate_cache(cache_dir, dataset, current_md5, files)
        else:
            while not os.path.exists(done_file):
                time.sleep(5)

        return os.path.join(cache_dir, "MDS")
    
    def merge_index(self, files: List[str], dataset: str) -> str:
        if len(files) == 0:
            logger.info(f"No json file found for {dataset} {self.split} set.")
            raise ValueError
        if len(files) == 1:
            return os.path.dirname(files[0])
        rank = int(os.environ.get("RANK", "0"))
        files.sort()
        current_md5 = compute_content_md5(files)
        cache_dir = os.path.join(os.environ.get("DATA_CACHE_ROOT", ".cache/data"), "unified_index", f"{current_md5}")
        done_file = os.path.join(cache_dir, ".done")

        if rank == 0:
            logger.info(f"{dataset} {self.split} set has {len(files)} index files.")
            cached = self.check_cache(cache_dir, dataset, current_md5)
            if not cached:
                os.makedirs(cache_dir, exist_ok=True)
                for i, file_path in enumerate(files):
                    link_name = f"part_{i:06d}"
                    link_path = os.path.join(cache_dir, link_name)
                    os.symlink(os.path.dirname(file_path), link_path)
                
                index_file_urls = []
                for i in range(len(files)):
                    part_dir = os.path.join(cache_dir, f"part_{i:06d}")
                    index_file_urls.append(os.path.join(part_dir, "index.json"))
                
                from streaming.base.util import merge_index
                merge_index(index_file_urls, cache_dir, keep_local=True)

                with open(done_file, "w") as f:
                    f.write(current_md5)
        else:
            while not os.path.exists(done_file):
                time.sleep(5)

        return cache_dir
       
        
