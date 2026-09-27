import copy
import glob
import hashlib
import json
import os
import random
import shutil
import time
from functools import lru_cache
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from mmengine.dataset import Compose
from streaming import MDSWriter
from streaming.base.util import merge_index
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers.utils import logging

logger = logging.get_logger(__name__)


def read_json(ann_file: str) -> Dict[str, Any]:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        ann_file (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    with open(ann_file, "r") as f:
        return json.load(f)


@lru_cache(maxsize=128)
def _read_json_cached(ann_file: str) -> Dict[str, Any]:
    """Cache episode and COT JSON reads within each data-loader worker."""
    return read_json(ann_file)


def compute_md5(file_paths: List[str]) -> str:
    """
    Computes the MD5 hash of the concatenated file paths.

    Args:
        file_paths (List[str]): List of file paths to hash.

    Returns:
        str: Hex digest of the MD5 hash.
    """
    path = "".join(file_paths)
    md5_hash = hashlib.md5()
    md5_hash.update(path.encode())
    return md5_hash.hexdigest()

def compute_content_md5(file_paths):
    if not file_paths:
        return "empty"
    md5 = hashlib.md5()
    for path in sorted(file_paths):
        md5.update(path.encode('utf-8'))
        with open(path, 'rb') as f:
            md5.update(f.read())
    return md5.hexdigest()

def list_all_json_file(paths: List[str], recursive: bool = True) -> List[str]:
    """
    Lists all JSON files in the given paths, including subdirectories if recursive is True.

    Args:
        paths (List[str]): List of directories or files to search for JSON files.
        recursive (bool): Whether to search recursively in subdirectories.

    Returns:
        List[str]: Sorted list of JSON file paths.
    """
    json_files = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".json"):
            json_files.append(path)
        elif os.path.isdir(path):
            pattern = os.path.join(path, "**", "*.json") if recursive else os.path.join(path, "*.json")
            json_files.extend(glob.glob(pattern, recursive=recursive))

    json_files.sort()
    return json_files

def write_shard(args):
    """
    Standalone function to write a shard to MDS format.
    Must be at module level to be picklable by multiprocessing.
    """
    rows, shard_dir, cols, size_limit = args
    os.makedirs(shard_dir, exist_ok=True)
    with MDSWriter(out=shard_dir, columns=cols, size_limit=size_limit) as writer:
        for row in rows:
            writer.write(row)
    return len(rows)

def check_paths(paths):
    """
    Check if all paths in the given list end with 'index.json'.

    Args:
        paths (list[str]): A list of file paths as strings.

    Returns:
        bool: True if all paths end with 'index.json', False otherwise.
    """
    if isinstance(paths, str):
        paths = [paths]
    return all(path.endswith("index.json") for path in paths)


class JsonDataset(Dataset):
    def __init__(self, params: Dict[str, Any], split: str = "train", val_set_name: Optional[str] = None):
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
        self._raw_json = False
        assert self.split == "train" or self.val_set_name is not None, "val_set_name must be provided for val set."

        batch_size = getattr(self.params, f"{self.split}_datasets").batch_size
        max_steps = params.get("max_steps", 1000)
        accumulate_grad_batches = params.get("accumulate_grad_batches", 1)

        if self.split == "train":
            self.max_samples = max_steps * batch_size * int(os.environ.get("WORLD_SIZE", 1)) * accumulate_grad_batches
        else:
            self.max_samples = None

        self.transforms = {}
        for source, source_params in getattr(self.params, f"{self.split}_datasets").sources.items():
            self.transforms[source] = Compose(source_params.transforms)

        self.load_data()
        self.samples = self.mix_data()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a sample by index, applies transformations, and handles errors.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: Transformed sample.

        Raises:
            RuntimeError: If after max_attempts, a valid sample cannot be processed.
        """
        if not self._raw_json:
            raise NotImplementedError("The JSON loader requires raw JSON+video paths, not an MDS index.")

        row = self.samples[index]
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                sample = {
                    "file": row["file"],
                    "frame_index": int(row["frame_index"]),
                    "dataset": row["dataset"],
                }
                sample.update(copy.deepcopy(_read_json_cached(row["file"])))

                cot_file = row.get("cot_file")
                if cot_file:
                    cot_data = copy.deepcopy(_read_json_cached(cot_file))
                    frames = cot_data.get("frames")
                    frame_index = sample["frame_index"]
                    if not isinstance(frames, list) or frame_index >= len(frames):
                        raise ValueError(
                            f"COT frame {frame_index} is missing in {cot_file}"
                        )
                    cot_frame = frames[frame_index]
                    if cot_frame.get("t", frame_index) != frame_index:
                        raise ValueError(
                            f"COT frame index mismatch in {cot_file}: "
                            f"expected {frame_index}, got {cot_frame.get('t')}"
                        )
                    sample["cot"] = cot_frame

                return self.transforms[row["dataset"]](sample)
            except Exception as error:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Unable to process JSON sample {row.get('file', '<unknown>')} "
                        f"after {max_attempts} attempts"
                    ) from error
                row = self.samples[random.randrange(len(self.samples))]

    def load_data(self) -> None:
        """
        Loads dataset configurations and samples for each source.
        For validation, keeps only the specified validation set.
        """
        data_params = getattr(self.params, f"{self.split}_datasets")
        for dataset, source in list(data_params["sources"].items()):
            if self.val_set_name and dataset != self.val_set_name:
                del data_params["sources"][dataset]
            else:
                paths = source["path"]
                source["files"] = list_all_json_file(paths)
                if not check_paths(paths):
                    self._raw_json = True
                    source["samples"] = self.list_raw_samples(
                        source["files"], dataset, source.get("cot_path")
                    )
                else:
                    source["samples"] = self.list_samples(source["files"], dataset)

        if self._raw_json:
            sources = getattr(self.params, f"{self.split}_datasets")["sources"]
            if any(check_paths(source["path"]) for source in sources.values()):
                raise ValueError("Raw JSON and MDS sources cannot be mixed in one JsonDataset")

    def mix_data(self) -> Union[pa.Table, List[Dict[str, Any]]]:
        """
        Mixes data from different sources based on weights and ensures the total number of samples matches max_samples.

        Returns:
            pa.Table: Mixed PyArrow Table containing samples.
        """

        if self._raw_json:
            return self.mix_raw_data()

        def to_large_string_table(table: pa.Table) -> pa.Table:
            """
            Converts string columns in a PyArrow table to large_string type.

            Args:
                table (pa.Table): Input PyArrow table.

            Returns:
                pa.Table: Table with string columns cast to large_string.
            """
            fields = []
            arrays = []
            for field in table.schema:
                col = table.column(field.name)
                if pa.types.is_string(field.type):
                    fields.append(pa.field(field.name, pa.large_string()))
                    arrays.append(col.cast(pa.large_string()))
                else:
                    fields.append(field)
                    arrays.append(col)
            new_schema = pa.schema(fields)
            return pa.Table.from_arrays(arrays, schema=new_schema)

        tables = []

        data_params = getattr(self.params, f"{self.split}_datasets")["sources"]

        if self.max_samples is None:
            for _, params in data_params.items():
                tables.append(to_large_string_table(params["samples"]))
            samples = pa.concat_tables(tables)
            self.params = None
            return samples

        total_weight = sum(params.get("weight", 1) for params in data_params.values())

        for _, params in data_params.items():
            weight = params.get("weight", 1)
            data_table = to_large_string_table(params["samples"])
            data_len = data_table.num_rows

            weight_ratio = weight / total_weight
            num_samples_to_sample = int(self.max_samples * weight_ratio)

            num_repeats = num_samples_to_sample // data_len
            num_remaining_samples = num_samples_to_sample % data_len

            if num_repeats > 0:
                repeated_table = pa.concat_tables([data_table] * num_repeats)
                tables.append(repeated_table)

            if num_remaining_samples > 0:
                idx = np.random.choice(data_len, size=num_remaining_samples, replace=False)
                remaining_table = data_table.take(pa.array(idx))
                tables.append(remaining_table)

        mixed_table = pa.concat_tables(tables)
        mixed_table = to_large_string_table(mixed_table)

        num_to_fill = self.max_samples - mixed_table.num_rows
        if num_to_fill > 0:
            idx = np.random.choice(mixed_table.num_rows, size=num_to_fill, replace=True)
            fill_table = mixed_table.take(pa.array(idx))
            mixed_table = pa.concat_tables([mixed_table, fill_table])
            mixed_table = to_large_string_table(mixed_table)

        if mixed_table.num_rows > self.max_samples:
            mixed_table = mixed_table.slice(0, self.max_samples)

        assert mixed_table.num_rows == self.max_samples

        return mixed_table

    def list_raw_samples(
        self, files: List[str], dataset: str, cot_path: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Build frame references directly from raw JSON episode files.

        The references keep the original JSON and video paths. No Feather or MDS
        cache is written for this loader.
        """
        if not files:
            raise ValueError(f"No JSON files found for {dataset} {self.split}")
        if isinstance(cot_path, (list, tuple)):
            if len(cot_path) != 1:
                raise ValueError("A raw JSON source accepts one cot_path root")
            cot_path = cot_path[0]

        samples = []
        for file in files:
            content = _read_json_cached(file)
            num_frames = int(content.get("num_frames", 0))
            if num_frames <= 0:
                logger.warning(f"File {file} has no frames, skipping")
                continue

            cot_file = None
            if cot_path:
                cot_file = self.cot_file_for_json(file, cot_path)
                if not os.path.isfile(cot_file):
                    raise FileNotFoundError(f"COT JSON not found for {file}: {cot_file}")
                cot_content = _read_json_cached(cot_file)
                cot_frames = cot_content.get("frames", [])
                if len(cot_frames) != num_frames:
                    raise ValueError(
                        f"Raw/COT frame count mismatch for {file}: "
                        f"raw={num_frames}, cot={len(cot_frames)}"
                    )

            samples.extend(
                {
                    "file": file,
                    "type": "action",
                    "frame_index": frame_index,
                    "dataset": dataset,
                    "cot_file": cot_file,
                }
                for frame_index in range(num_frames)
            )

        if not samples:
            raise ValueError(f"{dataset} {self.split} set has no valid frames")
        logger.info(f"{dataset} {self.split} raw JSON set has {len(samples)} frames")
        return samples

    @staticmethod
    def cot_file_for_json(raw_json_path: str, cot_root: str) -> str:
        chunk_name = os.path.basename(os.path.dirname(raw_json_path))
        episode_name = os.path.splitext(os.path.basename(raw_json_path))[0]
        return os.path.join(cot_root, chunk_name, f"{episode_name}_ecot.json")

    def mix_raw_data(self) -> List[Dict[str, Any]]:
        data_params = getattr(self.params, f"{self.split}_datasets")["sources"]
        source_samples = []
        for _, params in data_params.items():
            source_samples.append((params["samples"], params.get("weight", 1.0)))

        if self.max_samples is None:
            return [sample for samples, _ in source_samples for sample in samples]

        total_weight = sum(weight for _, weight in source_samples)
        mixed = []
        for samples, weight in source_samples:
            target = int(self.max_samples * weight / total_weight)
            if target <= 0:
                continue
            if target <= len(samples):
                indices = np.random.choice(len(samples), size=target, replace=False)
                mixed.extend(samples[int(index)] for index in indices)
            else:
                repeats, remainder = divmod(target, len(samples))
                mixed.extend(samples * repeats)
                if remainder:
                    indices = np.random.choice(len(samples), size=remainder, replace=False)
                    mixed.extend(samples[int(index)] for index in indices)

        while len(mixed) < self.max_samples:
            samples, _ = random.choices(
                source_samples, weights=[weight for _, weight in source_samples], k=1
            )[0]
            mixed.append(random.choice(samples))
        random.shuffle(mixed)
        return mixed[: self.max_samples]

    def list_samples(self, files: List[str], dataset: str) -> pa.Table:
        """
        Lists all samples from the given JSON files and caches them using feather format.

        Args:
            files (List[str]): List of JSON file paths.
            dataset (str): Name of the dataset.

        Returns:
            pa.Table: Table containing samples.
        """
        assert len(files) > 0, f"No json file found for {dataset} {self.split} set. Please remove it from the config."
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

        # load cache
        samples = feather.read_table(os.path.join(cache_dir, "index.feather"))
        if rank == 0:
            logger.info(f"{dataset} {self.split} has {len(samples)} samples.")
        return samples

    def check_cache(self, cache_dir: str, dataset: str, current_md5: str) -> bool:
        """
        Checks whether a valid cache exists for the given dataset.

        Args:
            cache_dir (str): Directory where the cache is stored.
            dataset (str): Name of the dataset.
            current_md5 (str): MD5 hash of current file paths.

        Returns:
            bool: True if the cache is valid, False otherwise.
        """
        done_file = os.path.join(cache_dir, ".done")
        if os.path.exists(cache_dir) and os.path.exists(done_file):
            with open(done_file, "r") as f:
                cached_md5 = f.read().strip()

            if cached_md5 == current_md5:
                logger.info(f"{dataset} {self.split} set has cache in {cache_dir}.")
                return True
            else:
                logger.warning(f"MD5 mismatch: cached {cached_md5} != current {current_md5}, removing")
                shutil.rmtree(cache_dir)
                return False
        else:
            logger.warning(f"{dataset} {self.split} dataset has no cache")
            return False

    def generate_cache(self, cache_dir: str, dataset: str, current_md5: str, files: List[str]) -> None:
        """
        Generates a cache by reading all JSON files and converting them to a PyArrow table.

        Args:
            cache_dir (str): Directory to store the cache.
            dataset (str): Name of the dataset.
            current_md5 (str): MD5 hash of current file paths.
            files (List[str]): List of JSON file paths.

        Returns:
            pa.Table: Table containing all samples.
        """
        os.makedirs(cache_dir, exist_ok=True)

        samples = []
        args = ((file, dataset) for file in files)
        with Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=len(files), desc="Processing Annotations") as pbar:
                for result in pool.starmap(self.generate_samples_from_json, args):
                    samples.extend(result)
                    pbar.update()

        original_state = random.getstate()
        random.seed(42)
        random.shuffle(samples)
        random.setstate(original_state)

        samples = pa.Table.from_pylist(samples)
        save_path = os.path.join(cache_dir, "index.feather")
        feather.write_feather(samples, save_path)

        column_encodings = {
            "file": "str",
            "type": "str",
            "frame_index": "int",
        }

        num_workers = max(32, len(samples) // 2000000)
        logger.info(f"Using {num_workers} worker to generate cache.")
        self.write_mds_parallel(
            table=samples,
            out_dir=os.path.join(cache_dir, "MDS"),
            columns=column_encodings,
            num_workers=num_workers,
            size_limit="128kb",
        )

        done_file = os.path.join(cache_dir, ".done")
        with open(done_file, "w") as f:
            f.write(current_md5)

        logger.info(f"Generate {dataset} {self.split} set cache in {cache_dir}.")
    
    @staticmethod
    def write_mds_parallel(
        table: pa.Table,
        out_dir: str,
        columns: Dict[str, str],
        num_workers: int = 8,
        size_limit: str = "128kb",
    ) -> None:
        """
        Write a PyArrow Table to MDS format in parallel.

        Data is split into shards and written to temporary subdirectories (shard_0000, ...).
        After writing, all indices are merged into a single index.json at out_dir.
        Optionally, the temporary shard directories are fully removed.

        Args:
            table (pa.Table): Input data table.
            out_dir (str): Output directory for the final MDS dataset.
            columns (Dict[str, str]): Column schema, e.g., {"col": "str"}.
            num_workers (int): Number of parallel processes.
            size_limit (str): MDS shard size limit (e.g., "128kb", "1MB").
        """
        os.makedirs(out_dir, exist_ok=True)
        num_rows = len(table)
        if num_rows == 0:
            print("Warning: Empty table, nothing to write.")
            return

        # Split data into chunks
        shard_size = (num_rows + num_workers - 1) // num_workers
        batches: List[List[Dict[str, Any]]] = [
            table.slice(i, shard_size).to_pylist()
            for i in range(0, num_rows, shard_size)
        ]

        # Generate task args: (data, shard_dir, columns, size_limit)
        shard_dirs = []  # Track for cleanup

        def get_task_args():
            for idx, batch in enumerate(batches):
                shard_dir = os.path.join(out_dir, f"shard_{idx:04d}")
                shard_dirs.append(shard_dir)
                yield (batch, shard_dir, columns, size_limit)

        # Parallel write using top-level function
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(write_shard, get_task_args()), total=len(batches), desc="Writing shards"))

        print(f"Finished writing {sum(results)} samples across {len(batches)} shards")

        # Merge all shard indices into the root index.json
        print("Merging shard indices...")
        merge_index(out_dir, keep_local=True)

        print(f"MDS dataset complete: {out_dir}")

    @staticmethod
    def generate_samples_from_json(file: str, dataset: str) -> List[Dict[str, Union[str, int]]]:
        """
        Generates a list of samples from a single JSON file.

        Args:
            file (str): Path to the JSON file.
            dataset (str): Name of the dataset.

        Returns:
            List[Dict[str, Union[str, int]]]: List of samples, each represented as a dictionary.
        """
        try:
            json_data = read_json(file)
        except Exception as e:
            raise ValueError(f"Error loading {file}: {e}")

        num_frames = json_data["num_frames"]
        return [
            {
                "file": file,
                "type": "action",
                "frame_index": frame_i,
            }
            for frame_i in range(num_frames)
        ]
