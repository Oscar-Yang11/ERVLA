#!/usr/bin/env python3

"""Load the official LIBERO-Plus package with its external asset bundle."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import yaml

from catalog import DEFAULT_LIBEROPLUS_ROOT


DEFAULT_CONFIG_DIR = Path(os.environ.get(
    "LIBERO_CONFIG_PATH", Path(__file__).resolve().parent / "libero_config"
))
DEFAULT_ASSETS_ROOT = Path(os.environ.get(
    "LIBERO_ASSETS_ROOT", DEFAULT_LIBEROPLUS_ROOT / "libero/libero/assets"
))


def load_official_runtime(
    *,
    liberoplus_root: Path | str = DEFAULT_LIBEROPLUS_ROOT,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    assets_root: Path | str = DEFAULT_ASSETS_ROOT,
) -> Tuple[Any, Any, Any]:
    """Return official benchmark module, environment class, and path helper."""

    root = Path(liberoplus_root).resolve()
    config = Path(config_dir).resolve()
    assets = Path(assets_root).resolve()

    required = {
        "official package": root / "libero/libero/__init__.py",
        "task classification": root
        / "libero/libero/benchmark/task_classification.json",
        "LIBERO config": config / "config.yaml",
        "asset bundle": assets,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LIBERO-Plus runtime files:\n" + "\n".join(missing))

    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # Resolve the config template against the selected checkout before imports
    # capture LIBERO's path helper.
    from libero import libero

    with (config / "config.yaml").open() as handle:
        paths = {key: str(root / value) for key, value in yaml.safe_load(handle).items()}
    paths["assets"] = str(assets)

    def get_libero_path(query_key):
        return paths[query_key]

    libero.get_libero_path = get_libero_path
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_file = Path(benchmark.__file__).resolve()
    if root not in benchmark_file.parents:
        raise RuntimeError(
            "Imported a different LIBERO package. "
            f"Expected code under {root}, got {benchmark_file}"
        )

    # The official package resolves these assets relative to its source tree.
    # Keep that code unchanged while using the separately downloaded asset bundle.
    package_root = assets.parent
    bddl_base = importlib.import_module("libero.libero.envs.bddl_base_domain")
    bddl_base.DIR_PATH = str(package_root / "envs")

    for module_name in (
        "articulated_objects",
        "custom_objects",
        "google_scanned_objects",
        "hope_objects",
        "site_object",
        "target_zones",
        "turbosquid_objects",
    ):
        module = importlib.import_module(f"libero.libero.envs.objects.{module_name}")
        if hasattr(module, "absolute_path"):
            module.absolute_path = package_root

    return benchmark, OffScreenRenderEnv, get_libero_path
