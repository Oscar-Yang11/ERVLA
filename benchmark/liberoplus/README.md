# MiBot LIBERO-Plus evaluation

Install the official LIBERO-Plus simulator and download its assets, then set
the checkout, assets, and Python paths from the repository root:

```bash
export LIBEROPLUS_ROOT=path/to/LIBERO-plus
export LIBERO_ASSETS_ROOT=path/to/LIBERO-plus/libero/libero/assets
export PY_BIN=path/to/liberoplus-env/bin/python
export SERVER_PY_BIN=path/to/ervla-env/bin/python
```

The evaluator reads the official task map, BDDL files, initial states, and
classification file from `LIBEROPLUS_ROOT`. Paths in
[libero_config/config.yaml](libero_config/config.yaml) are relative to that
checkout; `LIBERO_ASSETS_ROOT` can point to a separately downloaded asset bundle.

The evaluator preserves the LIBERO training contract: 256x256 Base and
Left-Wrist images rotated by 180 degrees, an 8D state, 10 predicted actions,
and five executed actions per replanning request. It writes one JSON record per
episode and summarizes success by suite, perturbation type, and difficulty.

The official classification leaves 121 `libero_goal` light-condition tasks
without a difficulty level. They are retained and reported as `unclassified`;
numeric difficulty filters exclude them.

Run one end-to-end smoke episode:

```bash
bash benchmark/liberoplus/run_liberoplus_dynamic.sh \
  --model-path path/to/hf/ervla-opensource-liberoplus \
  --suites libero_spatial --task-indices 0 --max-tasks 1 \
  --max-steps 5 --num-gpus 1 --num-workers 1 --vis 0
```

Run one complete task suite:

```bash
bash benchmark/liberoplus/run_liberoplus_dynamic.sh \
  --model-path path/to/hf/ervla-opensource-liberoplus \
  --suites libero_spatial --episodes-per-task 1 --vis 0
```

Run one perturbation type across all four suites:

```bash
bash benchmark/liberoplus/run_liberoplus_dynamic.sh \
  --model-path path/to/hf/ervla-opensource-liberoplus \
  --perturbations background_textures --episodes-per-task 1 --vis 0
```

Use `--launch-servers 0` when compatible servers are already listening on the
requested consecutive ports. Use `--env-only 1` to validate official task
assets and environment initialization without loading a model.
