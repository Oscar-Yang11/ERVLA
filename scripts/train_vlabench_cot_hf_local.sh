#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: train_vlabench_cot_hf_local.sh --nodes N --gpus N [options] [-- hydra_override ...]

Distributed options:
  --nodes N                 Number of nodes.
  --gpus N                  GPUs per node.
  --node-rank N             Rank of this node (default: 0).
  --master-addr ADDR        Address of node 0 for multi-node runs.
  --master-port PORT        Rendezvous port (default: 29500).

Training options:
  --max-steps N             Maximum optimizer steps (default: 1).
  --save-interval N         Checkpoint interval (default: max-steps).
  --val-check-interval N    Validation interval (default: max-steps).
  --batch-size N            Batch size per process (default: 1).
  --pretrained-model PATH   HF pretrained model directory.
  --output-dir PATH         Training output directory.
  --data-path PATH          VLABench JSON+video train/data directory.
  --cot-path PATH           VLABench COT directory.
  --project NAME            WandB project name.
  --exp-name NAME           Experiment name.
  --torchrun PATH           torchrun executable.
  -h, --help                Show this help.

Arguments after --, or unrecognized arguments, are passed to tools/train.py.
EOF
}

NODES="${NNODES:-}"
GPUS="${NPROC_PER_NODE:-}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VLABENCH_DATA="${VLABENCH_DATA:-}"
COT_DATA="${COT_DATA:-}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vlabench_cot_hf_local}"
PROJECT_NAME="${PROJECT_NAME:-vlabench_cot}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hf_pretrain_local}"
TORCHRUN="${TORCHRUN:-torchrun}"
HF_HOME="${HF_HOME:-.cache/huggingface}"
DATA_CACHE_ROOT="${DATA_CACHE_ROOT:-.cache/data}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --nodes|--nnodes)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      NODES="$2"
      shift 2
      ;;
    --gpus|--nproc-per-node)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --node-rank)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      NODE_RANK="$2"
      shift 2
      ;;
    --master-addr)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-port)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      MASTER_PORT="$2"
      shift 2
      ;;
    --max-steps)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      MAX_STEPS="$2"
      shift 2
      ;;
    --save-interval)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --val-check-interval)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      VAL_CHECK_INTERVAL="$2"
      shift 2
      ;;
    --batch-size)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      BATCH_SIZE="$2"
      shift 2
      ;;
    --pretrained-model)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      PRETRAINED_MODEL="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-path)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      VLABENCH_DATA="$2"
      shift 2
      ;;
    --cot-path)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      COT_DATA="$2"
      shift 2
      ;;
    --project)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      PROJECT_NAME="$2"
      shift 2
      ;;
    --exp-name)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --torchrun)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      TORCHRUN="$2"
      shift 2
      ;;
    --hf-home)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      HF_HOME="$2"
      shift 2
      ;;
    --data-cache-root)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      DATA_CACHE_ROOT="$2"
      shift 2
      ;;
    --max-length)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      MAX_LENGTH="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "${NODES}" && -n "${GPUS}" ]] || { echo "--nodes and --gpus are required" >&2; usage >&2; exit 2; }
[[ "${NODES}" =~ ^[1-9][0-9]*$ ]] || { echo "--nodes must be a positive integer" >&2; exit 2; }
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be a positive integer" >&2; exit 2; }
[[ "${NODE_RANK}" =~ ^[0-9]+$ && "${NODE_RANK}" -lt "${NODES}" ]] || { echo "Invalid --node-rank" >&2; exit 2; }
[[ "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] || { echo "--master-port must be a positive integer" >&2; exit 2; }
if ((NODES > 1)) && [[ "${MASTER_ADDR}" == 127.0.0.1 ]]; then
  echo "--master-addr must be the address of node 0 for a multi-node run" >&2
  exit 2
fi

[[ -n "${VLABENCH_DATA}" && -n "${COT_DATA}" && -n "${PRETRAINED_MODEL}" ]] || {
  echo "--data-path, --cot-path, and --pretrained-model are required" >&2
  exit 2
}

SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_STEPS}}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-${MAX_STEPS}}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DATA_CACHE_ROOT HF_HOME MAX_LENGTH
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export MLP_WORKER_NUM="${NODES}"

if ((NODES == 1)); then
  LAUNCH_ARGS=(--standalone --nproc_per_node="${GPUS}")
else
  LAUNCH_ARGS=(
    --nnodes="${NODES}"
    --nproc_per_node="${GPUS}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

TRAIN_ARGS=(
  +trainer.devices="${GPUS}"
  +trainer.log_every_n_steps=1
  trainer.num_nodes="${NODES}"
  trainer.max_steps="${MAX_STEPS}"
  trainer.save_interval="${SAVE_INTERVAL}"
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  trainer.num_sanity_val_steps=0
  trainer.project="${PROJECT_NAME}"
  trainer.exp_name="${EXPERIMENT_NAME}"
  trainer.default_root_dir="${OUTPUT_DIR}"
  data=vlabench/vlabench_primitive_qwen3vl_choice_CoT_fromPT
  "sources.vlabench_choice.train_path=[${VLABENCH_DATA}]"
  "sources.vlabench_choice.cot_path=${COT_DATA}"
  data.params.train_datasets.batch_size="${BATCH_SIZE}"
  model=P2_Choice_KI
  model.params.pretrained="${PRETRAINED_MODEL}"
  model.params.model.freeze_vlm=true
  model.params.model.training_repeat=1
  trainer.accumulate_grad_batches=1
)

echo "Launching VLABench: nodes=${NODES}, gpus/node=${GPUS}, node-rank=${NODE_RANK}"
exec "${TORCHRUN}" "${LAUNCH_ARGS[@]}" tools/train.py "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
