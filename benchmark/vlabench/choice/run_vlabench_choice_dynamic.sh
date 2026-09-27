#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export VLABENCH_ROOT="${VLABENCH_ROOT:-${REPO_ROOT}/third_party/VLABench}"
VLABENCH_REPO="${VLABENCH_ROOT}"

PY_BIN="${PY_BIN:-python}"
SERVER_PY_BIN="${SERVER_PY_BIN:-python}"
DISPATCH_PY="${REPO_ROOT}/benchmark/vlabench/choice/dispatch_vlabench_choice.py"
SUMMARIZE_PY="${REPO_ROOT}/benchmark/vlabench/summarize.py"
LAUNCHER_SH="${REPO_ROOT}/src/server/launch_choice_servers.sh"
HF_LAUNCHER_SH="${REPO_ROOT}/src/server/launch_hf_choice_servers.sh"

HOST=127.0.0.1
BASE_PORT=10086

NUM_WORKERS=""
NUM_GPUS=""
SERVERS_PER_GPU=1

LAUNCH_SERVERS=0
MODEL_ROOT=""
CKPT_PATH=""
SESSION_NAME="choice_ki_servers"
HF_MODEL_PATH=""
ACTION_STATS_CONFIG="${ACTION_STATS_CONFIG:-${REPO_ROOT}/configs/data/sources/vlabench_choice.yaml}"
WAIT_TIMEOUT=600

TASK_ID=vlabench_choice
REPLAN_STEPS=5
ACTION_CHUNK=10
IMAGE_SIZE=480

POS_OFFSET_X=0.0
POS_OFFSET_Y=-0.4
POS_OFFSET_Z=0.78

GRIPPER_THRESHOLD=0.2
GRIPPER_OPEN_VALUE=0.04

N_EP=50
VIS=1
TASKS=""
MODEL_NAME=""
SAVE_DIR=""

EVAL_TRACKS=(
  track_1_in_distribution
  track_2_cross_category
  track_3_common_sense
  track_4_semantic_instruction
  track_6_unseen_texture
)

usage() {
  cat <<'EOF'
Usage:
  bash run_vlabench_choice_dynamic.sh [options]

Core options:
  --model-name NAME         Model name used in result directory (required)
  --num-workers N           Number of dispatcher workers / ports to use
  --num-gpus N              Number of GPUs used for server launch
  --servers-per-gpu N       Number of servers per GPU
  --base-port PORT          Starting port
  --vis 0|1                 Whether to save videos
  --n-ep N                  Episodes per task
  --tasks "a b c"           Only run these tasks inside each track
  --save-dir PATH           Result directory (default: results/vlabench/...)

Optional auto-launch options:
  --launch-servers 0|1      Whether to auto-launch choice servers
  --model-root PATH         Experiment root dir containing config.py
  --ckpt-path PATH          Path to mp_rank_00_model_states.pt
  --session-name NAME       tmux session name
  --hf-model-path PATH      Local Hugging Face model directory
  --action-stats-config PATH
                            Training action stats used by the HF adapter
  --wait-timeout SEC        Server readiness timeout

Examples:
  # Already launched 64 servers manually:
  bash run_vlabench_choice_dynamic.sh \
    --model-name exp_xxx \
    --num-workers 64 \
    --base-port 10086 \
    --n-ep 50 --vis 1

  # Auto launch: 8 GPUs x 8 servers/GPU = 64 servers
  bash run_vlabench_choice_dynamic.sh \
    --model-name exp_xxx \
    --launch-servers 1 \
    --model-root path/to/model_root \
    --ckpt-path path/to/mp_rank_00_model_states.pt \
    --num-gpus 8 \
    --servers-per-gpu 8 \
    --base-port 10086 \
    --n-ep 50 --vis 1

  # HF-format checkpoint
  bash run_vlabench_choice_dynamic.sh \
    --model-name ervla-opensource-vlabench \
    --launch-servers 1 \
    --hf-model-path models/ervla-opensource-vlabench \
    --num-gpus 1 --servers-per-gpu 1 \
    --n-ep 1 --vis 0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --servers-per-gpu)
      SERVERS_PER_GPU="$2"
      shift 2
      ;;
    --launch-servers)
      LAUNCH_SERVERS="$2"
      shift 2
      ;;
    --model-root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --ckpt-path)
      CKPT_PATH="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --hf-model-path)
      HF_MODEL_PATH="$2"
      shift 2
      ;;
    --action-stats-config)
      ACTION_STATS_CONFIG="$2"
      shift 2
      ;;
    --wait-timeout)
      WAIT_TIMEOUT="$2"
      shift 2
      ;;
    --vis)
      VIS="$2"
      shift 2
      ;;
    --n-ep)
      N_EP="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_NAME}" ]]; then
  echo "[ERROR] --model-name is required"
  usage
  exit 1
fi

if ! [[ "${SERVERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --servers-per-gpu must be a positive integer"
  exit 1
fi

if ! [[ "${N_EP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --n-ep must be a positive integer"
  exit 1
fi

if ! [[ "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --base-port must be a positive integer"
  exit 1
fi

if [[ "${VIS}" != "0" && "${VIS}" != "1" ]]; then
  echo "[ERROR] --vis must be 0 or 1"
  exit 1
fi

if [[ "${LAUNCH_SERVERS}" != "0" && "${LAUNCH_SERVERS}" != "1" ]]; then
  echo "[ERROR] --launch-servers must be 0 or 1"
  exit 1
fi

if [[ -n "${NUM_WORKERS}" ]]; then
  if ! [[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num-workers must be a positive integer"
    exit 1
  fi
else
  if [[ -z "${NUM_GPUS}" ]]; then
    echo "[ERROR] Please provide either --num-workers, or (--num-gpus and optionally --servers-per-gpu)"
    exit 1
  fi
  if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num-gpus must be a positive integer"
    exit 1
  fi
  NUM_WORKERS=$((NUM_GPUS * SERVERS_PER_GPU))
fi

if [[ "${LAUNCH_SERVERS}" == "1" ]]; then
  if [[ -z "${NUM_GPUS}" ]]; then
    echo "[ERROR] --launch-servers 1 requires --num-gpus"
    exit 1
  fi
  if [[ -n "${HF_MODEL_PATH}" ]]; then
    echo "[launch] starting HF choice servers..."
    PY_BIN="${SERVER_PY_BIN}" bash "${HF_LAUNCHER_SH}" \
      "${HF_MODEL_PATH}" \
      "${NUM_GPUS}" \
      "${BASE_PORT}" \
      "${SESSION_NAME}" \
      "${SERVERS_PER_GPU}" \
      "${ACTION_STATS_CONFIG}"
  else
    if [[ -z "${MODEL_ROOT}" || -z "${CKPT_PATH}" ]]; then
      echo "[ERROR] --launch-servers 1 requires --model-root and --ckpt-path, or --hf-model-path"
      exit 1
    fi

    echo "[launch] starting choice servers..."
    PY_BIN="${SERVER_PY_BIN}" bash "${LAUNCHER_SH}" \
      "${MODEL_ROOT}" \
      "${CKPT_PATH}" \
      "${NUM_GPUS}" \
      "${BASE_PORT}" \
      "${SESSION_NAME}" \
      "${SERVERS_PER_GPU}"
  fi

  echo "[launch] waiting for ${NUM_WORKERS} server(s) to be ready..."
  server_ready=0
  elapsed=0
  while [ $server_ready -eq 0 ] && [ $elapsed -lt $WAIT_TIMEOUT ]; do
      server_ready=1
      for ((i=0; i<NUM_WORKERS; i++)); do
          port=$((BASE_PORT + i))
          if ! bash -c "echo > /dev/tcp/${HOST}/${port}" 2>/dev/null; then
              server_ready=0
              break
          fi
      done
      if [ $server_ready -eq 0 ]; then
          sleep 5
          elapsed=$((elapsed + 5))
          if (( elapsed % 30 == 0 )); then
              echo "[launch] still waiting... (${elapsed}s elapsed)"
          fi
      fi
  done

  if [ $server_ready -eq 0 ]; then
      echo "[ERROR] Servers not ready within ${WAIT_TIMEOUT}s"
      exit 1
  fi
  echo "[launch] all servers ready (took ${elapsed}s)"
fi

RUN_TAG=$(date +%Y%m%d_%H%M)
ROOT_SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/results/vlabench/${MODEL_NAME}_${RUN_TAG}}"

export PYTHONPATH="${REPO_ROOT}/benchmark/vlabench/choice:${VLABENCH_REPO}:${PYTHONPATH:-}"

mkdir -p "${ROOT_SAVE_DIR}"

echo "[config] model_name=${MODEL_NAME}"
echo "[config] num_workers=${NUM_WORKERS}"
echo "[config] base_port=${BASE_PORT}"
echo "[config] host=${HOST}"
echo "[config] save_root=${ROOT_SAVE_DIR}"

for EVAL_TRACK in "${EVAL_TRACKS[@]}"; do
  SAVE_DIR="${ROOT_SAVE_DIR}/${EVAL_TRACK}"
  mkdir -p "${SAVE_DIR}"

  CMD=(
    "${PY_BIN}" -u "${DISPATCH_PY}"
    --args.python-bin "${PY_BIN}"
    --args.host "${HOST}"
    --args.base-port "${BASE_PORT}"
    --args.num-workers "${NUM_WORKERS}"
    --args.task-id "${TASK_ID}"
    --args.n-episode "${N_EP}"
    --args.save-dir "${SAVE_DIR}"
    --args.replan-steps "${REPLAN_STEPS}"
    --args.action-chunk-size "${ACTION_CHUNK}"
    --args.image-size "${IMAGE_SIZE}"
    --args.metrics "success_rate intention_score progress_score"
    --args.pos-offset-x "${POS_OFFSET_X}"
    --args.pos-offset-y "${POS_OFFSET_Y}"
    --args.pos-offset-z "${POS_OFFSET_Z}"
    --args.gripper-threshold "${GRIPPER_THRESHOLD}"
    --args.gripper-open-value "${GRIPPER_OPEN_VALUE}"
    --args.eval-track "${EVAL_TRACK}"
  )

  if [[ "${VIS}" == "1" ]]; then
    CMD+=(--args.visulization)
  else
    CMD+=(--args.no-visulization)
  fi

  if [[ -n "${TASKS}" ]]; then
    CMD+=(--args.tasks "${TASKS}")
  fi

  echo "[dispatch][${EVAL_TRACK}] ${CMD[*]}"
  "${CMD[@]}" | tee "${SAVE_DIR}/dispatch.log"

  echo "========== DONE ${EVAL_TRACK} =========="
  echo "save_dir: ${SAVE_DIR}"
  echo "dispatch_log: ${SAVE_DIR}/dispatch.log"
  echo
done

echo "All tracks finished."
echo "root_save_dir: ${ROOT_SAVE_DIR}"

echo "[summary] generating summary artifacts..."
"${PY_BIN}" "${SUMMARIZE_PY}" \
  --root_dir "${ROOT_SAVE_DIR}" \
  --model_name "${MODEL_NAME}" \
  --out_dir "${ROOT_SAVE_DIR}" \
  --out_name "vlabench_summary.xlsx" \
  --fig_name "vlabench_figure.png" \
  --fig_stack_name "vlabench_figure_stack.png" \
  --md_name "vlabench_table.md"

echo "[summary] done"
echo "summary_xlsx: ${ROOT_SAVE_DIR}/vlabench_summary.xlsx"
echo "summary_png:  ${ROOT_SAVE_DIR}/vlabench_figure.png"
echo "stack_png:    ${ROOT_SAVE_DIR}/vlabench_figure_stack.png"
echo "summary_md:   ${ROOT_SAVE_DIR}/vlabench_table.md"
