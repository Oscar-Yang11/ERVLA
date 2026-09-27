#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LIBEROPLUS_ROOT="${LIBEROPLUS_ROOT:-${REPO_ROOT}/third_party/LIBERO-plus}"
PY_BIN="${PY_BIN:-python}"
SERVER_PY_BIN="${SERVER_PY_BIN:-python}"
DISPATCH_PY=${REPO_ROOT}/benchmark/liberoplus/dispatch_liberoplus.py
SERVER_LAUNCHER=${REPO_ROOT}/src/server/launch_hf_liberoplus_servers.sh
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/benchmark/liberoplus/libero_config}"
ASSETS_ROOT="${LIBERO_ASSETS_ROOT:-${LIBEROPLUS_ROOT}/libero/libero/assets}"
STATS_CONFIG="${ACTION_STATS_CONFIG:-${REPO_ROOT}/configs/data/sources/libero.yaml}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/ervla-opensource-liberoplus}"
MODEL_NAME=ervla-opensource-liberoplus
HOST=127.0.0.1
BASE_PORT=10086
NUM_GPUS=2
SERVERS_PER_GPU=1
NUM_WORKERS=""
SESSION_NAME=hf_liberoplus
WAIT_TIMEOUT=600
LAUNCH_SERVERS=1

SUITES="libero_spatial libero_object libero_goal libero_10"
PERTURBATIONS=""
TASK_INDICES=""
TASK_NAMES=""
DIFFICULTY_LEVELS=""
MAX_TASKS=0
EPISODES_PER_TASK=1
MAX_STEPS=0
VIS=0
ENV_ONLY=0
RESUME=1
SAVE_DIR=""

usage() {
    cat <<'EOF'
Usage: bash benchmark/liberoplus/run_liberoplus_dynamic.sh [options]

Selection:
  --suites "libero_spatial ..."       Default: all four suites
  --perturbations background_textures  One or more comma/space-separated aliases
  --task-indices "0 1"                Zero-based indices within each selected suite
  --task-names "name_a name_b"
  --difficulty-levels "1 2"
  --max-tasks N                        Global cap after filtering; 0 means all
  --episodes-per-task N                Official LIBERO-Plus default is 1

Execution:
  --model-path PATH
  --model-name NAME
  --num-gpus N
  --servers-per-gpu N
  --num-workers N
  --base-port PORT
  --max-steps N                        0 uses suite defaults; use 5 for smoke test
  --vis 0|1
  --save-dir PATH
  --launch-servers 0|1
  --env-only 0|1                       Initialize and step environments without a model
  --resume 0|1

Smoke test:
  bash benchmark/liberoplus/run_liberoplus_dynamic.sh \
    --suites libero_spatial --task-indices 0 --max-tasks 1 \
    --max-steps 5 --num-gpus 1 --num-workers 1 --vis 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH=$2; shift 2 ;;
        --model-name) MODEL_NAME=$2; shift 2 ;;
        --num-gpus) NUM_GPUS=$2; shift 2 ;;
        --servers-per-gpu) SERVERS_PER_GPU=$2; shift 2 ;;
        --num-workers) NUM_WORKERS=$2; shift 2 ;;
        --base-port) BASE_PORT=$2; shift 2 ;;
        --session-name) SESSION_NAME=$2; shift 2 ;;
        --wait-timeout) WAIT_TIMEOUT=$2; shift 2 ;;
        --launch-servers) LAUNCH_SERVERS=$2; shift 2 ;;
        --suites) SUITES=$2; shift 2 ;;
        --perturbations) PERTURBATIONS=$2; shift 2 ;;
        --task-indices) TASK_INDICES=$2; shift 2 ;;
        --task-names) TASK_NAMES=$2; shift 2 ;;
        --difficulty-levels) DIFFICULTY_LEVELS=$2; shift 2 ;;
        --max-tasks) MAX_TASKS=$2; shift 2 ;;
        --episodes-per-task) EPISODES_PER_TASK=$2; shift 2 ;;
        --max-steps) MAX_STEPS=$2; shift 2 ;;
        --vis) VIS=$2; shift 2 ;;
        --env-only) ENV_ONLY=$2; shift 2 ;;
        --resume) RESUME=$2; shift 2 ;;
        --save-dir) SAVE_DIR=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

for value in "${NUM_GPUS}" "${SERVERS_PER_GPU}" "${BASE_PORT}" "${EPISODES_PER_TASK}"; do
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_GPUS, SERVERS_PER_GPU, BASE_PORT, and EPISODES_PER_TASK must be positive integers"
        exit 1
    fi
done
for value in "${VIS}" "${ENV_ONLY}" "${RESUME}" "${LAUNCH_SERVERS}"; do
    if [[ "${value}" != 0 && "${value}" != 1 ]]; then
        echo "VIS, ENV_ONLY, RESUME, and LAUNCH_SERVERS must be 0 or 1"
        exit 1
    fi
done
if ! [[ "${MAX_TASKS}" =~ ^[0-9]+$ && "${MAX_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "MAX_TASKS and MAX_STEPS must be non-negative integers"
    exit 1
fi

if [[ -z "${NUM_WORKERS}" ]]; then
    NUM_WORKERS=$((NUM_GPUS * SERVERS_PER_GPU))
fi
if ! [[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_WORKERS must be a positive integer"
    exit 1
fi
if [[ -z "${SAVE_DIR}" ]]; then
    run_tag=$(date +%Y%m%d_%H%M%S)
    SAVE_DIR=${REPO_ROOT}/results/libero_plus/${MODEL_NAME}_${run_tag}
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_DIR}
export PYTHONPATH=${LIBEROPLUS_ROOT}:${REPO_ROOT}/benchmark/liberoplus:${PYTHONPATH:-}
unset DISPLAY

launched=0
cleanup() {
    if [[ ${launched} -eq 1 ]]; then
        tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${ENV_ONLY}" == 0 && "${LAUNCH_SERVERS}" == 1 ]]; then
    PY_BIN="${SERVER_PY_BIN}" bash "${SERVER_LAUNCHER}" \
        "${MODEL_PATH}" "${NUM_GPUS}" "${BASE_PORT}" "${SESSION_NAME}" \
        "${SERVERS_PER_GPU}" "${STATS_CONFIG}" libero_all
    launched=1

    echo "Waiting for ${NUM_WORKERS} server(s)..."
    ready=0
    elapsed=0
    while [[ ${ready} -eq 0 && ${elapsed} -lt ${WAIT_TIMEOUT} ]]; do
        ready=1
        for ((worker=0; worker<NUM_WORKERS; worker++)); do
            port=$((BASE_PORT + worker))
            if ! bash -c "echo > /dev/tcp/${HOST}/${port}" 2>/dev/null; then
                ready=0
                break
            fi
        done
        if [[ ${ready} -eq 0 ]]; then
            sleep 5
            elapsed=$((elapsed + 5))
        fi
    done
    if [[ ${ready} -eq 0 ]]; then
        echo "Servers were not ready within ${WAIT_TIMEOUT}s"
        exit 1
    fi
fi

mkdir -p "${SAVE_DIR}"
command=(
    "${PY_BIN}" -u "${DISPATCH_PY}"
    --args.host "${HOST}"
    --args.base-port "${BASE_PORT}"
    --args.num-workers "${NUM_WORKERS}"
    --args.python-bin "${PY_BIN}"
    --args.suites "${SUITES}"
    --args.max-tasks "${MAX_TASKS}"
    --args.episodes-per-task "${EPISODES_PER_TASK}"
    --args.max-steps "${MAX_STEPS}"
    --args.save-dir "${SAVE_DIR}"
    --args.liberoplus-root "${LIBEROPLUS_ROOT}"
    --args.libero-config-dir "${LIBERO_CONFIG_DIR}"
    --args.assets-root "${ASSETS_ROOT}"
)
[[ -n "${PERTURBATIONS}" ]] && command+=(--args.perturbations "${PERTURBATIONS}")
[[ -n "${TASK_INDICES}" ]] && command+=(--args.task-indices "${TASK_INDICES}")
[[ -n "${TASK_NAMES}" ]] && command+=(--args.task-names "${TASK_NAMES}")
[[ -n "${DIFFICULTY_LEVELS}" ]] && command+=(--args.difficulty-levels "${DIFFICULTY_LEVELS}")
[[ "${VIS}" == 1 ]] && command+=(--args.vis)
[[ "${ENV_ONLY}" == 1 ]] && command+=(--args.env-only)
[[ "${RESUME}" == 0 ]] && command+=(--args.no-resume)

echo "[dispatch] ${command[*]}"
"${command[@]}" 2>&1 | tee "${SAVE_DIR}/dispatch.log"
echo "LIBERO-Plus evaluation finished: ${SAVE_DIR}"
