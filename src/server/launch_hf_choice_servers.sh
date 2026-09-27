#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_HOME}"

PY_BIN="${PY_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash src/server/launch_hf_choice_servers.sh <model_path> <num_gpus> [base_port] [session_name] [servers_per_gpu] [stats_config]

Example:
  bash src/server/launch_hf_choice_servers.sh \
    path/to/hf/ervla-opensource-vlabench \
    1 10086 hf_choice_servers 1 \
    configs/data/sources/vlabench_choice.yaml
EOF
}

if [[ $# -lt 2 || $# -gt 6 ]]; then
    usage
    exit 1
fi

MODEL_PATH=$1
NUM_GPUS=$2
BASE_PORT=${3:-10086}
SESSION_NAME=${4:-hf_choice_servers}
SERVERS_PER_GPU=${5:-1}
STATS_CONFIG=${6:-${REPO_ROOT}/configs/data/sources/vlabench_choice.yaml}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Error: model path not found: ${MODEL_PATH}"
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: HF config.json not found under: ${MODEL_PATH}"
    exit 1
fi
if [[ ! -f "${STATS_CONFIG}" ]]; then
    echo "Error: action stats config not found: ${STATS_CONFIG}"
    exit 1
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_gpus must be a positive integer"
    exit 1
fi
if ! [[ "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: base_port must be a positive integer"
    exit 1
fi
if ! [[ "${SERVERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: servers_per_gpu must be a positive integer"
    exit 1
fi
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
    echo "Error: Python executable not found: ${PY_BIN}"
    exit 1
fi

PY_BIN="$(realpath -s "$(command -v "${PY_BIN}")")"
MODEL_PATH="$(realpath "${MODEL_PATH}")"
STATS_CONFIG="$(realpath "${STATS_CONFIG}")"
HF_HOME="$(realpath -m "${HF_HOME}")"

NUM_SERVERS=$((NUM_GPUS * SERVERS_PER_GPU))

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}"

setup_window() {
    local target=$1
    tmux send-keys -t "${target}" "export HF_HOME='${HF_HOME}'" Enter
    tmux send-keys -t "${target}" "export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'" Enter
    tmux send-keys -t "${target}" "export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'" Enter
    tmux send-keys -t "${target}" "export OMP_NUM_THREADS=1" Enter
    tmux send-keys -t "${target}" "export MKL_NUM_THREADS=1" Enter
    tmux send-keys -t "${target}" "export OPENBLAS_NUM_THREADS=1" Enter
    tmux send-keys -t "${target}" "export NUMEXPR_NUM_THREADS=1" Enter
    tmux send-keys -t "${target}" "export PYTHONUNBUFFERED=1" Enter
    tmux send-keys -t "${target}" "export PYTHONPATH='${REPO_ROOT}':\${PYTHONPATH:-}" Enter
    tmux send-keys -t "${target}" "cd '${REPO_ROOT}'" Enter
}

server_idx=0
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<SERVERS_PER_GPU; slot++)); do
        PORT=$((BASE_PORT + server_idx))
        WINDOW_NAME="g${gpu}-s$(printf "%02d" "${slot}")"

        if [[ ${server_idx} -eq 0 ]]; then
            tmux rename-window -t "${SESSION_NAME}:0" "${WINDOW_NAME}"
        else
            tmux new-window -d -t "${SESSION_NAME}" -n "${WINDOW_NAME}"
        fi

        setup_window "${SESSION_NAME}:${WINDOW_NAME}"
        CMD="CUDA_VISIBLE_DEVICES=${gpu} '${PY_BIN}' -u '${REPO_ROOT}/src/server/deploy_hf_choice.py' --model-path '${MODEL_PATH}' --port ${PORT} --robot-type vlabench_choice --action-stats-config '${STATS_CONFIG}'"
        tmux send-keys -t "${SESSION_NAME}:${WINDOW_NAME}" "${CMD}" Enter
        echo "[launch] server_idx=${server_idx} gpu=${gpu} slot=${slot} port=${PORT}"
        server_idx=$((server_idx + 1))
    done
done

echo
echo "Started ${NUM_SERVERS} HF choice server(s) in tmux session '${SESSION_NAME}'"
echo "Model path      : ${MODEL_PATH}"
echo "Stats config    : ${STATS_CONFIG}"
echo "Python          : ${PY_BIN}"
echo "Ports           : $(seq -s ', ' "${BASE_PORT}" "$((BASE_PORT + NUM_SERVERS - 1))")"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "Kill with:"
echo "  tmux kill-session -t ${SESSION_NAME}"
