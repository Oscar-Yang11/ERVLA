#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PY_BIN="${PY_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash src/server/launch_hf_liberoplus_servers.sh MODEL_PATH NUM_GPUS [BASE_PORT] [SESSION_NAME] [SERVERS_PER_GPU] [STATS_CONFIG] [STATS_SOURCE]

Example:
  bash src/server/launch_hf_liberoplus_servers.sh \
    path/to/hf/ervla-opensource-liberoplus \
    2 10086 hf_liberoplus 1
EOF
}

if [[ $# -lt 2 || $# -gt 7 ]]; then
    usage
    exit 1
fi

MODEL_PATH=$1
NUM_GPUS=$2
BASE_PORT=${3:-10086}
SESSION_NAME=${4:-hf_liberoplus}
SERVERS_PER_GPU=${5:-1}
STATS_CONFIG=${6:-${REPO_ROOT}/configs/data/sources/libero.yaml}
STATS_SOURCE=${7:-libero_all}
LOG_DIR="${REPO_ROOT}/outputs/server_logs/${SESSION_NAME}"

for value in "${NUM_GPUS}" "${BASE_PORT}" "${SERVERS_PER_GPU}"; do
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: NUM_GPUS, BASE_PORT, and SERVERS_PER_GPU must be positive integers"
        exit 1
    fi
done
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: Hugging Face checkpoint not found: ${MODEL_PATH}"
    exit 1
fi
if [[ ! -f "${STATS_CONFIG}" ]]; then
    echo "Error: action statistics config not found: ${STATS_CONFIG}"
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

mkdir -p "${HF_HOME}" "${LOG_DIR}"
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}"

server_index=0
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<SERVERS_PER_GPU; slot++)); do
        port=$((BASE_PORT + server_index))
        window="g${gpu}-s$(printf '%02d' "${slot}")"
        log_path="${LOG_DIR}/server_${server_index}_gpu_${gpu}_port_${port}.log"
        if [[ ${server_index} -eq 0 ]]; then
            tmux rename-window -t "${SESSION_NAME}:0" "${window}"
        else
            tmux new-window -d -t "${SESSION_NAME}" -n "${window}"
        fi
        command="CUDA_VISIBLE_DEVICES=${gpu} '${PY_BIN}' -u '${REPO_ROOT}/src/server/deploy_hf_liberoplus.py' --model-path '${MODEL_PATH}' --host 127.0.0.1 --port ${port} --device cuda:0 --action-stats-config '${STATS_CONFIG}' --action-stats-source '${STATS_SOURCE}' 2>&1 | tee '${log_path}'"
        tmux send-keys -t "${SESSION_NAME}:${window}" "export HF_HOME='${HF_HOME}' HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}' PYTHONPATH='${REPO_ROOT}':\${PYTHONPATH:-}" Enter
        tmux send-keys -t "${SESSION_NAME}:${window}" "cd '${REPO_ROOT}'" Enter
        tmux send-keys -t "${SESSION_NAME}:${window}" "${command}" Enter
        echo "[launch] server=${server_index} gpu=${gpu} port=${port} log=${log_path}"
        server_index=$((server_index + 1))
    done
done

echo "Started ${server_index} server(s) in tmux session '${SESSION_NAME}'"
echo "Ports: $(seq -s ', ' "${BASE_PORT}" "$((BASE_PORT + server_index - 1))")"
echo "Logs: ${LOG_DIR}"
