#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export HF_HOME="$(realpath -m "${HF_HOME:-${REPO_ROOT}/.cache/huggingface}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
PY_BIN="$(realpath -s "$(command -v "${PY_BIN:-python}")")"

usage() {
    cat <<'EOF'
Usage:
  bash src/server/launch_choice_servers.sh <model_root> <ckpt_path> <num_gpus> [base_port] [session_name] [servers_per_gpu]

Examples:
  # 8 张卡，每张卡 1 个 server => 共 8 个
  bash src/server/launch_choice_servers.sh path/to/model_root path/to/mp_rank_00_model_states.pt 8 10086 choice_ki_servers 1

  # 8 张卡，每张卡 8 个 server => 共 64 个
  bash src/server/launch_choice_servers.sh path/to/model_root path/to/mp_rank_00_model_states.pt 8 10086 choice_ki_servers 8

  # 2 张卡，每张卡 8 个 server => 共 16 个
  bash src/server/launch_choice_servers.sh path/to/model_root path/to/mp_rank_00_model_states.pt 2 10086 choice_ki_servers 8
EOF
}

if [ $# -lt 3 ]; then
    usage
    exit 1
fi

MODEL_ROOT="$(realpath "$1")"
CKPT_PATH="$(realpath "$2")"
NUM_GPUS=$3
BASE_PORT=${4:-10086}
SESSION_NAME=${5:-choice_ki_servers}
SERVERS_PER_GPU=${6:-1}

DEPLOY_MODULE=src.server.deploy_choice_ki
DEPLOY_PY=${REPO_ROOT}/src/server/deploy_choice_ki.py

if [ ! -d "$MODEL_ROOT" ]; then
    echo "Error: model_root not found: $MODEL_ROOT"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

if [ ! -f "$DEPLOY_PY" ]; then
    echo "Error: deploy_choice_ki.py not found: $DEPLOY_PY"
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_gpus must be a positive integer"
    exit 1
fi

if ! [[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: base_port must be a positive integer"
    exit 1
fi

if ! [[ "$SERVERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: servers_per_gpu must be a positive integer"
    exit 1
fi

NUM_SERVERS=$((NUM_GPUS * SERVERS_PER_GPU))

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME"

setup_window() {
    local target=$1
    tmux send-keys -t "$target" "export HF_HOME='${HF_HOME}'" Enter
    tmux send-keys -t "$target" "export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'" Enter
    tmux send-keys -t "$target" "export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'" Enter
    tmux send-keys -t "$target" "export OMP_NUM_THREADS=1" Enter
    tmux send-keys -t "$target" "export MKL_NUM_THREADS=1" Enter
    tmux send-keys -t "$target" "export OPENBLAS_NUM_THREADS=1" Enter
    tmux send-keys -t "$target" "export NUMEXPR_NUM_THREADS=1" Enter
    tmux send-keys -t "$target" "export PYTHONUNBUFFERED=1" Enter
    tmux send-keys -t "$target" "export PYTHONPATH='${REPO_ROOT}':\${PYTHONPATH:-}" Enter
    tmux send-keys -t "$target" "cd '${REPO_ROOT}'" Enter
}

server_idx=0
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<SERVERS_PER_GPU; slot++)); do
        PORT=$((BASE_PORT + server_idx))
        WINDOW_NAME="g${gpu}-s$(printf "%02d" "$slot")"

        if [ $server_idx -eq 0 ]; then
            tmux rename-window -t "${SESSION_NAME}:0" "$WINDOW_NAME"
        else
            tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME"
        fi

        setup_window "${SESSION_NAME}:${WINDOW_NAME}"

        CMD="CUDA_VISIBLE_DEVICES=${gpu} '${PY_BIN}' -m ${DEPLOY_MODULE} \
--model_root '${MODEL_ROOT}' \
--ckpt_path '${CKPT_PATH}' \
--port ${PORT}"

        tmux send-keys -t "${SESSION_NAME}:${WINDOW_NAME}" "$CMD" Enter
        echo "[launch] server_idx=${server_idx} gpu=${gpu} slot=${slot} port=${PORT}"
        server_idx=$((server_idx + 1))
    done
done

echo
echo "Started ${NUM_SERVERS} choice KI server(s) in tmux session '${SESSION_NAME}'"
echo "Model root      : ${MODEL_ROOT}"
echo "Checkpoint      : ${CKPT_PATH}"
echo "Num GPUs        : ${NUM_GPUS}"
echo "Servers / GPU   : ${SERVERS_PER_GPU}"
echo "Total servers   : ${NUM_SERVERS}"
echo "Ports           : $(seq -s ', ' "$BASE_PORT" "$((BASE_PORT + NUM_SERVERS - 1))")"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "Kill with:"
echo "  tmux kill-session -t ${SESSION_NAME}"
