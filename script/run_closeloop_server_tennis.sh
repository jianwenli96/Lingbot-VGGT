#!/usr/bin/env bash

set -euo pipefail

# Tennis inference service for issacsim_evaluate_tennis.py. Model paths and
# ZeroMQ endpoints can be overridden through environment variables; additional
# CLI arguments are forwarded to the Python entry point.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-tennis_i2va}"
MODEL_PATH="${MODEL_PATH:-/home/jdhc/lijianwen/Pretrained_models/lingbot-va-base}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-/home/jdhc/lijianwen/Pretrained_models/lingbot-tennis/omega_checkpoint_step_5000}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/home/jdhc/lijianwen/Pretrained_models/VGGT-Omega/vggt_omega_1b_512.pt}"
SAVE_ROOT="${SAVE_ROOT:-/home/jdhc/lijianwen/Codes/Lingbot-VGGT/outputs}"
INSTRUCTION="${INSTRUCTION:-{}"
# The fixed client binds its observation PUB socket and connects its action SUB
# socket to this host. With the current client interface both processes should
# run on the same machine, using 127.0.0.1.
ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
ZMQ_PORT="${ZMQ_PORT:-5563}"
ZMQ_ACT_PORT="${ZMQ_ACT_PORT:-5564}"
FUTURE_NUM_FRAMES="${FUTURE_NUM_FRAMES:-16}"
GRIPPER="${GRIPPER:-0.0}"
SEED="${SEED:-0}"

cmd=(
    "${PYTHON_BIN}" -m wan_va.tennis.closeloop_inference_tennis
    --config-name "${CONFIG_NAME}"
    --model-path "${MODEL_PATH}"
    --transformer-path "${TRANSFORMER_PATH}"
    --vggt-model-path "${VGGT_MODEL_PATH}"
    --instruction "${INSTRUCTION}"
    --save-root "${SAVE_ROOT}"
    --future-num-frames "${FUTURE_NUM_FRAMES}"
    --gripper "${GRIPPER}"
    --seed "${SEED}"
    --zmq-host "${ZMQ_HOST}"
    --zmq-port "${ZMQ_PORT}"
    --zmq-act-port "${ZMQ_ACT_PORT}"
)

source /home/jdhc/miniconda3/etc/profile.d/conda.sh
conda activate lingbot-vggt

pkill -f vla_infer_frames.py || true
pkill -f closeloop_inference_tennis.py || true

cd "${REPO_ROOT}"
LINGBOT_USE_NPU="${LINGBOT_USE_NPU:-0}" \
exec "${cmd[@]}" "$@"
