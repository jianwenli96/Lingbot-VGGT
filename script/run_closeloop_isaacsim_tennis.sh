#!/usr/bin/env bash

set -euo pipefail

# Launch the Isaac Sim tennis client. The VLA inference service should be
# started separately with script/run_closeloop_server_tennis.sh.
# Example: bash script/run_closeloop_isaacsim_tennis.sh --max_episodes 5 --max_steps 1000
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jdhc/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/jdhc/IsaacLab}"
ISAACSIM_SCRIPT="${ISAACSIM_SCRIPT:-${REPO_ROOT}/wan_va/tennis/isaacsim_evaluate_tennis.py}"
OBJECT_DIR="${OBJECT_DIR:-/home/jdhc/z00821918/tennis_robot/dynamicvla/objects}"
SAVE_ROOT="${SAVE_ROOT:-/home/jdhc/lijianwen/Codes/Lingbot-VGGT/outputs/client}"
ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
ZMQ_PORT="${ZMQ_PORT:-5563}"
ZMQ_ACT_PORT="${ZMQ_ACT_PORT:-5564}"
THROW_INTERVAL="${THROW_INTERVAL:-3.0}"
SEED="${SEED:-42}"
PHYSICS_DT="${PHYSICS_DT:-0.0333333}"
RENDER_DT="${RENDER_DT:-0.0333333}"
MAX_EPISODES="${MAX_EPISODES:-100}"
BALL_DIAMETER="${BALL_DIAMETER:-6.5}"

if [[ ! -f "${ISAACSIM_SCRIPT}" ]]; then
    printf 'Isaac Sim entry point not found: %s\n' "${ISAACSIM_SCRIPT}" >&2
    exit 1
fi
if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    printf 'IsaacLab launcher not found or not executable: %s\n' \
        "${ISAACLAB_ROOT}/isaaclab.sh" >&2
    exit 1
fi

# Isaac Sim's activation script appends to PYTHONPATH without a default.  It
# also leaves Isaac Sim packages visible to the base Python used by `conda`
# when this script is called from an already activated environment.  Avoid
# both cases by making activation idempotent and starting environment
# switching with a clean PYTHONPATH.
: "${PYTHONPATH:=}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    export PYTHONPATH=
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi
cd "${ISAACLAB_ROOT}"

cmd=(
    "${ISAACLAB_ROOT}/isaaclab.sh"
    -p "${ISAACSIM_SCRIPT}"
    --object_dir "${OBJECT_DIR}"
    --randomize_throw
    --enable_cameras
    --zmq_publish
    --control_arm
    --viz kit
    --throw_interval "${THROW_INTERVAL}"
    --seed "${SEED}"
    --physics_dt "${PHYSICS_DT}"
    --render_dt "${RENDER_DT}"
    --zmq_host "${ZMQ_HOST}"
    --zmq_port "${ZMQ_PORT}"
    --zmq_act_port "${ZMQ_ACT_PORT}"
    --save_video
    --video_dir "${SAVE_ROOT}"
    --max_episodes "${MAX_EPISODES}"
    --ball_diameter "${BALL_DIAMETER}"
    --clean_close
)

exec "${cmd[@]}" "$@"
