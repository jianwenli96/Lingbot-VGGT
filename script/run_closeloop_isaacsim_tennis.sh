#!/usr/bin/env bash

set -euo pipefail

# Launch the Isaac Sim tennis client. The VLA inference service should be
# started separately with script/run_closeloop_server_tennis.sh.
# Example: bash script/run_closeloop_isaacsim_tennis.sh --max-episodes 5 --max-steps 1000
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jdhc/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/jdhc/IsaacLab}"
ISAACSIM_SCRIPT="${ISAACSIM_SCRIPT:-${REPO_ROOT}/wan_va/tennis/closeloop_isaacsim_tennis.py}"
ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
ZMQ_OBSERVATION_PORT="${ZMQ_OBSERVATION_PORT:-5566}"
ZMQ_ACTION_PORT="${ZMQ_ACTION_PORT:-5567}"
SEED="${SEED:-0}"
MAX_EPISODES="${MAX_EPISODES:-100}"
INFERENCE_INPUT_DIR="${INFERENCE_INPUT_DIR:-/home/jdhc/lijianwen/Codes/Lingbot-VGGT/closeloop_outputs}"

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
    --seed "${SEED}"
    --max-episodes "${MAX_EPISODES}"
    --enable_cameras
    --viz kit
    --camera-width 480
    --camera-height 360
    --ball-distance 4.0
    --ball-distance-range 0.5
    --ball-height-range 0.81 1.0
    --ball-flight-time-range 1.4 1.6
    --ball-target-height 0.9
    --ball-landing-radius 1.4
    --ball-landing-forward-min 0.3
    --catch-radius 0.13
    --physics_dt 0.0333333
    --throw-period 1.7
    --observation-video-quality 9
    --rendering_mode quality
    --net-opacity 0
    --inference-control
    --inference-timeout 30
    --zmq-host "${ZMQ_HOST}"
    --zmq-observation-port "${ZMQ_OBSERVATION_PORT}"
    --zmq-action-port "${ZMQ_ACTION_PORT}"
    --hide-throw-markers
    --inference-frame-offsets 0 2 4 6 8 10 12 14 16
    --inference-once-per-episode
    --inference-save-inputs
    --inference-input-dir "${INFERENCE_INPUT_DIR}"
    --dataset-task "Catch the green tennis ball"
    --clean-close
)

exec "${cmd[@]}" "$@"
