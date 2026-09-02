"""Serve LingBot-VGGT tennis inference for issacsim_evaluate_tennis.py.

The fixed Isaac Sim client publishes one nine-frame, three-camera prefix and
the current end-effector pose for each episode.  The arm is stationary during
the prefix, so this service constructs the training-aligned action prefix as
zeros, runs the same video-prefix inference path as open-loop inference, then
converts the model's relative XYZ/Euler-XYZ predictions into absolute 8-D
Isaac Sim targets: [x, y, z, qx, qy, qz, qw, gripper].
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import VA_CONFIGS
from .openloop_inference_tennis import (
    VideoPrefixInference,
    flatten_predicted_actions,
)
from utils import init_logger, logger


VAE_TEMPORAL_FACTOR = 4
PREFIX_NUM_RGB_FRAMES = 9
DEFAULT_FUTURE_NUM_FRAMES = 16

# The simulator and training dataset use different names for the same ordered
# set of camera views.
CLIENT_CAMERA_ALIASES = {
    "opst_cam": "left_base_cam",
    "side_cam": "right_base_cam",
    "wrist_cam": "center_base_cam",
}


def _camera_frame(frame: dict, camera_key: str) -> np.ndarray:
    """Read one HWC RGB image using a model key or its simulator alias."""
    short_key = camera_key.rsplit(".", 1)[-1]
    client_key = CLIENT_CAMERA_ALIASES.get(short_key)
    for key in (camera_key, short_key, client_key):
        if key is None:
            continue
        if key in frame:
            image = np.asarray(frame[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Camera {key!r} must be an HWC RGB image, got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(
                    f"Camera {key!r} must be uint8 RGB like the open-loop "
                    f"video decoder, got {image.dtype}"
                )
            return np.ascontiguousarray(image)
    raise KeyError(
        f"Missing camera {camera_key!r}; received keys={sorted(frame)}"
    )


def build_prefix_observations(
    frames: list[dict],
    camera_keys: list[str],
    temporal_factor: int = VAE_TEMPORAL_FACTOR,
) -> dict:
    """Validate and convert the simulator RGB prefix into model observations."""
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("frames must be a non-empty list")
    if len(frames) < temporal_factor + 1:
        raise ValueError(
            f"RGB prefix needs at least {temporal_factor + 1} frames so its "
            "encoded prefix contains more than one latent frame"
        )
    if (len(frames) - 1) % temporal_factor:
        raise ValueError(
            f"RGB prefix length must be 1 mod {temporal_factor}, got {len(frames)}"
        )

    observations = []
    shapes_by_camera = {}
    for frame_id, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{frame_id}] must be a camera dictionary")
        observation = {
            camera_key: _camera_frame(frame, camera_key)
            for camera_key in camera_keys
        }
        for camera_key, image in observation.items():
            previous_shape = shapes_by_camera.setdefault(camera_key, image.shape)
            if image.shape != previous_shape:
                raise ValueError(
                    f"Camera {camera_key!r} changed shape inside the prefix: "
                    f"{previous_shape} -> {image.shape} at frame {frame_id}"
                )
        observations.append(observation)
    return {"obs": observations}


def _validate_quaternions(quaternions: np.ndarray, name: str) -> None:
    if not np.isfinite(quaternions).all():
        raise ValueError(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError(f"{name} contains a zero-length quaternion")


def build_zero_action_prefix(
    config,
    num_rgb_frames: int,
    temporal_factor: int = VAE_TEMPORAL_FACTOR,
) -> np.ndarray:
    """Build the open-loop action-prefix layout for a stationary robot."""
    if (num_rgb_frames - 1) % temporal_factor:
        raise ValueError(
            f"RGB prefix length must be 1 mod {temporal_factor}, got "
            f"{num_rgb_frames}"
        )
    latent_frames = (num_rgb_frames - 1) // temporal_factor + 1
    used_action_dim = len(config.used_action_channel_ids)
    if used_action_dim != 6:
        raise ValueError(
            f"Tennis stationary prefix requires 6 used action channels, got "
            f"{used_action_dim}"
        )
    # Same [D,F,H] raw-action layout produced by build_action_prefix() in
    # openloop_inference_tennis.py. preprocess_action() performs normalization.
    return np.zeros(
        (used_action_dim, latent_frames, config.action_per_frame),
        dtype=np.float32,
    )


def relative_to_sim_actions(
    relative_actions: torch.Tensor | np.ndarray,
    anchor_position: np.ndarray,
    anchor_quaternion_xyzw: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Convert model-relative 6-D actions to absolute Isaac Lab 8-D actions."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("Action conversion requires scipy") from exc

    relative = flatten_predicted_actions(relative_actions)
    if relative.ndim != 2 or relative.shape[1] != 6:
        raise ValueError(f"Expected predicted actions [T,6], got {relative.shape}")
    if not np.isfinite(relative).all():
        raise ValueError("Predicted actions contain NaN or Inf")

    anchor_position = np.asarray(anchor_position, dtype=np.float64).reshape(-1)
    anchor_quaternion_xyzw = np.asarray(
        anchor_quaternion_xyzw,
        dtype=np.float64,
    ).reshape(-1)
    if anchor_position.shape != (3,) or anchor_quaternion_xyzw.shape != (4,):
        raise ValueError(
            "Expected anchor position [3] and quaternion [4], got "
            f"{anchor_position.shape} and {anchor_quaternion_xyzw.shape}"
        )
    if not np.isfinite(anchor_position).all():
        raise ValueError("Anchor position contains NaN or Inf")
    _validate_quaternions(anchor_quaternion_xyzw, "anchor quaternion")

    anchor_rotation = Rotation.from_quat(anchor_quaternion_xyzw)
    absolute_positions = relative[:, :3] + anchor_position[None]
    absolute_quaternions_xyzw = (
        anchor_rotation
        * Rotation.from_euler("xyz", relative[:, 3:6])
    ).as_quat()
    grippers = np.full((len(relative), 1), gripper, dtype=np.float64)
    return np.concatenate(
        [absolute_positions, absolute_quaternions_xyzw, grippers],
        axis=1,
    ).astype(np.float32)


@torch.no_grad()
def infer_request(
    model: VideoPrefixInference,
    message: dict,
    instruction: str,
    gripper: float,
    future_num_frames: int,
) -> dict:
    """Run one request and return one complete absolute action response."""
    if not isinstance(message, dict):
        raise TypeError(f"Expected a dictionary request, got {type(message).__name__}")
    if "episode" not in message:
        raise ValueError("Request must contain episode")

    frames = message.get("frames")
    if (
        not isinstance(frames, (list, tuple))
        or len(frames) != PREFIX_NUM_RGB_FRAMES
    ):
        received = (
            len(frames)
            if isinstance(frames, (list, tuple))
            else type(frames).__name__
        )
        raise ValueError(
            f"Expected exactly {PREFIX_NUM_RGB_FRAMES} temporal frames, got "
            f"{received}"
        )
    prefix_obs = build_prefix_observations(
        frames,
        list(model.job_config.obs_cam_keys),
    )

    anchor_position = np.asarray(message.get("ee_pos_b"), dtype=np.float64)
    anchor_quaternion_xyzw = np.asarray(message.get("ee_quat_b"), dtype=np.float64)
    if anchor_position.shape != (3,):
        raise ValueError(f"Expected ee_pos_b [3], got {anchor_position.shape}")
    if anchor_quaternion_xyzw.shape != (4,):
        raise ValueError(
            f"Expected ee_quat_b [4], got {anchor_quaternion_xyzw.shape}"
        )
    if not np.isfinite(anchor_position).all():
        raise ValueError("ee_pos_b contains NaN or Inf")
    _validate_quaternions(anchor_quaternion_xyzw, "ee_quat_b")

    prefix_actions = build_zero_action_prefix(
        model.job_config,
        len(frames),
    )

    if (
        not isinstance(future_num_frames, int)
        or future_num_frames <= 0
        or future_num_frames % VAE_TEMPORAL_FACTOR
    ):
        raise ValueError(
            f"future_num_frames must be a positive multiple of "
            f"{VAE_TEMPORAL_FACTOR}, got {future_num_frames!r}"
        )
    future_latent_frames = future_num_frames // VAE_TEMPORAL_FACTOR
    num_chunks = math.ceil(
        (future_latent_frames + 1) / model.job_config.frame_chunk_size
    )

    start = time.perf_counter()
    all_latents, all_vggt_latents, relative_actions, _ = (
        model.generate_from_video_prefix(
            prefix_obs=prefix_obs,
            prefix_actions=prefix_actions,
            prompt=instruction,
            num_chunks=num_chunks,
            future_latent_frames=future_latent_frames,
        )
    )
    actions = relative_to_sim_actions(
        relative_actions,
        anchor_position,
        anchor_quaternion_xyzw,
        gripper,
    )
    del all_latents, all_vggt_latents, relative_actions

    expected_actions = future_latent_frames * model.job_config.action_per_frame
    if actions.shape != (expected_actions, 8):
        raise RuntimeError(
            f"Expected generated actions [{expected_actions},8], got {actions.shape}"
        )
    logger.info(
        "episode=%s: inferred %d absolute actions in %.1f ms",
        message["episode"],
        len(actions),
        (time.perf_counter() - start) * 1000,
    )
    return {
        "episode": message["episode"],
        "actions": actions.tolist(),
    }


def load_model(args: argparse.Namespace) -> VideoPrefixInference:
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    if config.env_type != "tennis_tshape":
        raise ValueError(
            f"Config {args.config_name!r} is not a tennis checkpoint"
        )
    if (
        config.action_per_frame != 8
        or list(config.used_action_channel_ids) != list(range(6))
    ):
        raise ValueError(
            "This protocol requires action_per_frame=8 and action channels 0:6"
        )

    config.rank = config.local_rank = 0
    config.world_size = 1
    config.save_root = str(Path(args.save_root).resolve())
    config.vae_temporal_factor = VAE_TEMPORAL_FACTOR
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.transformer_path is not None:
        transformer_path = Path(args.transformer_path).expanduser().resolve()
        if (transformer_path / "transformer").is_dir():
            config.transformer_path = str(transformer_path)
        elif transformer_path.name == "transformer" and transformer_path.is_dir():
            config.transformer_path = str(transformer_path.parent)
        else:
            raise FileNotFoundError(
                "--transformer-path must be a transformer directory or its "
                f"parent checkpoint: {transformer_path}"
            )
    if args.vggt_model_path is not None:
        config.vggt_pretrained_model_name_or_path = args.vggt_model_path
    if args.enable_offload is not None:
        config.enable_offload = args.enable_offload

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return VideoPrefixInference(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve LingBot-VGGT tennis actions over ZeroMQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", default="tennis_i2va", choices=VA_CONFIGS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--transformer-path", default=None)
    parser.add_argument("--vggt-model-path", default=None)
    parser.add_argument("--instruction", default="Catch the green tennis ball.")
    parser.add_argument("--save-root", default="outputs/simulator_inference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument(
        "--future-num-frames",
        type=int,
        default=DEFAULT_FUTURE_NUM_FRAMES,
        help=(
            "Number of future RGB frames to generate. Must be divisible by the "
            "VAE temporal factor; 16 frames produce 32 control actions."
        ),
    )
    parser.add_argument(
        "--enable-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--zmq-host",
        default="127.0.0.1",
        help=(
            "Host used by the fixed local client topology: connect to its "
            "observation PUB socket and bind the action PUB socket here."
        ),
    )
    parser.add_argument("--zmq-port", type=int, default=5563)
    parser.add_argument("--zmq-act-port", type=int, default=5564)
    return parser


def _error_response(message: object, error: Exception) -> dict:
    request = message if isinstance(message, dict) else {}
    return {
        "episode": request.get("episode"),
        # The fixed client only recognizes messages containing "actions". An
        # empty chunk releases its inference hold while preserving the error.
        "actions": [],
        "error": f"{type(error).__name__}: {error}",
    }


def run(args: argparse.Namespace) -> None:
    if not math.isfinite(args.gripper):
        raise ValueError("--gripper must be finite")
    if not args.zmq_host:
        raise ValueError("--zmq-host must not be empty")
    if args.future_num_frames <= 0:
        raise ValueError("--future-num-frames must be positive")
    if args.future_num_frames % VAE_TEMPORAL_FACTOR:
        raise ValueError(
            f"--future-num-frames must be divisible by {VAE_TEMPORAL_FACTOR}"
        )
    for option, port in (
        ("--zmq-port", args.zmq_port),
        ("--zmq-act-port", args.zmq_act_port),
    ):
        if not 1 <= port <= 65535:
            raise ValueError(f"{option} must be in [1, 65535], got {port}")
    if args.zmq_port == args.zmq_act_port:
        raise ValueError("Request and action ports must be different")

    import zmq

    context = zmq.Context.instance()
    frame_socket = context.socket(zmq.SUB)
    frame_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    frame_socket.connect(f"tcp://{args.zmq_host}:{args.zmq_port}")

    action_socket = context.socket(zmq.PUB)
    action_socket.bind(f"tcp://{args.zmq_host}:{args.zmq_act_port}")
    logger.info(
        "Receiving observations from tcp://%s:%d; publishing actions on "
        "tcp://%s:%d",
        args.zmq_host,
        args.zmq_port,
        args.zmq_host,
        args.zmq_act_port,
    )
    model = load_model(args)

    while True:
        message = frame_socket.recv_pyobj()
        try:
            response = infer_request(
                model=model,
                message=message,
                instruction=args.instruction,
                gripper=args.gripper,
                future_num_frames=args.future_num_frames,
            )
        except Exception as error:
            logger.exception("Inference request failed")
            response = _error_response(message, error)
        action_socket.send_pyobj(response)


def main() -> None:
    init_logger()
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
