"""Tennis closed-loop inference server."""
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
from .openloop_inference_tennis import VideoPrefixInference, flatten_predicted_actions
from diffusers.utils import export_to_video
from utils import init_logger, logger

VAE_TEMPORAL_FACTOR = 4
PREFIX_NUM_RGB_FRAMES = 9


def _unpack_ndarray(value, name):
    if not isinstance(value, dict) or not value.get("__ndarray_raw_v1__"):
        raise ValueError(f"{name} must use the __ndarray_raw_v1__ format")
    try:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(dim) for dim in value["shape"])
        array = np.frombuffer(value["data"], dtype=dtype).reshape(shape)
    except Exception as exc:
        raise ValueError(f"Invalid packed ndarray for {name}: {exc}") from exc
    return np.array(array, copy=True)


def _prefix_from_message(message, camera_keys):
    images = {
        name: _unpack_ndarray(message[f"observation.images.{name}"], name)
        for name in ("left", "right", "upper")
    }
    for name, image in images.items():
        if image.ndim != 4 or image.shape[0] != PREFIX_NUM_RGB_FRAMES \
                or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"observation.images.{name} must be uint8 [{PREFIX_NUM_RGB_FRAMES},H,W,3], got {image.shape} {image.dtype}")
    return {"obs": [
        {
            key: np.ascontiguousarray(
                images[key.rsplit(".", 1)[-1]][index]
            )
            for key in camera_keys
        }
        for index in range(PREFIX_NUM_RGB_FRAMES)
    ]}


@torch.no_grad()
def infer_request(model, message, instruction, future_num_frames,
                  video_output_dir=None, output_fps=15.0):
    if not isinstance(message, dict) or "episode" not in message or message.get("request_id") is None:
        raise ValueError("Request must contain episode and request_id")
    if model.job_config.env_type != "tennis_tshape":
        raise ValueError("This server requires a tennis_tshape checkpoint")
    if len(message.get("obs_frames", [])) != PREFIX_NUM_RGB_FRAMES:
        raise ValueError(f"Expected exactly {PREFIX_NUM_RGB_FRAMES} obs_frames")
    if future_num_frames <= 0 or future_num_frames % VAE_TEMPORAL_FACTOR:
        raise ValueError(f"future_num_frames must be a positive multiple of {VAE_TEMPORAL_FACTOR}")
    prefix_obs = _prefix_from_message(message, list(model.job_config.obs_cam_keys))
    latent_prefix_frames = (PREFIX_NUM_RGB_FRAMES - 1) // VAE_TEMPORAL_FACTOR + 1
    prefix_actions = np.zeros((len(model.job_config.used_action_channel_ids), latent_prefix_frames,
                               model.job_config.action_per_frame), dtype=np.float32)
    future_latent_frames = future_num_frames // VAE_TEMPORAL_FACTOR
    num_chunks = math.ceil((future_latent_frames + 1) / model.job_config.frame_chunk_size)
    start = time.perf_counter()
    all_latents, _, predicted, _ = model.generate_from_video_prefix(
        prefix_obs=prefix_obs, prefix_actions=prefix_actions, prompt=instruction,
        num_chunks=num_chunks, future_latent_frames=future_latent_frames,
    )
    # ``predicted`` is [D, F, H] (9 action channels, latent frames, eight
    # controls per frame).  Flatten in time-major order; a raw reshape would
    # interleave channels and time and produces physically incorrect motion.
    targets = flatten_predicted_actions(predicted)
    if not np.isfinite(targets).all():
        raise ValueError("Model returned NaN or Inf actions")
    # Model outputs cumulative offsets; the simulator consumes per-step deltas.
    targets[:, 2] = np.unwrap(targets[:, 2])
    deltas = np.diff(targets, axis=0, prepend=np.zeros((1, 9), dtype=targets.dtype))
    if video_output_dir is not None:
        video_dir = Path(video_output_dir) / f"episode-{int(message['episode']):06d}"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"request-{int(message['request_id']):06d}.mp4"
        decoded_video = model.decode_one_video(all_latents.to(model.device), "np")[0]
        export_to_video(decoded_video, str(video_path), fps=float(output_fps))
        logger.info("Saved inference video to %s", video_path)
    logger.info("episode=%s request=%s: inferred %d delta actions in %.1f ms",
                message["episode"], message["request_id"], len(deltas),
                (time.perf_counter() - start) * 1000)
    return {"episode": int(message["episode"]), "request_id": int(message["request_id"]),
            "action_mode": "delta", "actions": deltas.astype(np.float32).tolist()}


def load_model(args):
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    if config.env_type != "tennis_tshape":
        raise ValueError(f"Config {args.config_name!r} is not a tennis checkpoint")
    if config.action_per_frame != 8 or list(config.used_action_channel_ids) != list(range(9)):
        raise ValueError("Checkpoint must use action_per_frame=8 and channels 0:9")
    config.rank = config.local_rank = 0
    config.world_size = 1
    config.save_root = str(Path(args.save_root).resolve())
    config.vae_temporal_factor = VAE_TEMPORAL_FACTOR
    if args.model_path:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.transformer_path:
        checkpoint = Path(args.transformer_path).expanduser().resolve()
        config.transformer_path = str(checkpoint.parent if checkpoint.name == "transformer" else checkpoint)
    if args.vggt_model_path:
        config.vggt_pretrained_model_name_or_path = args.vggt_model_path
    if args.enable_offload is not None:
        config.enable_offload = args.enable_offload
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(args.seed)
    return VideoPrefixInference(config)


def build_parser():
    parser = argparse.ArgumentParser(description="Serve tennis 9-D delta actions over ZeroMQ")
    parser.add_argument("--config-name", default="tennis_i2va", choices=VA_CONFIGS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--transformer-path", default=None)
    parser.add_argument("--vggt-model-path", default=None)
    parser.add_argument("--instruction", default="Catch the green tennis ball.")
    parser.add_argument("--save-root", default="outputs/simulator_inference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--future-num-frames", type=int, default=16)
    parser.add_argument("--save-videos", action="store_true",
                        help="Save decoded prefix+rollout video for every request.")
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--enable-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, default=5566)
    parser.add_argument("--zmq-act-port", type=int, default=5567)
    return parser


def run(args):
    if not args.zmq_host or args.zmq_port == args.zmq_act_port:
        raise ValueError("Invalid ZeroMQ host or duplicate ports")
    import zmq
    context = zmq.Context.instance()
    observation_socket = context.socket(zmq.SUB)
    observation_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    observation_socket.connect(f"tcp://{args.zmq_host}:{args.zmq_port}")
    action_socket = context.socket(zmq.PUB)
    action_socket.bind(f"tcp://{args.zmq_host}:{args.zmq_act_port}")
    logger.info("Receiving observations on %s:%d; publishing actions on %s:%d",
                args.zmq_host, args.zmq_port, args.zmq_host, args.zmq_act_port)
    model = load_model(args)
    video_output_dir = (Path(args.save_root) / "videos") if args.save_videos else None
    while True:
        message = observation_socket.recv_pyobj()
        try:
            response = infer_request(
                model, message, args.instruction, args.future_num_frames,
                video_output_dir=video_output_dir, output_fps=args.output_fps,
            )
        except Exception as error:
            logger.exception("Inference request failed")
            response = {"episode": message.get("episode") if isinstance(message, dict) else None,
                        "request_id": message.get("request_id") if isinstance(message, dict) else None,
                        "action_mode": "delta", "actions": [],
                        "error": f"{type(error).__name__}: {error}"}
        action_socket.send_pyobj(response)


if __name__ == "__main__":
    init_logger()
    run(build_parser().parse_args())
