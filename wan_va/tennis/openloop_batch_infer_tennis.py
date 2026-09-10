"""Batch inference on entire validation set and compute average Euclidean distance.

Example:
    python -m wan_va.tennis.openloop_batch_infer_tennis \
        --config-name demo_i2av \
        --dataset-root /data/my_lerobot_dataset \
        --output-dir outputs/batch_val_results
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import VA_CONFIGS
from .openloop_inference_tennis import (
    VideoPrefixInference,
    load_lerobot_episode,
    load_lerobot_actions,
    flatten_predicted_actions,
    make_uniform_timestamps,
    build_action_prefix,
    tennis_pose
)
from ..utils import init_logger, logger


class EpisodeMetrics(NamedTuple):
    """Metrics for a single episode inference."""
    episode_index: int
    euclidean_distance: float
    gt_endpoint: tuple[float, float, float]
    pred_endpoint: tuple[float, float, float]
    success: bool
    error_message: str | None = None
    video_path: str | None = None


def decode_and_save_video(
    model,
    all_latents: torch.Tensor,
    prefix_latent_frames: int,
    episode_index: int,
    output_dir: Path,
    sample_fps: float,
    prefix_num_frames: int,
    mark_borders: bool = True,
    border_width: int = 6,
) -> str:
    """Decode latents and save as video file.

    Returns:
        Path to the saved video file.
    """
    # Clear cache before decoding to free memory from inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Move VAE to GPU if offloaded
    if model.enable_offload:
        model.vae = model.vae.to(model.device).to(model.dtype)

    # Decode video - process in chunks if needed to reduce memory
    video_latent = all_latents.to(model.device, model.vae.dtype)
    latents_mean = torch.tensor(
        model.vae.config.latents_mean, device=model.device, dtype=model.vae.dtype
    ).view(1, model.vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        model.vae.config.latents_std, device=model.device, dtype=model.vae.dtype
    ).view(1, model.vae.config.z_dim, 1, 1, 1)

    # Decode in temporal chunks to reduce peak memory
    temporal_chunks = 4  # Split into 4 temporal chunks
    num_frames = video_latent.shape[2]
    chunk_size = max(1, num_frames // temporal_chunks)

    decoded_chunks = []
    for i in range(0, num_frames, chunk_size):
        end_idx = min(i + chunk_size, num_frames)
        chunk_latent = video_latent[:, :, i:end_idx, :, :]

        decoded_chunk = model.vae.decode(
            chunk_latent / latents_std + latents_mean,
            return_dict=False,
        )[0]
        decoded_chunks.append(decoded_chunk.cpu())  # Move to CPU immediately

        # Clear GPU memory after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all decoded chunks
    decoded = torch.cat(decoded_chunks, dim=2)
    del decoded_chunks  # Free memory

    # Detach to remove gradients before postprocessing
    frames = VideoProcessor(vae_scale_factor=1).postprocess_video(
        decoded.detach(),
        output_type="np",
    )[0]
    del decoded  # Free memory immediately after postprocessing

    # Add borders to mark prefix vs generated frames
    if mark_borders:
        frames = _add_phase_borders(frames, prefix_num_frames, border_width)

    # Save video
    video_path = output_dir / f"episode_{episode_index:06d}.mp4"
    export_to_video(frames, str(video_path), fps=sample_fps)

    # Move VAE back to CPU if offloaded
    if model.enable_offload:
        model.vae = model.vae.to('cpu')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(video_path)


def _add_phase_borders(
    frames: np.ndarray,
    prefix_num_frames: int,
    border_width: int,
) -> np.ndarray:
    """Add red borders for prefix frames and green for generated frames."""
    marked_frames = []
    for frame_id, frame in enumerate(frames):
        marked = np.array(frame, copy=True)
        height, width = marked.shape[:2]
        effective_width = min(border_width, height // 2, width // 2)
        if effective_width < 1:
            marked_frames.append(frame)
            continue

        scale = 255 if np.issubdtype(marked.dtype, np.integer) else 1.0
        # Red for prefix, green for generated
        color = (
            np.array([scale, 0, 0], dtype=marked.dtype)
            if frame_id < prefix_num_frames
            else np.array([0, scale, 0], dtype=marked.dtype)
        )
        marked[:effective_width, :, :3] = color
        marked[-effective_width:, :, :3] = color
        marked[:, :effective_width, :3] = color
        marked[:, -effective_width:, :3] = color
        marked_frames.append(np.ascontiguousarray(marked))

    return np.stack(marked_frames, axis=0)


def get_all_episode_indices(dataset_root: str) -> list[int]:
    """Get all episode indices from LeRobot dataset.
    
    Directly use the same logic as lerobot_latent_dataset.py
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.constants import HF_LEROBOT_HOME
    import packaging
    
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    
    # Read info.json to get total_episodes
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    
    total_episodes = int(info.get("total_episodes", 0))
    
    if total_episodes == 0:
        raise ValueError(f"No episodes found in dataset: {dataset_root}")
    
    # Get all episode indices
    all_indices = list(range(total_episodes))
    
    logger.info(f"Found {total_episodes} episodes in dataset")
    return all_indices


def compute_endpoint_distance(
    future_actions: torch.Tensor | np.ndarray,
    prefix_actions: torch.Tensor | np.ndarray,
    dataset_metadata: dict,
    env_type: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Euclidean distance between GT and predicted endpoints."""
    episode_index = int(dataset_metadata["episode_index"])
    _, gt_actions = load_lerobot_actions(
        dataset_metadata["dataset_root"],
        episode_index,
    )
    predicted_relative = flatten_predicted_actions(future_actions)
    packed_input_actions = flatten_predicted_actions(prefix_actions)
    action_prefix_metadata = dataset_metadata.get("action_prefix") or {}
    action_per_frame = int(action_prefix_metadata.get("action_per_frame", 0))
    known_action_count = int(
        action_prefix_metadata.get("known_source_action_count", -1)
    )
    
    if action_per_frame <= 0 or known_action_count < 0:
        raise ValueError(
            "Distance computation requires action_per_frame and "
            "known_source_action_count in action-prefix metadata"
        )
    
    input_relative = packed_input_actions[
        action_per_frame : action_per_frame + known_action_count
    ]
    
    base_action = np.asarray(
        action_prefix_metadata.get("relative_pose_base_action", []),
        dtype=np.float64,
    )
    
    if env_type == "robotwin_tshape":
        if base_action.size < 11:
            raise ValueError("RoboTwin requires base_action with at least 11 channels")
        pred_endpoint = base_action[:3] + predicted_relative[-1, :3]
        gt_endpoint = gt_actions[-1, :3]
    elif env_type == "tennis_tshape":
        base_action = tennis_pose(base_action)
        gt_actions = tennis_pose(gt_actions)
        pred_endpoint = base_action[:3] + predicted_relative[-1, :3]
        gt_endpoint = gt_actions[-1, :3]
    else:
        if base_action.size < 3:
            raise ValueError("Single-arm requires base_action with at least 3 channels")
        pred_endpoint = base_action[:3] + predicted_relative[-1, :3]
        gt_endpoint = gt_actions[-1, :3]
    
    distance = float(np.linalg.norm(gt_endpoint - pred_endpoint))
    return distance, gt_endpoint, pred_endpoint


def run_single_episode(
    config,
    episode_index: int,
    args: argparse.Namespace,
    camera_offsets: dict,
    output_dir: Path,
) -> EpisodeMetrics:
    """Run inference on a single episode and return metrics."""
    try:
        target_timestamps = make_uniform_timestamps(
            args.prefix_num_frames,
            args.sample_fps,
            args.prefix_start_sec,
        )
        
        video_paths, prompt, dataset_metadata = load_lerobot_episode(
            dataset_root=args.dataset_root,
            episode_index=episode_index,
            expected_camera_keys=config.obs_cam_keys,
            prompt_time_sec=float(target_timestamps[-1]),
        )
        
        # Load actions for prefix
        action_timestamps, episode_actions = load_lerobot_actions(
            dataset_metadata["dataset_root"],
            episode_index,
        )
        
        prefix_actions, action_prefix_metadata = build_action_prefix(
            timestamps=action_timestamps,
            actions=episode_actions,
            target_timestamps=target_timestamps,
            dataset_fps=dataset_metadata["dataset_fps"],
            sample_fps=args.sample_fps,
            vae_temporal_factor=args.vae_temporal_factor,
            action_per_frame=config.action_per_frame,
            env_type=config.env_type,
            timestamp_tolerance=args.timestamp_tolerance,
        )
        dataset_metadata["action_prefix"] = action_prefix_metadata
        
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        model = VideoPrefixInference(config)
        prefix_obs, metadata = model.load_prefix_observations(
            video_paths=video_paths,
            camera_offsets=camera_offsets,
            target_timestamps=target_timestamps,
            timestamp_tolerance=args.timestamp_tolerance,
            max_camera_skew=args.max_camera_skew,
            rotate_clockwise_90=args.rotate_clockwise_90,
        )
        
        future_latent_frames = None
        num_chunks = args.num_chunks
        if args.future_num_frames is not None:
            future_latent_frames = (
                args.future_num_frames // args.vae_temporal_factor
            )
            num_chunks = math.ceil(
                (future_latent_frames + 1) / config.frame_chunk_size
            )
        
        all_latents, all_vggt_latents, future_actions, prefix_latent_frames = (
            model.generate_from_video_prefix(
                prefix_obs=prefix_obs,
                prefix_actions=prefix_actions,
                prompt=prompt,
                num_chunks=num_chunks,
                future_latent_frames=future_latent_frames,
            )
        )

        distance, gt_end, pred_end = compute_endpoint_distance(
            future_actions=future_actions,
            prefix_actions=prefix_actions,
            dataset_metadata=dataset_metadata,
            env_type=config.env_type,
        )

        # Save inference video
        video_path = None
        if not getattr(args, 'skip_video_save', False):
            video_path = decode_and_save_video(
                model=model,
                all_latents=all_latents.cpu(),
                prefix_latent_frames=prefix_latent_frames,
                episode_index=episode_index,
                output_dir=output_dir,
                sample_fps=args.sample_fps,
                prefix_num_frames=args.prefix_num_frames,
            )

        # Save latents and actions
        episode_out_dir = output_dir / f"episode_{episode_index:06d}"
        episode_out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(all_latents.cpu(), episode_out_dir / "latents.pt")
        torch.save(future_actions.cpu(), episode_out_dir / "actions.pt")
        if all_vggt_latents is not None:
            torch.save(all_vggt_latents.cpu(), episode_out_dir / "vggt_latents.pt")

        return EpisodeMetrics(
            episode_index=episode_index,
            euclidean_distance=distance,
            gt_endpoint=tuple(gt_end.tolist()),
            pred_endpoint=tuple(pred_end.tolist()),
            success=True,
            video_path=video_path,
        )
        
    except Exception as e:
        import traceback
        return EpisodeMetrics(
            episode_index=episode_index,
            euclidean_distance=0.0,
            gt_endpoint=(0.0, 0.0, 0.0),
            pred_endpoint=(0.0, 0.0, 0.0),
            success=False,
            error_message=f"{str(e)}\n{traceback.format_exc()}",
        )


def main() -> None:
    init_logger()
    parser = argparse.ArgumentParser(
        description="Batch inference on validation set and compute average distance."
    )
    parser.add_argument("--config-name", default="tennis_i2va", choices=VA_CONFIGS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--transformer-path", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--episode-range",
        type=str,
        default=None,
        help="Episode range to process (e.g., '0:10' for episodes 0-9, default: all)"
    )
    parser.add_argument("--output-dir", default="outputs/batch_val_results")
    parser.add_argument("--sample-fps", type=float, default=15.0)
    parser.add_argument("--prefix-num-frames", type=int, default=13)
    parser.add_argument("--prefix-start-sec", type=float, default=0.0)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--future-num-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae-temporal-factor", type=int, default=4)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.03)
    parser.add_argument("--max-camera-skew", type=float, default=0.03)
    parser.add_argument(
        "--rotate-clockwise-90",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--camera-offset", action="append", default=[])
    parser.add_argument(
        "--skip-video-save",
        action="store_true",
        help="Skip saving decoded video files to save time and disk space.",
    )
    
    args = parser.parse_args()
    
    # Get all episode indices from dataset
    all_episode_indices = get_all_episode_indices(args.dataset_root)
    
    # Apply episode range if specified
    if args.episode_range:
        parts = args.episode_range.split(":")
        if len(parts) == 2:
            start_idx = int(parts[0])
            end_idx = int(parts[1])
            episode_indices = all_episode_indices[start_idx:end_idx]
        else:
            raise ValueError(f"Invalid episode-range format: {args.episode_range}")
    else:
        episode_indices = all_episode_indices
    
    logger.info(f"Processing {len(episode_indices)} episodes")
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.save_root = str(output_dir)
    config.vae_temporal_factor = args.vae_temporal_factor
    
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.transformer_path is not None:
        transformer_path = Path(args.transformer_path).expanduser().resolve()
        if not transformer_path.is_dir():
            raise FileNotFoundError(f"Transformer directory not found: {transformer_path}")
        config.transformer_path = str(transformer_path)
    
    camera_offsets = {}
    for item in args.camera_offset:
        if "=" not in item:
            raise ValueError(f"camera-offset expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        camera_offsets[key.strip()] = float(value)
    
    all_metrics = []
    for episode_index in tqdm(episode_indices, desc="Processing episodes"):
        metrics = run_single_episode(
            config=config,
            episode_index=episode_index,
            args=args,
            camera_offsets=camera_offsets,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)
    
    successful_metrics = [m for m in all_metrics if m.success]
    failed_count = len(all_metrics) - len(successful_metrics)
    
    if successful_metrics:
        distances = [m.euclidean_distance for m in successful_metrics]
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        median_distance = np.median(distances)
    else:
        avg_distance = std_distance = min_distance = max_distance = median_distance = 0.0
    
    print("\n" + "="*60)
    print("BATCH INFERENCE RESULTS")
    print("="*60)
    print(f"Total episodes:     {len(all_metrics)}")
    print(f"Successful:         {len(successful_metrics)}")
    print(f"Failed:             {failed_count}")
    print(f"\nEndpoint Distance Statistics:")
    print(f"  Mean:   {avg_distance:.4f}")
    print(f"  Std:    {std_distance:.4f}")
    print(f"  Min:    {min_distance:.4f}")
    print(f"  Max:    {max_distance:.4f}")
    print(f"  Median: {median_distance:.4f}")
    print("="*60 + "\n")
    
    results = {
        "total_episodes": len(all_metrics),
        "successful": len(successful_metrics),
        "failed": failed_count,
        "statistics": {
            "mean_distance": float(avg_distance),
            "std_distance": float(std_distance),
            "min_distance": float(min_distance),
            "max_distance": float(max_distance),
            "median_distance": float(median_distance),
        },
        "per_episode": [
            {
                "episode_index": m.episode_index,
                "distance": m.euclidean_distance,
                "gt_endpoint": list(m.gt_endpoint),
                "pred_endpoint": list(m.pred_endpoint),
                "success": m.success,
                "error": m.error_message,
                "video_path": m.video_path,
            }
            for m in all_metrics
        ],
    }
    
    results_path = output_dir / "batch_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
