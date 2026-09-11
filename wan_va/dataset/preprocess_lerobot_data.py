#!/usr/bin/env python3
"""
Processing script using Wan2.2 VAE and text encoder initialization logic.

Supports:
- Step 1: Generate action_config in episodes.jsonl
- Step 2: Extract latents using Wan2.2 VAE
- Step 3: (Optional) Extract latents using VGGT-Omega aggregator
- Batch processing multiple datasets in a directory
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import multiprocessing as mp

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# Add Wan2.2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from Wan2.2 utils
from modules.utils import load_vae, load_text_encoder, load_tokenizer
from modules.vggt_adapter import VGGTAdapter


# ==================== Step 1: Generate action_config ====================

def generate_action_config(episode: Dict) -> List[Dict]:
    """Generate default action_config for an episode.

    Default: treat entire episode as single segment with task description.
    """
    episode_length = episode.get("length", 0)
    tasks = episode.get("tasks", [])
    task_text = tasks[0] if tasks else ""

    action_config = [
        {
            "start_frame": 0,
            "end_frame": episode_length,
            "action_text": task_text,
            "skill": ""  # Optional field
        }
    ]
    return action_config


def update_episodes_jsonl(dataset_path: Path) -> None:
    """Step 1: Add or update action_config in episodes.jsonl.

    Args:
        dataset_path: Path to LeRobot dataset root
    """
    episodes_path = dataset_path / "meta" / "episodes.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_path}")

    # Load existing episodes
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    # Update action_config
    updated_count = 0
    for episode in episodes:
        episode['action_config'] = generate_action_config(episode)
        updated_count += 1

    # Save updated episodes
    with open(episodes_path, 'w') as f:
        for episode in episodes:
            f.write(json.dumps(episode) + '\n')

    print(f"  ✓ Updated {updated_count}/{len(episodes)} episodes in {episodes_path}")


# ==================== Step 2: Extract latents ====================

def normalize_latents(latents, latents_mean, latents_std):
    """Normalize latents using VAE config."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def extract_latents_from_video(
    vae,
    video_path: str,
    action_text: str,
    text_encoder,
    tokenizer,
    fps: float = 12.5,
    height: int = 256,
    width: int = 320,
    start_frame: int = 0,
    end_frame: int = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "npu",
):
    """Extract latent features from video using Xi0 VAE (AutoencoderKLWan)."""
    import av  # Use PyAV for better codec support (including AV1)

    # Read video with PyAV
    container = av.open(video_path)
    stream = container.streams.video[0]

    ori_fps = float(stream.average_rate)
    total_frames = stream.frames
    video_height = stream.height
    video_width = stream.width

    if end_frame is None:
        end_frame = total_frames

    # Calculate frame indices to sample
    frame_indices = []
    frame_idx = start_frame
    frame_stride = max(1, int(ori_fps / fps))
    while frame_idx < end_frame:
        frame_indices.append(frame_idx)
        frame_idx += frame_stride

    if len(frame_indices) == 0:
        raise ValueError(f"No frames to extract from {video_path}")

    # Adjust frame count to satisfy constraint: (num_frames - 1) % 4 == 0
    # This is required by Wan VAE encoder
    num_frames = len(frame_indices)
    if (num_frames - 1) % 4 != 0:
        # Find the largest valid frame count <= current count
        # num_frames = 4k + 1, where k is integer
        # => k = (num_frames - 1) // 4
        # => valid_num_frames = 4k + 1
        valid_num_frames = ((num_frames - 1) // 4) * 4 + 1
        if valid_num_frames > 0:
            frame_indices = frame_indices[:valid_num_frames]
            print(f"Warning: Adjusted frame count from {num_frames} to {valid_num_frames} to satisfy VAE constraint")
        else:
            raise ValueError(f"Frame count {num_frames} cannot satisfy VAE constraint (need at least 1 frame)")

    # Read frames using PyAV (supports software AV1 decoding)
    frames = []
    frame_count = 0
    target_indices = set(frame_indices)

    container.seek(0)
    for frame in container.decode(video=0):
        if frame_count in target_indices:
            # Convert to numpy array (RGB format)
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        frame_count += 1
        if frame_count >= end_frame:
            break
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # Convert to tensor
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).float().permute(3, 0, 1, 2).unsqueeze(0)
    # Shape: (1, C, F, H, W) - batch, channels, frames, height, width

    # Resize spatial dimensions (height and width) for each frame
    frames = frames.squeeze(0).permute(1, 0, 2, 3)  # (F, C, H, W)
    ori_frames = frames.clone()  # Keep original frames for VGGT-Omega
    frames = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=False)
    # Reshape back to (1, C, F, H, W)
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
    ori_frames = ori_frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, F, H, W)

    # Normalize to [-1, 1]
    frames = frames / 255.0 * 2.0 - 1.0
    frames = frames.to(device).to(dtype)
    ori_frames = ori_frames / 255.0 * 2.0 - 1.0
    ori_frames = ori_frames.to(device).to(dtype)

    # Encode with VAE (AutoencoderKLWan)
    with torch.no_grad():
        # Use vae.encode() directly - it handles patchify and encoder internally
        posterior = vae.encode(frames).latent_dist
        # Use mean (mu) instead of sample for deterministic latent representation
        latents = posterior.mean

        # Normalize latents using VAE config
        latents_mean = torch.tensor(vae.config.latents_mean).to(latents.device)
        latents_std = torch.tensor(vae.config.latents_std).to(latents.device)
        latents = normalize_latents(latents, latents_mean, 1.0 / latents_std)

    # Get latent shape
    latent_num_frames = latents.shape[2]
    latent_height = latents.shape[3]
    latent_width = latents.shape[4]

    # Flatten latents: (B, C, F, H, W) -> (B*F*H*W, C)
    # Note: batch dimension is included in the flattening
    latents_flat = rearrange(latents, 'b c f h w -> (b f h w) c')

    # Encode text
    prompt = action_text.strip()
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        text_encoder_device = next(text_encoder.parameters()).device
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder_device),
            mask.to(text_encoder_device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        text_emb = prompt_embeds[0]

    # Prepare output
    output = {
        'latent': latents_flat.cpu().to(dtype),
        'latent_num_frames': latent_num_frames,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'video_num_frames': len(frame_indices),
        'video_height': video_height,
        'video_width': video_width,
        'text_emb': text_emb.cpu().to(dtype),
        'text': action_text,
        'frame_ids': frame_indices,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'fps': fps,
        'ori_fps': int(ori_fps),
    }

    return output, ori_frames


def extract_vggt_latents_from_video(
    vggt_adapter,
    video_multiview_frames: list,
    video_multiview_latents: list,
    view_keys: Optional[List[str]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "npu"
):
    """Extract latent features from video using VGGT-Omega aggregator."""
    if vggt_adapter is None:
        raise ValueError("vggt_adapter must not be None")
    if not video_multiview_frames:
        raise ValueError("video_multiview_frames must contain at least one view")
    if len(video_multiview_frames) != len(video_multiview_latents):
        raise ValueError(
            "video_multiview_frames and video_multiview_latents must have the same number of views "
            f"(got {len(video_multiview_frames)} and {len(video_multiview_latents)})"
        )
    if view_keys is not None and len(view_keys) != len(video_multiview_frames):
        raise ValueError(
            "view_keys must have the same number of views as video_multiview_frames "
            f"(got {len(view_keys)} and {len(video_multiview_frames)})"
        )

    expected_start_frame = start_frame
    expected_end_frame = end_frame
    view_frame_tensors = []

    for view_idx, (frames, latent_data) in enumerate(zip(video_multiview_frames, video_multiview_latents)):
        if frames is None:
            raise ValueError(f"View {view_idx} has no frame tensor")
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        if frames.ndim != 5:
            raise ValueError(
                f"View {view_idx} frames must have shape [1, C, F, H, W] or [C, F, H, W], "
                f"got {tuple(frames.shape)}"
            )
        if frames.shape[0] != 1:
            raise ValueError(f"View {view_idx} frames must have batch size 1, got {frames.shape[0]}")

        latent_start_frame = latent_data.get("start_frame") if isinstance(latent_data, dict) else None
        latent_end_frame = latent_data.get("end_frame") if isinstance(latent_data, dict) else None
        video_num_frames = latent_data.get("video_num_frames") if isinstance(latent_data, dict) else None

        if latent_start_frame is not None and expected_start_frame is not None and latent_start_frame != expected_start_frame:
            raise ValueError(
                f"View {view_idx} frame/latent start_frame mismatch: "
                f"{expected_start_frame} != {latent_start_frame}"
            )
        if latent_end_frame is not None and expected_end_frame is not None and latent_end_frame != expected_end_frame:
            raise ValueError(
                f"View {view_idx} frame/latent end_frame mismatch: "
                f"{expected_end_frame} != {latent_end_frame}"
            )
        if video_num_frames != frames.shape[2]:
            raise ValueError(
                f"View {view_idx} video frame number mismatch: "
                f"{frames.shape[2]} != {video_num_frames}"
            )

        view_frame_tensors.append(frames[0].float())

    num_video_frames = view_frame_tensors[0].shape[1]
    for view_idx, frame_tensors in enumerate(view_frame_tensors):
        if frame_tensors.shape[1] != num_video_frames:
            raise ValueError(
                f"View {view_idx} has {frame_tensors.shape[1]} frames, expected {num_video_frames}"
            )

    resized_view_frame_tensors = []
    resized_target_size = (
        max(int(frame_tensors.shape[2]) for frame_tensors in view_frame_tensors),
        max(int(frame_tensors.shape[3]) for frame_tensors in view_frame_tensors),
    )
    for frame_tensors in view_frame_tensors:
        # extract_latents_from_video returns [-1, 1]; VGGTAdapter expects [0, 1].
        frame_tensors = ((frame_tensors + 1.0) * 0.5).clamp(0.0, 1.0)
        frame_tensors = rearrange(frame_tensors, "c f h w -> f c h w")
        frame_tensors = F.interpolate(frame_tensors, size=resized_target_size, mode="bilinear", align_corners=False)
        resized_view_frame_tensors.append(frame_tensors)

    # [B, C, 3, H, W], where each time step is one VGGT batch item.
    vggt_images = torch.stack(resized_view_frame_tensors, dim=2).to(device=device, dtype=dtype)

    # 分批处理避免显存溢出
    encode_batch_size = getattr(vggt_adapter, "encode_batch_size", 20)  # 可通过adapter属性配置
    if vggt_images.shape[0] <= encode_batch_size:
        encoded = vggt_adapter.encode(vggt_images,
                                      return_dict=True,
                                      device=vggt_images.device,
                                      torch_dtype=dtype)
        vggt_latents = encoded["latents"].detach().cpu().to(dtype)
    else:
        # 分批encode后合并
        all_latents = []
        for i in range(0, vggt_images.shape[0], encode_batch_size):
            batch_images = vggt_images[i:i + encode_batch_size]
            encoded = vggt_adapter.encode(batch_images,
                                          return_dict=True,
                                          device=batch_images.device,
                                          torch_dtype=dtype)
            all_latents.append(encoded["latents"].detach().cpu().to(dtype))
        vggt_latents = torch.cat(all_latents, dim=0)

    output = {
        "vggt_latents": rearrange(vggt_latents, "b i j c -> (b i j) c"),
        'vggt_latent_num_frames': vggt_latents.shape[0],
        'vggt_latent_height': vggt_latents.shape[1],
        'vggt_latent_width': vggt_latents.shape[2],
        "vggt_image_size": getattr(vggt_adapter, "default_image_size", None),
        "vggt_latent_frame_mode": getattr(vggt_adapter, "latent_frame_mode", None),
        "num_video_frames": num_video_frames,
        "view_keys": view_keys,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
    return output


def get_target_resolution(video_key: str, base_height: int, base_width: int, env_type: str = None) -> tuple:
    """Get target resolution for a video key based on environment type.

    For robotwin_tshape mode:
    - cam_high uses base resolution (base_height, base_width)
    - wrist cameras use half resolution (base_height // 2, base_width // 2)

    Args:
        video_key: Video key (e.g., 'observation.images.cam_high')
        base_height: Base height resolution
        base_width: Base width resolution
        env_type: Environment type (e.g., 'robotwin_tshape')

    Returns:
        tuple: (height, width) target resolution
    """

    if env_type == 'robotwin_tshape':
        wrist_list = ['left_wrist', 'right_wrist']
    elif env_type == 'tennis_tshape':
        wrist_list = ['side_cam', 'wrist_cam']
    else:
        wrist_list = []

    # Check if this is a wrist camera
    is_wrist = any(wrist_key in video_key for wrist_key in wrist_list)
    if is_wrist:
        print(f"Found {video_key} and set half target resolution.")
        # Use half resolution for wrist cameras
        return base_height // 2, base_width // 2

    # Default: use base resolution
    return base_height, base_width


# ==================== Worker function for parallel processing ====================

def process_dataset_worker(args_dict: dict):
    """Worker function that processes a subset of episodes on a specific device.

    This is designed to be called via torch.multiprocessing.Process.
    """
    device = args_dict['device']
    dataset_path = Path(args_dict['dataset_path'])
    args = argparse.Namespace(**args_dict['namespace_args'])
    episode_indices = args_dict['episode_indices']

    print(f"[Worker {device}] Started, processing {len(episode_indices)} episodes")

    # Build model paths
    vae_path = Path(args.pretrained_model_path) / "vae"
    text_encoder_path = Path(args.pretrained_model_path) / "text_encoder"
    tokenizer_path = Path(args.pretrained_model_path) / "tokenizer"

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load models on this device
    print(f"[Worker {device}] Loading models...")
    vae = load_vae(str(vae_path), dtype, device)
    text_encoder = load_text_encoder(str(text_encoder_path), dtype, device)
    tokenizer = load_tokenizer(str(tokenizer_path))

    vggt_adapter = None
    if not args.skip_step3:
        print(f"[Worker {device}] Loading VGGT-Omega adapter...")
        vggt_adapter_config = {
            "default_image_size": args.vggt_image_size,
            "latent_frame_mode": args.vggt_latent_frame_mode,
            "latent_dimension": args.vggt_latent_dimension
        }
        vggt_adapter = VGGTAdapter.from_pretrained(
            vggt_adapter_config=vggt_adapter_config,
            vggt_pretrained_path=args.vggt_pretrained_model_path,
            device=device,
            torch_dtype=dtype,
        )
        vggt_adapter.eval()

    print(f"[Worker {device}] Models loaded")

    # Read episodes
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    all_episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            if line.strip():
                all_episodes.append(json.loads(line))

    # Filter episodes for this worker
    episodes = [all_episodes[i] for i in episode_indices]

    # Auto-detect video keys
    video_keys = args.video_keys
    if video_keys is None:
        videos_dir = dataset_path / "videos"
        if videos_dir.exists():
            chunk_dirs = list(videos_dir.glob("chunk-*"))
            if chunk_dirs:
                video_keys = [d.name for d in chunk_dirs[0].iterdir() if d.is_dir()]

    if not video_keys:
        print(f"[Worker {device}] No video keys found, skipping")
        return

    # Detect or use specified env_type
    env_type = args.env_type
    if env_type is None:
        if any('left_wrist' in vk for vk in video_keys) and any('right_wrist' in vk for vk in video_keys):
            env_type = 'robotwin_tshape'
            print(f"[Worker {device}] Auto-detected env_type: {env_type}")
        else:
            print(f"[Worker {device}] No env_type detected, using base resolution for all cameras")
    else:
        print(f"[Worker {device}] Using env_type: {env_type}")

    # Create latents directories
    latents_dir = dataset_path / "latents"
    latents_dir.mkdir(exist_ok=True)
    if not args.skip_step3:
        vggt_latents_dir = dataset_path / "vggt_latents"
        vggt_latents_dir.mkdir(exist_ok=True)

    # Process episodes assigned to this worker
    for episode in tqdm(episodes, desc=f"[Worker {device}] Processing episodes", position=int(device.split(':')[-1])):
        episode_index = episode['episode_index']
        episode_chunk = episode.get('episode_chunk', 0)
        action_configs = episode['action_config']

        for acfg in action_configs:
            start_frame = acfg['start_frame']
            end_frame = acfg['end_frame']
            action_text = acfg['action_text']

            if not args.skip_step3:
                video_multiview_frames = []
                video_multiview_latents = []
                video_multiview_keys = []

            for video_key in video_keys:
                video_file = (
                    dataset_path / "videos" / f"chunk-{episode_chunk:03d}" /
                    video_key / f"episode_{episode_index:06d}.mp4"
                )

                if not video_file.exists():
                    continue

                latent_key_dir = latents_dir / f"chunk-{episode_chunk:03d}" / video_key
                latent_key_dir.mkdir(parents=True, exist_ok=True)
                latent_file = latent_key_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"

                target_height, target_width = get_target_resolution(
                    video_key, args.height, args.width, env_type
                )

                try:
                    latent_data, frames = extract_latents_from_video(
                        vae, str(video_file), action_text, text_encoder, tokenizer,
                        args.fps, target_height, target_width,
                        start_frame, end_frame, dtype, device
                    )
                    torch.save(latent_data, latent_file)
                    if not args.skip_step3:
                        video_multiview_frames.append(frames)
                        video_multiview_latents.append(latent_data)
                        video_multiview_keys.append(video_key)
                except Exception as e:
                    print(f"\n[Worker {device}] Error processing {video_file}: {e}")
                    raise

            if not args.skip_step3:
                if len(video_multiview_keys) != len(video_keys):
                    missing_keys = sorted(set(video_keys) - set(video_multiview_keys))
                    print(
                        f"\n[Worker {device}] Skipping VGGT-Omega in episode {episode_index} "
                        f"frames {start_frame}-{end_frame}: missing/failed views {missing_keys}"
                    )
                    continue

                vggt_latent_key_dir = vggt_latents_dir / f"chunk-{episode_chunk:03d}"
                vggt_latent_key_dir.mkdir(parents=True, exist_ok=True)
                vggt_latent_file = (vggt_latent_key_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth")
                try:
                    vggt_latent_data = extract_vggt_latents_from_video(
                        vggt_adapter,
                        video_multiview_frames,
                        video_multiview_latents,
                        view_keys=video_multiview_keys,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        dtype=dtype,
                        device=device,
                    )
                    torch.save(vggt_latent_data, vggt_latent_file)
                except Exception as e:
                    print(f"\n[Worker {device}] Error processing VGGT-Omega in episode {episode_index}: {e}")
                    raise

    print(f"[Worker {device}] Completed {len(episode_indices)} episodes")


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset using Wan2.2 VAE and text encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Single GPU (serial, original behavior):
              python preprocess_lerobot_data.py --input-dir /path/to/datasets

              # 8-GPU parallel (distribute episodes across 8 NPU cards):
              python preprocess_lerobot_data.py --input-dir /path/to/datasets --num-devices 8
        """
    )
    parser.add_argument("--input-dir", type=str, help="Path to directory containing multiple datasets")
    parser.add_argument("--pretrained-model-path", type=str,
                        default="/efs-gy1/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base",
                        help="Path to pretrained model root directory (containing vae/, text_encoder/, tokenizer/)")
    parser.add_argument("--vggt-pretrained-model-path", type=str,
                        default="/efs-gy1/lijianwen/Pretrained_models/VGGT/VGGT-Omega/vggt_omega_1b_512.pt",
                        help="Path to VGGT-Omega checkpoint.")
    parser.add_argument("--fps", type=float, default=15, help="Target FPS")
    parser.add_argument("--height", type=int, default=256, help="Target height")
    parser.add_argument("--width", type=int, default=320, help="Target width")
    parser.add_argument("--env-type", type=str, default=None,
                        help="Environment type (e.g., 'robotwin_tshape'). If not specified, auto-detect from dataset.")
    parser.add_argument("--vggt-image-size", type=int, nargs=2, default=[512, 512],
                        help="VGGT-Omega default image size [height width]")
    parser.add_argument("--vggt-latent-frame-mode", type=str, default="concat",
                        help="VGGT-Omega latent frame mode (e.g., 'concat', 'every_first')")
    parser.add_argument("--vggt-latent-dimension", type=int, default=2048,
                        help="VGGT-Omega latent dimension")
    parser.add_argument("--num-devices", type=int, default=1,
                        help="Number of NPU devices to use for parallel processing (default: 1 = serial)")
    parser.add_argument("--device-ids", type=str, default=None,
                        help="Comma-separated device IDs to use (e.g., '0,1,2,3'). Default: 0..num-devices-1")
    parser.add_argument("--device", type=str, default="npu:0", help="Device to use (only used when num-devices=1)")
    parser.add_argument("--video-keys", type=str, nargs='+', help="Video keys to process")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--skip-step1", action="store_true", help="Skip Step 1 (action_config generation)")
    parser.add_argument("--skip-step2", action="store_true", help="Skip Step 2 (latent extraction)")
    parser.add_argument("--skip-step3", action="store_true", help="Skip Step 3 (VGGT-Omega latent extraction)")

    args = parser.parse_args()

    # Validate input arguments
    if not args.input_dir:
        parser.error("Argument --input-dir must be specified")

    # Get list of datasets to process
    dataset_paths = []

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all LeRobot datasets (directories with meta/info.json)
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir() and (subdir / "meta" / "info.json").exists():
                dataset_paths.append(subdir)

        if len(dataset_paths) == 0:
            if input_dir.is_dir() and (input_dir / "meta" / "info.json").exists():
                dataset_paths.append(input_dir)

        print(f"Found {len(dataset_paths)} datasets in {input_dir}:")
        for p in dataset_paths:
            print(f"  - {p.name}")

    if not dataset_paths:
        raise ValueError("No datasets found to process")

    # Step 1 is always done serially (it's lightweight and modifies the same episodes.jsonl)
    if not args.skip_step1:
        print("\nStep 1: Generating action_config (serial)...")
        for dataset_path in dataset_paths:
            print(f"  Processing {dataset_path.name}...")
            update_episodes_jsonl(dataset_path)

    # Steps 2 & 3
    if not args.skip_step2:
        # Determine number of devices
        num_devices = args.num_devices

        if args.device_ids is not None:
            device_ids = [int(x.strip()) for x in args.device_ids.split(',')]
            num_devices = len(device_ids)
        else:
            device_ids = list(range(num_devices))

        device_id_strs = [f"npu:{did}" for did in device_ids]

        print(f"\n{'='*60}")
        print(f"Configuration:")
        print(f"  Devices: {num_devices} ({', '.join(device_id_strs)})")
        print(f"  Dtype: {args.dtype}")
        print(f"  Target resolution: {args.height}x{args.width} @ {args.fps} fps")
        print(f"  Pretrained model: {args.pretrained_model_path}")
        print(f"  VGGT-Omega image size: {args.vggt_image_size[0]}x{args.vggt_image_size[1]}")
        print(f"  VGGT-Omega latent frame mode: {args.vggt_latent_frame_mode}")
        print(f"  VGGT-Omega latent dimension: {args.vggt_latent_dimension}")
        print(f"  Steps: Step2={not args.skip_step2}, Step3={not args.skip_step3}")
        print(f"{'='*60}\n")

        # Process each dataset
        for dataset_idx, dataset_path in enumerate(dataset_paths, 1):
            print(f"\n{'='*60}")
            print(f"[{dataset_idx}/{len(dataset_paths)}] Processing: {dataset_path.name}")
            print(f"{'='*60}\n")

            # Read episodes to split work
            episodes_path = dataset_path / "meta" / "episodes.jsonl"
            all_episodes = []
            with open(episodes_path, 'r') as f:
                for line in f:
                    if line.strip():
                        all_episodes.append(json.loads(line))

            total_episodes = len(all_episodes)
            print(f"  Total episodes: {total_episodes}")

            if num_devices == 1:
                # Serial processing
                single_worker_args = {
                    'device': args.device,
                    'dataset_path': str(dataset_path),
                    'namespace_args': vars(args),
                    'episode_indices': list(range(total_episodes)),
                }
                process_dataset_worker(single_worker_args)
            else:
                # Parallel processing: split episodes into chunks
                # Ensure each chunk has approximately equal size
                chunk_size = (total_episodes + num_devices - 1) // num_devices
                chunks = []
                for i in range(num_devices):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, total_episodes)
                    if start_idx < total_episodes:
                        chunks.append(list(range(start_idx, end_idx)))

                print(f"  Splitting {total_episodes} episodes across {len(chunks)} workers:")
                for i, chunk in enumerate(chunks):
                    print(f"    {device_id_strs[i]}: episodes {chunk[0]}-{chunk[-1]} ({len(chunk)} episodes)")

                # Spawn worker processes
                ctx = mp.get_context('spawn')
                processes = []
                for i, chunk in enumerate(chunks):
                    worker_args = {
                        'device': device_id_strs[i],
                        'dataset_path': str(dataset_path),
                        'namespace_args': vars(args),
                        'episode_indices': chunk,
                    }
                    p = ctx.Process(
                        target=process_dataset_worker,
                        args=(worker_args,),
                        name=f"Worker-{device_id_strs[i]}",
                    )
                    p.start()
                    processes.append(p)

                # Wait for all workers to finish
                for p in processes:
                    p.join()

                # Check for failures
                failed = False
                for p in processes:
                    if p.exitcode != 0:
                        print(f"Worker {p.name} failed with exit code {p.exitcode}")
                        failed = True

                if failed:
                    print(f"\n  ✗ Some workers failed for dataset {dataset_path.name}")
                else:
                    print(f"\n  ✓ All workers completed successfully for {dataset_path.name}")

            latents_dir = dataset_path / "latents"
            print(f"  ✓ Latents saved to {latents_dir}")
            if not args.skip_step3:
                vggt_latents_dir = dataset_path / "vggt_latents"
                print(f"  ✓ VGGT-Omega latents saved to {vggt_latents_dir}")

    print(f"\n{'='*60}")
    print(f"✓ All datasets processed successfully!")
    print(f"{'='*60}\n")

    # Generate empty_emb.pt in input_dir
    if args.input_dir:
        input_dir = Path(args.input_dir)
        empty_emb_path = input_dir / "empty_emb.pt"

        # Create a zero tensor with shape (512, 4096) and dtype bfloat16
        # This matches the content in /efs-gy1/dgh/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50/empty_emb.pt
        empty_emb = torch.zeros(512, 4096, dtype=torch.bfloat16)
        torch.save(empty_emb, empty_emb_path)

        print(f"✓ Generated empty_emb.pt at {empty_emb_path}")


if __name__ == "__main__":
    main()
