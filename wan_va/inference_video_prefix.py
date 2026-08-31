"""Standalone multi-frame video-prefix inference for LingBot-VGGT.

This entry point resolves an episode directly from a LeRobot dataset, samples
its synchronized camera videos and actions by timestamp, packs the action
prefix with the same alignment used during training, pre-fills the joint
video/VGGT and action KV caches, and continues LingBot-VGGT's rollout from the
last observed video/VGGT/action block.

Example:
    python -m wan_va.inference_video_prefix \
        --config-name robotwin_i2av \
        --dataset-root /data/my_lerobot_dataset \
        --episode-index 0 \
        --sample-fps 15 \
        --prefix-num-frames 13 \
        --num-chunks 10 \
        --output-video outputs/prefix_i2va.mp4

PyAV is required because frame indices and container FPS are not reliable for
timestamp-accurate sampling, especially for variable-frame-rate videos.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

if os.environ.get("LINGBOT_USE_NPU") == "1":
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "LINGBOT_USE_NPU=1 requires torch-npu and its CANN runtime"
        ) from exc

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

from .configs import VA_CONFIGS
from .modules.utils import load_vae
from .utils import data_seq_to_patch, init_logger, logger
from .wan_va_server import VA_Server


def make_uniform_timestamps(
    num_frames: int,
    sample_fps: float,
    start_sec: float = 0.0,
) -> np.ndarray:
    """Build timestamps without changing or re-encoding the source video."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError(f"start_sec must be non-negative, got {start_sec}")

    return start_sec + np.arange(num_frames, dtype=np.float64) / sample_fps


def sample_video_by_timestamps(
    video_path: str | os.PathLike[str],
    target_timestamps: Iterable[float],
    max_time_error: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Select original RGB frames whose PTS is nearest to each target time.

    Target and returned timestamps are relative to the first decoded video
    frame. No FPS conversion, interpolation, duplication, or re-encoding is
    performed.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "Timestamp-based sampling requires PyAV. Install it with `pip "
            "install av`."
        ) from exc

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")

    targets = np.asarray(list(target_timestamps), dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("target_timestamps must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(targets)) or np.any(targets < 0):
        raise ValueError("target_timestamps must be finite and non-negative")
    if np.any(np.diff(targets) <= 0):
        raise ValueError("target_timestamps must be strictly increasing")
    if not math.isfinite(max_time_error) or max_time_error < 0:
        raise ValueError("max_time_error must be finite and non-negative")

    selected_frames: list[np.ndarray] = []
    selected_times: list[float] = []
    selected_pts: list[int] = []
    target_id = 0
    first_pts_time: float | None = None
    previous: tuple[float, np.ndarray, int] | None = None

    def append_selected(candidate: tuple[float, np.ndarray, int]) -> None:
        nonlocal target_id
        actual_time, rgb, pts = candidate
        target_time = float(targets[target_id])
        error = abs(actual_time - target_time)
        if error > max_time_error:
            raise RuntimeError(
                f"No frame in {path} is close enough to target timestamp "
                f"{target_time:.6f}s: nearest={actual_time:.6f}s, "
                f"error={error:.6f}s, tolerance={max_time_error:.6f}s"
            )
        if selected_pts and pts == selected_pts[-1]:
            raise RuntimeError(
                f"The same source frame (PTS={pts}) in {path} was selected "
                "for two target timestamps. Lower --sample-fps or increase "
                "the source frame rate."
            )
        selected_frames.append(np.ascontiguousarray(rgb))
        selected_times.append(actual_time)
        selected_pts.append(pts)
        target_id += 1

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            absolute_time = float(frame.pts * stream.time_base)
            if first_pts_time is None:
                first_pts_time = absolute_time
            relative_time = absolute_time - first_pts_time

            if previous is not None and relative_time < previous[0]:
                raise RuntimeError(
                    f"Non-monotonic video PTS encountered in {path}: "
                    f"{relative_time:.6f}s after {previous[0]:.6f}s"
                )

            current = (
                relative_time,
                frame.to_ndarray(format="rgb24"),
                int(frame.pts),
            )

            while target_id < len(targets) and targets[target_id] <= relative_time:
                if previous is None:
                    selected = current
                else:
                    prev_error = abs(previous[0] - targets[target_id])
                    curr_error = abs(current[0] - targets[target_id])
                    selected = previous if prev_error <= curr_error else current
                append_selected(selected)

            previous = current
            if target_id == len(targets):
                break

    # A target just beyond the final PTS can still legitimately be nearest to
    # the final frame, provided it is inside the caller's tolerance.
    while target_id < len(targets) and previous is not None:
        append_selected(previous)

    if target_id != len(targets):
        raise RuntimeError(
            f"Video {path} ended before target timestamp "
            f"{targets[target_id]:.6f}s"
        )

    return (
        selected_frames,
        np.asarray(selected_times, dtype=np.float64),
        np.asarray(selected_pts, dtype=np.int64),
    )


def parse_key_value(items: list[str], value_type, option_name: str) -> dict:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"{option_name} expects KEY=VALUE, but received: {item!r}"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{option_name} contains an empty key: {item!r}")
        if key in result:
            raise ValueError(f"Duplicate key for {option_name}: {key}")
        result[key] = value_type(raw_value)
    return result


def _read_jsonl_record(
    path: Path,
    key: str,
    value: int,
) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            if record.get(key) == value:
                return record
    return None


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(200).startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )


def load_lerobot_episode(
    dataset_root: str | os.PathLike[str],
    episode_index: int,
    expected_camera_keys: Iterable[str],
    prompt_time_sec: float,
) -> tuple[dict[str, str], str, dict]:
    """Resolve camera videos and task text from a LeRobot dataset episode."""
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"LeRobot metadata does not exist: {info_path}"
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LeRobot metadata JSON: {info_path}") from exc

    total_episodes = int(info.get("total_episodes", 0))
    if episode_index < 0 or (
        total_episodes > 0 and episode_index >= total_episodes
    ):
        raise ValueError(
            f"episode_index={episode_index} is outside [0, "
            f"{total_episodes - 1}] for {root}"
        )
    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    if chunk_size <= 0:
        raise ValueError(f"Invalid LeRobot chunk size: {chunk_size}")
    episode_chunk = episode_index // chunk_size

    features = info.get("features", {})
    dataset_video_keys = {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    expected_camera_keys = list(expected_camera_keys)
    missing_cameras = sorted(set(expected_camera_keys) - dataset_video_keys)
    if missing_cameras:
        raise ValueError(
            f"LeRobot dataset is missing configured video features "
            f"{missing_cameras}; available={sorted(dataset_video_keys)}"
        )

    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4",
    )
    format_values = {
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
        "chunk_index": episode_chunk,
    }
    video_paths = {}
    lfs_pointers = []
    for camera_key in expected_camera_keys:
        try:
            relative_path = video_template.format(
                video_key=camera_key,
                **format_values,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unsupported LeRobot video_path template: {video_template!r}"
            ) from exc
        video_path = root / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Episode {episode_index} video does not exist: {video_path}"
            )
        if _is_git_lfs_pointer(video_path):
            lfs_pointers.append(str(video_path))
        video_paths[camera_key] = str(video_path)
    if lfs_pointers:
        raise RuntimeError(
            "LeRobot video files are Git LFS pointer files rather than MP4 "
            "content. Fetch the dataset media first (for example, `git lfs "
            "pull`) and retry. Pointer files: " + ", ".join(lfs_pointers)
        )

    episode = _read_jsonl_record(
        root / "meta" / "episodes.jsonl",
        "episode_index",
        episode_index,
    )
    if episode is None:
        episode = _read_jsonl_record(
            root / "meta" / "episodes_ori.jsonl",
            "episode_index",
            episode_index,
        )
    if episode is None:
        raise RuntimeError(
            f"Cannot find episode {episode_index} in meta/episodes.jsonl "
            "or meta/episodes_ori.jsonl"
        )

    dataset_fps = float(info.get("fps", 0))
    if not math.isfinite(dataset_fps) or dataset_fps <= 0:
        raise ValueError(f"Invalid LeRobot dataset fps: {dataset_fps}")
    prompt_frame = int(math.floor(prompt_time_sec * dataset_fps + 1e-6))
    prompt = None
    selected_action_config = None
    for action_config in episode.get("action_config", []):
        start_frame = int(action_config.get("start_frame", 0))
        end_frame = int(action_config.get("end_frame", start_frame))
        if start_frame <= prompt_frame < end_frame:
            candidate = action_config.get("action_text")
            if isinstance(candidate, str) and candidate.strip():
                prompt = candidate.strip()
                selected_action_config = action_config
                break
    if prompt is None:
        tasks = episode.get("tasks", [])
        if tasks and isinstance(tasks[0], str) and tasks[0].strip():
            prompt = tasks[0].strip()
    if prompt is None:
        raise RuntimeError(
            f"Episode {episode_index} has no usable task text in "
            "meta/episodes.jsonl"
        )

    dataset_metadata = {
        "dataset_root": str(root),
        "lerobot_codebase_version": info.get("codebase_version"),
        "dataset_fps": dataset_fps,
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
        "episode_length": episode.get("length"),
        "episode_tasks": episode.get("tasks", []),
        "prompt_time_sec": prompt_time_sec,
        "prompt_frame": prompt_frame,
        "selected_action_config": selected_action_config,
        "resolved_prompt": prompt,
    }
    return video_paths, prompt, dataset_metadata


class VideoPrefixInference(VA_Server):
    """VGGT VA server extended with timestamp-sampled prefix inference."""

    @torch.no_grad()
    def encode_video_prefix(
        self,
        prefix_obs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode synchronized VAE and VGGT prefix latents causally.

        Wan's streaming encoder initializes its temporal caches from one frame.
        Feeding the full prefix into an empty cache can make residual branches
        disagree on temporal length (for example, 25 versus 13). The online
        LingBot-VGGT path avoids this by encoding one initial frame and then one
        temporal-compression group at a time. VGGT is encoded over the exact
        same RGB groups so its merged latent timeline stays synchronized with
        the VAE timeline.
        """
        observations = prefix_obs["obs"]
        if not observations:
            raise ValueError("Video prefix contains no observations")

        temporal_factor = getattr(
            self.job_config,
            "vae_temporal_factor",
            4,
        )
        chunks = [observations[:1]]
        for start in range(1, len(observations), temporal_factor):
            chunks.append(observations[start : start + temporal_factor])

        latent_chunks = []
        vggt_chunks = []
        for chunk_id, observations_chunk in enumerate(chunks):
            if chunk_id > 0 and len(observations_chunk) != temporal_factor:
                logger.warning(
                    f"Encoding an incomplete VAE temporal chunk with "
                    f"{len(observations_chunk)} RGB frames; expected "
                    f"{temporal_factor}"
                )
            latent_chunk = self._encode_obs({"obs": observations_chunk})
            if latent_chunk is not None and latent_chunk.shape[2] > 0:
                latent_chunks.append(latent_chunk)
                vggt_chunk = self._encode_vggt_obs(
                    {"obs": observations_chunk},
                    target_num_frames=latent_chunk.shape[2],
                )
                if vggt_chunk.shape[2] != latent_chunk.shape[2]:
                    raise RuntimeError(
                        "VAE/VGGT prefix encoders produced different temporal "
                        f"lengths for RGB chunk {chunk_id}: "
                        f"VAE={latent_chunk.shape[2]}, "
                        f"VGGT={vggt_chunk.shape[2]}"
                    )
                vggt_chunks.append(vggt_chunk)

        if not latent_chunks or not vggt_chunks:
            raise RuntimeError("Video prefix produced no VAE/VGGT latent chunks")
        prefix_latents = torch.cat(latent_chunks, dim=2)
        prefix_vggt_latents = torch.cat(vggt_chunks, dim=2)
        if prefix_latents.shape[2] != prefix_vggt_latents.shape[2]:
            raise RuntimeError(
                "Encoded VAE/VGGT prefix lengths differ: "
                f"VAE={prefix_latents.shape[2]}, "
                f"VGGT={prefix_vggt_latents.shape[2]}"
            )
        return prefix_latents, prefix_vggt_latents

    @torch.no_grad()
    def prefill_prefix_cache(
        self,
        prefix_latents: torch.Tensor,
        prefix_vggt_latents: torch.Tensor,
        prefix_actions: torch.Tensor,
    ) -> int:
        """Commit synchronized known VAE, VGGT, and action history."""
        if prefix_latents.ndim != 5:
            raise ValueError(
                f"Expected prefix latents [B,C,F,H,W], got "
                f"{tuple(prefix_latents.shape)}"
            )
        expected_action_shape = (
            1,
            self.job_config.action_dim,
            prefix_latents.shape[2],
            self.action_per_frame,
            1,
        )
        if tuple(prefix_actions.shape) != expected_action_shape:
            raise ValueError(
                f"Expected synchronized prefix actions "
                f"{expected_action_shape}, got {tuple(prefix_actions.shape)}"
            )
        expected_vggt_shape = (
            prefix_latents.shape[0],
            self.vggt_latent_dimension,
            prefix_latents.shape[2],
            self.vggt_latent_height,
            self.vggt_latent_width,
        )
        if tuple(prefix_vggt_latents.shape) != expected_vggt_shape:
            raise ValueError(
                f"Expected synchronized prefix VGGT latents "
                f"{expected_vggt_shape}, got "
                f"{tuple(prefix_vggt_latents.shape)}"
            )
        if prefix_latents.shape[2] == 0:
            return 0

        chunk_size = self.job_config.frame_chunk_size
        cursor = 0
        while cursor < prefix_latents.shape[2]:
            latent_chunk = prefix_latents[:, :, cursor : cursor + chunk_size]
            vggt_chunk = prefix_vggt_latents[:, :, cursor : cursor + chunk_size]
            action_chunk = prefix_actions[:, :, cursor : cursor + chunk_size]
            input_dict = self._prepare_latent_input(
                latent_model_input=latent_chunk,
                vggt_model_input=vggt_chunk,
                action_model_input=action_chunk,
                latent_t=0,
                vggt_t=0,
                action_t=0,
                frame_st_id=cursor,
            )
            latent_input = self._repeat_input_for_cfg(
                input_dict["latent_res_lst"]
            )
            vggt_input = self._repeat_input_for_cfg(
                input_dict["vggt_res_lst"]
            )
            self.transformer(
                latent_input,
                vggt_dict=vggt_input,
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )
            cursor += latent_chunk.shape[2]

        return cursor

    @torch.no_grad()
    def infer_conditioned_chunk(
        self,
        frame_st_id: int,
        latent_cond: torch.Tensor | None = None,
        vggt_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Generate one chunk, optionally fixing its first observed block."""
        frame_chunk_size = self.job_config.frame_chunk_size
        if latent_cond is not None:
            expected = (
                1,
                48,
                1,
                self.latent_height,
                self.latent_width,
            )
            if tuple(latent_cond.shape) != expected:
                raise ValueError(
                    f"Expected latent condition shape {expected}, got "
                    f"{tuple(latent_cond.shape)}"
                )
            latent_cond = latent_cond.to(device=self.device, dtype=self.dtype)

        if vggt_cond is not None:
            expected_vggt_cond = (
                1,
                self.vggt_latent_dimension,
                1,
                self.vggt_latent_height,
                self.vggt_latent_width,
            )
            if tuple(vggt_cond.shape) != expected_vggt_cond:
                raise ValueError(
                    f"Expected VGGT condition shape {expected_vggt_cond}, "
                    f"got {tuple(vggt_cond.shape)}"
                )
            vggt_cond = vggt_cond.to(
                device=self.device,
                dtype=self.dtype,
            )

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        vggt_latents = torch.randn(
            1,
            self.vggt_latent_dimension,
            frame_chunk_size,
            self.vggt_latent_height,
            self.vggt_latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )

        self.scheduler.set_timesteps(self.job_config.num_inference_steps)
        self.vggt_scheduler.set_timesteps(
            self.job_config.vggt_num_inference_steps
        )
        self.action_scheduler.set_timesteps(
            self.job_config.action_num_inference_steps
        )
        video_timesteps = F.pad(
            self.scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        vggt_timesteps = F.pad(
            self.vggt_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if self.job_config.video_exec_step != -1:
            video_timesteps = video_timesteps[
                : self.job_config.video_exec_step
            ]
            vggt_timesteps = vggt_timesteps[
                : self.job_config.video_exec_step
            ]
        if len(video_timesteps) != len(vggt_timesteps):
            raise ValueError(
                "Joint video/VGGT diffusion requires equal timestep counts, "
                f"got {len(video_timesteps)} and {len(vggt_timesteps)}"
            )
        action_timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )

        joint_timesteps = zip(video_timesteps, vggt_timesteps)
        for step_id, (timestep, vggt_timestep) in enumerate(
            tqdm(
                joint_timesteps,
                total=len(video_timesteps),
                desc="video+vggt",
            )
        ):
            last_step = step_id == len(video_timesteps) - 1
            input_dict = self._prepare_latent_input(
                latent_model_input=latents,
                vggt_model_input=vggt_latents,
                action_model_input=None,
                latent_t=timestep,
                vggt_t=vggt_timestep,
                latent_cond=latent_cond,
                vggt_cond=vggt_cond,
                frame_st_id=frame_st_id,
            )
            latent_input = self._repeat_input_for_cfg(
                input_dict["latent_res_lst"]
            )
            vggt_input = self._repeat_input_for_cfg(
                input_dict["vggt_res_lst"]
            )
            video_flow, vggt_flow = self.transformer(
                latent_input,
                vggt_dict=vggt_input,
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False,
            )

            if not last_step or self.job_config.video_exec_step != -1:
                video_flow = data_seq_to_patch(
                    self.job_config.patch_size,
                    video_flow,
                    frame_chunk_size,
                    self.latent_height,
                    self.latent_width,
                    batch_size=2 if self.use_cfg else 1,
                )
                if self.job_config.guidance_scale > 1:
                    video_flow = video_flow[1:] + self.job_config.guidance_scale * (
                        video_flow[:1] - video_flow[1:]
                    )
                else:
                    video_flow = video_flow[:1]
                latents = self.scheduler.step(
                    video_flow,
                    timestep,
                    latents,
                    return_dict=False,
                )
                vggt_flow = data_seq_to_patch(
                    self.job_config.vggt_patch_size,
                    vggt_flow,
                    frame_chunk_size,
                    self.vggt_latent_height,
                    self.vggt_latent_width,
                    batch_size=2 if self.use_cfg else 1,
                )
                if self.job_config.vggt_guidance_scale > 1:
                    vggt_flow = vggt_flow[1:] + (
                        self.job_config.vggt_guidance_scale
                        * (vggt_flow[:1] - vggt_flow[1:])
                    )
                else:
                    vggt_flow = vggt_flow[:1]
                vggt_latents = self.vggt_scheduler.step(
                    vggt_flow,
                    vggt_timestep,
                    vggt_latents,
                    return_dict=False,
                )

            if latent_cond is not None:
                latents[:, :, :1] = latent_cond
            if vggt_cond is not None:
                vggt_latents[:, :, :1] = vggt_cond

        if action_cond is not None:
            expected_action_cond = (
                1,
                self.job_config.action_dim,
                1,
                self.action_per_frame,
                1,
            )
            if tuple(action_cond.shape) != expected_action_cond:
                raise ValueError(
                    f"Expected action condition {expected_action_cond}, got "
                    f"{tuple(action_cond.shape)}"
                )
            action_cond = action_cond.to(device=self.device, dtype=self.dtype)

        for step_id, timestep in enumerate(tqdm(action_timesteps, desc="action")):
            last_step = step_id == len(action_timesteps) - 1
            input_dict = self._prepare_latent_input(
                latent_model_input=None,
                vggt_model_input=None,
                action_model_input=actions,
                action_t=timestep,
                action_cond=action_cond,
                frame_st_id=frame_st_id,
            )
            action_flow = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )

            if not last_step:
                action_flow = rearrange(
                    action_flow,
                    "b (f n) c -> b c f n 1",
                    f=frame_chunk_size,
                )
                if self.job_config.action_guidance_scale > 1:
                    action_flow = action_flow[1:] + (
                        self.job_config.action_guidance_scale
                        * (action_flow[:1] - action_flow[1:])
                    )
                else:
                    action_flow = action_flow[:1]
                actions = self.action_scheduler.step(
                    action_flow,
                    timestep,
                    actions,
                    return_dict=False,
                )

            if action_cond is not None:
                actions[:, :, :1] = action_cond

        actions[:, ~self.action_mask] *= 0
        return self.postprocess_action(actions), latents, vggt_latents

    def load_prefix_observations(
        self,
        video_paths: dict[str, str],
        camera_offsets: dict[str, float],
        target_timestamps: np.ndarray,
        timestamp_tolerance: float,
        max_camera_skew: float,
        rotate_clockwise_90: bool = True,
    ) -> tuple[dict, dict]:
        if not math.isfinite(max_camera_skew) or max_camera_skew < 0:
            raise ValueError("max_camera_skew must be finite and non-negative")
        expected_keys = set(self.job_config.obs_cam_keys)
        provided_keys = set(video_paths)
        if provided_keys != expected_keys:
            missing = sorted(expected_keys - provided_keys)
            extra = sorted(provided_keys - expected_keys)
            raise ValueError(
                f"Video keys must exactly match config obs_cam_keys. "
                f"Missing={missing}, extra={extra}, "
                f"expected={self.job_config.obs_cam_keys}"
            )
        unknown_offsets = set(camera_offsets) - expected_keys
        if unknown_offsets:
            raise ValueError(f"Offsets supplied for unknown cameras: {unknown_offsets}")

        frames_by_camera = {}
        canonical_times_by_camera = {}
        pts_by_camera = {}
        for camera_key in self.job_config.obs_cam_keys:
            offset = camera_offsets.get(camera_key, 0.0)
            camera_targets = target_timestamps + offset
            if np.any(camera_targets < 0):
                raise ValueError(
                    f"Camera offset makes a target timestamp negative for "
                    f"{camera_key}: offset={offset}"
                )
            frames, actual_times, pts = sample_video_by_timestamps(
                video_paths[camera_key],
                camera_targets,
                max_time_error=timestamp_tolerance,
            )

            # Rotate every camera view 90 degrees clockwise after timestamp
            # sampling and before constructing the model observation. np.rot90
            # uses counter-clockwise-positive k, so k=-1 means clockwise 90°.
            if rotate_clockwise_90:
                frames = [
                    np.ascontiguousarray(
                        np.rot90(frame, k=-1, axes=(0, 1))
                    )
                    for frame in frames
                ]

            frames_by_camera[camera_key] = frames
            # Convert camera-local times back to the common time axis.
            canonical_times_by_camera[camera_key] = actual_times - offset
            pts_by_camera[camera_key] = pts

        stacked_times = np.stack(
            [canonical_times_by_camera[key] for key in self.job_config.obs_cam_keys],
            axis=0,
        )
        per_frame_skew = stacked_times.max(axis=0) - stacked_times.min(axis=0)
        worst_skew = float(per_frame_skew.max())
        if worst_skew > max_camera_skew:
            worst_id = int(per_frame_skew.argmax())
            raise RuntimeError(
                f"Multi-camera timestamp skew is too large at prefix frame "
                f"{worst_id}: skew={worst_skew:.6f}s, "
                f"allowed={max_camera_skew:.6f}s"
            )

        observations = [
            {
                camera_key: frames_by_camera[camera_key][frame_id]
                for camera_key in self.job_config.obs_cam_keys
            }
            for frame_id in range(len(target_timestamps))
        ]
        metadata = {
            "target_timestamps": target_timestamps.tolist(),
            "actual_timestamps": {
                key: canonical_times_by_camera[key].tolist()
                for key in self.job_config.obs_cam_keys
            },
            "source_pts": {
                key: pts_by_camera[key].tolist()
                for key in self.job_config.obs_cam_keys
            },
            "max_camera_skew": worst_skew,
            "rotate_clockwise_90": rotate_clockwise_90,
        }
        return {"obs": observations}, metadata

    @torch.no_grad()
    def generate_from_video_prefix(
        self,
        prefix_obs: dict,
        prefix_actions: np.ndarray,
        prompt: str,
        num_chunks: int,
        future_latent_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")

        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(prompt)
        prefix_latents, prefix_vggt_latents = self.encode_video_prefix(prefix_obs)
        if prefix_latents is None or prefix_latents.shape[2] < 1:
            raise RuntimeError("Video prefix produced no latent frames")

        logger.info(
            f"Encoded {len(prefix_obs['obs'])} RGB frames into "
            f"{prefix_latents.shape[2]} latent frames: "
            f"VAE shape={tuple(prefix_latents.shape)}, "
            f"VGGT shape={tuple(prefix_vggt_latents.shape)}"
        )

        normalized_prefix_actions = self.preprocess_action(prefix_actions).to(
            device=self.device,
            dtype=self.dtype,
        )
        if normalized_prefix_actions.shape[2] != prefix_latents.shape[2]:
            raise ValueError(
                f"Action/video prefix latent lengths differ: actions="
                f"{normalized_prefix_actions.shape[2]}, "
                f"video={prefix_latents.shape[2]}"
            )

        known_history = prefix_latents[:, :, :-1]
        known_vggt_history = prefix_vggt_latents[:, :, :-1]
        known_action_history = normalized_prefix_actions[:, :, :-1]
        history_len = self.prefill_prefix_cache(
            known_history,
            known_vggt_history,
            known_action_history,
        )
        last_observed_latent = prefix_latents[:, :, -1:]
        last_observed_vggt = prefix_vggt_latents[:, :, -1:]
        last_observed_action = normalized_prefix_actions[:, :, -1:]

        generated_latents = []
        generated_vggt_latents = []
        generated_actions = []
        frame_cursor = history_len

        for chunk_id in range(num_chunks):
            condition = last_observed_latent if chunk_id == 0 else None
            vggt_condition = last_observed_vggt if chunk_id == 0 else None
            action_condition = last_observed_action if chunk_id == 0 else None
            actions, latents, vggt_latents = self.infer_conditioned_chunk(
                frame_st_id=frame_cursor,
                latent_cond=condition,
                vggt_cond=vggt_condition,
                action_cond=action_condition,
            )

            if chunk_id == 0:
                # The first VAE/VGGT/action block is the observed anchor.
                generated_latents.append(latents[:, :, 1:])
                generated_vggt_latents.append(vggt_latents[:, :, 1:])
                generated_actions.append(torch.from_numpy(actions[:, 1:]))
            else:
                generated_latents.append(latents)
                generated_vggt_latents.append(vggt_latents)
                generated_actions.append(torch.from_numpy(actions))

            frame_cursor += latents.shape[2]

        future_latents = torch.cat(generated_latents, dim=2)
        future_vggt_latents = torch.cat(generated_vggt_latents, dim=2)
        future_actions = torch.cat(generated_actions, dim=1)
        if future_latent_frames is not None:
            if future_latent_frames <= 0:
                raise ValueError("future_latent_frames must be positive")
            if future_latents.shape[2] < future_latent_frames:
                raise RuntimeError(
                    f"Generated only {future_latents.shape[2]} future latent "
                    f"frames, but {future_latent_frames} were requested"
                )
            future_latents = future_latents[:, :, :future_latent_frames]
            future_vggt_latents = future_vggt_latents[
                :, :, :future_latent_frames
            ]
            future_actions = future_actions[:, :future_latent_frames]
        all_latents = torch.cat([prefix_latents, future_latents], dim=2)
        all_vggt_latents = torch.cat(
            [prefix_vggt_latents, future_vggt_latents],
            dim=2,
        )
        return (
            all_latents,
            all_vggt_latents,
            future_actions,
            prefix_latents.shape[2],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LingBot-VGGT i2va inference from a PTS-sampled multi-frame "
            "episode prefix in a LeRobot dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-name",
        default="robotwin_i2av",
        choices=VA_CONFIGS,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Override wan22_pretrained_model_name_or_path from the selected "
            "config."
        ),
    )
    parser.add_argument(
        "--transformer-path",
        default=None,
        help=(
            "A standalone Transformer directory or its parent checkpoint "
            "directory. When omitted, use <model-path>/transformer. VAE, "
            "tokenizer, and text encoder still load from --model-path."
        ),
    )
    parser.add_argument(
        "--vggt-model-path",
        default=None,
        help=(
            "Override vggt_pretrained_model_name_or_path from the selected "
            "config (the VGGTOmega .pt checkpoint)."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "LeRobot dataset root containing meta/info.json, "
            "meta/episodes.jsonl, and videos/."
        ),
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="LeRobot episode_index to use as the video prefix source.",
    )
    parser.add_argument(
        "--camera-offset",
        action="append",
        default=[],
        metavar="CAMERA_KEY=SECONDS",
        help=(
            "Optional offset where camera-local target time equals common "
            "target time plus this value; may be negative."
        ),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=15.0,
        help="Spacing of requested timestamps; source video is not resampled.",
    )
    parser.add_argument("--prefix-num-frames", type=int, default=13)
    parser.add_argument("--prefix-start-sec", type=float, default=0.0)
    parser.add_argument(
        "--rotate-clockwise-90",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rotate every sampled camera frame 90 degrees clockwise before "
            "it is passed to the model."
        ),
    )
    parser.add_argument("--timestamp-tolerance", type=float, default=0.03)
    parser.add_argument("--max-camera-skew", type=float, default=0.03)
    parser.add_argument("--vae-temporal-factor", type=int, default=4)
    parser.add_argument(
        "--allow-unaligned-prefix",
        action="store_true",
        help="Allow k not equal to 1 modulo the VAE temporal factor.",
    )
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument(
        "--future-num-frames",
        type=int,
        default=None,
        help=(
            "Generate exactly this many future RGB frames. The value must be "
            "divisible by --vae-temporal-factor; --num-chunks is then derived "
            "automatically and excess latent frames are trimmed."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-root", default="outputs/video_prefix_i2va")
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--output-actions", default=None)
    parser.add_argument("--output-latents", default=None)
    parser.add_argument("--output-metadata", default=None)
    parser.add_argument(
        "--output-trajectory",
        default=None,
        help=(
            "Output path for the automatically generated 3-D end-effector "
            "trajectory plot. Defaults to <save-root>/eef_trajectories_3d.png."
        ),
    )
    parser.add_argument(
        "--plot-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot input, full GT, and predicted trajectories after inference.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="MP4 playback metadata only; defaults to --sample-fps.",
    )
    parser.add_argument(
        "--mark-borders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mark reconstructed input frames with a red border and generated "
            "frames with a green border in the output MP4."
        ),
    )
    parser.add_argument(
        "--border-width",
        type=int,
        default=6,
        help="Width in output pixels of the red/green phase border.",
    )
    parser.add_argument(
        "--enable-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config VAE/text-encoder CPU offload setting.",
    )
    parser.add_argument(
        "--vae-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use spatial tiling during VAE decode to reduce peak memory.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("separate", "in-process", "skip"),
        default="separate",
        help=(
            "Release the sampling model and decode in a fresh subprocess "
            "(recommended), decode in the sampling process, or skip MP4 "
            "rendering."
        ),
    )
    parser.add_argument(
        "--decode-only-latents",
        default=None,
        metavar="PATH",
        help="Load a saved latent .pt file and render it without sampling.",
    )
    return parser


def add_phase_borders(
    frames: Iterable[np.ndarray],
    prefix_num_frames: int,
    border_width: int,
) -> list[np.ndarray]:
    """Add red input borders and green generated-frame borders."""
    frames = list(frames)
    if not 0 <= prefix_num_frames <= len(frames):
        raise ValueError(
            f"prefix_num_frames={prefix_num_frames} is incompatible with "
            f"decoded frame count {len(frames)}"
        )
    if border_width <= 0:
        raise ValueError(f"border_width must be positive, got {border_width}")

    marked_frames = []
    for frame_id, frame in enumerate(frames):
        marked = np.array(frame, copy=True)
        if marked.ndim != 3 or marked.shape[-1] < 3:
            raise ValueError(
                f"Expected an HWC RGB frame, got shape {marked.shape}"
            )
        height, width = marked.shape[:2]
        effective_width = min(border_width, height // 2, width // 2)
        if effective_width < 1:
            raise ValueError(f"Frame is too small for a border: {marked.shape}")

        scale = 255 if np.issubdtype(marked.dtype, np.integer) else 1.0
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
    return marked_frames


def load_lerobot_actions(
    dataset_root: str | os.PathLike[str],
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and absolute actions for one LeRobot episode."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Action-prefix inference requires pyarrow to read LeRobot parquet data"
        ) from exc

    root = Path(dataset_root).expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    episode_chunk = episode_index // chunk_size
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    parquet_path = root / template.format(
        episode_chunk=episode_chunk,
        chunk_index=episode_chunk,
        episode_index=episode_index,
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Cannot build the action prefix because episode parquet is missing: "
            f"{parquet_path}"
        )
    table = pq.read_table(parquet_path, columns=["timestamp", "action"])
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 3:
        raise ValueError(
            f"Expected non-empty LeRobot actions [T,D] with D >= 3, "
            f"got {actions.shape}"
        )
    return timestamps, actions


def make_pose_relative_to_first(pose: np.ndarray) -> np.ndarray:
    """Match the RoboTwin relative-pose transform used by training."""
    if len(pose) == 0:
        return np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected pose [T,7], got {pose.shape}")
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError(
            "RoboTwin action-prefix preparation requires scipy"
        ) from exc

    rotation = Rotation.from_quat(pose[:, 3:7])
    first_rotation = Rotation.from_quat(
        np.repeat(pose[:1, 3:7], len(pose), axis=0)
    )
    relative_translation = pose[:, :3] - pose[:1, :3]
    relative_rotation = (first_rotation.inv() * rotation).as_quat()
    return np.concatenate([relative_translation, relative_rotation], axis=1)


def make_pose_6d_relative_to_first(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()

    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError(
            "RoboTwin action-prefix preparation requires scipy"
        ) from exc

    rot = Rotation.from_euler("xyz", pose[:, 3:6])
    first_rot = Rotation.from_euler(
        "xyz",
        np.tile(pose[:1, 3:6], (pose.shape[0], 1)),
    )

    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_euler = relative_rot.as_euler("xyz")

    relative_pose = np.concatenate(
        [relative_trans, relative_euler],
        axis=1,
    )

    return relative_pose


def build_action_prefix(
    timestamps: np.ndarray,
    actions: np.ndarray,
    target_timestamps: np.ndarray,
    dataset_fps: float,
    sample_fps: float,
    vae_temporal_factor: int,
    action_per_frame: int,
    env_type: str,
    timestamp_tolerance: float,
) -> tuple[np.ndarray, dict]:
    """Pack known actions exactly like the training dataset's causal prefix."""
    ratio = dataset_fps / sample_fps
    source_stride = int(round(ratio))
    if source_stride <= 0 or not math.isclose(
        ratio,
        source_stride,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Action-conditioned prefix requires dataset_fps/sample_fps to "
            f"be an integer; got {dataset_fps}/{sample_fps}={ratio:.8f}"
        )
    expected_action_per_frame = source_stride * vae_temporal_factor
    if action_per_frame != expected_action_per_frame:
        raise ValueError(
            f"Checkpoint action_per_frame={action_per_frame} is incompatible "
            f"with source stride {source_stride} and VAE temporal factor "
            f"{vae_temporal_factor}; expected {expected_action_per_frame}. "
            f"Use sample_fps={dataset_fps * vae_temporal_factor / action_per_frame:g} "
            "to match training."
        )
    if (len(target_timestamps) - 1) % vae_temporal_factor != 0:
        raise ValueError(
            "Action-conditioned prefixes require prefix_num_frames = 1 mod "
            f"{vae_temporal_factor}; got {len(target_timestamps)}"
        )

    input_indices = np.asarray(
        [int(np.abs(timestamps - target).argmin()) for target in target_timestamps],
        dtype=np.int64,
    )
    action_time_errors = np.abs(timestamps[input_indices] - target_timestamps)
    if float(action_time_errors.max()) > timestamp_tolerance:
        worst = int(action_time_errors.argmax())
        raise RuntimeError(
            f"No action is close enough to input timestamp "
            f"{target_timestamps[worst]:.6f}s: selected "
            f"{timestamps[input_indices[worst]]:.6f}s, error "
            f"{action_time_errors[worst]:.6f}s"
        )
    index_steps = np.diff(input_indices)
    if len(index_steps) and not np.all(index_steps == source_stride):
        raise RuntimeError(
            f"Action indices are not uniformly spaced by source stride "
            f"{source_stride}: {input_indices.tolist()}"
        )

    first_index = int(input_indices[0])
    last_index = int(input_indices[-1])
    # Training prepends one zero action block. The remaining F-1 blocks contain
    # every source-rate action from the first observed frame up to, but not
    # including, the final observed frame. No future action is leaked.
    history = np.asarray(actions[first_index:last_index], dtype=np.float64).copy()
    expected_history = (len(target_timestamps) - 1) * source_stride
    if len(history) != expected_history:
        raise RuntimeError(
            f"Expected {expected_history} known source actions, got "
            f"{len(history)} from rows [{first_index}, {last_index})"
        )

    if env_type == "robotwin_tshape":
        if actions.shape[1] < 16:
            raise ValueError(
                f"RoboTwin action history must have at least 16 channels, "
                f"got {actions.shape}"
            )
        if len(history):
            left = make_pose_relative_to_first(history[:, :7])
            right = make_pose_relative_to_first(history[:, 8:15])
            history = np.concatenate(
                [left, history[:, 7:8], right, history[:, 15:16]],
                axis=1,
            )
    elif env_type == "tennis_tshape":
        if actions.shape[1] < 6:
            raise ValueError(
                f"Tennis action history must have at least 6 channels, "
                f"got {actions.shape}"
            )
        if len(history):
            history = make_pose_6d_relative_to_first(history[:, :7])

    zero_anchor = np.zeros(
        (action_per_frame, history.shape[1] if len(history) else actions.shape[1]),
        dtype=np.float64,
    )
    packed_time_major = np.concatenate([zero_anchor, history], axis=0)
    prefix_latent_frames = (
        (len(target_timestamps) - 1) // vae_temporal_factor + 1
    )
    expected_packed = prefix_latent_frames * action_per_frame
    if len(packed_time_major) != expected_packed:
        raise RuntimeError(
            f"Packed action prefix has {len(packed_time_major)} controls; "
            f"expected {expected_packed}"
        )
    packed = packed_time_major.reshape(
        prefix_latent_frames,
        action_per_frame,
        packed_time_major.shape[1],
    ).transpose(2, 0, 1)
    metadata = {
        "input_action_indices": input_indices.tolist(),
        "input_action_timestamps": timestamps[input_indices].tolist(),
        "action_source_stride": source_stride,
        "action_per_frame": action_per_frame,
        "prefix_action_shape": list(packed.shape),
        "relative_pose_base_index": first_index,
        "relative_pose_base_timestamp": float(timestamps[first_index]),
        # Relative actions alone cannot recover the world-frame trajectory.
        # Persist the absolute pose anchor that was part of the inference input
        # so plotting never has to borrow the input segment from the GT curve.
        "relative_pose_base_action": np.asarray(
            actions[first_index], dtype=np.float64
        ).tolist(),
        "known_source_action_range": [first_index, last_index],
        "known_source_action_count": len(history),
    }
    # Training returns the packed action tensor as float32.
    return np.ascontiguousarray(packed, dtype=np.float32), metadata


def flatten_predicted_actions(actions: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert LingBot-VGGT [D,F,H] actions to time-major [F*H,D]."""
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[0] < 3:
        raise ValueError(f"Expected predicted actions [D,F,H], got {actions.shape}")
    return actions.transpose(1, 2, 0).reshape(-1, actions.shape[0])


def set_equal_3d_limits(axis, point_sets: list[np.ndarray]) -> None:
    points = np.concatenate(point_sets, axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 1e-3) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_arm_trajectory(
    axis,
    gt: np.ndarray,
    input_absolute: np.ndarray,
    predicted: np.ndarray,
    arm_name: str,
    color: str,
    marker: str,
) -> None:
    """Draw one arm; color denotes arm and line style denotes GT/prediction."""
    axis.plot(
        *gt.T,
        color=color,
        linewidth=2.2,
        alpha=0.78,
        linestyle="-",
        label=f"{arm_name} GT (full episode)",
    )
    input_and_prediction = np.concatenate([input_absolute, predicted], axis=0)
    axis.plot(
        *input_and_prediction.T,
        color=color,
        linewidth=2.7,
        linestyle="--",
        label=f"{arm_name} input + prediction",
    )
    axis.scatter(
        *input_absolute[-1],
        color=color,
        marker=marker,
        s=42,
        edgecolor="white",
        linewidth=0.8,
        label=f"{arm_name} input endpoint",
    )
    axis.scatter(
        *predicted[-1],
        color=color,
        marker="*",
        s=95,
        edgecolor="white",
        linewidth=0.8,
        label=f"{arm_name} prediction end",
    )


def save_trajectory_plot(
    future_actions: torch.Tensor | np.ndarray,
    prefix_actions: torch.Tensor | np.ndarray,
    dataset_metadata: dict,
    env_type: str,
    output_path: Path,
) -> None:
    """Save a combined 3-D plot immediately after inference completes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            "Trajectory reconstruction requires action_per_frame and "
            "known_source_action_count in action-prefix metadata"
        )
    # The first block is the synthetic causal zero anchor used by training,
    # not a physical robot pose. Everything after it is exactly what was fed
    # to inference, already expressed relative to the saved absolute anchor.
    input_relative = packed_input_actions[
        action_per_frame : action_per_frame + known_action_count
    ]
    if len(input_relative) != known_action_count:
        raise ValueError(
            f"Packed prefix contains {len(input_relative)} physical controls; "
            f"metadata expects {known_action_count}"
        )

    # GT and prediction lengths are intentionally independent. Only their xyz
    # channel layouts must agree semantically. postprocess_action has already
    # de-normalized predictions into the checkpoint's action representation.
    trajectories = []
    if env_type == "robotwin_tshape":
        if gt_actions.shape[1] < 11 or predicted_relative.shape[1] < 11:
            raise ValueError(
                "robotwin_tshape requires dual-arm action layouts with xyz at "
                f"GT/pred channels 0:3 and 8:11; got GT={gt_actions.shape}, "
                f"prediction={predicted_relative.shape}"
            )
        if input_relative.shape[1] < 11:
            raise ValueError(
                f"RoboTwin input actions require at least 11 channels, got "
                f"{input_relative.shape}"
            )
        base_action = np.asarray(
            action_prefix_metadata.get("relative_pose_base_action", []),
            dtype=np.float64,
        )
        if base_action.size < 11:
            raise ValueError(
                "RoboTwin absolute reconstruction requires "
                "relative_pose_base_action in action-prefix metadata"
            )
        # Both dashed segments now come exclusively from inference data:
        # packed relative input actions and de-normalized model predictions.
        # GT is used only for the independent solid reference trajectory.
        trajectories = [
            (
                "Left",
                gt_actions[:, :3],
                base_action[None, :3] + input_relative[:, :3],
                base_action[None, :3] + predicted_relative[:, :3],
                "#0072B2",
                "o",
            ),
            (
                "Right",
                gt_actions[:, 8:11],
                base_action[None, 8:11] + input_relative[:, 8:11],
                base_action[None, 8:11] + predicted_relative[:, 8:11],
                "#D55E00",
                "^",
            ),
        ]
    elif env_type == "tennis_tshape":
        # Tennis uses 6D action: [x, y, z, rx, ry, rz]
        if gt_actions.shape[1] < 6 or predicted_relative.shape[1] < 3:
            raise ValueError(
                "tennis requires 6D action layout with xyz at channels 0:3; "
                f"got GT={gt_actions.shape}, prediction={predicted_relative.shape}"
            )
        if input_relative.shape[1] < 3:
            raise ValueError(
                f"Tennis input actions require at least 3 channels, got "
                f"{input_relative.shape}"
            )
        base_action = np.asarray(
            action_prefix_metadata.get("relative_pose_base_action", []),
            dtype=np.float64,
        )
        if base_action.size < 3:
            raise ValueError(
                "Tennis absolute reconstruction requires "
                "relative_pose_base_action in action-prefix metadata"
            )
        trajectories = [
            (
                "Arm",
                gt_actions[:, :3],
                base_action[None, :3] + input_relative[:, :3],
                base_action[None, :3] + predicted_relative[:, :3],
                "#0072B2",
                "o",
            )
        ]
    elif gt_actions.shape[1] == 14 and predicted_relative.shape[1] >= 11:
        # Franka GT is [left_pose(7), right_pose(7)], while postprocess_action
        # returns [left_pose(7), left_gripper, right_pose(7), right_gripper].
        trajectories = [
            (
                "Left", gt_actions[:, :3], input_relative[:, :3],
                predicted_relative[:, :3], "#0072B2", "o"
            ),
            (
                "Right",
                gt_actions[:, 7:10],
                input_relative[:, 8:11],
                predicted_relative[:, 8:11],
                "#D55E00",
                "^",
            ),
        ]
    elif gt_actions.shape[1] >= 16 and predicted_relative.shape[1] >= 11:
        trajectories = [
            (
                "Left", gt_actions[:, :3], input_relative[:, :3],
                predicted_relative[:, :3], "#0072B2", "o"
            ),
            (
                "Right",
                gt_actions[:, 8:11],
                input_relative[:, 8:11],
                predicted_relative[:, 8:11],
                "#D55E00",
                "^",
            ),
        ]
    else:
        # Single-arm datasets such as LIBERO use a 7-D action. The first three
        # channels are plotted directly in the same action space as GT.
        trajectories = [
            (
                "Arm",
                gt_actions[:, :3],
                input_relative[:, :3],
                predicted_relative[:, :3],
                "#0072B2",
                "o",
            )
        ]

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    for arm_name, gt_xyz, input_xyz, predicted_xyz, color, marker in trajectories:
        plot_arm_trajectory(
            axis,
            gt_xyz,
            input_xyz,
            predicted_xyz,
            arm_name=arm_name,
            color=color,
            marker=marker,
        )
    endpoint_lines = ["Endpoint coordinates (x, y, z)"]
    for arm_name, gt_xyz, _, predicted_xyz, _, _ in trajectories:
        gt_end = gt_xyz[-1]
        prediction_end = predicted_xyz[-1]
        # 计算欧式距离
        euclidean_distance = np.linalg.norm(gt_end - prediction_end)
        endpoint_lines.extend(
            [
                f"{arm_name} GT:   "
                f"({gt_end[0]:.4f}, {gt_end[1]:.4f}, {gt_end[2]:.4f})",
                f"{arm_name} Pred: "
                f"({prediction_end[0]:.4f}, {prediction_end[1]:.4f}, "
                f"{prediction_end[2]:.4f})",
                f"{arm_name} Distance: {euclidean_distance:.4f}",
            ]
        )
    axis.text2D(
        0.98,
        0.98,
        "\n".join(endpoint_lines),
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.grid(True, alpha=0.22)
    set_equal_3d_limits(
        axis,
        [
            points
            for _, gt, input_xyz, predicted, _, _ in trajectories
            for points in (gt, input_xyz, predicted)
        ],
    )
    axis.legend(
        loc="upper left",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#dddddd",
    )
    figure.suptitle(
        f"LingBot-VGGT 3-D trajectory — episode {episode_index}\n"
        f"solid: GT, dashed: input + prediction "
        f"({len(predicted_relative)} control points)",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    logger.info(
        f"Saved 3-D trajectory plot to {output_path} "
        f"(input={len(input_relative)}, GT={len(gt_actions)}, "
        f"prediction={len(predicted_relative)})"
    )


@torch.no_grad()
def decode_saved_latents(args: argparse.Namespace) -> None:
    if args.output_fps is not None and args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")
    if args.border_width <= 0:
        raise ValueError("--border-width must be positive")

    latent_path = Path(args.decode_only_latents).resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent file does not exist: {latent_path}")
    output_video = Path(
        args.output_video
        or Path(args.save_root).resolve() / "prefix_i2va.mp4"
    ).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(latent_path, map_location="cpu", weights_only=True)
    latents = payload["latents"] if isinstance(payload, dict) else payload
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError(
            f"Expected [B,C,F,H,W] tensor in {latent_path}, got "
            f"{type(latents).__name__} {getattr(latents, 'shape', None)}"
        )

    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    model_path = (
        args.model_path
        if args.model_path is not None
        else config.wan22_pretrained_model_name_or_path
    )
    dtype = config.param_dtype
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    vae = load_vae(
        os.path.join(model_path, "vae"),
        torch_dtype=dtype,
        torch_device=device,
    )
    if args.vae_tiling:
        vae.enable_tiling()

    latents = latents.to(device=device, dtype=vae.dtype)
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=vae.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=vae.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    decoded = vae.decode(
        latents / latents_std + latents_mean,
        return_dict=False,
    )[0]
    frames = VideoProcessor(vae_scale_factor=1).postprocess_video(
        decoded,
        output_type="np",
    )[0]
    prefix_num_frames = (
        payload.get("prefix_num_rgb_frames")
        if isinstance(payload, dict)
        else args.prefix_num_frames
    )
    if prefix_num_frames is None:
        if not isinstance(payload, dict) or "prefix_latent_frames" not in payload:
            raise ValueError(
                "Cannot determine the input/generated frame boundary from "
                f"latent file {latent_path}"
            )
        prefix_latent_frames = int(payload["prefix_latent_frames"])
        prefix_num_frames = (
            (prefix_latent_frames - 1) * args.vae_temporal_factor + 1
        )
    if args.mark_borders:
        frames = add_phase_borders(
            frames,
            prefix_num_frames=int(prefix_num_frames),
            border_width=args.border_width,
        )
    output_fps = args.output_fps or args.sample_fps
    export_to_video(frames, str(output_video), fps=output_fps)
    logger.info(
        f"Decoded {len(frames)} RGB frames from {tuple(latents.shape)} "
        f"latents to {output_video} at {output_fps} FPS"
    )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LingBot-VGGT inference requires a CUDA device")
    if args.decode_only_latents is not None:
        decode_saved_latents(args)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "This standalone entry point currently supports one GPU. Launch it "
            "with plain `python -m`, not multi-process torchrun."
        )
    if args.vae_temporal_factor <= 0:
        raise ValueError("--vae-temporal-factor must be positive")
    if args.vae_temporal_factor != 4:
        raise ValueError(
            "LingBot-VGGT concat-mode alignment currently requires "
            "--vae-temporal-factor=4"
        )
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if args.future_num_frames is not None and args.future_num_frames <= 0:
        raise ValueError("--future-num-frames must be positive")
    if args.output_fps is not None and args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")
    if args.border_width <= 0:
        raise ValueError("--border-width must be positive")
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for sampling")
    if (
        (args.prefix_num_frames - 1) % args.vae_temporal_factor != 0
        and not args.allow_unaligned_prefix
    ):
        raise ValueError(
            f"--prefix-num-frames must satisfy k = 1 mod "
            f"{args.vae_temporal_factor}; got {args.prefix_num_frames}. "
            "Choose an aligned k or pass --allow-unaligned-prefix."
        )

    camera_offsets = parse_key_value(
        args.camera_offset,
        float,
        "--camera-offset",
    )
    target_timestamps = make_uniform_timestamps(
        args.prefix_num_frames,
        args.sample_fps,
        args.prefix_start_sec,
    )

    save_root = Path(args.save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    output_video = Path(args.output_video or save_root / "prefix_i2va.mp4").resolve()
    output_actions = Path(
        args.output_actions or save_root / "prefix_i2va_actions.pt"
    ).resolve()
    output_latents = Path(
        args.output_latents or save_root / "prefix_i2va_latents.pt"
    ).resolve()
    output_metadata = Path(
        args.output_metadata or save_root / "prefix_i2va_metadata.json"
    ).resolve()
    output_trajectory = Path(
        args.output_trajectory or save_root / "eef_trajectories_3d.png"
    ).resolve()
    for path in (
        output_video,
        output_actions,
        output_latents,
        output_metadata,
        output_trajectory,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.save_root = str(save_root)
    config.vae_temporal_factor = args.vae_temporal_factor
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.transformer_path is not None:
        transformer_path = Path(args.transformer_path).expanduser().resolve()
        if not transformer_path.is_dir():
            raise FileNotFoundError(
                f"Transformer directory does not exist: {transformer_path}"
            )
        config.transformer_path = str(transformer_path)
    if args.enable_offload is not None:
        config.enable_offload = args.enable_offload

    video_paths, prompt, dataset_metadata = load_lerobot_episode(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        expected_camera_keys=config.obs_cam_keys,
        prompt_time_sec=float(target_timestamps[-1]),
    )
    action_timestamps, episode_actions = load_lerobot_actions(
        dataset_metadata["dataset_root"],
        args.episode_index,
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
    logger.info(
        f"Prepared training-aligned action prefix {prefix_actions.shape}: "
        f"source stride={action_prefix_metadata['action_source_stride']}, "
        f"known controls={action_prefix_metadata['known_source_action_count']}"
    )

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
        if args.future_num_frames % args.vae_temporal_factor != 0:
            raise ValueError(
                "--future-num-frames must be divisible by "
                f"--vae-temporal-factor ({args.vae_temporal_factor}); got "
                f"{args.future_num_frames}"
            )
        future_latent_frames = (
            args.future_num_frames // args.vae_temporal_factor
        )
        # The first conditioned chunk contributes frame_chunk_size - 1 new
        # latent frames; every later chunk contributes frame_chunk_size.
        num_chunks = math.ceil(
            (future_latent_frames + 1) / config.frame_chunk_size
        )
        logger.info(
            f"Requested {args.future_num_frames} future RGB frames -> "
            f"{future_latent_frames} future latent frames -> "
            f"{num_chunks} inference chunks"
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

    prediction_control_count = int(
        future_actions.shape[1] * future_actions.shape[2]
    )
    prediction_start_index = int(
        action_prefix_metadata["known_source_action_range"][1]
    )
    if prediction_start_index >= len(action_timestamps):
        raise RuntimeError(
            f"Prediction starts at action row {prediction_start_index}, but "
            f"episode contains only {len(action_timestamps)} rows"
        )
    prediction_timestamps = (
        float(action_timestamps[prediction_start_index])
        + np.arange(prediction_control_count, dtype=np.float64)
        / dataset_metadata["dataset_fps"]
    )
    action_prefix_metadata.update(
        {
            "prediction_start_action_index": prediction_start_index,
            "prediction_control_count": prediction_control_count,
            "prediction_action_timestamps": prediction_timestamps.tolist(),
        }
    )

    torch.save(
        {
            "actions": future_actions,
            "input_actions": torch.from_numpy(prefix_actions),
            "action_prefix_metadata": action_prefix_metadata,
            "prefix_latent_frames": prefix_latent_frames,
            "sample_fps": args.sample_fps,
        },
        output_actions,
    )
    # Persist both normalized latent streams before VAE decode so an OOM during
    # rendering does not require rerunning the expensive sampling stage.
    all_latents = all_latents.cpu()
    all_vggt_latents = all_vggt_latents.cpu()
    generated_future_latent_frames = int(
        all_latents.shape[2] - prefix_latent_frames
    )
    torch.save(
        {
            "latents": all_latents,
            "vggt_latents": all_vggt_latents,
            "prefix_latent_frames": prefix_latent_frames,
            "prefix_num_rgb_frames": args.prefix_num_frames,
            "future_latent_frames": generated_future_latent_frames,
        },
        output_latents,
    )

    output_fps = args.output_fps or args.sample_fps
    metadata.update(
        {
            "config_name": args.config_name,
            "transformer_path": getattr(config, "transformer_path", None),
            "vggt_model_path": config.vggt_pretrained_model_name_or_path,
            "prompt": prompt,
            "sample_fps": args.sample_fps,
            "prefix_num_rgb_frames": args.prefix_num_frames,
            "prefix_num_latent_frames": int(prefix_latent_frames),
            "num_generated_chunks": num_chunks,
            "future_num_rgb_frames": args.future_num_frames,
            "future_num_latent_frames": generated_future_latent_frames,
            "vae_latent_shape": list(all_latents.shape),
            "vggt_latent_shape": list(all_vggt_latents.shape),
            "output_fps": output_fps,
            "video_paths": video_paths,
            "camera_offsets": camera_offsets,
            "output_video": str(output_video),
            "output_actions": str(output_actions),
            "output_latents": str(output_latents),
            "output_trajectory": (
                str(output_trajectory) if args.plot_trajectory else None
            ),
            "decode_mode": args.decode_mode,
            "mark_borders": args.mark_borders,
            "border_width": args.border_width,
            "lerobot": dataset_metadata,
        }
    )
    output_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Plot before the optional decoder process replacement below. This keeps
    # trajectory generation part of the same inference command in every
    # decode mode, including "skip" and "separate".
    if args.plot_trajectory:
        save_trajectory_plot(
            future_actions=future_actions,
            prefix_actions=prefix_actions,
            dataset_metadata=dataset_metadata,
            env_type=config.env_type,
            output_path=output_trajectory,
        )

    if args.decode_mode == "skip":
        logger.info(f"Saved latents to {output_latents}; MP4 decode skipped")
        return
    if args.decode_mode == "separate":
        logger.info(
            "Releasing sampling-model memory before launching the VAE-only "
            "decoder subprocess"
        )
        decode_args = [
            sys.executable,
            "-m",
            "wan_va.inference_video_prefix",
            "--config-name",
            args.config_name,
            "--model-path",
            str(config.wan22_pretrained_model_name_or_path),
            "--decode-only-latents",
            str(output_latents),
            "--output-video",
            str(output_video),
            "--output-fps",
            str(output_fps),
            "--vae-temporal-factor",
            str(args.vae_temporal_factor),
            "--border-width",
            str(args.border_width),
        ]
        decode_args.append("--vae-tiling" if args.vae_tiling else "--no-vae-tiling")
        decode_args.append("--mark-borders" if args.mark_borders else "--no-mark-borders")

        # A child decoder cannot safely share the sampling model's NPU memory.
        # Drop the entire server (including its VAE and prompt embeddings),
        # force Python reference collection, and return cached device blocks to
        # the allocator before starting and waiting for the decoder process.
        model.transformer.clear_cache(model.cache_name)
        model.streaming_vae.clear_cache()
        if model.streaming_vae_half is not None:
            model.streaming_vae_half.clear_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Starting decoder subprocess: {decode_args}")
        subprocess.run(decode_args, check=True)
        logger.info(f"Saved video to {output_video}")
        logger.info(f"Saved actions to {output_actions}")
        logger.info(f"Saved timestamp metadata to {output_metadata}")
        return

    model.transformer.clear_cache(model.cache_name)
    model.streaming_vae.clear_cache()
    if model.streaming_vae_half is not None:
        model.streaming_vae_half.clear_cache()
        del model.streaming_vae_half
    del model.transformer
    del model.vggt_adapter
    del model.text_encoder
    del model.streaming_vae
    gc.collect()
    torch.cuda.empty_cache()

    if model.enable_offload:
        model.vae = model.vae.to(model.device).to(model.dtype)
    if args.vae_tiling:
        model.vae.enable_tiling()
    decoded_video = model.decode_one_video(
        all_latents.to(model.device),
        "np",
    )[0]
    if args.mark_borders:
        decoded_video = add_phase_borders(
            decoded_video,
            prefix_num_frames=args.prefix_num_frames,
            border_width=args.border_width,
        )
    export_to_video(decoded_video, str(output_video), fps=output_fps)

    logger.info(f"Saved video to {output_video}")
    logger.info(f"Saved actions to {output_actions}")
    logger.info(f"Saved timestamp metadata to {output_metadata}")


def main() -> None:
    init_logger()
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
