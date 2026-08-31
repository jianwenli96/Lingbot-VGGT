# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception as e:
    pass

import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from modules.vggt_adapter import VGGTAdapter
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
    merge_vggt_latents,
)

# for embedding offload
import hashlib

def md5_hash(s):
    return hashlib.md5(s.encode()).hexdigest()

class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.vggt_scheduler = FlowMatchScheduler(shift=self.job_config.vggt_snr_shift,
                                                 sigma_min=0.0,
                                                 extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.vggt_scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        transformer_load_path = job_config.transformer_path if hasattr(job_config, 'transformer_path') \
            and os.path.exists(job_config.transformer_path) else job_config.wan22_pretrained_model_name_or_path
        transformer_load_path = os.path.join(transformer_load_path, 'transformer')
        logger.info(f"Loading transformer from {transformer_load_path} ...")
        self.transformer = load_transformer(
            transformer_load_path,
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch"
        )
        self.transformer.eval().requires_grad_(False).to(self.dtype).to(self.device)

        self.env_type = job_config.env_type
        self.streaming_vae_half = None

        self.vggt_latent_dimension = getattr(job_config, "vggt_latent_dimension", 2048)
        self.vggt_latent_width = getattr(job_config, "vggt_latent_width", 17)
        self.vggt_latent_height = getattr(job_config, "vggt_latent_height", 4 * len(job_config.obs_cam_keys))
        vggt_adapter_config = {
            "default_image_size": getattr(job_config, "vggt_image_size", [512, 512]),
            "latent_frame_mode": getattr(job_config, "vggt_latent_frame_mode", "concat"),
            "latent_dimension": self.vggt_latent_dimension
        }
        if isinstance(vggt_adapter_config["default_image_size"], int):
            vggt_adapter_config["default_image_size"] = [vggt_adapter_config["default_image_size"]] * 2
        self.vggt_adapter = VGGTAdapter.from_pretrained(
            vggt_adapter_config=vggt_adapter_config,
            vggt_pretrained_path=job_config.vggt_pretrained_model_name_or_path,
            device='cpu',
            torch_dtype=self.dtype,
        )
        self.vggt_adapter.eval()

        if 'tshape' in self.env_type:
            self.streaming_vae_half = WanVAEStreamingWrapper(self.vae)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              vggt_model_input,
                              action_model_input,
                              latent_t=0,
                              vggt_t=0,
                              action_t=0,
                              latent_cond=None,
                              vggt_cond=None,
                              action_cond=None,
                              frame_st_id=0):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()

        patch_size = self.job_config.patch_size
        vggt_patch_size = self.job_config.vggt_patch_size

        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if vggt_model_input is not None:
            input_dict['vggt_res_lst'] = {
                'noisy_latents':
                vggt_model_input,
                'timesteps':
                torch.ones([vggt_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * vggt_t,
                'grid_id':
                get_mesh_id(vggt_model_input.shape[-3] // vggt_patch_size[0],
                            vggt_model_input.shape[-2] // vggt_patch_size[1],
                            vggt_model_input.shape[-1] // vggt_patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if vggt_cond is not None:
                input_dict['vggt_res_lst'][
                    'noisy_latents'][:, :, 0:1] = vggt_cond[:, :, 0:1]
                input_dict['vggt_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        torch.cuda.empty_cache()
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if 'tshape' in self.env_type:
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if 'tshape' in self.env_type:
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)

        torch.cuda.empty_cache()
        return video_latent.to(self.device)

    def _encode_vggt_obs(self, obs, target_num_frames):
        torch.cuda.empty_cache()
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        view_frame_tensors = []
        for key in self.job_config.obs_cam_keys:
            frames = torch.from_numpy(
                np.stack([each[key] for each in images])).float().permute(3, 0, 1, 2)
            view_frame_tensors.append(frames)
        resized_target_size = (
            max(int(frames.shape[2]) for frames in view_frame_tensors),
            max(int(frames.shape[3]) for frames in view_frame_tensors),
        )
        resized_view_frame_tensors = []
        for frames in view_frame_tensors:
            frames = (frames / 255.0).clamp(0.0, 1.0)
            frames = rearrange(frames, "c f h w -> f c h w")
            frames = F.interpolate(frames,
                                   size=resized_target_size,
                                   mode="bilinear",
                                   align_corners=False)
            resized_view_frame_tensors.append(frames)
        vggt_images = torch.stack(resized_view_frame_tensors, dim=2).to(
            device=self.device, dtype=self.dtype)
        self.vggt_adapter.to(self.device)  # 移到 GPU
        encoded = self.vggt_adapter.encode(vggt_images,
                                           return_dict=True,
                                           device=self.device.type,
                                           torch_dtype=self.dtype)
        self.vggt_adapter.to('cpu')  # 移回 CPU
        torch.cuda.empty_cache()
        # The images length would be 1, 4, or 8, thus set pad_first to False for 4 and 8
        vggt_latents = merge_vggt_latents(
            encoded["latents"].detach(),
            target_num_frames=target_num_frames,
            frame_mode=self.job_config.vggt_latent_frame_mode,
            pad_first=True if len(images) == 1 else False
        )

        # Normalize vggt latents first
        vggt_latents = self.transformer.vggt_pre_norm(vggt_latents)
        vggt_latents = rearrange(vggt_latents, "f h w c -> 1 c f h w")

        torch.cuda.empty_cache()
        return vggt_latents.to(self.device, self.dtype)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.vggt_guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self.init_vggt = None
        self.pred_latent_lst = []  # 累积 latents 用于最后解码
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if 'tshape' in self.env_type:
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        vggt_patch_size = self.job_config.vggt_patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        vggt_token_per_chunk = (self.job_config.frame_chunk_size *
                                self.vggt_latent_height * self.vggt_latent_width) // (
                                    vggt_patch_size[0] * vggt_patch_size[1] *
                                    vggt_patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            vggt_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            prompt_hash = md5_hash(prompt)
            # search md5 hash
            if os.path.exists(os.path.join("prompts", prompt_hash)):
                self.prompt_embeds = torch.load(
                    os.path.join("prompts", prompt_hash),
                    map_location="cpu"
                ).to(self.device)
                self.negative_prompt_embeds = torch.load(
                    os.path.join("prompts", "negative_prompt_embeds"),
                    map_location="cpu"
                ).to(self.device)
            else:
                self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=self.use_cfg,
                    num_videos_per_prompt=1,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    max_sequence_length=512,
                    device=self.device,
                    dtype=self.dtype,
                )

        self.exp_name = f"{prompt_hash}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            init_vggt = self._encode_vggt_obs(obs, init_latent.shape[2])
            self.init_latent = init_latent
            self.init_vggt = init_vggt

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        vggt_latents = torch.randn(1,
                                   self.vggt_latent_dimension,
                                   frame_chunk_size,
                                   self.vggt_latent_height,
                                   self.vggt_latent_width,
                                   device=self.device,
                                   dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        vggt_inference_step = self.job_config.vggt_num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.vggt_scheduler.set_timesteps(vggt_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        vggt_timesteps = self.vggt_scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)
        vggt_timesteps = F.pad(vggt_timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]
            vggt_timesteps = vggt_timesteps[:video_step]

        if len(timesteps) != len(vggt_timesteps):
            raise ValueError(
                "Joint video/VGGT diffusion requires equal timestep counts, "
                f"got {len(timesteps)} and {len(vggt_timesteps)}."
            )

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Joint Video/VGGT Generation Loop
            joint_timesteps = zip(timesteps, vggt_timesteps)
            for i, (t, vggt_t) in enumerate(tqdm(joint_timesteps, total=len(timesteps))):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                vggt_cond = init_vggt[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    vggt_latents,
                    None,
                    t,
                    vggt_t,
                    0,
                    latent_cond,
                    vggt_cond,
                    None,
                    frame_st_id=frame_st_id)

                latent_input = self._repeat_input_for_cfg(input_dict['latent_res_lst'])
                vggt_input = self._repeat_input_for_cfg(input_dict['vggt_res_lst'])
                video_noise_pred, vggt_noise_pred = self.transformer(
                    latent_input,
                    vggt_dict=vggt_input,
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)
                    vggt_noise_pred = data_seq_to_patch(
                        self.job_config.vggt_patch_size, vggt_noise_pred,
                        frame_chunk_size, self.vggt_latent_height,
                        self.vggt_latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.vggt_guidance_scale > 1:
                        vggt_noise_pred = vggt_noise_pred[1:] + self.job_config.vggt_guidance_scale * (vggt_noise_pred[:1] - vggt_noise_pred[1:])
                    else:
                        vggt_noise_pred = vggt_noise_pred[:1]
                    vggt_latents = self.vggt_scheduler.step(vggt_noise_pred,
                                                            vggt_t,
                                                            vggt_latents,
                                                            return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                vggt_latents[:, :, 0:1] = vggt_cond if frame_st_id == 0 else vggt_latents[:, :, 0:1]

            # 2. Action Generation Loop
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    None,
                    actions,
                    t,
                    t,
                    t,
                    None,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        # 累积 latents 用于最后解码
        self.pred_latent_lst.append(latents.detach().cpu())

        actions = self.postprocess_action(actions)

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(vggt_latents, os.path.join(self.exp_save_root, f'vggt_latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        torch.cuda.empty_cache()
        return actions, latents, vggt_latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        vggt_model_input = self._encode_vggt_obs(obs, latent_model_input.shape[2])
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent
            vggt_model_input = torch.cat(
                [self.init_vggt, vggt_model_input],
                dim=2) if vggt_model_input is not None else self.init_vggt
        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {vggt_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                vggt_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            latent_input = self._repeat_input_for_cfg(input_dict['latent_res_lst'])
            vggt_input = self._repeat_input_for_cfg(input_dict['vggt_res_lst'])
            self.transformer(latent_input,
                             vggt_dict=vggt_input,
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)
        decode_latents = obs.get('decode_latents', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        elif decode_latents:
            logger.info(f"################# Decode Latents #################")
            video_path = self.decode_and_save_latents()
            return dict(video_path=video_path)
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video

    @torch.no_grad()
    def decode_and_save_latents(self):
        """
        解码累积的 latents 并保存为视频。
        复用 generate() 的模式。
        """
        if not self.pred_latent_lst:
            logger.warning("No latents accumulated to decode")
            return None

        logger.info(f"Decoding {len(self.pred_latent_lst)} accumulated latents...")

        # 拼接所有 latents
        pred_latent = torch.cat(self.pred_latent_lst, dim=2)
        logger.info(f"Concatenated latent shape: {pred_latent.shape}")

        # 初始化 video_processor（如果需要）
        if not hasattr(self, 'video_processor') or self.video_processor is None:
            from diffusers.video_processor import VideoProcessor
            self.video_processor = VideoProcessor(vae_scale_factor=1)

        # 将 VAE 移到 GPU
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        # 解码
        decoded_video = self.decode_one_video(pred_latent.to(self.device), 'np')[0]
        logger.info(f"Decoded video shape: {decoded_video.shape}")

        # 保存视频
        output_path = os.path.join(self.exp_save_root, 'predicted_video.mp4')
        export_to_video(decoded_video, output_path, fps=10)
        logger.info(f"Predicted video saved to {output_path}")

        # # VAE 移回 CPU
        if self.enable_offload:
            self.vae = self.vae.to('cpu')

        torch.cuda.empty_cache()
        return output_path

    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_vggt_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents, vggt_latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_vggt_lst.append(vggt_latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_vggt = torch.cat(pred_vggt_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
