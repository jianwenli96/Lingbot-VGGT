# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_tennis_cfg = EasyDict(__name__='Config: VA tennis')
va_tennis_cfg.update(va_shared_cfg)

va_tennis_cfg.wan22_pretrained_model_name_or_path = '/efs-mi-east4-2/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base'

va_tennis_cfg.attn_window = 16
va_tennis_cfg.frame_chunk_size = 2
va_tennis_cfg.env_type = 'tennis_tshape'

va_tennis_cfg.height = 256
va_tennis_cfg.width = 320
va_tennis_cfg.action_dim = 30
va_tennis_cfg.action_per_frame = 8
va_tennis_cfg.obs_cam_keys = [
    'observation.images.upper', 'observation.images.left',
    'observation.images.right'
]
va_tennis_cfg.guidance_scale = 5
va_tennis_cfg.vggt_guidance_scale = 5
va_tennis_cfg.action_guidance_scale = 1

va_tennis_cfg.num_inference_steps = 3
va_tennis_cfg.vggt_num_inference_steps = 3
va_tennis_cfg.video_exec_step = -1
va_tennis_cfg.action_num_inference_steps = 5

va_tennis_cfg.snr_shift = 5.0
va_tennis_cfg.vggt_snr_shift = 5.0
va_tennis_cfg.action_snr_shift = 1.0

va_tennis_cfg.used_action_channel_ids = list(range(0, 6))

inverse_used_action_channel_ids = [
    len(va_tennis_cfg.used_action_channel_ids)
] * va_tennis_cfg.action_dim
for i, j in enumerate(va_tennis_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_tennis_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_tennis_cfg.action_norm_method = 'quantiles'
va_tennis_cfg.norm_stat = {
    "q01": [
        -0.6256, -0.2168,  0.0000, -0.0076, -0.0187, -0.0153
    ] + [0.] * 24,
    "q99": [
        0.6297, 0.7398, 0.0337, 0.0784, 0.0186, 0.0158
    ] + [0.] * 24,
}

# VGGTOmega config. Keep these values aligned with the training config and
va_tennis_cfg.vggt_pretrained_model_name_or_path = "/efs-mi-east4-2/lijianwen/Pretrained_models/VGGT/VGGT-Omega/vggt_omega_1b_512.pt"
va_tennis_cfg.vggt_image_size = 512
va_tennis_cfg.vggt_latent_frame_mode = "concat"
va_tennis_cfg.vggt_latent_dimension = 2048
va_tennis_cfg.vggt_latent_height = 12
va_tennis_cfg.vggt_latent_width = 17
