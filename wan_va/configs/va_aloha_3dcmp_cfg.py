# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_aloha_3dcmp_cfg = EasyDict(__name__='Config: VA robotwin')
va_aloha_3dcmp_cfg.update(va_shared_cfg)

va_aloha_3dcmp_cfg.wan22_pretrained_model_name_or_path = '/efs-gy1/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base'

va_aloha_3dcmp_cfg.attn_window = 30
va_aloha_3dcmp_cfg.frame_chunk_size = 2
va_aloha_3dcmp_cfg.env_type = 'aloha_tshape'

va_aloha_3dcmp_cfg.height = 256
va_aloha_3dcmp_cfg.width = 320
va_aloha_3dcmp_cfg.action_dim = 30
va_aloha_3dcmp_cfg.action_per_frame = 12 # 16
va_aloha_3dcmp_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_aloha_3dcmp_cfg.guidance_scale = 5
va_aloha_3dcmp_cfg.vggt_guidance_scale = 5
va_aloha_3dcmp_cfg.action_guidance_scale = 1

va_aloha_3dcmp_cfg.num_inference_steps = 3 # 25
va_aloha_3dcmp_cfg.vggt_num_inference_steps = 3 # 25
va_aloha_3dcmp_cfg.video_exec_step = -1
va_aloha_3dcmp_cfg.action_num_inference_steps = 10 # 50

va_aloha_3dcmp_cfg.snr_shift = 5.0
va_aloha_3dcmp_cfg.vggt_snr_shift = 5.0
va_aloha_3dcmp_cfg.action_snr_shift = 1.0

va_aloha_3dcmp_cfg.used_action_channel_ids = \
    list(range(0, 7)) + list(range(7, 14))  + \
    list(range(14, 20)) + list(range(21, 27)) + \
    list(range(28, 29)) + list(range(29, 30))

inverse_used_action_channel_ids = [
    len(va_aloha_3dcmp_cfg.used_action_channel_ids)
] * va_aloha_3dcmp_cfg.action_dim
for i, j in enumerate(va_aloha_3dcmp_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_aloha_3dcmp_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_aloha_3dcmp_cfg.action_norm_method = 'quantiles'
va_aloha_3dcmp_cfg.norm_stat = {
    "q01": [
         7.5801e+01, -3.0468e+02,  1.2295e+02,
        -4.2067e-02, -7.8078e-01, -1.7006e-01, -6.4037e-01,
         5.7825e+01, -3.6821e+01,  2.5075e+02,
        -4.0566e-01, -9.9744e-01, -2.7361e-01, -6.0712e-01,
        -6.0421e-01,  2.8999e-01, -1.5773e+00,
        -1.2113e-01, -9.6753e-01, -3.5434e-01,  0,
        -1.2336e-01,  2.1831e-01, -2.0167e+00,
        -5.4478e-01, -5.2733e-02, -7.5088e-01,  0,
         0,  0
    ],
    "q99": [
         4.6041e+02,  7.2614e+00,  4.0368e+02,
         5.3347e-01,  8.3967e-01,  8.2274e-02,  7.8521e-01,
         3.5154e+02,  1.6308e+02,  5.6607e+02,
         3.4989e-01,  9.9715e-01,  2.4807e-01,  7.9752e-01,
         3.9633e-02,  2.2753e+00, -1.6814e-01,
         7.7171e-01,  2.3956e-01,  1.8283e-01,  0,
         6.3643e-01,  1.7597e+00, -3.4030e-01,
         8.9727e-01,  1.2279e+00,  2.3820e-01,  0,
         8.7600e-02,  9.9500e-02
    ]
}

# VGGTOmega config. Keep these values aligned with the training config and
va_aloha_3dcmp_cfg.vggt_pretrained_model_name_or_path = "/efs-gy1/lijianwen/Pretrained_models/VGGT/VGGT-Omega/vggt_omega_1b_512.pt"
va_aloha_3dcmp_cfg.vggt_image_size = 512
va_aloha_3dcmp_cfg.vggt_latent_frame_mode = "concat"
va_aloha_3dcmp_cfg.vggt_latent_dimension = 2048
va_aloha_3dcmp_cfg.vggt_latent_height = 12
va_aloha_3dcmp_cfg.vggt_latent_width = 17
