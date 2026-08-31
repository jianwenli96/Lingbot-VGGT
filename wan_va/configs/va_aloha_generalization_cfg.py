# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_aloha_generalization_cfg = EasyDict(__name__='Config: VA generalization')
va_aloha_generalization_cfg.update(va_shared_cfg)

va_aloha_generalization_cfg.wan22_pretrained_model_name_or_path = '/efs-gy1/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base'

va_aloha_generalization_cfg.attn_window = 30
va_aloha_generalization_cfg.frame_chunk_size = 2
va_aloha_generalization_cfg.env_type = 'aloha_tshape'

va_aloha_generalization_cfg.height = 256
va_aloha_generalization_cfg.width = 320
va_aloha_generalization_cfg.action_dim = 30
va_aloha_generalization_cfg.action_per_frame = 24 # 16
va_aloha_generalization_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_aloha_generalization_cfg.guidance_scale = 5
va_aloha_generalization_cfg.vggt_guidance_scale = 5
va_aloha_generalization_cfg.action_guidance_scale = 1

va_aloha_generalization_cfg.num_inference_steps = 3 # 25
va_aloha_generalization_cfg.vggt_num_inference_steps = 3 # 25
va_aloha_generalization_cfg.video_exec_step = -1
va_aloha_generalization_cfg.action_num_inference_steps = 10 # 50

va_aloha_generalization_cfg.snr_shift = 5.0
va_aloha_generalization_cfg.vggt_snr_shift = 5.0
va_aloha_generalization_cfg.action_snr_shift = 1.0

va_aloha_generalization_cfg.used_action_channel_ids = \
    list(range(0, 7)) + list(range(7, 14))  + \
    list(range(14, 20)) + list(range(21, 27)) + \
    list(range(28, 29)) + list(range(29, 30))

inverse_used_action_channel_ids = [
    len(va_aloha_generalization_cfg.used_action_channel_ids)
] * va_aloha_generalization_cfg.action_dim
for i, j in enumerate(va_aloha_generalization_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_aloha_generalization_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_aloha_generalization_cfg.action_norm_method = 'quantiles'
va_aloha_generalization_cfg.norm_stat = {
    "q01": [
        -2.2401e+01, -3.1585e+02,  3.5548e+01,
        -4.2655e-01, -9.9786e-01, -4.1843e-01, -6.7376e-01,
         1.5433e+01, -1.4080e+02,  9.4240e+01,
        -8.9584e-01, -9.9928e-01, -5.4577e-01, -6.8716e-01,
        -1.2203e+00,  6.7683e-03, -2.3217e+00,
        -1.7452e+00, -1.1845e+00, -2.0770e+00,  0,
        -4.8179e-01,  6.2450e-03, -2.1160e+00,
        -1.7855e+00, -1.2148e+00, -2.1158e+00,  0,
         0, 0],
    "q99": [
         4.8799e+02,  1.8949e+02,  5.7112e+02,
         5.8027e-01,  9.9751e-01,  4.6846e-01,  8.0343e-01,
         4.8425e+02,  3.7353e+02,  5.5808e+02,
         4.8264e-01,  9.9871e-01,  5.1068e-01,  7.5782e-01,
         5.6142e-01,  2.5641e+00, -2.9901e-01,
         1.7382e+00,  1.2648e+00,  1.7831e+00,  0,
         1.2083e+00,  2.4977e+00, -2.4279e-01,
         1.7399e+00,  1.2432e+00,  1.9818e+00,  0,
         9.9700e-02,  9.9500e-02]
}

# VGGTOmega config. Keep these values aligned with the training config and
va_aloha_generalization_cfg.vggt_pretrained_model_name_or_path = "/efs-gy1/lijianwen/Pretrained_models/VGGT/VGGT-Omega/vggt_omega_1b_512.pt"
va_aloha_generalization_cfg.vggt_image_size = 512
va_aloha_generalization_cfg.vggt_latent_frame_mode = "concat"
va_aloha_generalization_cfg.vggt_latent_dimension = 2048
va_aloha_generalization_cfg.vggt_latent_height = 12
va_aloha_generalization_cfg.vggt_latent_width = 17
