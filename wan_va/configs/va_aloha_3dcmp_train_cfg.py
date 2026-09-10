# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_aloha_3dcmp_cfg import va_aloha_3dcmp_cfg
import os

va_aloha_3dcmp_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_aloha_3dcmp_train_cfg.update(va_aloha_3dcmp_cfg)

va_aloha_3dcmp_train_cfg.resume_from = '/efs-mi-east4-2/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base'

va_aloha_3dcmp_train_cfg.save_root = './train_out/aloha_poc_3dcmp_tasks'
va_aloha_3dcmp_train_cfg.dataset_path = '/efs-mi-east4-2/lijianwen/Datasets/Aloha/poc_3d_cmp'
va_aloha_3dcmp_train_cfg.empty_emb_path = os.path.join(va_aloha_3dcmp_train_cfg.dataset_path, 'empty_emb.pt')
va_aloha_3dcmp_train_cfg.enable_wandb = True
va_aloha_3dcmp_train_cfg.load_worker = 16
va_aloha_3dcmp_train_cfg.save_interval = 500
va_aloha_3dcmp_train_cfg.gc_interval = 50
va_aloha_3dcmp_train_cfg.cfg_prob = 0.1
va_aloha_3dcmp_train_cfg.random_frame_cut = False
va_aloha_3dcmp_train_cfg.min_frames = 16
va_aloha_3dcmp_train_cfg.max_frames = 24

# Training parameters
va_aloha_3dcmp_train_cfg.learning_rate = 1e-4
va_aloha_3dcmp_train_cfg.beta1 = 0.9
va_aloha_3dcmp_train_cfg.beta2 = 0.95
va_aloha_3dcmp_train_cfg.weight_decay = 0.1
va_aloha_3dcmp_train_cfg.warmup_steps = 10
va_aloha_3dcmp_train_cfg.batch_size = 1
va_aloha_3dcmp_train_cfg.gradient_accumulation_steps = 1
va_aloha_3dcmp_train_cfg.num_steps = 3000
