# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_tennis_cfg import va_tennis_cfg
import os

va_tennis_train_cfg = EasyDict(__name__='Config: VA tennis train')
va_tennis_train_cfg.update(va_tennis_cfg)

va_tennis_train_cfg.resume_from = '/efs-mi-east4-2/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base'

va_tennis_train_cfg.save_root = './train_out/va_tennis_tasks'
va_tennis_train_cfg.dataset_path = '/efs-mi-east4-2/lijianwen/Datasets/Tennis/tennis_train'
va_tennis_train_cfg.empty_emb_path = os.path.join(va_tennis_train_cfg.dataset_path, 'empty_emb.pt')
va_tennis_train_cfg.enable_wandb = True
va_tennis_train_cfg.load_worker = 16
va_tennis_train_cfg.save_interval = 500
va_tennis_train_cfg.gc_interval = 50
va_tennis_train_cfg.cfg_prob = 0.1
va_tennis_train_cfg.random_frame_cut = False
va_tennis_train_cfg.min_frames = 16
va_tennis_train_cfg.max_frames = 24

# Training parameters
va_tennis_train_cfg.learning_rate = 1e-4
va_tennis_train_cfg.beta1 = 0.9
va_tennis_train_cfg.beta2 = 0.95
va_tennis_train_cfg.weight_decay = 0.1
va_tennis_train_cfg.warmup_steps = 10
va_tennis_train_cfg.batch_size = 1
va_tennis_train_cfg.gradient_accumulation_steps = 2
va_tennis_train_cfg.num_steps = 5000
