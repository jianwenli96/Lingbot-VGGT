# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_tennis_cfg import va_tennis_cfg

va_tennis_i2va_cfg = EasyDict(__name__='Config: VA tennis i2va')
va_tennis_i2va_cfg.update(va_tennis_cfg)

va_tennis_i2va_cfg.transformer_path = '/path/to/finetune/transformer'
va_tennis_i2va_cfg.input_img_path = 'example/piper-overfit'
va_tennis_i2va_cfg.num_chunks_to_infer = 10
va_tennis_i2va_cfg.prompt = "Catch the green tennis ball"
va_tennis_i2va_cfg.infer_mode = 'i2va'
