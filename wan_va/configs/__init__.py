# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_server import va_robotwin_server_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_aloha_3dcmp_train_cfg import va_aloha_3dcmp_train_cfg
from .va_aloha_3dcmp_server import va_aloha_3dcmp_server_cfg
from .va_aloha_generalization_train_cfg import va_aloha_generalization_train_cfg
from .va_aloha_generalization_server import va_aloha_generalization_server_cfg
from .va_tennis_cfg import va_tennis_cfg
from .va_tennis_i2va import va_tennis_i2va_cfg
from .va_tennis_train_cfg import va_tennis_train_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'robotwin_server': va_robotwin_server_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'aloha_3dcmp_train': va_aloha_3dcmp_train_cfg,
    'aloha_3dcmp_server': va_aloha_3dcmp_server_cfg,
    'aloha_generalization_train': va_aloha_generalization_train_cfg,
    'aloha_generalization_server': va_aloha_generalization_server_cfg,
    'tennis': va_tennis_cfg,
    'tennis_i2va': va_tennis_i2va_cfg,
    'tennis_train': va_tennis_train_cfg
}