"""Minimal single-environment tennis scene — nothing attached.

Scene identical to evaluate_tennis_0804_new.py (white room, table, Franka,
tennis ball, 3 base cameras, lights). Ball is thrown randomly each episode.
No VLA, no dataset collection, no catch logic, no markers.

Usage:
    source /home/jdhc/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
    cd /home/jdhc/IsaacLab && ./isaaclab.sh -p \\
        /home/jdhc/z00821918/tennis_robot/dynamicvla/dynamic-vla/simulations/evaluate_tennis_0813_new.py \\
        --object_dir /home/jdhc/z00821918/tennis_robot/dynamicvla/objects \\
        --randomize_throw --viz kit --enable_cameras
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import MISSING

import numpy as np
import scipy.spatial.transform
import torch

# ── Isaac Sim AppLauncher (must run before isaac* imports) ────────────────
from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser(description="Minimal single-env tennis scene.")
    parser.add_argument("--object_dir", default="../objects", help="Object asset root.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--render_dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--randomize_throw", action="store_true")
    parser.add_argument("--camera_width", type=int, default=480)
    parser.add_argument("--camera_height", type=int, default=360)
    parser.add_argument("--room_length", type=float, default=10.5)
    parser.add_argument("--room_width", type=float, default=6.0)
    parser.add_argument("--room_height", type=float, default=3.0)
    parser.add_argument("--table_height", type=float, default=0.40)
    parser.add_argument("--table_length", type=float, default=1.20)
    parser.add_argument("--table_width", type=float, default=0.80)
    parser.add_argument("--robot_base_z_offset", type=float, default=0.005)
    parser.add_argument("--ball_radius", type=float, default=0.0325)
    parser.add_argument("--ball_mass", type=float, default=0.058)
    parser.add_argument("--ball_start_distance", type=float, default=4.0)
    parser.add_argument("--ball_start_height", type=float, default=1.0)
    parser.add_argument("--clean_close", action="store_true")
    # VLA frame publishing via ZeroMQ (model runs separately in dy-vla env)
    parser.add_argument("--zmq_publish", action="store_true",
                        help="Publish 9 temporal RGB frames + EE pose over ZeroMQ PUB socket.")
    parser.add_argument("--zmq_host", type=str, default="127.0.0.1")
    parser.add_argument("--zmq_port", type=int, default=5563)
    parser.add_argument("--control_arm", action="store_true",
                        help="Receive VLA actions over ZeroMQ and drive the arm via IK.")
    parser.add_argument("--zmq_act_port", type=int, default=5564,
                        help="ZeroMQ SUB port for receiving action chunks.")
    parser.add_argument("--inference-delay", "--inference_delay", type=float, default=0.0,
                        help="Minimum wall-clock hold after publishing an observation, "
                             "in seconds. Default: 0 (wait only for the model).")
    parser.add_argument("--vla_match_dataset_video", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="对发布的观测帧做 AV1/yuv420p 有损往返, 与训练视频管线对齐 "
                        "(默认开启; 用 --no-vla_match_dataset_video 关闭).")
    parser.add_argument("--print_ee", action="store_true",
                        help="Print EE (panda_hand) world pose every step.")
    parser.add_argument("--print_ee_every", type=int, default=1,
                        help="Print EE pose every N steps when --print_ee is set.")
    parser.add_argument("--throw_interval", type=float, default=5.0,
                        help="Wait time (seconds) between ball throws.")
    parser.add_argument("--save_video", "--save", action="store_true",
                        help="Save stitched 3-camera videos per episode (works with --viz none headless).")
    parser.add_argument("--video_dir", default="../output/vla_tennis_0813",
                        help="Output directory for saved videos.")
    parser.add_argument("--record_depth", action="store_true",
                        help="在保存的视频中加入深度图面板 (每相机一行: RGB+深度+掩码).")
    parser.add_argument("--record_segmentation", action="store_true",
                        help="在保存的视频中加入实例分割掩码面板.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
if args_cli.inference_delay < 0.0:
    raise SystemExit("[ERROR] --inference-delay must be non-negative")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports (after AppLauncher) ─────────────────────────────────
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms  # noqa: E402

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, os.path.join(PROJECT_HOME, "simulations"))
sys.path.insert(0, PROJECT_HOME)

# Match the 30 Hz source-frame indices used by openloop inference: 9 RGB
# observations sampled at 15 Hz.  Logical frame 0 is the first rendered frame
# after the episode starts.
VLA_OBSERVATION_STEPS = tuple(range(0, 17, 2))


# ── Helpers ───────────────────────────────────────────────────────────────
def _abs_path(path):
    return path if os.path.isabs(path) else os.path.abspath(path)


def _preview(color, roughness=0.65):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=0.0)


def _static_box(prim_path, size, pos, color=(1.0, 1.0, 1.0)):
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.25,
            ),
            visual_material=_preview(color),
        ),
    )


def _wxyz_from_xyzw(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _look_at_opengl_quat(camera_pos, target_pos):
    """Return wxyz quaternion for an OpenGL camera frame looking at target_pos."""
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    forward = target_pos - camera_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = -forward
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _wxyz_from_xyzw(scipy.spatial.transform.Rotation.from_matrix(rot_mat).as_quat())


def _camera_cfg(parent_path, pos, quat, convention, args):
    data_types = ["rgb"]
    if args.record_depth:
        data_types.append("distance_to_image_plane")
    if args.record_segmentation:
        # instance_id_segmentation_fast = 原始 id 图 (非 RGBA 可视化)
        data_types.append("instance_id_segmentation_fast")
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}" + parent_path,
        update_period=args.render_dt,
        height=args.camera_height,
        width=args.camera_width,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.3,
            focus_distance=400.0,
            horizontal_aperture=4.6,
            clipping_range=(0.01, 10000.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=quat, convention=convention),
    )


def _ball_asset_path(object_dir):
    path = os.path.join(_abs_path(object_dir), "tennis_ball", "green_yellow_tennis_ball.usd")
    if not os.path.exists(path):
        raise FileNotFoundError("Tennis ball USD not found: %s" % path)
    return path


# ── Franka config (same as evaluate_tennis_0804_new.py) ───────────────────
LOCAL_FRANKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # The arm is position/IK controlled.  Keep its links suspended so
            # gravity does not pull the end-effector away from the commanded
            # pose; the tennis ball below intentionally keeps gravity enabled.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.610,
            "panda_joint5": 0.0,
            "panda_joint6": 3.752,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=4.0,
            stiffness=800.0,
            damping=120.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=5.0,
            stiffness=800.0,
            damping=120.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
            friction=50,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# ── Scene config ──────────────────────────────────────────────────────────
@configclass
class TennisSceneCfg(InteractiveSceneCfg):
    """Scene placeholders filled by ``make_scene_cfg``."""
    floor: AssetBaseCfg = MISSING
    back_wall: AssetBaseCfg = MISSING
    front_wall: AssetBaseCfg = MISSING
    left_wall: AssetBaseCfg = MISSING
    right_wall: AssetBaseCfg = MISSING
    table_top: AssetBaseCfg = MISSING
    table_leg_fl: AssetBaseCfg = MISSING
    table_leg_fr: AssetBaseCfg = MISSING
    table_leg_bl: AssetBaseCfg = MISSING
    table_leg_br: AssetBaseCfg = MISSING
    robot: ArticulationCfg = MISSING
    tennis_ball: RigidObjectCfg = MISSING
    center_base_cam: CameraCfg | None = None
    left_base_cam: CameraCfg | None = None
    right_base_cam: CameraCfg | None = None
    dome_light: AssetBaseCfg = MISSING
    distant_light: AssetBaseCfg = MISSING


def make_scene_cfg(args):
    env_spacing = max(args.room_length, args.room_width) + 1.0
    cfg = TennisSceneCfg(
        num_envs=args.num_envs,
        env_spacing=env_spacing,
        lazy_sensor_update=False,
        replicate_physics=True,
    )

    half_len = args.room_length / 2.0
    half_wid = args.room_width / 2.0
    wall_thickness = 0.05
    floor_thickness = 0.05
    wall_center_z = args.room_height / 2.0

    cfg.floor = _static_box(
        "{ENV_REGEX_NS}/Floor",
        (args.room_length, args.room_width, floor_thickness),
        (0.0, 0.0, -floor_thickness / 2.0),
        (1.0, 1.0, 1.0),
    )
    cfg.back_wall = _static_box(
        "{ENV_REGEX_NS}/BackWall",
        (wall_thickness, args.room_width, args.room_height),
        (-half_len, 0.0, wall_center_z),
        (1.0, 1.0, 1.0),
    )
    cfg.front_wall = _static_box(
        "{ENV_REGEX_NS}/FrontWall",
        (wall_thickness, args.room_width, args.room_height),
        (half_len, 0.0, wall_center_z),
        (1.0, 1.0, 1.0),
    )
    cfg.left_wall = _static_box(
        "{ENV_REGEX_NS}/LeftWall",
        (args.room_length, wall_thickness, args.room_height),
        (0.0, half_wid, wall_center_z),
        (1.0, 1.0, 1.0),
    )
    cfg.right_wall = _static_box(
        "{ENV_REGEX_NS}/RightWall",
        (args.room_length, wall_thickness, args.room_height),
        (0.0, -half_wid, wall_center_z),
        (1.0, 1.0, 1.0),
    )

    table_top_thickness = 0.06
    leg_width = 0.055
    table_top_z = args.table_height - table_top_thickness / 2.0
    table_color = (0.92, 0.92, 0.90)
    cfg.table_top = _static_box(
        "{ENV_REGEX_NS}/TableTop",
        (args.table_length, args.table_width, table_top_thickness),
        (0.0, 0.0, table_top_z),
        table_color,
    )
    leg_z = args.table_height / 2.0 - table_top_thickness / 2.0
    leg_height = args.table_height - table_top_thickness
    leg_x = args.table_length / 2.0 - 0.12
    leg_y = args.table_width / 2.0 - 0.10
    for attr, pos in {
        "table_leg_fl": (leg_x, leg_y, leg_z),
        "table_leg_fr": (leg_x, -leg_y, leg_z),
        "table_leg_bl": (-leg_x, leg_y, leg_z),
        "table_leg_br": (-leg_x, -leg_y, leg_z),
    }.items():
        setattr(cfg, attr, _static_box("{ENV_REGEX_NS}/" + attr,
                                        (leg_width, leg_width, leg_height),
                                        pos, table_color))

    cfg.robot = LOCAL_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.robot.init_state.pos = (0.0, 0.0, args.table_height + args.robot_base_z_offset)

    cfg.tennis_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TennisBall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(args.ball_start_distance, 0.0, args.ball_start_height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ball_asset_path(args.object_dir),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=10.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    if args.enable_cameras:
        cfg.left_base_cam = _camera_cfg(
            "/Robot/LeftBaseCamera",
            [0.0, 0.25, 1.0],
            (-0.695, 0.1288, 0.69528, -0.1288),  # xyzw
            "opengl", args,
        )
        cfg.right_base_cam = _camera_cfg(
            "/Robot/RightBaseCamera",
            [0.0, -0.25, 1.0],
            (-0.6837, -0.247, 0.629, 0.2755),  # xyzw
            "opengl", args,
        )
        cfg.center_base_cam = _camera_cfg(
            "/Robot/CenterBaseCamera",
            (0.3, 0.0, 0.0),
            (-0.798, 0.4469, 0.279, -0.293),  # xyzw
            "opengl", args,
        )

    cfg.dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            enable_color_temperature=True,
            color_temperature=6500,
            intensity=450.0,
        ),
    )
    cfg.distant_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DistantLight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -2.0, 4.0),
            rot=_look_at_opengl_quat([0.0, -2.0, 4.0], [0.0, 0.0, 0.5]),
        ),
        spawn=sim_utils.DistantLightCfg(
            enable_color_temperature=True,
            color_temperature=6500,
            intensity=1800.0,
        ),
    )
    return cfg


# ── Ball throw ────────────────────────────────────────────────────────────
def _sample_p2p_throw(rng, max_retries=200):
    """Point-to-point throw: (pos_3d, vel_3d, landing_3d)."""
    g = 9.81
    z_fixed = rng.uniform(0.81, 1.0)
    theta_lo = np.deg2rad(30.0)
    theta_hi = np.deg2rad(75.0)

    for _ in range(max_retries):
        angle_a = rng.uniform(0.0, 2.0 * np.pi)
        radius_a = rng.uniform(0.0, 1.0)
        a_x = 4.0 + radius_a * np.cos(angle_a)
        a_y = 0.0 + radius_a * np.sin(angle_a)

        for _ in range(100):
            angle_b = rng.uniform(0.0, 2.0 * np.pi)
            radius_b = rng.uniform(0.0, 0.7)
            b_x = radius_b * np.cos(angle_b)
            b_y = radius_b * np.sin(angle_b)
            if b_x >= 0.3:
                break
        else:
            b_x = 0.3 + rng.uniform(0.0, 0.7)
            b_y = 0.0

        t_target = rng.uniform(1.2, 1.4)
        dx = b_x - a_x
        dy = b_y - a_y
        d_xy = np.sqrt(dx * dx + dy * dy)
        if d_xy < 1e-6:
            continue
        v_xy = d_xy / t_target
        v_z = 0.5 * g * t_target

        theta = np.arctan2(v_z, v_xy)
        if theta < theta_lo or theta > theta_hi:
            continue

        v_x = v_xy * dx / d_xy
        v_y = v_xy * dy / d_xy
        pos = np.array([a_x, a_y, z_fixed], dtype=np.float64)
        vel = np.array([v_x, v_y, v_z], dtype=np.float64)
        land = np.array([b_x, b_y, z_fixed], dtype=np.float64)
        return pos, vel, land

    z_fb = 0.81
    t_target = 1.0
    v_xy = 3.5 / t_target
    v_z = 0.5 * g * t_target
    pos = np.array([4.0, 0.0, z_fb], dtype=np.float64)
    vel = np.array([-v_xy, 0.0, v_z], dtype=np.float64)
    land = np.array([0.5, 0.0, z_fb], dtype=np.float64)
    return pos, vel, land


def _reset_ball(scene, args, rng):
    """Reset tennis ball with throw parameters."""
    ball = scene["tennis_ball"]
    pos_list, vel_list = [], []
    for _ in range(args.num_envs):
        p, v, _ = _sample_p2p_throw(rng)
        pos_list.append(p)
        vel_list.append(v)
    pos_np = np.stack(pos_list, axis=0)
    vel_np = np.stack(vel_list, axis=0)
    root_pose = ball.data.default_root_pose.clone()
    root_pose[:, :3] = scene.env_origins + torch.from_numpy(pos_np).float().to(ball.device)
    root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=ball.device).repeat(args.num_envs, 1)
    root_vel = ball.data.default_root_vel.clone()
    root_vel[:, :3] = torch.from_numpy(vel_np).float().to(ball.device)
    root_vel[:, 3:6] = torch.tensor([0.0, 25.0, 0.0], device=ball.device).repeat(args.num_envs, 1)
    ball.write_root_link_pose_to_sim_index(root_pose=root_pose)
    ball.write_root_com_velocity_to_sim_index(root_velocity=root_vel)
    ball.reset()


def _reset_robot(scene):
    """Reset robot to default pose (same as evaluate_tennis_0804_new.py)."""
    robot = scene["robot"]
    root_pose = robot.data.default_root_pose.clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel)
    robot.set_joint_position_target_index(target=robot.data.default_joint_pos)


# ── VLA frame capture (model inference runs separately in dy-vla env) ────
def _capture_cams(scene):
    """Capture the 3 base cameras as uint8 (H, W, 3) dict."""
    frames = {}
    for key in ["left_base_cam", "right_base_cam", "center_base_cam"]:
        if key in scene.sensors:
            f = scene.sensors[key].data.output["rgb"][0].detach().cpu().numpy()
            if f.shape[-1] == 4:
                f = f[..., :3]
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            frames[key] = f
    return frames


_DEPTH_CMAP_ANCHORS = np.array(
    [[0, 0, 0.5], [0, 0.5, 1], [0, 1, 0.7], [1, 0.85, 0], [0.8, 0, 0]],
    dtype=np.float32,
)


def _depth_to_rgb(depth):
    """Depth (H,W) float32 m → RGB (H,W,3) uint8; inf(无命中背景)渲染为黑色."""
    v = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(v)
    if valid.any():
        lo, hi = float(v[valid].min()), float(v[valid].max())
    else:
        lo, hi = 0.0, 1.0
    vc = np.clip(v, lo, hi)
    t = np.clip((vc - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    scaled = t * (len(_DEPTH_CMAP_ANCHORS) - 1)
    i0 = np.floor(scaled).astype(int)
    i1 = np.minimum(i0 + 1, len(_DEPTH_CMAP_ANCHORS) - 1)
    frac = np.clip(scaled - i0, 0.0, 1.0)[..., None]
    rgb = _DEPTH_CMAP_ANCHORS[i0] * (1 - frac) + _DEPTH_CMAP_ANCHORS[i1] * frac
    rgb = (rgb * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _seg_to_rgb(ids):
    """Instance id 图 (H,W) int → RGB (H,W,3) uint8 (0=黑, 其他确定性哈希着色)."""
    ids = np.asarray(ids, dtype=np.int64)
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for uid in np.unique(ids):
        if uid == 0:
            continue
        h = (uid * 2654435761) % (2 ** 32)
        color = np.array(
            [(h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF)], dtype=np.uint8
        )
        out[ids == uid] = color
    return out


def _encode_video_file(frames, output_path, fps=30):
    """Encode (N, H, W, 3) uint8 frames to an AV1 mp4 file (headless-safe)."""
    import av
    from PIL import Image

    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"Expected non-empty video frames (N, H, W, 3), got {frames.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with av.open(output_path, "w") as output:
        stream = output.add_stream("libsvtav1", fps, options={"g": "2", "crf": "30"})
        stream.pix_fmt = "yuv420p"
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(Image.fromarray(image))):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _get_ee_pose_w(scene):
    """End-effector (panda_hand) world pose: pos(3), quat_xyzw(4)."""
    robot = scene["robot"]
    ee_idx = int(robot.find_bodies("panda_hand")[0][0])
    pose = robot.data.body_pose_w[:, ee_idx][0].cpu().numpy()
    return pose[:3], pose[3:7]


def _get_ee_pose_b(scene):
    """End-effector (panda_hand) pose in robot BASE frame: pos(3), quat_xyzw(4).

    Consistent with the training dataset (state/action recorded in base frame).
    """
    robot = scene["robot"]
    ee_idx = int(robot.find_bodies("panda_hand")[0][0])
    ee_pose_w = robot.data.body_pose_w[:, ee_idx]
    root_pose_w = robot.data.root_pose_w.torch
    # subtract_frame_transforms returns a (pos, quat) tuple — unpack properly
    pos_b, quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    return pos_b[0].cpu().numpy(), quat_b[0].cpu().numpy()


class ArmController:
    """Task-space IK controller driving the Franka arm from 8D VLA actions.

    Action format (base frame, same as training data):
        [pos(3), quat_xyzw(4), gripper(1)]
    With no command, holds the default pose.
    """

    def __init__(self, scene, device):
        self.robot = scene["robot"]
        self.device = device
        self.ee_frame_idx = int(self.robot.find_bodies("panda_hand")[0][0])
        self.ee_jacobi_idx = self.ee_frame_idx - 1
        self.arm_joint_ids = self.robot.find_joints("panda_joint.*")[0]
        self.finger_joint_ids = self.robot.find_joints("panda_finger.*")[0]
        self._arm_ids_t = torch.tensor(self.arm_joint_ids, device=device)
        self._gripper_open = torch.full(
            (1, len(self.finger_joint_ids)), 0.04, device=device, dtype=torch.float32)
        self._gripper_closed = torch.zeros_like(self._gripper_open)
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)
        self.command = None
        self.gripper_target = None

    def reset(self):
        self.diff_ik.reset()
        self.command = None
        self.gripper_target = None

    def set_action(self, action):
        """Set IK target from an 8D action (base frame)."""
        self.command = torch.from_numpy(action[:7]).float().to(self.device).unsqueeze(0)
        self.command[:, 3:7] = torch.nn.functional.normalize(self.command[:, 3:7], dim=-1)
        self.diff_ik.set_command(self.command)
        if len(self.finger_joint_ids) > 0:
            grip = torch.tensor([[0.04, 0.04]], device=self.device) if action[7] > 0.0 \
                else torch.zeros(1, 2, device=self.device)
            self.gripper_target = grip

    def apply(self):
        if self.command is None:
            # 无命令时保持当前位姿 (目标=当前关节角), 而不是回到默认初始位姿
            self.robot.set_joint_position_target_index(
                target=self.robot.data.joint_pos[:, self._arm_ids_t],
                joint_ids=self.arm_joint_ids)
            if len(self.finger_joint_ids) > 0:
                self.robot.set_joint_position_target_index(
                    target=self._gripper_open, joint_ids=self.finger_joint_ids)
            return
        jacobian_w = self.robot.data.body_link_jacobian_w.torch
        jacobian_w = jacobian_w[:, self.ee_jacobi_idx, :, :][:, :, self._arm_ids_t]
        root_quat_w = self.robot.data.root_pose_w.torch[:, 3:7]
        base_rot = matrix_from_quat(quat_inv(root_quat_w))
        jacobian_w[:, :3, :] = torch.bmm(base_rot, jacobian_w[:, :3, :])
        jacobian_w[:, 3:, :] = torch.bmm(base_rot, jacobian_w[:, 3:, :])
        ee_pos_b, ee_quat_b = self._get_ee_pose_b_t()
        joint_pos = self.robot.data.joint_pos[:, self._arm_ids_t]
        arm_target = self.diff_ik.compute(ee_pos_b, ee_quat_b, jacobian_w, joint_pos)
        self.robot.set_joint_position_target_index(
            target=arm_target, joint_ids=self.arm_joint_ids)
        if self.gripper_target is not None and len(self.finger_joint_ids) > 0:
            self.robot.set_joint_position_target_index(
                target=self.gripper_target, joint_ids=self.finger_joint_ids)

    def _get_ee_pose_b_t(self):
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_frame_idx]
        root_pose_w = self.robot.data.root_pose_w.torch
        return subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])


def _dataset_video_roundtrip(frames):
    """Encode frames with libx264 (crf=0, yuv444p, 无损) then decode rgb24.

    Matches the dataset video pipeline so live inference frames have the
    same (lossless) compression distribution as training frames.
    """
    import io
    import av
    from PIL import Image

    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected frames with shape (N, H, W, 3), got %s" % (frames.shape,))

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as output:
        # libx264 crf=0 + yuv444p = 无损, 与数据集视频管线一致
        stream = output.add_stream(
            "libx264",
            30,
            options={"crf": "0"},
        )
        stream.pix_fmt = "yuv444p"
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        for image in frames:
            for packet in stream.encode(
                av.VideoFrame.from_image(Image.fromarray(image))
            ):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    buffer.seek(0)
    decoded = []
    with av.open(buffer) as container:
        for frame in container.decode(container.streams.video[0]):
            decoded.append(frame.to_ndarray(format="rgb24"))
    if len(decoded) != len(frames):
        raise RuntimeError(
            "AV1 round-trip decoded %d frames, expected %d"
            % (len(decoded), len(frames))
        )
    return np.stack(decoded).astype(np.uint8, copy=False)


def _make_zmq_publisher(host, port):
    """Create a ZeroMQ PUB socket for VLA frame publishing."""
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[VLA] ZeroMQ PUB ready on tcp://{host}:{port}", flush=True)
    return sock


def _publish_frames(zmq_sock, episode_idx, frames, ee_pos_b, ee_quat_b):
    """Publish the 9-frame camera prefix + EE pose (base frame) over ZeroMQ."""
    msg = {
        "episode": episode_idx,
        "frames": frames,       # list[dict[cam_name, uint8 (H, W, 3)]]
        "ee_pos_b": ee_pos_b,   # (3,) base frame — consistent with training
        "ee_quat_b": ee_quat_b,  # (4,) xyzw base frame
    }
    zmq_sock.send_pyobj(msg)
    print(f"[VLA] Published {len(frames)} temporal frames for episode {episode_idx}",
          flush=True)


def _make_zmq_action_sub(host, port):
    """Create a ZeroMQ SUB socket for receiving VLA action chunks."""
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"[VLA] ZeroMQ action SUB connected to tcp://{host}:{port}", flush=True)
    return sock


def _drain_actions(act_sock):
    """Non-blocking drain of action messages; returns the latest chunk or None."""
    import zmq
    latest = None
    try:
        while True:
            msg = act_sock.recv_pyobj(flags=zmq.NOBLOCK)
            if isinstance(msg, dict) and "actions" in msg:
                latest = msg
    except zmq.Again:
        pass
    return latest


# ── End-effector visual adjustments (same as evaluate_tennis_0804_new.py) ──
def _hide_gripper(num_envs):
    """Hide hand+fingers prims and disable their collision."""
    import omni.usd
    from pxr import Sdf, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for _ei in range(num_envs):
        for _name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            _fp = f"/World/envs/env_{_ei}/Robot/{_name}"
            _fprim = stage.GetPrimAtPath(_fp)
            if _fprim:
                _fprim.CreateAttribute(
                    "visibility", Sdf.ValueTypeNames.Token, False
                ).Set("invisible")
                try:
                    _coll_prim = stage.GetPrimAtPath(f"{_fp}/collisions")
                    if _coll_prim:
                        for _child in _coll_prim.GetAllChildren():
                            _physx = PhysxSchema.PhysxCollisionAPI.Apply(_child)
                            _physx.CreateCollisionGroupAttr().Set(0)
                except Exception:
                    pass
    print("[INFO] Hand+fingers hidden & collision off.", flush=True)


def _spawn_end_ring(num_envs):
    """Spawn the end-ring mesh on panda_link7 with collision enabled.

    Fixed pose, same as 0804_new: translate z=0.24, rotY=90°, rotX=45°.
    The ring mesh gets UsdPhysics collision so the ball can be caught.
    """
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    _ring_obj = os.path.normpath(os.path.join(PROJECT_HOME, "objects", "end_ring.obj"))
    if not os.path.isfile(_ring_obj):
        print(f"[WARN] End-ring mesh not found: {_ring_obj}", flush=True)
        return
    _verts, _faces = [], []
    with open(_ring_obj) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("v "):
                _parts = _line.split()
                _verts.append(tuple(float(_p) for _p in _parts[1:4]))
            elif _line.startswith("f "):
                _faces.append([int(_p.split("/")[0]) - 1 for _p in _line.split()[1:4]])
    _ring_stage = omni.usd.get_context().get_stage()
    for _ei in range(num_envs):
        _ring_path = f"/World/envs/env_{_ei}/Robot/panda_link7/EndRing"
        _xform = UsdGeom.Xform.Define(_ring_stage, _ring_path)
        _xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.24))
        _xform.AddRotateYOp().Set(90.0)
        _xform.AddRotateXOp().Set(45.0)
        _mesh_path = _ring_path + "/ring_mesh"
        _mesh = UsdGeom.Mesh.Define(_ring_stage, _mesh_path)
        _mesh.CreatePointsAttr().Set([Gf.Vec3f(*v) for v in _verts])
        _mesh.CreateFaceVertexCountsAttr().Set([len(f) for f in _faces])
        _mesh.CreateFaceVertexIndicesAttr().Set([vi for f in _faces for vi in f])
        _mesh.CreateDoubleSidedAttr().Set(True)
        _mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
        # Collision: ring participates in physics as part of panda_link7
        UsdPhysics.CollisionAPI.Apply(_mesh.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(_mesh.GetPrim())
    print("[INFO] End-ring mesh spawned with collision on all envs.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=args_cli.physics_dt,
        render_interval=max(1, int(round(args_cli.render_dt / args_cli.physics_dt))),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()
    sim.set_camera_view(eye=[3.2, -3.0, 2.1], target=[0.0, 0.0, 0.7])

    scene = InteractiveScene(make_scene_cfg(args_cli))
    # Trigger PHYSICS_READY event so articulations fully initialize
    # (_data, actuators, etc.) before we touch robot/ball state.
    sim.reset()

    # ── Extend panda_joint6 upper limit by 90° (与 0804 训练采集一致) ────────
    # 默认 Franka USD 限位 ±3.753 rad (±215°); 上界 +π/2 → ≈305°, 接低球需要
    import warp as wp
    _robot = scene["robot"]
    _limits = _robot.data.joint_pos_limits.torch  # (num_envs, num_joints, 2)
    _joint6_idx = _robot.find_joints("panda_joint6")[0][0]
    _new_upper = _limits[0, _joint6_idx, 1].item() + np.pi / 2.0
    _limits[:, _joint6_idx, 1] = _new_upper
    _robot.root_view.set_dof_limits(
        wp.clone(_robot.data._joint_pos_limits, device="cpu"),
        indices=wp.array(np.arange(args_cli.num_envs), dtype=wp.int32, device="cpu"),
    )
    _soft_limits = _robot.data.soft_joint_pos_limits.torch
    _soft_limits[:, _joint6_idx, 1] = _new_upper

    # ── Hide end-effector (hand+fingers) & spawn end-ring mesh ─────────────
    _hide_gripper(args_cli.num_envs)
    _spawn_end_ring(args_cli.num_envs)

    rng = np.random.RandomState(args_cli.seed)
    # Same call order as evaluate_tennis_0804_new.py:
    # write robot/ball state to sim, then scene.reset() applies everything.
    _reset_robot(scene)
    _reset_ball(scene, args_cli, rng)
    scene.reset()
    scene.write_data_to_sim()
    sim.play()

    print("[INFO] Tennis scene ready. num_envs=%d randomize_throw=%s dt=%.4f"
          % (args_cli.num_envs, args_cli.randomize_throw, sim_dt), flush=True)

    total_steps = 0
    episode_idx = 0
    episode_steps = 0
    sim_time = 0.0
    episode_flight_time = 1.5  # ball flight time per episode
    observation_frames = []  # camera dictionaries at logical steps 0,2,...,16
    episode_video_frames = [] if args_cli.save_video else None
    throw_pending_since = None  # sim_time when the last episode ended

    # ── ZeroMQ publisher for VLA frames ──
    zmq_sock = _make_zmq_publisher(args_cli.zmq_host, args_cli.zmq_port) \
        if args_cli.zmq_publish else None

    # ── Arm control: IK controller + action SUB socket ──
    controller = None
    act_sock = None
    action_queue = []
    inference_waiting = False
    inference_action_received = False
    inference_resume_after = 0.0
    if args_cli.control_arm:
        controller = ArmController(scene, "cuda" if torch.cuda.is_available() else "cpu")
        controller.reset()
        act_sock = _make_zmq_action_sub(args_cli.zmq_host, args_cli.zmq_act_port)
        print("[INFO] Arm control ON: VLA actions will drive the robot.", flush=True)

    while simulation_app.is_running():
        if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
            break
        if sim.is_stopped():
            break

        # ── Arm control: consume next action from queue ──
        if controller is not None:
            if act_sock is not None:
                chunk = _drain_actions(act_sock)
                if chunk is not None:
                    action_queue = [np.asarray(a, dtype=np.float32) for a in chunk["actions"]]
                    inference_action_received = True
                    print(f"[VLA] Episode {episode_idx}: received {len(action_queue)}-step "
                          f"action chunk from episode {chunk.get('episode', '?')}", flush=True)
            if action_queue:
                controller.set_action(action_queue.pop(0))
            controller.apply()

        # Keep the complete scene frozen while the external VLA process is
        # computing the action chunk.  In particular, this holds the ball at
        # its observation-time pose without writing a PhysX velocity (which is
        # forbidden by Direct GPU API).  Rendering/application updates keep the
        # GUI responsive; simulated time does not advance until an action is
        # received.
        if inference_waiting:
            if inference_action_received and time.monotonic() >= inference_resume_after:
                inference_waiting = False
                print(f"[VLA] Episode {episode_idx}: inference hold released", flush=True)
            else:
                scene.write_data_to_sim()
                sim.render()
                simulation_app.update()
                continue

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        sim_time += sim_dt
        total_steps += 1
        episode_steps += 1

        # ── Per-episode video capture (headless-safe, works with --viz none) ──
        if episode_video_frames is not None:
            _vframes = _capture_cams(scene)
            _vnames = [n for n in ("left_base_cam", "right_base_cam", "center_base_cam")
                       if n in _vframes]
            if _vnames:
                if args_cli.record_depth or args_cli.record_segmentation:
                    # 每相机一行 [RGB, 深度, 掩码], 三排垂直堆叠 (与 viz_dataset.py 同布局)
                    _rows = []
                    for _n in _vnames:
                        _panels = [_vframes[_n]]
                        if args_cli.record_depth:
                            _d = scene.sensors[_n].data.output[
                                "distance_to_image_plane"][0].detach().cpu().numpy()
                            _panels.append(_depth_to_rgb(np.squeeze(_d)))
                        if args_cli.record_segmentation:
                            _s = scene.sensors[_n].data.output[
                                "instance_id_segmentation_fast"][0].detach().cpu().numpy()
                            _s = np.squeeze(_s)
                            if _s.ndim == 3 and _s.shape[-1] > 1:
                                _s = _s[..., 0]
                            _panels.append(_seg_to_rgb(_s.astype(np.int32)))
                        _rows.append(np.concatenate(_panels, axis=1))
                    episode_video_frames.append(np.concatenate(_rows, axis=0))
                else:
                    episode_video_frames.append(
                        np.concatenate([_vframes[n] for n in _vnames], axis=1))

        # ── Real-time EE pose print ──
        if args_cli.print_ee and total_steps % args_cli.print_ee_every == 0:
            ee_pos_w, ee_quat_w = _get_ee_pose_w(scene)
            print(f"[EE] ep={episode_idx} step={episode_steps} t={sim_time:.3f}s  "
                  f"pos=[{ee_pos_w[0]:.4f} {ee_pos_w[1]:.4f} {ee_pos_w[2]:.4f}]  "
                  f"quat=[{ee_quat_w[0]:.4f} {ee_quat_w[1]:.4f} {ee_quat_w[2]:.4f} {ee_quat_w[3]:.4f}]",
                  flush=True)

        # ── VLA frame capture: logical steps 0,2,...,16 → ZeroMQ ──
        if zmq_sock is not None:
            # episode_steps is incremented after sim.step(), so the first valid
            # rendered camera image is logical frame 0.
            observation_step = episode_steps - 1
            if observation_step in VLA_OBSERVATION_STEPS:
                observation_frames.append(_capture_cams(scene))
                print(f"[VLA] Episode {episode_idx}: captured frame at step "
                      f"{observation_step}", flush=True)
            if observation_step == VLA_OBSERVATION_STEPS[-1]:
                ee_pos_b, ee_quat_b = _get_ee_pose_b(scene)
                if len(observation_frames) == len(VLA_OBSERVATION_STEPS):
                    # Align live frames with the dataset video encoding pipeline
                    if args_cli.vla_match_dataset_video:
                        _cam_names = ["left_base_cam", "right_base_cam", "center_base_cam"]
                        _cam_names = [c for c in _cam_names
                                      if all(c in frame for frame in observation_frames)]
                        for _cn in _cam_names:
                            _sequence = np.stack(
                                [frame[_cn] for frame in observation_frames]
                            )  # (9, H, W, 3)
                            _decoded = _dataset_video_roundtrip(_sequence)
                            for _frame, _image in zip(observation_frames, _decoded):
                                _frame[_cn] = _image
                        print(f"[VLA] Episode {episode_idx}: applied lossless h264/yuv444p "
                              f"roundtrip to {len(_cam_names)} cameras", flush=True)
                    _publish_frames(zmq_sock, episode_idx, observation_frames,
                                    ee_pos_b, ee_quat_b)
                    if controller is not None:
                        inference_waiting = True
                        inference_action_received = False
                        inference_resume_after = (
                            time.monotonic() + float(args_cli.inference_delay)
                        )
                        if args_cli.inference_delay > 0.0:
                            print(
                                f"[VLA] Episode {episode_idx}: holding scene for at least "
                                f"{args_cli.inference_delay:.3f}s while awaiting inference",
                                flush=True,
                            )
                observation_frames = []

        # Episode end: ball has landed → wait throw_interval, then re-throw
        if throw_pending_since is None and episode_steps > 10 \
                and episode_steps * sim_dt >= episode_flight_time:
            throw_pending_since = sim_time
            print("[INFO] Episode %d ended after %.2f s (%d steps). Next throw in %.1f s."
                  % (episode_idx, episode_steps * sim_dt, episode_steps,
                     args_cli.throw_interval), flush=True)
            # Save the stitched per-episode video (if requested)
            if episode_video_frames is not None and episode_video_frames:
                _vpath = os.path.join(args_cli.video_dir, f"ep{episode_idx:04d}_stitched.mp4")
                _encode_video_file(np.stack(episode_video_frames), _vpath)
                print(f"[INFO] Saved video → {_vpath}", flush=True)
                episode_video_frames = []

        if throw_pending_since is not None \
                and sim_time - throw_pending_since >= args_cli.throw_interval:
            episode_idx += 1
            episode_steps = 0
            observation_frames = []
            throw_pending_since = None
            if controller is not None:
                controller.reset()
                action_queue = []
                inference_waiting = False
                inference_action_received = False
                inference_resume_after = 0.0
            # 每集把机械臂瞬移回初始位姿 (与训练数据一致: 每集开头臂在初始位姿静止)
            _reset_robot(scene)
            _reset_ball(scene, args_cli, rng)
            scene.write_data_to_sim()

    print("[INFO] Done.", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        if args_cli.clean_close:
            simulation_app.close(wait_for_replicator=False)
        else:
            os._exit(exit_code)
