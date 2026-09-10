"""Standalone 3DGS plate CatchIt scene viewer (Isaac Lab environment).

Opens the prebuilt 3DGS plate scene (same scene used by
/mnt/data7t/h00940284/catch_it_isaac_sim/scripts/drive_3dgs_plate_catch_it.sh)
and applies runtime calibration overrides. Scene-only: no keyboard driving;
extension hooks (driving / capture / evaluation) go into the main loop.

Usage:
    ./isaaclab.sh -p .../evaluate_tennis_0824_new.py --viz kit
    ./isaaclab.sh -p .../evaluate_tennis_0824_new.py --viz kit --skip-splat
    ./isaaclab.sh -p .../evaluate_tennis_0824_new.py --viz kit \\
        --plate-translate 0 -0.115864 0.1 --plate-size 30 30 --plate-visible
    ./isaaclab.sh -p .../evaluate_tennis_0824_new.py --viz none --play

Notes:
    - Physics runs continuously and an _ArmHoldController keeps the arm at its
      initial pose: every step it re-issues the hold target, pins the
      free-floating base (same as catch_it_env's set_world_poses hold), and
      zeroes joint velocities (catch_it_env._zero_articulation_motion).
      Use --static to pause physics for pure viewing.
    - The 3DGS splat is ~744MB and loads at open_stage; use --skip-splat to
      clear the reference and view the rest of the scene without it.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import pickle
import select
import shutil
import sys
import termios
import time
import traceback
import tty
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


DEFAULT_SCENE_USD = (
    "/home/jdhc/z00821918/tennis_robot/dynamicvla/dynamic-vla/simulations/"
    "assets/catch_it_x5_double_3dgs_plate_scene_patched.usda"
)
SCENE_ASSET_ROOT = "/mnt/data7t/h00940284/catch_it_isaac_sim"
PRIM_ENV = "/World/HS3DGS"
PRIM_ROBOT = "/World/CatchIt"
PRIM_PLATE = "/World/DrivePlate"
PRIM_SOURCE_BALL = "/World/CatchIt/object/object"
PRIM_BALL = "/World/TennisBall"
PRIM_RING = "/World/CatchIt/base_link/ring"
GROUND_MASK_PRIM_PATHS = (
    "/World/CatchIt/worldBody/floor",
    PRIM_PLATE,
)
ARM_MASK_PRIM_PATHS = (
    "/World/CatchIt/base_link/arm_base",
    "/World/CatchIt/base_link/link1",
    "/World/CatchIt/base_link/link2",
    "/World/CatchIt/base_link/link3",
    "/World/CatchIt/base_link/link4",
    "/World/CatchIt/base_link/link5",
    "/World/CatchIt/base_link/link6",
    "/World/CatchIt/base_link/ring",
)
SEGMENTATION_CLASS_IDS = {
    "background": 0,
    "tennis_ball": 1,
    "ground": 2,
    "robot_arm": 3,
}
PLATE_SHADER = "/World/Looks/DrivePlateTransparent/PreviewSurface"
NET_VISUAL_ROOT_PATH = "/World/CatchIt/base_link/ring/visuals"
NET_VISUAL_PATH = f"{NET_VISUAL_ROOT_PATH}/ring_rim"
RING_TORUS_VISUAL_PATH = "/World/CatchIt/base_link/ring/mujoco_ring_torus_visual"
_BALL_ASSET_RELATIVE = Path("objects/tennis_ball/green_yellow_tennis_ball.usd")
_BALL_ASSET_CANDIDATES = (
    Path(__file__).resolve().parents[2] / _BALL_ASSET_RELATIVE,
    Path("/home/jdhc/z00821918/tennis_robot/dynamicvla") / _BALL_ASSET_RELATIVE,
)
BALL_ASSET_USD = str(next(
    (candidate for candidate in _BALL_ASSET_CANDIDATES if candidate.is_file()),
    _BALL_ASSET_CANDIDATES[0],
))
PERSP_CAM = "/OmniverseKit_Persp"
ARM_BASE_CAMERA_PATHS = (
    "/World/CatchIt/base_link/arm_base/arm_base_camera_left",
    "/World/CatchIt/base_link/arm_base/arm_base_camera_right",
    "/World/CatchIt/base_link/arm_base/arm_base_camera_upward",
)
EXTRA_ARM_BASE_CAMERA_PATH = "/World/CatchIt/base_link/arm_base/obs_camera"
NDARRAY_WIRE_MARKER = "__ndarray_raw_v1__"
# The Isaac Sim client accepts action chunks of variable temporal length.  The
# WAM protocol still requires nine delta channels per action.
INFERENCE_DELTA_STEPS = None
# Default arm initial pose in radians.
ARM_HOME_RAD = (0.0, -0.35, -0.44, 0.0, -0.79, 1.57)
# Match patch_usd_physics_from_mjcf.py and the MJCF arm actuator limits.
ARM_DRIVE_STIFFNESS = (300.0, 400.0, 400.0, 50.0, 200.0, 20.0)
ARM_DRIVE_DAMPING = (40.0, 40.0, 40.0, 5.0, 10.0, 1.0)
ARM_DRIVE_MAX_FORCE = (100.0,) * 6
# Articulation-root pose measured immediately after the working
# drive_3dgs_plate_catch_it.sh -> IsaacCatchItEnv.reset() path.  This is NOT
# the /World/CatchIt/base_link pose: the MJCF-to-USD import leaves a fixed
# transform between the PhysX articulation root and base_link.  In the upright
# pose the root is at z=0.671 while base_link is at z=0.105.  The legacy
# Isaac Core API reports the measured quaternion as WXYZ; Isaac Lab expects
# XYZW, so (w, x, y, z) is reordered below as (x, y, z, w).
ROBOT_ROOT_POS = (0.0, 0.0, 0.671)
ROBOT_ROOT_QUAT_XYZW = (-0.405079, -0.579580, 0.405539, 0.579254)
# Upright base_link world position for the default scene calibration.  CatchIt
# drives forward along its local +Y axis (drive_3dgs_plate_catch_it.sh default).
BASE_LINK_POS = (0.0, -0.115864, 0.105)


def ground_mask_from_depth(
    depth,
    camera_intrinsic,
    world_from_camera_usd,
    ground_z=0.0,
    z_tolerance=0.05,
):
    """Return pixels whose depth back-projects near a world-Z ground plane.

    ``depth`` is Replicator ``distance_to_image_plane`` (optical-axis depth).
    Pixels are first unprojected in the OpenCV optical frame (+X right, +Y
    down, +Z forward), then converted to the USD camera frame (+X right, +Y
    up, -Z forward) before applying ``world_from_camera_usd``.
    """
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    transform = np.asarray(world_from_camera_usd, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(
            f"camera_intrinsic must be 3x3, got shape {intrinsic.shape}"
        )
    if transform.shape != (4, 4):
        raise ValueError(
            f"world_from_camera_usd must be 4x4, got shape {transform.shape}"
        )
    if float(z_tolerance) <= 0.0:
        raise ValueError("z_tolerance must be positive")

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    pixel_v, pixel_u = np.indices(depth.shape, dtype=np.float32)
    x_usd = (pixel_u - cx) * depth / fx
    y_usd = -(pixel_v - cy) * depth / fy
    z_usd = -depth
    world_z = (
        transform[2, 0] * x_usd
        + transform[2, 1] * y_usd
        + transform[2, 2] * z_usd
        + transform[2, 3]
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    return valid & (np.abs(world_z - float(ground_z)) <= float(z_tolerance))


def relabel_actions_from_future_states(episode_buffer, lookahead_steps=2):
    """Set action[t] to state[min(t + lookahead_steps, final_step)]."""
    if lookahead_steps < 0:
        raise ValueError("lookahead_steps must be non-negative")

    size = int(episode_buffer.get("size", 0))
    states = episode_buffer.get("observation.state")
    if states is None or len(states) != size:
        raise ValueError(
            "observation.state length does not match the episode buffer size"
        )
    if size == 0:
        raise ValueError("cannot relabel an empty episode")

    state_array = np.asarray(states, dtype=np.float32)
    source_indices = np.minimum(
        np.arange(size, dtype=np.int64) + int(lookahead_steps), size - 1
    )
    episode_buffer["action"] = [
        state_array[index].copy() for index in source_indices
    ]
    return source_indices


def _pack_ndarray(array):
    """Encode an ndarray using only pickle-stable Python built-in types."""
    array = np.ascontiguousarray(array)
    return {
        NDARRAY_WIRE_MARKER: True,
        "dtype": array.dtype.str,
        "shape": tuple(int(value) for value in array.shape),
        "data": array.tobytes(order="C"),
    }


class _KeyboardDriveInput:
    """Viewport/terminal W-S-A-D input matching keyboard_drive_3dgs_scene.py."""

    def __init__(self, enable_viewport_keyboard):
        self.pressed = set()
        self.quit_requested = False
        self._old_termios = None
        self._input = self._keyboard = self._keyboard_sub = None
        if sys.stdin.isatty():
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        if enable_viewport_keyboard:
            try:
                import carb.input
                try:
                    import omni.appwindow
                except ImportError:
                    import omni.kit.app
                    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                        "omni.appwindow", True
                    )
                    import omni.appwindow
                window = omni.appwindow.get_default_app_window()
                if window is not None:
                    self._input = carb.input.acquire_input_interface()
                    self._keyboard = window.get_keyboard()
                    self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                        self._keyboard, self._on_keyboard_event
                    )
            except Exception as exc:
                print(f"[WARN] viewport keyboard unavailable: {exc}", flush=True)

    def _on_keyboard_event(self, event, *_args):
        import carb.input
        mapping = {
            carb.input.KeyboardInput.W: "w", carb.input.KeyboardInput.S: "s",
            carb.input.KeyboardInput.A: "a", carb.input.KeyboardInput.D: "d",
            carb.input.KeyboardInput.Q: "q", carb.input.KeyboardInput.ESCAPE: "q",
            carb.input.KeyboardInput.SPACE: "space",
        }
        key = mapping.get(event.input)
        if key is None:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key == "q":
                self.quit_requested = True
            elif key == "space":
                self.pressed.clear()
            else:
                self.pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(key)
        return True

    def poll(self):
        if sys.stdin.isatty():
            while select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1).lower()
                if ch in {"w", "a", "s", "d"}:
                    self.pressed = {ch}
                elif ch in {" ", "\n"}:
                    self.pressed.clear()
                elif ch in {"q", "\x1b"}:
                    self.quit_requested = True
        return set(self.pressed)

    def close(self):
        if self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    self._keyboard, self._keyboard_sub
                )
            except Exception:
                pass
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="3DGS plate CatchIt scene viewer (same scene as drive_3dgs_plate_catch_it.sh)."
    )
    parser.add_argument("--scene-usd", default=DEFAULT_SCENE_USD,
                        help="Prebuilt combined scene usda to open.")
    # Calibration interfaces (same names/defaults as build_3dgs_plate_catch_it_scene.py)
    parser.add_argument("--env-translate", nargs=3, type=float,
                        default=(13.0, -20.115864, 2.5), metavar=("X", "Y", "Z"))
    parser.add_argument("--env-rotate", nargs=3, type=float,
                        default=(0.0, 0.0, -50.0), metavar=("RX", "RY", "RZ"),
                        help="Degrees, same convention as the build script.")
    parser.add_argument("--robot-translate", nargs=3, type=float,
                        default=(0.0, -0.115864, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--robot-rotate", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"))
    parser.add_argument("--plate-translate", nargs=3, type=float,
                        default=(0.0, -0.115864, -0.01), metavar=("X", "Y", "Z"))
    parser.add_argument("--plate-rotate", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"))
    parser.add_argument("--plate-size", nargs=2, type=float,
                        default=(50.0, 50.0), metavar=("X", "Y"))
    parser.add_argument("--plate-thickness", type=float, default=0.02)
    parser.add_argument("--plate-color", nargs=3, type=float,
                        default=(0.0, 0.18, 1.0), metavar=("R", "G", "B"))
    parser.add_argument("--plate-opacity", type=float, default=0.0,
                        help="Drive plate visual opacity (0 = invisible but collidable).")
    parser.add_argument("--plate-visible", action="store_true",
                        help="Render the drive plate for debugging.")
    parser.add_argument(
        "--net-opacity",
        type=float,
        default=0.0,
        help="Catching-net visual opacity in [0, 1]; default 0 is fully transparent.",
    )
    # Runtime behavior
    parser.add_argument("--skip-splat", action="store_true",
                        help="Clear the 3DGS reference at runtime (rest of scene still loads).")
    parser.add_argument("--arm-init", nargs=6, type=float, default=ARM_HOME_RAD,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="Initial arm joint angles in RADIANS. "
                        "Default: [0, -0.35, -0.44, 0, -0.79, 1.57].")
    parser.add_argument("--static", action="store_true",
                        help="Pause physics after initialization.")
    parser.add_argument("--keep-playing", action="store_true",
                        help="Deprecated compatibility flag; physics now plays by default.")
    parser.add_argument("--settle-steps", type=int, default=60,
                        help="Deprecated compatibility option; default kinematic mode does not settle.")
    parser.add_argument("--camera-view", nargs=6, type=float, default=None,
                        metavar=("EX", "EY", "EZ", "TX", "TY", "TZ"),
                        help="Set the perspective camera eye/look-at target.")
    parser.add_argument("--arm-base-camera-left", nargs=3, type=float,
                        default=(-0.27, -0.234, 0.989), metavar=("X", "Y", "Z"),
                        help="Left arm-base camera local position in metres.")
    parser.add_argument("--arm-base-camera-right", nargs=3, type=float,
                        default=(0.27, -0.234, 0.989), metavar=("X", "Y", "Z"),
                        help="Right arm-base camera local position in metres.")
    parser.add_argument("--arm-base-camera-pitch", type=float, default=15.0,
                        help="Upward pitch for both arm-base cameras in degrees.")
    parser.add_argument("--arm-base-camera-local-z-rotation", type=float,
                        default=-90.0,
                        help="Left/right camera rotation about each camera's local Z "
                        "axis in degrees; -90 is clockwise.")
    parser.add_argument("--arm-base-camera-upward", nargs=3, type=float,
                        default=(0.0, 0.25, 0.1724), metavar=("X", "Y", "Z"),
                        help="Upward arm-base camera local position in metres.")
    parser.add_argument("--arm-base-camera-upward-extra-pitch", type=float,
                        default=30.0,
                        help="Additional upward-camera pitch in degrees.")
    parser.add_argument("--camera-width", type=int, default=480,
                        help="Arm-base camera render width in pixels.")
    parser.add_argument("--camera-height", type=int, default=360,
                        help="Arm-base camera render height in pixels.")
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=95,
        metavar="1-100",
        help="External JPEG quality for dataset images (default: 95). 100 is highest quality.",
    )
    parser.add_argument("--print-joints", type=int, default=30, metavar="N",
                        help="Print measured arm joint angles (RADIANS) every N "
                        "loop steps. 0 disables. Default: 30.")
    parser.add_argument("--print-base", type=int, default=0, metavar="N",
                        help="Print measured root/base pose and velocity every N steps. "
                        "0 disables. Default: 0.")
    parser.add_argument("--print-camera-extrinsics", type=int, default=10,
                        metavar="N",
                        help="Print the exact T_world_camera_usd saved to the dataset "
                        "every N frames in each episode. 0 disables. Default: 10.")
    parser.add_argument("--max-steps", type=int, default=0, metavar="N",
                        help="Exit after N update-loop steps. 0 runs until closed.")
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=100,
                        metavar="N", help="Exit after N completed throw episodes.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for tennis-ball throws.")
    parser.add_argument("--physics_dt", type=float, default=0.0333333,
                        help="Physics timestep in seconds. Default: 0.0333333 (30 Hz).")
    parser.add_argument("--ball-distance", type=float, default=4.0,
                        help="Launch-disk center distance in front of base_link (metres).")
    parser.add_argument("--ball-distance-range", type=float, default=1.0,
                        help="Half-length of the uniform launch line along the forward axis.")
    parser.add_argument("--ball-height-range", nargs=2, type=float,
                        default=(0.81, 1.0), metavar=("MIN", "MAX"),
                        help="Launch world-Z range in metres.")
    parser.add_argument("--ball-target-height", type=float, default=0.7,
                        help="Red-disk/ball-center target world Z in metres.")
    parser.add_argument("--ball-flight-time-range", nargs=2, type=float,
                        default=(1.2, 1.4), metavar=("MIN", "MAX"),
                        help="Random point-to-point flight time in seconds.")
    parser.add_argument("--ball-landing-radius", type=float, default=1.4,
                        help="Landing-point radius around base_link.")
    parser.add_argument("--ball-landing-forward-min", type=float, default=0.3,
                        help="Minimum forward coordinate of the landing point.")
    parser.add_argument("--ball-landing-lateral-max", type=float, default=1.4,
                        help="Maximum absolute lateral coordinate; values up to the landing radius are unrestricted.")
    parser.add_argument("--throw-period", type=float, default=2.0,
                        help="Seconds between automatic rethrows. 0 throws once.")
    parser.add_argument("--hide-throw-markers", action="store_true",
                        help="Hide the throw-point disks and camera-position spheres.")
    parser.add_argument("--arm-start-step", type=int, default=20,
                        help="Episode step at which arm and base movement unlocks; "
                        "the base is locked before this step.")
    parser.add_argument("--arm-move-steps", type=int, default=20,
                        help="Number of steps used to interpolate from home to the IK solution.")
    parser.add_argument("--base-landing-share", type=float, default=0.7,
                        help="Fraction of landing-target X/Y displacement assigned to "
                             "base translation; the arm handles the remainder and all Z. "
                             "Must be in [0, 1]. Default: 0.7.")
    parser.add_argument("--linear-speed", type=float, default=0.25,
                        help="W/S kinematic base speed in metres/second.")
    parser.add_argument("--turn-speed", type=float, default=0.6,
                        help="A/D kinematic base yaw speed in radians/second.")
    parser.add_argument("--scripted-keys", default="",
                        help="Headless/debug held keys, for example 'w' or 'wa'.")
    parser.add_argument("--clean-close", action="store_true",
                        help="Call SimulationApp.close() on exit (default: fast os._exit).")
    parser.add_argument("--save-lerobot-dataset", action="store_true",
                        help="Record synchronized data; DynamicVLA-only mode writes LeRobot v2.1.")
    parser.add_argument(
        "--dynamicvla-only-dataset",
        action="store_true",
        help=(
            "Save only the features required by DynamicVLA and encode one "
            "independent MP4 per camera per episode using the "
            "LeRobot v2.1 layout, JSONL metadata, and per-episode statistics."
        ),
    )
    parser.add_argument(
        "--lerobot-dataset-dir",
        default=str(Path(__file__).resolve().parents[1] / "data/tennis_0826_lerobot_v2"),
        help="Exact output directory (v2.1 in DynamicVLA-only mode; v3 for full diagnostics).",
    )
    parser.add_argument("--dataset-num-episodes", type=int, default=50,
                        help="Number of complete throw episodes to record. Default: 50.")
    parser.add_argument("--throw-analysis-interval", type=int, default=500,
                        help="Rewrite cumulative throw analysis every N saved episodes. "
                             "Default: 500.")
    parser.add_argument("--action-lookahead-steps", type=int, default=2,
                        help="Relabel action[t] as state[t+N] before saving; tail frames "
                             "use the final state. Default: 2.")
    parser.add_argument("--dataset-task", default="Catch the green tennis ball",
                        help="Natural-language task stored in LeRobot metadata.")
    parser.add_argument("--overwrite-lerobot-dataset", action="store_true",
                        help="Replace an existing dataset output directory.")
    parser.add_argument("--ball-mask-prim-path", default=PRIM_BALL,
                        help="Prim path whose instance IDs form the tennis-ball class.")
    parser.add_argument(
        "--ground-mask-z",
        type=float,
        default=0.0,
        help="Ground-plane world Z used to derive the ground mask from depth.",
    )
    parser.add_argument(
        "--ground-mask-z-tolerance",
        type=float,
        default=0.05,
        help="Maximum absolute world-Z error for ground pixels in metres. Default: 0.05.",
    )
    parser.add_argument("--record-depth", action="store_true",
                        help="Record per-camera distance-to-image-plane depth in metres.")
    parser.add_argument("--inference-control", action="store_true",
                        help="Drive the base and arm exclusively from 9-D VLA actions over ZeroMQ.")
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--zmq-observation-port", type=int, default=5563)
    parser.add_argument("--zmq-action-port", type=int, default=5564)
    parser.add_argument("--inference-frame-offsets", nargs="+", type=int,
                        default=tuple(range(0, 17, 2)), metavar="FRAME",
                        help="Absolute episode frame numbers used as input. "
                        "The default matches isaacsim_evaluate_tennis.py: "
                        "0,2,4,...,16 (nine RGB frames).")
    parser.add_argument("--inference-replan-steps", type=int, default=20,
                        help="Request a new action chunk after this many executed actions.")
    parser.add_argument(
        "--inference-once-per-episode", action="store_true",
        help="Publish only the first observation triplet in each episode; after the "
             "returned action chunk is exhausted, hold the last action.",
    )
    parser.add_argument("--inference-timeout", type=float, default=30.0,
                        help="Seconds to wait for each inference response.")
    parser.add_argument("--catch-radius", type=float, default=0.13,
                        help="Maximum ball-to-ring-center distance counted as a catch (m).")
    parser.add_argument(
        "--inference-save-inputs", action="store_true",
        help="Save a 3x3 input grid and observation-camera episode video.",
    )
    parser.add_argument(
        "--inference-input-dir",
        default=str(Path(__file__).resolve().parents[1] / "data/inference_inputs"),
        help=(
            "Directory for inference input grids and observation-camera videos. "
            "Its contents are cleared at startup when --inference-save-inputs is enabled."
        ),
    )
    parser.add_argument(
        "--observation-video-quality",
        type=int,
        default=9,
        metavar="1-10",
        help="H.264 quality for the obs_camera MP4 (1-10, default: 9).",
    )
    parser.add_argument(
        "--observation-video-width",
        type=int,
        default=960,
        help="obs_camera MP4 render width in pixels (default: 960).",
    )
    parser.add_argument(
        "--observation-video-height",
        type=int,
        default=720,
        help="obs_camera MP4 render height in pixels (default: 720).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
# AppLauncher stores --viz in visualizer (list[str] or None). Keep a normalized
# name for headless camera synchronization checks.
args_cli.viz = ",".join(args_cli.visualizer) if args_cli.visualizer else "none"
# Dataset recording and inference both require rendered camera observations.
# Enable the camera extensions automatically so their commands do not need to
# repeat AppLauncher's --enable_cameras flag.
if (
    args_cli.viz == "none"
    or args_cli.save_lerobot_dataset
    or args_cli.inference_control
):
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after AppLauncher (Isaac Lab convention) ────────────────────────
import carb  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

# ── Scene open / calibration / view helpers ─────────────────────────────────
def _validate_inputs(args):
    if not os.path.isfile(args.scene_usd):
        raise SystemExit(f"[ERROR] scene usda not found: {args.scene_usd}")
    if not os.path.isdir(SCENE_ASSET_ROOT):
        raise SystemExit(
            f"[ERROR] asset root not mounted: {SCENE_ASSET_ROOT} "
            "(the scene usda references this mount via absolute paths)."
        )
    if not os.path.isfile(BALL_ASSET_USD):
        raise SystemExit(f"[ERROR] tennis ball USD not found: {BALL_ASSET_USD}")
    if float(args.plate_size[0]) <= 0.0 or float(args.plate_size[1]) <= 0.0:
        raise SystemExit("[ERROR] --plate-size values must be positive")
    if int(args.max_steps) < 0:
        raise SystemExit("[ERROR] --max-steps must be non-negative")
    if int(args.max_episodes) <= 0:
        raise SystemExit("[ERROR] --max-episodes must be positive")
    if int(args.print_camera_extrinsics) < 0:
        raise SystemExit("[ERROR] --print-camera-extrinsics must be non-negative")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise SystemExit("[ERROR] --camera-width and --camera-height must be positive")
    if not 1 <= args.image_jpeg_quality <= 100:
        raise SystemExit("[ERROR] --image-jpeg-quality must be in [1, 100]")
    if not 1 <= args.observation_video_quality <= 10:
        raise SystemExit("[ERROR] --observation-video-quality must be in [1, 10]")
    if args.observation_video_width <= 0 or args.observation_video_height <= 0:
        raise SystemExit(
            "[ERROR] --observation-video-width and --observation-video-height "
            "must be positive"
        )
    if args.ground_mask_z_tolerance <= 0.0:
        raise SystemExit("[ERROR] --ground-mask-z-tolerance must be positive")
    if args.ball_distance <= 0.0 or args.ball_distance_range < 0.0:
        raise SystemExit("[ERROR] ball distance must be positive and range non-negative")
    if args.ball_distance - args.ball_distance_range <= 0.0:
        raise SystemExit("[ERROR] ball distance range reaches behind base_link")
    for name, values in (("height", args.ball_height_range),
                         ("flight-time", args.ball_flight_time_range)):
        if values[0] <= 0.0 or values[1] < values[0]:
            raise SystemExit(f"[ERROR] invalid ball {name} range: {tuple(values)}")
    if args.ball_landing_forward_min <= 0.0:
        raise SystemExit("[ERROR] --ball-landing-forward-min must be positive")
    if args.ball_landing_lateral_max <= 0.0:
        raise SystemExit("[ERROR] --ball-landing-lateral-max must be positive")
    if args.ball_landing_radius < args.ball_landing_forward_min:
        raise SystemExit("[ERROR] landing radius must cover landing-forward-min")
    if args.ball_target_height <= 0.0:
        raise SystemExit("[ERROR] --ball-target-height must be positive")
    if args.throw_period < 0.0:
        raise SystemExit("[ERROR] --throw-period must be non-negative")
    if args.catch_radius < 0.0:
        raise SystemExit("[ERROR] --catch-radius must be non-negative")
    if args.physics_dt <= 0.0:
        raise SystemExit("[ERROR] --physics_dt must be positive")
    if args.arm_start_step < 0 or args.arm_move_steps <= 0:
        raise SystemExit("[ERROR] arm-start-step must be >= 0 and arm-move-steps > 0")
    if not 0.0 <= args.base_landing_share <= 1.0:
        raise SystemExit("[ERROR] --base-landing-share must be in [0, 1]")
    if args.dataset_num_episodes <= 0:
        raise SystemExit("[ERROR] --dataset-num-episodes must be positive")
    if args.throw_analysis_interval <= 0:
        raise SystemExit("[ERROR] --throw-analysis-interval must be positive")
    if not args.dataset_task.strip():
        raise SystemExit("[ERROR] --dataset-task must not be empty")
    if args.save_lerobot_dataset and args.static:
        raise SystemExit("[ERROR] dataset recording is incompatible with --static")
    if args.save_lerobot_dataset and args.throw_period <= 0.0:
        raise SystemExit("[ERROR] dataset recording requires --throw-period > 0")
    if args.action_lookahead_steps < 0:
        raise SystemExit("[ERROR] --action-lookahead-steps must be non-negative")
    if args.inference_control and not args.enable_cameras:
        raise SystemExit("[ERROR] --inference-control requires --enable_cameras")
    if args.inference_control and args.static:
        raise SystemExit("[ERROR] --inference-control is incompatible with --static")
    frame_indices = tuple(int(value) for value in args.inference_frame_offsets)
    if any(value < 0 for value in frame_indices):
        raise SystemExit("[ERROR] --inference-frame-offsets must be non-negative")
    if len(set(frame_indices)) != len(frame_indices):
        raise SystemExit("[ERROR] --inference-frame-offsets must be distinct")
    if args.inference_replan_steps <= 0 or args.inference_timeout <= 0.0:
        raise SystemExit("[ERROR] inference replan steps and timeout must be positive")
    if not str(args.inference_input_dir).strip():
        raise SystemExit("[ERROR] --inference-input-dir must not be empty")


def _clear_inference_input_dir(args):
    """Start each input-recording run with an empty output directory."""
    if not args.inference_save_inputs:
        return

    output_dir = Path(args.inference_input_dir).expanduser().resolve()
    if output_dir in Path(__file__).resolve().parents:
        raise SystemExit(
            f"[ERROR] refusing to clear protected directory: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in output_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f"[INFO] Cleared inference input directory: {output_dir}", flush=True)


def _open_scene(args):
    """Open the prebuilt usda (first stage: AppLauncher creates none) and validate."""
    import omni.usd

    t0 = time.perf_counter()
    omni.usd.get_context().open_stage(str(Path(args.scene_usd).resolve()))
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise SystemExit("[ERROR] open_stage failed; no stage available.")
    elapsed = time.perf_counter() - t0
    print(f"[INFO] Scene opened in {elapsed:.1f}s: {args.scene_usd}", flush=True)

    for prim_path, label in (
        (PRIM_ENV, "3DGS environment"),
        (PRIM_ROBOT, "CatchIt robot"),
        (PRIM_PLATE, "drive plate"),
        (PRIM_SOURCE_BALL, "source object rigid body"),
        ("/physicsScene", "physics scene"),
    ):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise SystemExit(f"[ERROR] prim '{prim_path}' ({label}) missing in scene usda.")
        print(f"[INFO] prim OK: {prim_path} ({label})", flush=True)

    if args.skip_splat:
        gauss_prim = stage.GetPrimAtPath(PRIM_ENV)
        gauss_prim.GetReferences().ClearReferences()
        print("[INFO] --skip-splat: cleared 3DGS reference (xform shell kept).", flush=True)
    return stage


def _apply_xform(stage, prim_path, translate=None, rotate_deg=None, scale=None):
    """Apply runtime xform override, reusing existing ops (authored order kept)."""
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    if not xform:
        raise SystemExit(f"[ERROR] prim not found for calibration: {prim_path}")
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    if translate is not None:
        op = ops.get("xformOp:translate") or xform.AddTranslateOp()
        op.Set(Gf.Vec3d(*[float(v) for v in translate]))
    if rotate_deg is not None:
        # rotateXYZ is authored in degrees, same as the build script
        op = ops.get("xformOp:rotateXYZ") or xform.AddRotateXYZOp()
        op.Set(Gf.Vec3f(*[float(v) for v in rotate_deg]))
    if scale is not None:
        op = ops.get("xformOp:scale") or xform.AddScaleOp()
        op.Set(Gf.Vec3f(*[float(v) for v in scale]))


def _apply_plate_looks(stage, args):
    plate = UsdGeom.Cube(stage.GetPrimAtPath(PRIM_PLATE))
    if not plate:
        raise SystemExit(f"[ERROR] drive plate prim missing: {PRIM_PLATE}")
    UsdGeom.Imageable(plate.GetPrim()).CreateVisibilityAttr(
        "inherited" if args.plate_visible else "invisible"
    )
    rgb = tuple(max(0.0, min(1.0, float(v))) for v in args.plate_color)
    alpha = max(0.0, min(1.0, float(args.plate_opacity)))
    plate.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    plate.CreateDisplayOpacityAttr([alpha])
    # The plate also binds a UsdPreviewSurface; keep its inputs in sync.
    shader = UsdShade.Shader.Get(stage, Sdf.Path(PLATE_SHADER))
    if shader:
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*rgb))
        shader.GetInput("opacity").Set(alpha)


def _apply_net_looks(stage, args):
    """Make the solid catching-net disk transparent while retaining the rim."""
    alpha = max(0.0, min(1.0, float(args.net_opacity)))
    visual_root = stage.GetPrimAtPath(NET_VISUAL_ROOT_PATH)
    ring_prim = stage.GetPrimAtPath(RING_TORUS_VISUAL_PATH)
    if not visual_root.IsValid():
        raise SystemExit(
            f"[ERROR] catching-net visual root missing: {NET_VISUAL_ROOT_PATH}"
        )
    # The imported ring visual is instanceable. De-instance only this small
    # visual subtree so its child opacity can be overridden in the main stage.
    if visual_root.IsInstance():
        visual_root.SetInstanceable(False)
    net_prim = stage.GetPrimAtPath(NET_VISUAL_PATH)
    if not net_prim.IsValid() or not net_prim.IsA(UsdGeom.Gprim):
        raise SystemExit(
            f"[ERROR] catching-net visual prim missing: {NET_VISUAL_PATH}"
        )
    if not ring_prim.IsValid():
        raise SystemExit(f"[ERROR] catching-ring visual missing: {RING_TORUS_VISUAL_PATH}")

    net = UsdGeom.Gprim(net_prim)
    net.CreateDisplayOpacityAttr([alpha])
    if alpha <= 0.0:
        net.MakeInvisible()
    else:
        net.MakeVisible()
    print(
        f"[INFO] Catching net opacity={alpha:.2f}; "
        "blue outer ring and collision geometry retained.",
        flush=True,
    )


def _apply_calibration(args, stage):
    _apply_xform(stage, PRIM_ENV, translate=args.env_translate,
                 rotate_deg=args.env_rotate)
    _apply_xform(stage, PRIM_ROBOT, translate=args.robot_translate,
                 rotate_deg=args.robot_rotate)
    _apply_xform(stage, PRIM_PLATE, translate=args.plate_translate,
                 rotate_deg=args.plate_rotate)
    _apply_xform(stage, PRIM_PLATE,
                 scale=(args.plate_size[0], args.plate_size[1], args.plate_thickness))
    _apply_plate_looks(stage, args)
    # The scene's original object has no renderable geometry/material. Disable
    # it and spawn the same complete tennis-ball USD used by evaluate_0804.
    stage.GetPrimAtPath(PRIM_SOURCE_BALL).SetActive(False)
    print(f"[INFO] Disabled physics-only source object: {PRIM_SOURCE_BALL}", flush=True)
    print(
        "[INFO] Calibration applied: env=%s/%sdeg robot=%s/%sdeg "
        "plate=%s size=%sx%st%.3f opacity=%.2f visible=%s"
        % (tuple(args.env_translate), tuple(args.env_rotate),
           tuple(args.robot_translate), tuple(args.robot_rotate),
           tuple(args.plate_translate), args.plate_size[0], args.plate_size[1],
           args.plate_thickness, args.plate_opacity, args.plate_visible),
        flush=True,
    )


def _set_initial_view(args, stage):
    """Position the perspective camera (--camera-view) or frame the robot."""
    if getattr(args, "headless", False):
        return
    try:
        if args.camera_view is not None:
            cam = UsdGeom.Camera.Get(stage, Sdf.Path(PERSP_CAM))
            if cam:
                gf_cam = Gf.Camera()
                gf_cam.SetPosition(Gf.Vec3d(*args.camera_view[:3]))
                gf_cam.SetLookAtPoint(Gf.Vec3d(*args.camera_view[3:]))
                cam.SetFromCamera(gf_cam, Usd.TimeCode.Default())
                print(f"[INFO] Perspective camera set: eye={tuple(args.camera_view[:3])} "
                      f"target={tuple(args.camera_view[3:])}", flush=True)
                return
            print("[WARN] Perspective camera prim not found; skipped.", flush=True)
            return
        # Default: frame the robot in the active viewport
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp is not None:
            vp_util.frame_viewport_prims(vp, [PRIM_ROBOT])
            print("[INFO] Viewport framed on the CatchIt robot.", flush=True)
    except Exception as exc:  # viewport APIs are optional / window may not be ready
        print(f"[WARN] view setup skipped ({type(exc).__name__}): {exc}", flush=True)


def _set_arm_base_camera_positions(args, stage):
    """Set all arm-base camera positions and their relative upward pitch."""
    parent = "/World/CatchIt/base_link/arm_base"
    reference_prim = stage.GetPrimAtPath(f"{parent}/arm_base_camera_left")
    reference_ops = UsdGeom.Xformable(reference_prim).GetOrderedXformOps()
    reference_transform_op = next(
        op for op in reference_ops
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    )
    base_camera_rotation = np.asarray(
        Gf.Matrix4d(reference_transform_op.Get()), dtype=np.float64
    )[:3, :3].copy()
    left_right_local_z = float(args.arm_base_camera_local_z_rotation)
    specs = (
        ("arm_base_camera_left", args.arm_base_camera_left,
         float(args.arm_base_camera_pitch), left_right_local_z),
        ("arm_base_camera_right", args.arm_base_camera_right,
         float(args.arm_base_camera_pitch), left_right_local_z),
        ("arm_base_camera_upward", args.arm_base_camera_upward,
         float(args.arm_base_camera_pitch)
         + float(args.arm_base_camera_upward_extra_pitch), left_right_local_z),
        ("obs_camera", (-0.7, -1.5, 2.0),
         -15.0, left_right_local_z),
    )
    for name, position, camera_pitch_deg, local_z_deg in specs:
        path = f"{parent}/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Arm-base camera prim not found: {path}")
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        transform_ops = [
            op for op in ordered_ops
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform
        ]
        translate_ops = [
            op for op in ordered_ops
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        position_vec = Gf.Vec3d(*[float(v) for v in position])
        if transform_ops:
            # The imported cameras store rotation and translation in one
            # matrix. Replace its translation instead of adding another op.
            transform = Gf.Matrix4d(transform_ops[0].Get())
            pitch = math.radians(camera_pitch_deg)
            c, s = math.cos(pitch), math.sin(pitch)
            # USD/Gf uses row-vector transforms. Pre-multiplying by this
            # local-X rotation pitches the optical -Z axis upward.
            pitch_rotation = np.array(
                [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
                dtype=np.float64,
            )
            local_z = math.radians(local_z_deg)
            cz, sz = math.cos(local_z), math.sin(local_z)
            # Apply the requested rotation in the camera's own Z axis. With
            # the row-vector convention, -90 degrees is clockwise.
            local_z_rotation = np.array(
                [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            camera_rotation = (
                local_z_rotation @ pitch_rotation @ base_camera_rotation
            )
            for row in range(3):
                for column in range(3):
                    transform[row, column] = float(camera_rotation[row, column])
            transform.SetTranslateOnly(position_vec)
            transform_ops[0].Set(transform)
            # Remove a stale translate op authored by an older launcher run.
            for op in translate_ops:
                prim.RemoveProperty(op.GetOpName())
        elif translate_ops:
            translate_ops[0].Set(position_vec)
        else:
            xformable.AddTranslateOp().Set(position_vec)
        local_transform = xformable.GetLocalTransformation()
        if isinstance(local_transform, tuple):
            local_transform = local_transform[0]
        actual_position = tuple(float(v) for v in local_transform.ExtractTranslation())
        if not np.allclose(actual_position, position, atol=1.0e-6):
            raise RuntimeError(
                f"Camera local-position verification failed: {path} "
                f"expected={tuple(position)} actual={actual_position}"
            )
        print(
            f"[INFO] Camera local position verified: {path}={actual_position}; "
            f"upward_pitch={camera_pitch_deg:.1f}deg; "
            f"local_z_rotation={local_z_deg:.1f}deg",
            flush=True,
        )


def _create_extra_arm_base_camera(stage):
    """Inherit the complete left-camera configuration, then override the pose."""
    source = stage.GetPrimAtPath(ARM_BASE_CAMERA_PATHS[0])
    if not source.IsValid() or not source.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Left camera prim not found: {ARM_BASE_CAMERA_PATHS[0]}")
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(EXTRA_ARM_BASE_CAMERA_PATH))
    target = camera.GetPrim()
    # Preserve composed attributes, metadata, API schemas, relationships and
    # time samples. _set_arm_base_camera_positions authors the local pose.
    target.GetReferences().AddInternalReference(source.GetPath())
    target.SetDisplayName("obs_camera")
    return EXTRA_ARM_BASE_CAMERA_PATH


class _ObservationVideoRecorder:
    """Stream obs_camera to one MP4 per throw when input saving is enabled."""

    def __init__(self, args):
        self.enabled = bool(args.inference_save_inputs)
        self.writer = None
        self.request_dirs = set()
        self.render_product = None
        self.annotator = None
        if not self.enabled:
            return
        import imageio.v2 as imageio

        self.imageio = imageio
        self.output_dir = Path(args.inference_input_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = 1.0 / float(args.physics_dt)
        self.quality = int(args.observation_video_quality)
        self.resolution = (
            int(args.observation_video_width),
            int(args.observation_video_height),
        )
        self.run_id = f"{time.time_ns()}"
        self.render_product = rep.create.render_product(
            EXTRA_ARM_BASE_CAMERA_PATH,
            self.resolution,
        )
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.render_product])

    def capture(self, episode):
        if not self.enabled:
            return
        rgba = np.asarray(self.annotator.get_data())
        if rgba.size == 0:
            return
        if rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError(f"Unexpected observation camera RGB shape: {rgba.shape}")
        rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
        if self.writer is None:
            episode_dir = self.output_dir / f"episode-{episode:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            self.path = episode_dir / (
                f"observation_camera_{self.run_id}_episode-{episode:06d}.mp4"
            )
            self.writer = self.imageio.get_writer(
                str(self.path), format="FFMPEG", mode="I", fps=self.fps,
                codec="libx264", macro_block_size=1, pixelformat="yuv444p",
                quality=self.quality,
            )
        self.writer.append_data(rgb)

    def save_episode(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        destinations = sorted(self.request_dirs)
        if destinations:
            destination = destinations[0] / self.path.name
            self.path.replace(destination)
            self.path = destination
            for directory in destinations[1:]:
                shutil.copy2(self.path, directory / self.path.name)
        self.request_dirs.clear()
        print(
            f"[VIDEO] obs_camera: {self.path} "
            f"(resolution={self.resolution[0]}x{self.resolution[1]}, "
            f"H.264 quality={self.quality}/10)",
            flush=True,
        )

    def close(self):
        try:
            self.save_episode()
        finally:
            if self.annotator is not None:
                self.annotator.detach([self.render_product])
                self.render_product.destroy()


def _create_arm_base_camera_render_products(stage, args):
    """Create fixed-resolution render products for all three robot cameras."""
    resolution = (int(args.camera_width), int(args.camera_height))
    render_products = []
    for camera_path in ARM_BASE_CAMERA_PATHS:
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Arm-base camera prim not found: {camera_path}")
        render_products.append(rep.create.render_product(camera_path, resolution))
    print(
        f"[INFO] Camera render products created: count={len(render_products)}, "
        f"resolution={resolution[0]}x{resolution[1]}.",
        flush=True,
    )
    return tuple(render_products)


def _apply_semantic_class(stage, prim_path, instance_name, class_name):
    """Apply one stable Replicator semantic class to a prim subtree."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Segmentation prim not found: {prim_path}")
    api = Semantics.SemanticsAPI.Apply(prim, instance_name)
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set(class_name)


def _label_segmentation_classes(stage, args):
    """Label visible instances used by the four-class dataset mask."""
    _apply_semantic_class(
        stage,
        args.ball_mask_prim_path,
        "DatasetTennisBall",
        "tennis_ball",
    )
    for index, prim_path in enumerate(GROUND_MASK_PRIM_PATHS):
        _apply_semantic_class(
            stage, prim_path, f"DatasetGround{index}", "ground"
        )
    for index, prim_path in enumerate(ARM_MASK_PRIM_PATHS):
        _apply_semantic_class(
            stage, prim_path, f"DatasetRobotArm{index}", "robot_arm"
        )
    print(
        "[DATASET] Applied semantic classes: "
        f"tennis_ball={args.ball_mask_prim_path}; "
        f"ground={GROUND_MASK_PRIM_PATHS}; robot_arm={ARM_MASK_PRIM_PATHS}",
        flush=True,
    )


class _LeRobotRecorder:
    """Record DynamicVLA data in v2.1, or full diagnostic captures in v3."""

    CAMERA_NAMES = ("left", "right", "upper")

    def __init__(self, args, render_products, stage):
        self.args = args
        self.stage = stage
        self.enabled = bool(args.save_lerobot_dataset)
        self.dataset = None
        self.saved_episodes = 0
        self.rgb_annotators = []
        self.seg_annotators = []
        self.depth_annotators = []
        self.mask_class_nonempty_frames = {
            camera_name: {
                class_name: 0
                for class_name in SEGMENTATION_CLASS_IDS
                if class_name != "background"
            }
            for camera_name in self.CAMERA_NAMES
        }
        self._warned_missing_instance_classes = set()
        self.camera_intrinsics = {}
        self.camera_calibration_metadata = {}
        self.prim_world_from_cameras = {}
        self.prim_world_base_position = None
        self.initial_runtime_base_position = None
        self._last_world_from_cameras = {}
        self.output_dir = None
        self.throw_analysis_dir = None
        self.throw_records = []
        self.current_throw = None
        self._last_throw_analysis_count = 0
        if not self.enabled:
            return

        def diagnostic_dataset_type():
            """Load the legacy image writer only for full diagnostic captures."""
            import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from PIL import Image as PILImage

            # Keep image files external to Parquet.  Both LeRobot 0.3.x and newer
            # releases call this module-level helper before writing episode data.
            # ``add_frame`` writes each image to the standard image path.
            lerobot_dataset_module.embed_images = lambda dataset: dataset

            class _PathOnlyImageDataset(LeRobotDataset):
                """Keep image files on disk instead of embedding their bytes in Parquet.

                LeRobot's default writer calls ``embed_images`` before writing each
                episode.  The image files have already been written by ``add_frame``;
                storing only their paths keeps the standard dataset layout and Image
                feature semantics while making the Parquet files small.
                """

                _preserve_image_files = True
                _jpeg_quality = int(args.image_jpeg_quality)
                _encode_camera_frames_as_video = bool(args.dynamicvla_only_dataset)

                def _get_image_file_path(self, episode_index, image_key, frame_index):
                    """Use PNG video sources or JPEG persistent image files."""
                    path = super()._get_image_file_path(
                        episode_index=episode_index,
                        image_key=image_key,
                        frame_index=frame_index,
                    )
                    if self._encode_camera_frames_as_video:
                        return path
                    return path.with_suffix(".jpg")

                def _save_image(self, image, fpath):
                    """Write lossless temporary video frames or persistent JPEGs."""
                    if hasattr(image, "detach"):
                        image = image.detach().cpu().numpy()
                    if isinstance(image, PILImage.Image):
                        pil_image = image.convert("RGB")
                    else:
                        array = np.asarray(image)
                        if array.ndim == 3 and array.shape[0] in (1, 3):
                            array = np.moveaxis(array, 0, -1)
                        if np.issubdtype(array.dtype, np.floating):
                            if float(array.max(initial=0.0)) <= 1.0:
                                array = array * 255.0
                            array = np.clip(array, 0.0, 255.0)
                        pil_image = PILImage.fromarray(array.astype(np.uint8)).convert("RGB")
                    if Path(fpath).suffix.lower() == ".png":
                        pil_image.save(fpath, format="PNG")
                    else:
                        pil_image.save(
                            fpath,
                            format="JPEG",
                            quality=self._jpeg_quality,
                            subsampling=0,
                            optimize=True,
                        )

                def clear_episode_buffer(self, delete_images=True):
                    """Reset the buffer without deleting external image files.

                    LeRobot 0.3.x only deletes images when an async writer exists,
                    while newer versions expose an explicit ``delete_images`` flag.
                    Handle both APIs so the recorder works with either IsaacLab
                    environment.
                    """
                    if self._preserve_image_files:
                        try:
                            return super().clear_episode_buffer(delete_images=False)
                        except TypeError:
                            writer = self.image_writer
                            self.image_writer = None
                            try:
                                return super().clear_episode_buffer()
                            finally:
                                self.image_writer = writer
                    try:
                        return super().clear_episode_buffer(delete_images=delete_images)
                    except TypeError:
                        return super().clear_episode_buffer()

                def _save_episode_data(self, episode_buffer):
                    """Write exactly one episode per Parquet file on newer LeRobot."""
                    save_episode_data = super()._save_episode_data
                    metadata = save_episode_data(episode_buffer)
                    # The stock writer keeps appending episodes until its size limit.
                    # Marking it closed makes the next episode advance file_index.
                    self._close_writer()
                    self._writer_closed_for_reading = True
                    return metadata

                def _save_episode_video(self, video_key, episode_index):
                    """Encode one independent MP4 per camera and episode.

                    LeRobot v3 normally concatenates several episodes into a large
                    video shard.  DynamicVLA datasets use the same visible layout as
                    ``data/DataSet_Lerobot`` instead: one file named after the global
                    episode index under each camera directory.
                    """
                    from lerobot.datasets.utils import write_info
                    from lerobot.datasets.video_utils import (
                        get_video_duration_in_s,
                        get_video_info,
                    )

                    temporary_video_path = self._encode_temporary_episode_video(
                        video_key, episode_index
                    )
                    chunk_index = int(episode_index) // int(self.meta.chunks_size)
                    output_path = self.root / self.meta.video_path.format(
                        video_key=video_key,
                        chunk_index=chunk_index,
                        file_index=int(episode_index),
                        episode_chunk=chunk_index,
                        episode_index=int(episode_index),
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temporary_video_path), str(output_path))
                    shutil.rmtree(str(temporary_video_path.parent))

                    duration_s = get_video_duration_in_s(output_path)
                    if episode_index == 0:
                        self.meta.info["features"][video_key]["info"] = (
                            get_video_info(output_path)
                        )
                        write_info(self.meta.info, self.meta.root)

                    return {
                        "episode_index": int(episode_index),
                        f"videos/{video_key}/chunk_index": chunk_index,
                        f"videos/{video_key}/file_index": int(episode_index),
                        f"videos/{video_key}/from_timestamp": 0.0,
                        f"videos/{video_key}/to_timestamp": float(duration_s),
                    }

                def _save_episode_table(self, episode_buffer, episode_index):
                    import datasets
                    from lerobot.datasets.utils import hf_transform_to_torch

                    episode_dict = {key: episode_buffer[key] for key in self.hf_features}
                    episode_dataset = datasets.Dataset.from_dict(
                        episode_dict, features=self.hf_features, split="train"
                    )
                    self.hf_dataset = datasets.concatenate_datasets(
                        [self.hf_dataset, episode_dataset]
                    )
                    self.hf_dataset.set_transform(hf_transform_to_torch)
                    episode_path = self.root / self.meta.get_data_file_path(
                        ep_index=episode_index
                    )
                    episode_path.parent.mkdir(parents=True, exist_ok=True)
                    episode_dataset.to_parquet(episode_path)
            return _PathOnlyImageDataset

        output_dir = Path(args.lerobot_dataset_dir).expanduser().resolve()
        if output_dir == Path(output_dir.anchor):
            raise RuntimeError(f"Refusing unsafe dataset path: {output_dir}")
        if output_dir.exists():
            if not args.overwrite_lerobot_dataset:
                raise FileExistsError(
                    f"LeRobot output already exists: {output_dir}; pass "
                    "--overwrite-lerobot-dataset to replace it"
                )
            shutil.rmtree(output_dir)

        height, width = int(args.camera_height), int(args.camera_width)
        minimal = bool(args.dynamicvla_only_dataset)
        self.output_dir = output_dir
        if not minimal:
            self._initialize_camera_calibration(stage, width, height)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (15,),
                "names": (
                    "base_x_m", "base_y_m", "base_yaw_rad",
                    "joint_1_pos_rad", "joint_2_pos_rad", "joint_3_pos_rad",
                    "joint_4_pos_rad", "joint_5_pos_rad", "joint_6_pos_rad",
                    "link6_x_world_m", "link6_y_world_m", "link6_z_world_m",
                    "link6_roll_world_rad", "link6_pitch_world_rad",
                    "link6_yaw_world_rad",
                ),
            },
            "action": {
                "dtype": "float32",
                "shape": (15,),
                "names": (
                    "base_target_x_m", "base_target_y_m", "base_target_yaw_rad",
                    "joint_1_target_rad", "joint_2_target_rad", "joint_3_target_rad",
                    "joint_4_target_rad", "joint_5_target_rad", "joint_6_target_rad",
                    "link6_target_x_world_m", "link6_target_y_world_m",
                    "link6_target_z_world_m", "link6_target_roll_world_rad",
                    "link6_target_pitch_world_rad", "link6_target_yaw_world_rad",
                ),
            },
        }
        if not minimal:
            features["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (6,),
                "names": (
                    "ball_pos_x_world_m", "ball_pos_y_world_m", "ball_pos_z_world_m",
                    "ball_vel_x_world_m_per_s", "ball_vel_y_world_m_per_s",
                    "ball_vel_z_world_m_per_s",
                ),
            }
        for camera_name in self.CAMERA_NAMES:
            if not minimal:
                features[f"observation.camera_extrinsics.{camera_name}"] = {
                    "dtype": "float32",
                    "shape": (16,),
                    # Row-major USD-camera-to-world transform.  See the calibration
                    # metadata for the OpenCV-optical to USD axis conversion.
                    "names": tuple(
                        f"T_world_camera_{row}{column}"
                        for row in range(4)
                        for column in range(4)
                    ),
                }
            prefixes = ("observation.images",) if minimal else (
                "observation.images", "observation.segmentation"
            )
            for prefix in prefixes:
                features[f"{prefix}.{camera_name}"] = {
                    # DynamicVLA RGB is encoded into independent per-episode
                    # MP4 files. Full diagnostic datasets retain external images.
                    "dtype": (
                        "video"
                        if minimal and prefix == "observation.images"
                        else "image"
                    ),
                    "shape": (height, width, 3),
                    "names": ("height", "width", "channels"),
                }
            if args.record_depth and not minimal:
                features[f"observation.depth.{camera_name}"] = {
                    "dtype": "float32",
                    "shape": (height, width, 1),
                    "names": ("height", "width", "channel"),
                }

        if minimal:
            from lerobot_v21_writer import LeRobotV21Writer

            self.dataset = LeRobotV21Writer(
                root=output_dir,
                fps=int(round(1.0 / float(args.physics_dt))),
                robot_type="catchit_tennis_sim",
                features=features,
            )
        else:
            self.dataset = diagnostic_dataset_type().create(
                repo_id="local/tennis_0826_lerobot_v3",
                root=output_dir,
                fps=int(round(1.0 / float(args.physics_dt))),
                robot_type="catchit_tennis_sim",
                features=features,
                use_videos=False,
                video_backend="pyav",
                image_writer_threads=0,
            )
        self.throw_analysis_dir = output_dir / "analysis"
        self.throw_analysis_dir.mkdir(parents=True, exist_ok=True)
        if minimal:
            print(
                "[DATASET] LeRobot v2.1 DynamicVLA-only recording enabled: left/right/upper "
                "frames are encoded as one independent MP4 per camera per episode; "
                "Parquet stores episode-local timestamps; meta includes tasks, "
                "episodes, and numeric/RGB episode statistics in JSONL.",
                flush=True,
            )
        else:
            print(
                f"[DATASET] RGB images are saved as external JPEG files "
                f"(quality={args.image_jpeg_quality}, subsampling=0); "
                "Parquet stores image paths only (bytes=null); one Parquet per episode.",
                flush=True,
            )
        calibration_path = output_dir / "meta" / "camera_calibration.json"
        if not minimal:
            calibration_path.write_text(
            json.dumps(
                {
                    "camera_model": "pinhole",
                    "image_width": width,
                    "image_height": height,
                    "intrinsic_convention": (
                        "K maps OpenCV optical coordinates (+X right, +Y down, "
                        "+Z forward) to pixels"
                    ),
                    "extrinsic_convention": (
                        "Per-frame T_world_camera_usd remains in each data Parquet; "
                        "the USD camera frame is +X right, +Y up, -Z forward"
                    ),
                    "depth_backprojection": (
                        "p_world = T_world_camera_usd @ diag(1,-1,-1,1) @ "
                        "[depth*(u-cx)/fx, depth*(v-cy)/fy, depth, 1]"
                    ),
                    "cameras": self.camera_calibration_metadata,
                },
                indent=2,
            ) + "\n"
            )
            print(f"[DATASET] Fixed camera intrinsics saved once: {calibration_path}", flush=True)
        segmentation_metadata_path = output_dir / "meta" / "segmentation_classes.json"
        if not minimal:
            segmentation_metadata_path.write_text(
            json.dumps(
                {
                    "encoding": "uint8_class_id_repeated_in_lossless_rgb_image",
                    "decode": "class_id = round(decoded_rgb[..., 0] * 255) for float tensors",
                    "class_priority_low_to_high": ["ground", "robot_arm", "tennis_ball"],
                    "classes": {
                        "0": {
                            "name": "background",
                            "display_color_rgb": [0, 0, 0],
                        },
                        "1": {
                            "name": "tennis_ball",
                            "display_color_rgb": [0, 255, 0],
                            "prim_paths": [args.ball_mask_prim_path],
                            "source": "instance_segmentation_fast",
                        },
                        "2": {
                            "name": "ground",
                            "display_color_rgb": [210, 160, 60],
                            "prim_paths": list(GROUND_MASK_PRIM_PATHS),
                            "source": "instance segmentation plus depth backprojection",
                            "world_z_m": float(args.ground_mask_z),
                            "world_z_tolerance_m": float(args.ground_mask_z_tolerance),
                        },
                        "3": {
                            "name": "robot_arm",
                            "display_color_rgb": [0, 128, 255],
                            "prim_paths": list(ARM_MASK_PRIM_PATHS),
                            "includes_end_effector_ring": True,
                            "source": "instance_segmentation_fast",
                        },
                    },
                },
                indent=2,
            ) + "\n"
            )
            print(
                f"[DATASET] Segmentation class metadata saved: "
                f"{segmentation_metadata_path}",
                flush=True,
            )
        for render_product in render_products:
            rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb.attach([render_product])
            self.rgb_annotators.append(rgb)
            if not minimal:
                seg = rep.AnnotatorRegistry.get_annotator(
                    "instance_segmentation_fast",
                    init_params={"colorize": False},
                    device="cpu",
                )
                seg.attach([render_product])
                self.seg_annotators.append(seg)
                # Ground belongs to the 3DGS scene rather than a separable visible
                # mesh, so depth is required to derive its world-Z mask even when
                # the caller does not request a stored observation.depth feature.
                depth = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane", device="cpu"
                )
                depth.attach([render_product])
                self.depth_annotators.append(depth)
        print(
            f"[DATASET] LeRobot {self.dataset.meta.info['codebase_version']} "
            f"recording enabled: root={output_dir}; "
            f"episodes={args.dataset_num_episodes}; cameras={self.CAMERA_NAMES}; "
            f"dynamicvla_only={minimal}",
            flush=True,
        )

    def _initialize_camera_calibration(self, stage, width, height):
        """Cache K and fixed USD-camera transforms using column-vector matrices."""
        xform_cache = UsdGeom.XformCache()
        base_prim = stage.GetPrimAtPath("/World/CatchIt/base_link")
        if not base_prim.IsValid():
            raise RuntimeError("Robot base prim not found: /World/CatchIt/base_link")
        # Gf matrices use row vectors; transpose to the conventional column-vector form.
        world_from_base = np.asarray(
            xform_cache.GetLocalToWorldTransform(base_prim), dtype=np.float64
        ).T
        self.prim_world_base_position = world_from_base[:3, 3].copy()
        for camera_name, camera_path in zip(self.CAMERA_NAMES, ARM_BASE_CAMERA_PATHS):
            prim = stage.GetPrimAtPath(camera_path)
            camera = UsdGeom.Camera(prim)
            focal_length = float(camera.GetFocalLengthAttr().Get())
            horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
            vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
            if focal_length <= 0.0 or horizontal_aperture <= 0.0 or vertical_aperture <= 0.0:
                raise RuntimeError(f"Invalid camera calibration attributes: {camera_path}")
            # Omniverse renders pinhole cameras with square pixels and currently
            # derives both pixel focal lengths from the horizontal aperture.  The
            # authored vertical aperture may have a different aspect ratio from
            # the render product, so using it directly would save the wrong fy.
            fx = focal_length * float(width) / horizontal_aperture
            fy = fx
            effective_vertical_aperture = (
                horizontal_aperture * float(height) / float(width)
            )
            self.camera_intrinsics[camera_name] = np.array(
                [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.camera_calibration_metadata[camera_name] = {
                "prim_path": camera_path,
                "K": self.camera_intrinsics[camera_name].tolist(),
                "fx": fx,
                "fy": fy,
                "cx": width / 2.0,
                "cy": height / 2.0,
                "focal_length": focal_length,
                "horizontal_aperture": horizontal_aperture,
                "vertical_aperture": effective_vertical_aperture,
                "authored_vertical_aperture": vertical_aperture,
                "aperture_conform_policy": "match_horizontally",
                "distortion": None,
            }
            world_from_camera = np.asarray(
                xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64
            ).T
            self.prim_world_from_cameras[camera_name] = world_from_camera.copy()
            print(
                f"[DATASET] Camera calibration {camera_name}: "
                f"fx={fx:.3f}, fy={fy:.3f}, cx={width / 2.0:.3f}, "
                f"cy={height / 2.0:.3f}",
                flush=True,
            )

    def _world_from_cameras_for_base_pose(self, base_pose):
        """Use static Prim extrinsics plus pure runtime chassis displacement."""
        if not self.prim_world_from_cameras:
            raise RuntimeError("Camera Prim calibration was not initialized")
        runtime_base_position = np.asarray(base_pose[:3], dtype=np.float64)
        if self.initial_runtime_base_position is None:
            self.initial_runtime_base_position = runtime_base_position.copy()
        base_displacement = runtime_base_position - self.initial_runtime_base_position
        transforms = {}
        for camera_name in self.CAMERA_NAMES:
            transform = self.prim_world_from_cameras[camera_name].copy()
            # The camera Prim supplies the initial world extrinsic. PhysX moves
            # the base_link without updating the USD Prim, so add the measured
            # base-link world displacement to the camera translation.
            transform[:3, 3] += base_displacement
            transforms[camera_name] = transform
        return transforms

    def print_camera_extrinsics(self, episode, episode_step):
        """Print the exact matrices that add_frame() stores for this frame."""
        for camera_name, transform in self._last_world_from_cameras.items():
            print(
                f"[CAMERA-EXTRINSIC] episode={episode} frame={episode_step} "
                f"camera={camera_name} T_world_camera_usd=\n"
                f"{np.array2string(transform, precision=8, suppress_small=False, separator=', ')}",
                flush=True,
            )

    def start_episode(self, episode, launch, landing, flight_steps):
        """Start recording one planned throw and its measured flight path."""
        if not self.enabled:
            return
        self.current_throw = {
            "episode": int(episode),
            "launch_world_m": np.asarray(launch, dtype=np.float64).tolist(),
            "target_landing_world_m": np.asarray(
                landing, dtype=np.float64
            ).tolist(),
            "flight_steps": int(flight_steps),
            "planned_flight_time_s": (
                int(flight_steps) * float(self.args.physics_dt)
            ),
            # The first point is the commanded launch pose before physics steps.
            "trajectory_world_m": [
                np.asarray(launch, dtype=np.float64).tolist()
            ],
        }

    def _record_ball_trajectory_point(self, ball_position_world):
        if self.current_throw is None:
            return
        trajectory = self.current_throw["trajectory_world_m"]
        if len(trajectory) >= self.current_throw["flight_steps"] + 1:
            return
        trajectory.append(
            np.asarray(ball_position_world, dtype=np.float64).tolist()
        )

    @staticmethod
    def _scalar_statistics(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    @staticmethod
    def _point_statistics(points):
        points = np.asarray(points, dtype=np.float64)
        return {
            "count": int(points.shape[0]),
            "mean_xyz": np.mean(points, axis=0).tolist(),
            "std_xyz": np.std(points, axis=0).tolist(),
            "min_xyz": np.min(points, axis=0).tolist(),
            "max_xyz": np.max(points, axis=0).tolist(),
        }

    def _write_throw_analysis(self):
        """Rewrite statistics and plots from every completed throw so far."""
        if not self.throw_records or self.throw_analysis_dir is None:
            return

        launches = np.asarray(
            [item["launch_world_m"] for item in self.throw_records],
            dtype=np.float64,
        )
        target_landings = np.asarray(
            [item["target_landing_world_m"] for item in self.throw_records],
            dtype=np.float64,
        )
        actual_landings = np.asarray(
            [item["actual_landing_world_m"] for item in self.throw_records],
            dtype=np.float64,
        )
        target_errors = np.asarray(
            [item["target_error_m"] for item in self.throw_records],
            dtype=np.float64,
        )
        flight_times = np.asarray(
            [item["planned_flight_time_s"] for item in self.throw_records],
            dtype=np.float64,
        )
        trajectories = [
            np.asarray(item["trajectory_world_m"], dtype=np.float64)
            for item in self.throw_records
        ]
        points_per_trajectory = np.asarray(
            [len(trajectory) for trajectory in trajectories], dtype=np.float64
        )
        apex_heights = np.asarray(
            [np.max(trajectory[:, 2]) for trajectory in trajectories],
            dtype=np.float64,
        )
        path_lengths = np.asarray([
            np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
            if len(trajectory) > 1 else 0.0
            for trajectory in trajectories
        ], dtype=np.float64)

        statistics = {
            "completed_episodes": int(len(self.throw_records)),
            "episode_index_range": [
                int(self.throw_records[0]["episode"]),
                int(self.throw_records[-1]["episode"]),
            ],
            "analysis_interval_episodes": int(self.args.throw_analysis_interval),
            "coordinate_frame": "world",
            "point_units": "m",
            "launch_point": self._point_statistics(launches),
            "target_landing_point": self._point_statistics(target_landings),
            "actual_landing_point": self._point_statistics(actual_landings),
            "target_error_m": self._scalar_statistics(target_errors),
            "planned_flight_time_s": self._scalar_statistics(flight_times),
            "trajectory": {
                "total_recorded_points": int(points_per_trajectory.sum()),
                "points_per_episode": self._scalar_statistics(
                    points_per_trajectory
                ),
                "apex_world_z_m": self._scalar_statistics(apex_heights),
                "path_length_m": self._scalar_statistics(path_lengths),
            },
        }
        statistics_path = self.throw_analysis_dir / "throw_statistics.json"
        temporary_statistics_path = statistics_path.with_name(
            f".{statistics_path.name}.tmp"
        )
        with temporary_statistics_path.open("w", encoding="utf-8") as stream:
            json.dump(statistics, stream, indent=2)
            stream.write("\n")
        temporary_statistics_path.replace(statistics_path)

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynamicvla-matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        distribution_figure = plt.figure(
            figsize=(11, 5), constrained_layout=True
        )
        launch_axes = distribution_figure.add_subplot(1, 2, 1)
        landing_axes = distribution_figure.add_subplot(1, 2, 2)

        launch_axes.scatter(launches[:, 0], launches[:, 1], s=12, c="green", alpha=0.65)
        launch_axes.set_title("Launch point distribution")
        landing_axes.scatter(
            target_landings[:, 0], target_landings[:, 1],
            s=12, c="red", alpha=0.55, label="target",
        )
        landing_axes.scatter(
            actual_landings[:, 0], actual_landings[:, 1],
            s=12, c="blue", alpha=0.55, label="actual",
        )
        landing_axes.set_title("Landing point distribution")
        landing_axes.legend()
        for axes in (launch_axes, landing_axes):
            axes.set_xlabel("world X (m)")
            axes.set_ylabel("world Y (m)")
            axes.grid(True, alpha=0.25)
            axes.set_aspect("equal", adjustable="datalim")

        distribution_path = (
            self.throw_analysis_dir / "throw_point_distribution.png"
        )
        temporary_distribution_path = distribution_path.with_name(
            f".{distribution_path.name}.tmp.png"
        )
        distribution_figure.savefig(temporary_distribution_path, dpi=160)
        plt.close(distribution_figure)
        temporary_distribution_path.replace(distribution_path)

        trajectory_figure = plt.figure(
            figsize=(8, 7), constrained_layout=True
        )
        trajectory_axes = trajectory_figure.add_subplot(1, 1, 1, projection="3d")

        plotted_indices = np.unique(np.linspace(
            0,
            len(self.throw_records) - 1,
            min(len(self.throw_records), 200),
            dtype=int,
        ))
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(plotted_indices)))
        for color, record_index in zip(colors, plotted_indices):
            trajectory = trajectories[int(record_index)]
            trajectory_axes.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, alpha=0.45, linewidth=0.8,
            )
        trajectory_axes.scatter(
            launches[:, 0], launches[:, 1], launches[:, 2],
            s=8, c="green", alpha=0.6,
        )
        trajectory_axes.scatter(
            actual_landings[:, 0], actual_landings[:, 1], actual_landings[:, 2],
            s=8, c="red", alpha=0.6,
        )
        trajectory_axes.set_title(
            f"Measured flight trajectories (episodes={len(self.throw_records)})"
        )
        trajectory_axes.set_xlabel("world X (m)")
        trajectory_axes.set_ylabel("world Y (m)")
        trajectory_axes.set_zlabel("world Z (m)")

        trajectory_path = self.throw_analysis_dir / "throw_trajectories.png"
        temporary_trajectory_path = trajectory_path.with_name(
            f".{trajectory_path.name}.tmp.png"
        )
        trajectory_figure.savefig(temporary_trajectory_path, dpi=160)
        plt.close(trajectory_figure)
        temporary_trajectory_path.replace(trajectory_path)
        print(
            f"[THROW-ANALYSIS] Completed cumulative analysis for episodes "
            f"{self.throw_records[0]['episode']}-{self.throw_records[-1]['episode']}: "
            f"{statistics_path}, {distribution_path}, {trajectory_path}",
            flush=True,
        )

    def _maybe_write_throw_analysis(self, force=False):
        """Write a new cumulative checkpoint once per interval or final tail."""
        completed = len(self.throw_records)
        if completed == 0 or completed == self._last_throw_analysis_count:
            return False
        interval = int(self.args.throw_analysis_interval)
        if not force and completed % interval != 0:
            return False
        self._write_throw_analysis()
        self._last_throw_analysis_count = completed
        return True

    def _finish_throw_diagnostics(self):
        """Accumulate one completed throw and emit due cumulative analysis."""
        if self.current_throw is None or self.throw_analysis_dir is None:
            return
        record = self.current_throw
        trajectory = np.asarray(record["trajectory_world_m"], dtype=np.float64)
        target = np.asarray(record["target_landing_world_m"], dtype=np.float64)
        actual_landing = trajectory[-1]
        record["actual_landing_world_m"] = actual_landing.tolist()
        record["target_error_m"] = float(np.linalg.norm(actual_landing - target))
        record["recorded_trajectory_points"] = int(len(trajectory))
        self.throw_records.append(record)
        self.current_throw = None
        self._maybe_write_throw_analysis()

    @staticmethod
    def _data_and_info(payload):
        if isinstance(payload, dict) and "data" in payload:
            return np.asarray(payload["data"]), payload.get("info", {})
        return np.asarray(payload), {}

    @staticmethod
    def _label_contains_path(label, prim_path):
        if isinstance(label, dict):
            return any(
                _LeRobotRecorder._label_contains_path(value, prim_path)
                for value in label.values()
            )
        if isinstance(label, (tuple, list)):
            return any(
                _LeRobotRecorder._label_contains_path(value, prim_path)
                for value in label
            )
        return prim_path in str(label)

    @staticmethod
    def _label_contains_class(label, class_name):
        if isinstance(label, dict):
            for key, value in label.items():
                if str(key).lower() == "class" and str(value) == class_name:
                    return True
                if _LeRobotRecorder._label_contains_class(value, class_name):
                    return True
            return False
        if isinstance(label, (tuple, list)):
            return any(
                _LeRobotRecorder._label_contains_class(value, class_name)
                for value in label
            )
        return str(label) == class_name

    @classmethod
    def _matching_instance_ids(cls, info, class_name, prim_paths):
        """Find per-frame instance IDs by semantic class or descendant path."""
        if not isinstance(info, dict):
            return []
        matched = set()
        for mapping_name in ("idToSemantics", "idToLabels"):
            mapping = info.get(mapping_name, {})
            if not isinstance(mapping, dict):
                continue
            for instance_id, label in mapping.items():
                matches_class = cls._label_contains_class(label, class_name)
                matches_path = any(
                    cls._label_contains_path(label, prim_path)
                    for prim_path in prim_paths
                )
                if not (matches_class or matches_path):
                    continue
                try:
                    matched.add(int(instance_id))
                except (TypeError, ValueError):
                    continue
        return sorted(matched)

    def _rgb(self, annotator):
        frame, _ = self._data_and_info(annotator.get_data())
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(f"Unexpected Replicator RGB shape: {frame.shape}")
        return np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)

    def _semantic_mask_rgb(
        self,
        annotator,
        depth,
        camera_name,
        world_from_camera_usd,
    ):
        """Build an RGB-repeated uint8 semantic class-ID mask."""
        ids, info = self._data_and_info(annotator.get_data())
        ids = np.squeeze(ids)
        if ids.ndim != 2:
            raise RuntimeError(f"Unexpected instance segmentation shape: {ids.shape}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != ids.shape:
            raise RuntimeError(
                f"Segmentation/depth shape mismatch: {ids.shape} versus {depth.shape}"
            )

        class_ids = np.zeros(ids.shape, dtype=np.uint8)
        ground_from_depth = ground_mask_from_depth(
            depth,
            self.camera_intrinsics[camera_name],
            world_from_camera_usd,
            ground_z=self.args.ground_mask_z,
            z_tolerance=self.args.ground_mask_z_tolerance,
        )
        class_ids[ground_from_depth] = SEGMENTATION_CLASS_IDS["ground"]

        instance_ids_by_class = {
            "ground": self._matching_instance_ids(
                info, "ground", GROUND_MASK_PRIM_PATHS
            ),
            "robot_arm": self._matching_instance_ids(
                info, "robot_arm", ARM_MASK_PRIM_PATHS
            ),
            "tennis_ball": self._matching_instance_ids(
                info, "tennis_ball", (self.args.ball_mask_prim_path,)
            ),
        }
        # Higher-priority dynamic objects overwrite the depth-derived ground.
        for class_name in ("ground", "robot_arm", "tennis_ball"):
            instance_ids = instance_ids_by_class[class_name]
            if instance_ids:
                class_ids[np.isin(ids, instance_ids)] = SEGMENTATION_CLASS_IDS[
                    class_name
                ]
            elif (
                class_name != "ground"
                and class_name not in self._warned_missing_instance_classes
            ):
                print(
                    f"[WARN] No visible instance ID matched class {class_name!r}; "
                    "writing that class as empty until it becomes visible.",
                    flush=True,
                )
                self._warned_missing_instance_classes.add(class_name)
        return np.repeat(class_ids[:, :, None], 3, axis=2)

    def _depth(self, annotator):
        depth, _ = self._data_and_info(annotator.get_data())
        depth = np.squeeze(depth).astype(np.float32)
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected depth shape: {depth.shape}")
        # Replicator uses inf for rays without a geometry hit.  Store zero for
        # invalid rays so LeRobot statistics remain finite and deterministic.
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
        return np.ascontiguousarray(depth[:, :, None], dtype=np.float32)

    @staticmethod
    def _rpy_from_xyzw(quaternion):
        """Convert an xyzw quaternion to world-frame XYZ roll, pitch, yaw."""
        x, y, z, w = (float(value) for value in quaternion)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 0.0:
            raise RuntimeError("Received a zero-length body quaternion")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        roll = math.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @classmethod
    def _yaw_from_xyzw(cls, quaternion):
        return float(cls._rpy_from_xyzw(quaternion)[2])

    @staticmethod
    def _pose_matrix_from_xyzw(position, quaternion):
        """Return a conventional column-vector T_world_body matrix."""
        x, y, z, w = (float(value) for value in quaternion)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 0.0:
            raise RuntimeError("Received a zero-length base quaternion")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        return matrix

    def add_frame(
        self, articulation, arm_indices, base_body_index, link6_body_index,
        controller, tennis_ball
    ):
        if not self.enabled:
            return
        if link6_body_index is None:
            raise RuntimeError("Cannot record link6 pose: articulation body 'link6' is missing")
        body_poses = articulation.data.body_pose_w.torch
        base_pose = body_poses[0, base_body_index].detach().cpu().numpy()
        link6_pose = body_poses[0, link6_body_index].detach().cpu().numpy()
        link6_rpy = self._rpy_from_xyzw(link6_pose[3:])
        joint_pos = articulation.data.joint_pos
        joint_pos = getattr(joint_pos, "torch", joint_pos)
        joints = joint_pos[0, arm_indices].detach().cpu().numpy()
        state = np.concatenate((
            np.array([
                base_pose[0], base_pose[1], self._yaw_from_xyzw(base_pose[3:]),
            ], dtype=np.float32),
            np.asarray(joints, dtype=np.float32),
            np.asarray(link6_pose[:3], dtype=np.float32),
            link6_rpy,
        ))
        desired = controller.desired_base_pose
        action = np.concatenate((
            np.array([
                desired[0], desired[1], self._yaw_from_xyzw(desired[3:]),
            ], dtype=np.float32),
            controller.target[0].detach().cpu().numpy().astype(np.float32),
            np.asarray(link6_pose[:3], dtype=np.float32),
            link6_rpy,
        ))
        ball_position_world = (
            tennis_ball.data.root_com_pos_w.torch[0].detach().cpu().numpy()
        )
        ball_velocity_world = (
            tennis_ball.data.root_com_lin_vel_w.torch[0].detach().cpu().numpy()
        )
        self._record_ball_trajectory_point(ball_position_world)
        minimal = bool(self.args.dynamicvla_only_dataset)
        frame = {
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "task": self.args.dataset_task,
        }
        if not minimal:
            environment_state = np.concatenate((
                np.asarray(ball_position_world, dtype=np.float32),
                np.asarray(ball_velocity_world, dtype=np.float32),
            ))
            frame["observation.environment_state"] = environment_state.astype(np.float32)
            world_from_cameras = self._world_from_cameras_for_base_pose(base_pose)
            self._last_world_from_cameras = world_from_cameras
            for camera_name, world_from_camera in world_from_cameras.items():
                frame[f"observation.camera_extrinsics.{camera_name}"] = (
                    world_from_camera.astype(np.float32).reshape(-1)
                )
        for camera_index, (camera_name, rgb_annotator) in enumerate(
            zip(self.CAMERA_NAMES, self.rgb_annotators)
        ):
            frame[f"observation.images.{camera_name}"] = self._rgb(rgb_annotator)
            if not minimal:
                seg_annotator = self.seg_annotators[camera_index]
                depth_annotator = self.depth_annotators[camera_index]
                depth = self._depth(depth_annotator)
                mask = self._semantic_mask_rgb(
                    seg_annotator,
                    depth[:, :, 0],
                    camera_name,
                    world_from_cameras[camera_name],
                )
                frame[f"observation.segmentation.{camera_name}"] = mask
                mask_ids = mask[:, :, 0]
                for class_name, class_id in SEGMENTATION_CLASS_IDS.items():
                    if class_name != "background" and np.any(mask_ids == class_id):
                        self.mask_class_nonempty_frames[camera_name][class_name] += 1
                if self.args.record_depth:
                    frame[f"observation.depth.{camera_name}"] = depth
        self.dataset.add_frame(frame)

    def save_episode(self):
        if not self.enabled:
            return
        episode_buffer = self.dataset.episode_buffer
        source_indices = relabel_actions_from_future_states(
            episode_buffer, self.args.action_lookahead_steps
        )
        self.dataset.save_episode()
        self.saved_episodes += 1
        self._finish_throw_diagnostics()
        print(
            f"[DATASET] Saved episode {self.saved_episodes}/"
            f"{self.args.dataset_num_episodes}; action[t]=state[min(t+"
            f"{self.args.action_lookahead_steps}, {len(source_indices) - 1})]",
            flush=True,
        )

    @property
    def complete(self):
        return self.enabled and self.saved_episodes >= self.args.dataset_num_episodes

    def finalize(self):
        if self.dataset is not None:
            episode_buffer = getattr(self.dataset, "episode_buffer", None)
            if episode_buffer is not None and episode_buffer.get("size", 0) > 0:
                print(
                    f"[DATASET] Discarding incomplete final episode with "
                    f"{episode_buffer['size']} frames.",
                    flush=True,
                )
                self.dataset._preserve_image_files = False
                self.dataset.clear_episode_buffer(delete_images=True)
            self.dataset.finalize()
            self._maybe_write_throw_analysis(force=True)
            print(
                f"[DATASET] Finalized LeRobot {self.dataset.meta.info['codebase_version']} "
                f"dataset with "
                f"{self.saved_episodes} complete episodes at "
                f"{Path(self.args.lerobot_dataset_dir).expanduser().resolve()}",
                flush=True,
            )
            if not self.args.dynamicvla_only_dataset:
                print(
                    f"[DATASET] Non-empty semantic-class frames: "
                    f"{self.mask_class_nonempty_frames}",
                    flush=True,
                )
                missing_ball_cameras = [
                    camera_name
                    for camera_name, counts in self.mask_class_nonempty_frames.items()
                    if counts["tennis_ball"] == 0
                ]
                if self.saved_episodes > 0 and missing_ball_cameras:
                    raise RuntimeError(
                        "No tennis-ball pixels were recorded for cameras: "
                        f"{missing_ball_cameras}"
                    )
                missing_dataset_classes = [
                    class_name
                    for class_name in ("ground", "robot_arm")
                    if sum(
                        counts[class_name]
                        for counts in self.mask_class_nonempty_frames.values()
                    ) == 0
                ]
                if self.saved_episodes > 0 and missing_dataset_classes:
                    raise RuntimeError(
                        "No pixels were recorded anywhere for semantic classes: "
                        f"{missing_dataset_classes}"
                    )
        for annotator in (
            *self.rgb_annotators, *self.seg_annotators, *self.depth_annotators
        ):
            try:
                annotator.detach()
            except Exception:
                pass


class _VLAInferenceBridge:
    """Exchange synchronized observations and 9-D action chunks over ZeroMQ."""

    CAMERA_NAMES = ("left", "right", "upper")

    def __init__(self, args, render_products):
        import zmq

        self.args = args
        self.frame_indices = tuple(sorted(
            int(value) for value in args.inference_frame_offsets
        ))
        self.frame_index_set = set(self.frame_indices)
        self.inference_start_step = max(self.frame_indices)
        self.history = {}
        self.action_queue = deque()
        self.executed_since_plan = 0
        self.published_this_episode = False
        self.last_action_request_key = None
        self.request_id = 0
        self.awaiting_action = False
        self.pending_since = None
        self.input_save_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-input-writer")
            if args.inference_save_inputs else None
        )
        self.input_save_futures = []
        self.context = zmq.Context()
        self.observation_socket = self.context.socket(zmq.PUB)
        self.observation_socket.bind(
            f"tcp://{args.zmq_host}:{args.zmq_observation_port}"
        )
        self.action_socket = self.context.socket(zmq.SUB)
        self.action_socket.connect(f"tcp://{args.zmq_host}:{args.zmq_action_port}")
        self.action_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.input_dir = Path(args.inference_input_dir).expanduser().resolve()
        if args.inference_save_inputs:
            self.input_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_annotators = []
        for render_product in render_products:
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach([render_product])
            self.rgb_annotators.append(annotator)
        print(
            f"[VLA] Inference control enabled: observations=tcp://{args.zmq_host}:"
            f"{args.zmq_observation_port}, actions=tcp://{args.zmq_host}:"
            f"{args.zmq_action_port}",
            flush=True,
        )
        print(
            f"[VLA] Input frame indices={self.frame_indices}; "
            + (f"saved input grid={self.input_dir}; " if args.inference_save_inputs else "")
            + ("once per episode" if args.inference_once_per_episode else ""),
            flush=True,
        )

    @staticmethod
    def _rgb(annotator):
        payload = annotator.get_data()
        data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        image = np.asarray(data)
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"Unexpected inference RGB shape: {image.shape}")
        return np.array(image[:, :, :3], dtype=np.uint8, order="C", copy=True)

    @staticmethod
    def _state(articulation, arm_indices, base_body_index):
        base_pose = articulation.data.body_pose_w.torch[
            0, base_body_index
        ].detach().cpu().numpy()
        joint_pos = articulation.data.joint_pos
        joint_pos = getattr(joint_pos, "torch", joint_pos)
        joints = joint_pos[0, arm_indices].detach().cpu().numpy()
        return np.concatenate((
            np.asarray([
                base_pose[0], base_pose[1],
                _LeRobotRecorder._yaw_from_xyzw(base_pose[3:]),
            ], dtype=np.float32),
            np.asarray(joints, dtype=np.float32),
        )).astype(np.float32)

    def reset_episode(self):
        self.history.clear()
        self.action_queue.clear()
        self.executed_since_plan = 0
        self.published_this_episode = False
        self.last_action_request_key = None
        self.awaiting_action = False
        self.pending_since = None

    @property
    def control_ready(self):
        """Allow robot control only after this episode published an observation."""
        return self.published_this_episode

    def capture(self, episode, episode_step, articulation, arm_indices, base_body_index):
        self._reap_input_saves()
        self._receive_actions(episode)
        if episode_step in self.frame_index_set:
            self.history[int(episode_step)] = {
                "step": int(episode_step),
                "state": self._state(articulation, arm_indices, base_body_index),
                "images": {
                    name: self._rgb(annotator)
                    for name, annotator in zip(self.CAMERA_NAMES, self.rgb_annotators)
                },
            }
        if episode_step < self.inference_start_step:
            return
        needs_plan = (
            not self.action_queue
            and not self.awaiting_action
            and not (self.args.inference_once_per_episode and self.published_this_episode)
        )
        needs_replan = (
            not self.args.inference_once_per_episode
            and self.executed_since_plan >= self.args.inference_replan_steps
        )
        if needs_replan:
            self.action_queue.clear()
            self.awaiting_action = False
        if needs_plan or needs_replan:
            selected = tuple(self.history[index] for index in self.frame_indices)
            self.request_id += 1
            message = {
                "episode": int(episode),
                "request_id": self.request_id,
                "obs_frames": [item["step"] for item in selected],
                "observation.state": _pack_ndarray(
                    np.stack([item["state"] for item in selected])
                ),
                "task": self.args.dataset_task,
            }
            for camera_name in self.CAMERA_NAMES:
                message[f"observation.images.{camera_name}"] = _pack_ndarray(
                    np.stack([item["images"][camera_name] for item in selected])
                )
            self._save_input_images(episode, selected)
            if self.args.inference_save_inputs:
                message["input_image_dir"] = str(self.last_input_directory[1].resolve())
            # Publish every request exactly once. The server's recv_pyobj()
            # accepts this pickle payload directly.
            payload = pickle.dumps(
                message, protocol=pickle.HIGHEST_PROTOCOL
            )
            self.observation_socket.send(payload, copy=False)
            self.published_this_episode = True
            self.awaiting_action = True
            self.pending_since = time.monotonic()
            print(
                f"[VLA] Published episode={episode} request={self.request_id} "
                f"frames={message['obs_frames']}", flush=True,
            )

    def _save_input_images(self, episode, selected):
        """Queue a clockwise-rotated 3x3 PNG without blocking simulation."""
        if not self.args.inference_save_inputs:
            return
        request_dir = (
            self.input_dir
            / f"episode-{int(episode):06d}"
            / f"request-{int(self.request_id):06d}"
        )
        request_dir.mkdir(parents=True, exist_ok=True)
        self.last_input_directory = (int(episode), request_dir)
        future = self.input_save_executor.submit(
            self._write_input_grid,
            request_dir / "inputs_grid_3x3.png",
            tuple(sorted(selected, key=lambda item: int(item["step"]))),
        )
        self.input_save_futures.append(future)

    @classmethod
    def _write_input_grid(cls, output_path, rows):
        """Build and compress one grid on the background writer thread."""
        from PIL import Image

        grid = np.concatenate([
            np.concatenate([
                np.rot90(item["images"][name], k=-1)
                for name in cls.CAMERA_NAMES
            ], axis=1)
            for item in rows
        ], axis=0)
        Image.fromarray(grid.astype(np.uint8), mode="RGB").save(
            output_path
        )

    def _reap_input_saves(self):
        """Surface completed writer errors and retain unfinished jobs."""
        pending = []
        for future in self.input_save_futures:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self.input_save_futures = pending

    def _receive_actions(self, episode):
        import zmq

        expected_request_key = None
        if self.control_ready and self.awaiting_action:
            expected_request_key = (int(episode), int(self.request_id))
        latest = None
        try:
            while True:
                message = self.action_socket.recv_pyobj(flags=zmq.NOBLOCK)
                if not isinstance(message, dict) or "actions" not in message:
                    continue
                request_key = (
                    int(message.get("episode", episode)),
                    int(message.get("request_id", -1)),
                )
                if request_key == expected_request_key:
                    latest = message
        except zmq.Again:
            pass
        if latest is None:
            return
        if latest.get("action_mode") != "delta":
            raise RuntimeError(
                "Expected inference response action_mode='delta', got "
                f"{latest.get('action_mode')!r}"
            )
        actions = np.asarray(latest["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] != 9:
            raise RuntimeError(
                "Expected VLA delta actions shaped (N, 9), with N > 0; got "
                f"{actions.shape}"
            )
        request_key = (
            int(latest.get("episode", episode)),
            int(latest.get("request_id", -1)),
        )
        if request_key == self.last_action_request_key:
            return
        self.last_action_request_key = request_key
        self.action_queue = deque(actions)
        self.executed_since_plan = 0
        self.awaiting_action = False
        self.pending_since = None
        print(
            "[VLA] Received delta actions "
            f"episode={request_key[0]} request={request_key[1]} "
            f"source_action_index={latest.get('source_action_index', '?')} "
            "columns=[dx, dy, dyaw, dj1, dj2, dj3, dj4, dj5, dj6]:\n"
            + np.array2string(
                actions,
                precision=6,
                suppress_small=False,
                floatmode="fixed",
                max_line_width=200,
            ),
            flush=True,
        )
        print(
            "[VLA] Five-step accumulated delta: "
            + np.array2string(
                actions.sum(axis=0, dtype=np.float64),
                precision=6,
                suppress_small=False,
                floatmode="fixed",
                max_line_width=200,
            ),
            flush=True,
        )

    def poll_actions(self, episode):
        """Poll without blocking Kit and enforce the response timeout."""
        self._receive_actions(episode)
        if not self.awaiting_action:
            return
        now = time.monotonic()
        if self.pending_since is not None and now - self.pending_since >= self.args.inference_timeout:
            raise TimeoutError(
                f"No VLA action received within {self.args.inference_timeout:.1f}s"
            )

    def pop_action(self):
        if not self.control_ready or not self.action_queue:
            return None
        self.executed_since_plan += 1
        return self.action_queue.popleft()

    def close(self):
        if self.input_save_executor is not None:
            self.input_save_executor.shutdown(wait=True)
            for future in self.input_save_futures:
                future.result()
            self.input_save_futures.clear()
        for annotator in self.rgb_annotators:
            try:
                annotator.detach()
            except Exception:
                pass
        self.observation_socket.close(linger=0)
        self.action_socket.close(linger=0)
        self.context.term()


def _apply_arm_state(stage, arm_rad):
    """Author initial state and drive targets for joint1..6.

    The catch_it USD defines the kinematic chain via joint body rels with flat
    link prims, so link xforms cannot pose the arm — only the physics-cooked
    articulation state can. Isaac articulation APIs use radians, but raw USD
    angular state/drive attributes use degrees. The matching drive target
    prevents gravity (or an old zero target) from moving the arm while the
    initial pose is cooked.
    """
    for i, (rad, stiffness, damping, max_force) in enumerate(
        zip(arm_rad, ARM_DRIVE_STIFFNESS, ARM_DRIVE_DAMPING, ARM_DRIVE_MAX_FORCE),
        start=1,
    ):
        prim = stage.GetPrimAtPath(f"/World/CatchIt/joints/joint{i}")
        if not prim.IsValid():
            print(f"[WARN] arm state: joint prim missing: joint{i}", flush=True)
            continue
        attr = prim.GetAttribute("physics:state:angular:physics:position")
        if not attr.IsValid():
            attr = prim.CreateAttribute(
                "physics:state:angular:physics:position", Sdf.ValueTypeNames.Float
            )
        angle_deg = math.degrees(float(rad))
        attr.Set(angle_deg)

        target_attr = prim.GetAttribute("drive:X:physics:targetPosition")
        if not target_attr.IsValid():
            target_attr = prim.CreateAttribute(
                "drive:X:physics:targetPosition", Sdf.ValueTypeNames.Float
            )
        target_attr.Set(angle_deg)

        for attr_name, value in (
            ("drive:X:physics:stiffness", stiffness),
            ("drive:X:physics:damping", damping),
            ("drive:X:physics:maxForce", max_force),
        ):
            drive_attr = prim.GetAttribute(attr_name)
            if not drive_attr.IsValid():
                drive_attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            drive_attr.Set(value)
    print(
        f"[INFO] Arm initial state/drive target set: {tuple(arm_rad)} rad "
        f"({tuple(math.degrees(float(v)) for v in arm_rad)} deg USD); "
        f"stiffness={ARM_DRIVE_STIFFNESS}, maxForce={ARM_DRIVE_MAX_FORCE}",
        flush=True,
    )


def _capture_physics_scene_settings(stage):
    """Read solver settings off the stage's physics scene.

    SimulationContext deletes every existing PhysicsScene prim and creates a
    fresh one with default solver settings, which drops whatever the patched
    scene authored (iteration counts, TGS flags, ...). Capture them first so
    they can be re-applied.
    """
    settings = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        for attr in prim.GetAttributes():
            if not attr.IsValid() or attr.GetName() in (
                "physics:gravityDirection",
                "physics:gravityMagnitude",
            ):
                continue  # SimulationCfg re-authors gravity itself
            try:
                settings[attr.GetName()] = attr.Get()
            except Exception:
                continue
        break  # only the first physics scene matters
    return settings


def _apply_physics_scene_settings(stage, prim_path, settings):
    """Re-apply captured solver settings to the new physics scene prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for name, value in settings.items():
        try:
            attr = prim.GetAttribute(name)
            if not attr.IsValid():
                attr = prim.CreateAttribute(name)
            attr.Set(value)
        except Exception:
            continue
    if settings:
        print(f"[INFO] Re-applied {len(settings)} solver setting(s) to {prim_path}: "
              f"{sorted(settings)}", flush=True)


def _find_articulation_root(stage):
    """Find the same articulation root used by keyboard_drive_3dgs_scene.py."""
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith(PRIM_ROBOT) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return path
    raise RuntimeError(f"No articulation root found below {PRIM_ROBOT}")


def _create_runtime_arm(stage, arm_rad, device, physics_dt):
    """Initialize and pose the arm through the runtime Articulation API."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    # The referenced scene already owns /physicsScene. Use a distinct context
    # path so Isaac Lab can create its tensor simulation view without trying
    # to overwrite the referenced prim. SimulationContext replaces every
    # existing physics scene with a default one, so carry the scene's solver
    # settings (iteration counts, TGS flags) over to the new prim.
    solver_settings = _capture_physics_scene_settings(stage)
    sim = SimulationContext(
        SimulationCfg(
            physics_prim_path="/IsaacLabPhysicsScene",
            device=str(device),
            dt=float(physics_dt),
        )
    )
    _apply_physics_scene_settings(stage, "/IsaacLabPhysicsScene", solver_settings)
    art_path = _find_articulation_root(stage)
    initial_joint_pos = {
        f"joint{i}": float(rad) for i, rad in enumerate(arm_rad, start=1)
    }
    # Use the requested arm pose as the articulation reset state; steering and
    # driving DOFs are initialized to zero.
    initial_joint_pos.update({"steer_.*": 0.0, "drive_.*": 0.0})
    # NOTE: no Isaac Lab actuators are configured. Their explicit-force PD
    # model destabilizes this heavy arm under the default TGS solver (see the
    # "enable_external_forces_every_iteration=False may cause noisy velocities"
    # warning). Instead the controller issues PhysX NATIVE dof position targets
    # (root_view.set_dof_position_targets), which use the USD-authored drive
    # stiffness/damping integrated implicitly by the solver — the same path as
    # catch_it_env.py's articulation.set_joint_position_targets().
    articulation = Articulation(
        cfg=ArticulationCfg(
            prim_path=art_path,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=ROBOT_ROOT_POS,
                rot=ROBOT_ROOT_QUAT_XYZW,
                joint_pos=initial_joint_pos,
            ),
            actuators={},
        )
    )
    tennis_ball = RigidObject(
        cfg=RigidObjectCfg(
            prim_path=PRIM_BALL,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=BALL_ASSET_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=10.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.058),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
            ),
        )
    )
    ball_root = stage.GetPrimAtPath(PRIM_BALL)
    render_prims = [prim for prim in Usd.PrimRange(ball_root) if prim.IsA(UsdGeom.Gprim)]
    material_targets = []
    for prim in render_prims:
        material_targets.extend(
            UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets()
        )
    if not render_prims or not material_targets:
        raise RuntimeError(
            f"Spawned tennis ball has incomplete visuals: geometry={len(render_prims)} "
            f"materials={material_targets}"
        )
    print(
        f"[INFO] 0804 tennis-ball USD spawned: {BALL_ASSET_USD}; "
        f"geometry={len(render_prims)} material={material_targets}",
        flush=True,
    )
    sim.reset()

    dof_names = list(articulation.joint_names)
    arm_names = [f"joint{i}" for i in range(1, 7)]
    missing = [name for name in arm_names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Arm DOFs missing from articulation: {missing}; DOFs={dof_names}")
    arm_indices = [dof_names.index(name) for name in arm_names]
    steer_names = ["steer_fl", "steer_fr", "steer_rl", "steer_rr"]
    drive_names = ["drive_fl", "drive_fr", "drive_rl", "drive_rr"]
    missing_base = [n for n in (*steer_names, *drive_names) if n not in dof_names]
    if missing_base:
        raise RuntimeError(f"Base DOFs missing from articulation: {missing_base}; DOFs={dof_names}")
    steer_indices = [dof_names.index(n) for n in steer_names]
    drive_indices = [dof_names.index(n) for n in drive_names]
    _hold_runtime_arm(articulation, arm_indices, arm_rad)
    # ── Diagnostic: what drive gains/targets does the cooked sim actually have?
    articulation.update(sim.get_physics_dt())
    stiffness = articulation.data.joint_stiffness.torch[0].tolist()
    damping = articulation.data.joint_damping.torch[0].tolist()
    targets = getattr(articulation.data, "joint_pos_target", None)
    target_vals = targets.torch[0].tolist() if targets is not None else None
    print(
        f"[DEBUG] cooked dofs={dof_names}\n"
        f"[DEBUG] stiffness={[round(v, 2) for v in stiffness]}\n"
        f"[DEBUG] damping={[round(v, 2) for v in damping]}\n"
        f"[DEBUG] pos_target={None if target_vals is None else [round(v, 4) for v in target_vals]}",
        flush=True,
    )
    print(
        f"[INFO] Runtime articulation initialized: root={art_path}, "
        f"arm_indices={arm_indices}",
        flush=True,
    )
    return sim, articulation, arm_indices, steer_indices, drive_indices, tennis_ball


def _hold_runtime_arm(articulation, arm_indices, arm_rad):
    """Write the initial root/joint state and configure the arm hold target."""
    root_pose = torch.tensor(
        [[*ROBOT_ROOT_POS, *ROBOT_ROOT_QUAT_XYZW]],
        dtype=torch.float32,
        device=articulation.device,
    )
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=articulation.device)
    positions = torch.tensor(
        [arm_rad], dtype=torch.float32, device=articulation.device
    )
    zeros = torch.zeros_like(positions)
    articulation.write_root_pose_to_sim_index(root_pose=root_pose)
    articulation.write_root_velocity_to_sim_index(root_velocity=root_velocity)
    articulation.write_joint_position_to_sim_index(
        position=positions, joint_ids=arm_indices
    )
    articulation.write_joint_velocity_to_sim_index(
        velocity=zeros, joint_ids=arm_indices
    )
    _set_arm_target(articulation, arm_indices, arm_rad)


def _set_arm_target(articulation, arm_indices, arm_rad):
    """Set PhysX native drive targets without teleporting the actual joints.

    Mirrors omni.isaac.core's ``set_joint_position_targets`` used by
    catch_it_env.py: the USD-authored drive stiffness/damping (see
    ARM_DRIVE_STIFFNESS/DAMPING, applied in _apply_arm_state) act on these
    targets inside the solver. Wheel DOFs keep their measured positions.
    """
    import warp as wp

    device = articulation.device
    all_indices = torch.arange(
        articulation.num_joints, dtype=torch.int32, device=device
    )
    joint_pos = articulation.data.joint_pos
    full_target = getattr(joint_pos, "torch", joint_pos).clone()
    full_target[:, arm_indices] = torch.tensor(
        [arm_rad], dtype=torch.float32, device=device
    )
    # The omni.physics.tensors frontend expects warp arrays (device is
    # inferred from the torch tensors).
    view = articulation.root_view
    view.set_dof_position_targets(
        wp.from_torch(full_target.contiguous()),
        wp.from_torch(all_indices.contiguous()),
    )
    view.set_dof_velocity_targets(
        wp.from_torch(torch.zeros_like(full_target).contiguous()),
        wp.from_torch(all_indices.contiguous()),
    )


class _ArmHoldController:
    """Per-step kinematic hold for the free base and initial arm pose.

    Mirrors TennisVLAController.apply() from evaluate_tennis_0822_ok_4_zab.py
    (a per-step target write before stepping physics), plus the two pieces the
    CatchIt reference env needs because the MJCF base is a free-floating
    articulation root (see catch_it_env.py):
      - pin the base pose every step (``set_world_poses`` equivalent),
      - zero all joint velocities every step (``_zero_articulation_motion``
        equivalent), so wheels/arm do not wind up against the drives.
    The imported drive gains cook as zero in the Isaac Lab articulation, so a
    target-only hold lets the heavy arm oscillate and react against the base.
    For this scene launcher, both positions and zero velocities are restored
    before rendering. A later evaluation controller can replace this hold.
    """

    def __init__(self, articulation, arm_indices, steer_indices, drive_indices, arm_rad):
        self.articulation = articulation
        self.arm_indices = list(arm_indices)
        self.steer_indices = list(steer_indices)
        self.drive_indices = list(drive_indices)
        self.all_indices = list(range(articulation.num_joints))
        self.arm_rad = arm_rad
        device = articulation.device
        self.root_pose = torch.tensor(
            [[*ROBOT_ROOT_POS, *ROBOT_ROOT_QUAT_XYZW]],
            dtype=torch.float32,
            device=device,
        )
        self.root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
        self.joint_velocity = torch.zeros(
            (1, articulation.num_joints), dtype=torch.float32, device=device
        )
        self.joint_position = torch.zeros(
            (1, articulation.num_joints), dtype=torch.float32, device=device
        )
        self.target = torch.tensor(
            [arm_rad], dtype=torch.float32, device=device
        )
        self.home_target = self.target.clone()
        self.joint_position[:, self.arm_indices] = self.target
        self.root_yaw = 0.0
        self.base_reference_pose = None
        self.desired_base_pose = None
        self.automatic_base_offset_xy = np.zeros(2, dtype=np.float64)
        self._kinematic_drive_positions = None
        self._kinematic_steer_positions = None
        self._last_wheel_pose = None

    @staticmethod
    def _yaw_from_pose(pose):
        x, y, z, w = (float(v) for v in pose[3:])
        return math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

    @staticmethod
    def _ik_base(vx, vy, wz):
        r, track, base = 0.1, 0.364, 0.494
        if abs(vy) < 0.01: vy = 0.0
        if abs(vx) < 0.01: vx = 0.0
        if abs(wz) < 0.01: wz = 0.0
        if vx == 0.0 and vy == 0.0 and wz == 0.0:
            return np.zeros(4), np.zeros(4)
        sign = math.copysign(1.0, vy) if vy else 1.0
        if abs(vx) > 0.01:
            steer = np.clip(-math.atan(vx / (vy + 1e-5)), -1.570, 1.570)
            return np.full(4, steer), np.full(4, sign * math.hypot(vy, vx) / r)
        radius = abs(vy / wz) if abs(wz) > 1e-8 else float("inf")
        if radius < 0.47644:
            f = math.copysign(math.pi / 2.0, wz)
            steer = np.array([f, f, -f, -f])
        else:
            fl = math.atan(wz*base / (2*vy - wz*track + 1e-6))
            fr = math.atan(wz*base / (2*vy + wz*track + 1e-6))
            steer = np.array([fl, fr, -fl, -fr])
        hb, ht = base/2.0, track/2.0
        drive = sign * np.array([
            math.hypot(vy-wz*ht, wz*hb), math.hypot(vy+wz*ht, wz*hb),
            math.hypot(vy-wz*ht, wz*hb), math.hypot(vy+wz*ht, wz*hb),
        ]) / r
        return steer, drive

    def update_wheel_animation(self, dt):
        """Reference-script behavior: IK velocity, integrate wheel qpos, write qpos."""
        pose = self.desired_base_pose
        if pose is None:
            return
        pose = pose.copy()
        if self._last_wheel_pose is None:
            self._last_wheel_pose = pose
            return
        yaw = self._yaw_from_pose(pose)
        prev_yaw = self._yaw_from_pose(self._last_wheel_pose)
        dxy = (pose[:2] - self._last_wheel_pose[:2]) / float(dt)
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy = c*dxy[0] + s*dxy[1], -s*dxy[0] + c*dxy[1]
        wz = math.atan2(math.sin(yaw-prev_yaw), math.cos(yaw-prev_yaw)) / float(dt)
        steer, drive = self._ik_base(vx, vy, wz)
        if self._kinematic_drive_positions is None:
            jp = self.articulation.data.joint_pos
            jp = getattr(jp, "torch", jp)
            self._kinematic_drive_positions = jp[0, self.drive_indices].detach().cpu().numpy().astype(np.float64)
        self._kinematic_steer_positions = steer
        self._kinematic_drive_positions = (self._kinematic_drive_positions + drive * float(dt) + math.pi) % (2*math.pi) - math.pi
        self._last_wheel_pose = pose

    @staticmethod
    def _pose_matrix(pose):
        pose = np.asarray(pose, dtype=np.float64)
        x, y, z, w = pose[3:]
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = pose[:3]
        return transform

    @staticmethod
    def _matrix_pose(transform):
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
        return np.concatenate((transform[:3, 3], quat))

    def set_base_reference(self, base_pose):
        self.base_reference_pose = np.asarray(base_pose, dtype=np.float64).copy()
        self.desired_base_pose = self.base_reference_pose.copy()

    def correct_root_for_physical_base(self, measured_root_pose, measured_base_pose):
        """Shift the selected PhysX root so the actual base_link stays desired."""
        if self.desired_base_pose is None:
            return
        desired = self._pose_matrix(self.desired_base_pose)
        measured_base = self._pose_matrix(measured_base_pose)
        measured_root = self._pose_matrix(measured_root_pose)
        corrected_root = desired @ np.linalg.inv(measured_base) @ measured_root
        corrected_pose = self._matrix_pose(corrected_root)
        self.root_pose[0] = torch.as_tensor(
            corrected_pose, dtype=torch.float32, device=self.articulation.device
        )

    def drive_base(self, forward_speed, yaw_rate, dt):
        """Move the articulation root along CatchIt's local +Y direction."""
        self.root_yaw += float(yaw_rate) * float(dt)
        c, s = math.cos(self.root_yaw), math.sin(self.root_yaw)
        self.root_pose[0, 0] += -s * float(forward_speed) * float(dt)
        self.root_pose[0, 1] += c * float(forward_speed) * float(dt)
        if self.desired_base_pose is not None:
            self.desired_base_pose[0] += -s * float(forward_speed) * float(dt)
            self.desired_base_pose[1] += c * float(forward_speed) * float(dt)
        half = 0.5 * self.root_yaw
        yaw_q = torch.tensor(
            [0.0, 0.0, math.sin(half), math.cos(half)],
            dtype=torch.float32, device=self.articulation.device,
        )
        base_q = torch.tensor(
            ROBOT_ROOT_QUAT_XYZW, dtype=torch.float32,
            device=self.articulation.device,
        )
        x1, y1, z1, w1 = yaw_q
        x2, y2, z2, w2 = base_q
        self.root_pose[0, 3:] = torch.stack((
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ))
        if self.base_reference_pose is not None:
            desired_q = self._matrix_pose(
                self._pose_matrix([0.0, 0.0, 0.0, *yaw_q.detach().cpu().tolist()])
                @ self._pose_matrix([0.0, 0.0, 0.0, *self.base_reference_pose[3:]])
            )[3:]
            self.desired_base_pose[3:] = desired_q

    def set_automatic_base_offset_xy(self, offset_world_xy):
        """Set an episode-relative automatic X/Y offset without losing keyboard motion."""
        offset_world_xy = np.asarray(offset_world_xy, dtype=np.float64).reshape(2)
        increment = offset_world_xy - self.automatic_base_offset_xy
        self.root_pose[0, :2] += torch.as_tensor(
            increment, dtype=torch.float32, device=self.articulation.device
        )
        if self.desired_base_pose is not None:
            self.desired_base_pose[:2] += increment
        self.automatic_base_offset_xy = offset_world_xy.copy()

    def set_arm_position(self, arm_rad):
        values = torch.as_tensor(
            arm_rad, dtype=torch.float32, device=self.articulation.device
        ).reshape(1, 6)
        self.target.copy_(values)
        self.arm_rad = tuple(float(v) for v in values[0].detach().cpu().tolist())

    def apply_inference_delta(self, action):
        """Accumulate one [dx, dy, dyaw, djoint1..djoint6] command."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid 9-D inference action: shape={action.shape}")
        if self.desired_base_pose is None or self.base_reference_pose is None:
            raise RuntimeError("Base reference must be initialized before inference")
        self.desired_base_pose[0:2] += action[0:2]
        self.root_yaw += float(action[2])
        half = 0.5 * self.root_yaw
        yaw_pose = [0.0, 0.0, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]
        reference_rotation = [0.0, 0.0, 0.0, *self.base_reference_pose[3:]]
        self.desired_base_pose[3:] = self._matrix_pose(
            self._pose_matrix(yaw_pose) @ self._pose_matrix(reference_rotation)
        )[3:]
        arm_target = self.target[0].detach().cpu().numpy() + action[3:9]
        self.set_arm_position(arm_target)

    def set_home(self):
        self.set_arm_position(self.home_target)

    def reset_episode(self):
        """Restore every robot DOF and the free root to episode defaults."""
        self.set_home()
        self.root_yaw = 0.0
        self.automatic_base_offset_xy.fill(0.0)
        self.root_pose[0] = torch.tensor(
            [*ROBOT_ROOT_POS, *ROBOT_ROOT_QUAT_XYZW],
            dtype=torch.float32, device=self.articulation.device,
        )
        if self.base_reference_pose is not None:
            self.desired_base_pose = self.base_reference_pose.copy()
        self._kinematic_drive_positions = None
        self._kinematic_steer_positions = None
        self._last_wheel_pose = None
        self.joint_position[:, self.arm_indices] = self.home_target
        self.articulation.write_root_pose_to_sim_index(root_pose=self.root_pose)
        self.articulation.write_root_velocity_to_sim_index(
            root_velocity=self.root_velocity
        )
        self.articulation.write_joint_position_to_sim_index(
            position=self.joint_position, joint_ids=self.all_indices
        )
        self.articulation.write_joint_velocity_to_sim_index(
            velocity=self.joint_velocity, joint_ids=self.all_indices
        )
        _set_arm_target(self.articulation, self.arm_indices, self.arm_rad)

    def apply(self):
        """Keep the mobile base fixed and command only xArm joint1--joint6."""
        self.articulation.write_root_pose_to_sim_index(root_pose=self.root_pose)
        self.articulation.write_root_velocity_to_sim_index(
            root_velocity=self.root_velocity
        )
        self.articulation.write_joint_position_to_sim_index(
            position=self.target, joint_ids=self.arm_indices
        )
        self.articulation.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.target), joint_ids=self.arm_indices
        )
        _set_arm_target(
            self.articulation,
            self.arm_indices,
            self.target[0].detach().cpu().tolist(),
        )
        if self._kinematic_drive_positions is not None:
            self.articulation.write_joint_position_to_sim_index(
                position=torch.as_tensor(self._kinematic_steer_positions, dtype=torch.float32, device=self.articulation.device).reshape(1, 4),
                joint_ids=self.steer_indices,
            )
            self.articulation.write_joint_position_to_sim_index(
                position=torch.as_tensor(self._kinematic_drive_positions, dtype=torch.float32, device=self.articulation.device).reshape(1, 4),
                joint_ids=self.drive_indices,
            )


class _LandingArmMotion:
    """Move a base_link-oriented point at link6 to the landing point."""

    def __init__(self, controller, args):
        import mujoco
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        self.controller = controller
        self.args = args
        self.home = np.asarray(args.arm_init, dtype=np.float64)
        self.solution = self.home.copy()
        self.base_translation_world_xy = np.zeros(2, dtype=np.float64)
        self.ball_target_world = None
        self.ball_target_step = None
        self._mujoco = mujoco
        self._least_squares = least_squares
        self._Rotation = Rotation
        arm_xml = Path(SCENE_ASSET_ROOT) / "assets/urdf/xarm6_right.xml"
        self.model = mujoco.MjModel.from_xml_path(str(arm_xml))
        self.data = mujoco.MjData(self.model)
        # The control point starts at link6, but this offset is expressed in
        # the fixed base_link/arm-base axes rather than the rotating link6 axes.
        self.base_oriented_target_offset = np.array(
            [0.0, 0.10, 0.025], dtype=np.float64
        )
        self.data.qpos[:6] = self.home
        self._mujoco.mj_fwdPosition(self.model, self.data)
        self.home_link6_rotation = self.data.body("link6").xmat.reshape(3, 3).copy()
        self.home_control_pos = (
            self.data.body("link6").xpos.copy() + self.base_oriented_target_offset
        )

    def solve_landing(self, landing_world):
        """Place the base_link-oriented link6 control point at the landing."""
        yaw = math.radians(float(self.args.robot_rotate[2]))
        c, s = math.cos(yaw), math.sin(yaw)
        rot_world_arm = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local_arm_base_offset = np.array([0.0, 0.115864, 0.4038])
        robot_translation = np.asarray(self.args.robot_translate, dtype=np.float64)
        arm_base_world = robot_translation + rot_world_arm @ local_arm_base_offset
        control_target_arm = rot_world_arm.T @ (
            np.asarray(landing_world, dtype=np.float64) - arm_base_world
        )
        horizontal_delta_arm = control_target_arm[:2] - self.home_control_pos[:2]
        base_translation_arm = (
            float(self.args.base_landing_share) * horizontal_delta_arm
        )
        arm_control_target = control_target_arm.copy()
        arm_control_target[:2] -= base_translation_arm
        self.base_translation_world_xy = (
            rot_world_arm[:2, :2] @ base_translation_arm
        )

        def link6_pose(q):
            self.data.qpos[:6] = q
            self._mujoco.mj_fwdPosition(self.model, self.data)
            link6_pos = self.data.body("link6").xpos.copy()
            link6_rot = self.data.body("link6").xmat.reshape(3, 3).copy()
            control_pos = link6_pos + self.base_oriented_target_offset
            return control_pos, link6_rot

        def residual(q):
            link6_pos, link6_rot = link6_pose(q)
            orientation_error = self._Rotation.from_matrix(
                self.home_link6_rotation.T @ link6_rot
            ).as_rotvec()
            return np.concatenate(
                (20.0 * (link6_pos - arm_control_target),
                 5.0 * orientation_error,
                 0.01 * (q - self.home))
            )

        lower = self.model.jnt_range[:6, 0].copy()
        upper = self.model.jnt_range[:6, 1].copy()
        # Unlimited joints are represented by a zero range in MuJoCo.
        unlimited = np.logical_not(self.model.jnt_limited[:6].astype(bool))
        lower[unlimited], upper[unlimited] = -2.0 * math.pi, 2.0 * math.pi
        result = self._least_squares(
            residual,
            np.clip(self.home, lower, upper),
            bounds=(lower, upper),
            max_nfev=300,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        solution = np.asarray(result.x, dtype=np.float64).reshape(6)
        achieved_link6, achieved_rotation = link6_pose(solution)
        error = float(np.linalg.norm(achieved_link6 - arm_control_target))
        orientation_error = float(np.linalg.norm(
            self._Rotation.from_matrix(
                self.home_link6_rotation.T @ achieved_rotation
            ).as_rotvec()
        ))
        if (not result.success or not np.all(np.isfinite(solution))
                or error > 0.015 or orientation_error > math.radians(0.5)):
            self.solution = self.home.copy()
            raise RuntimeError(
                f"xArm6 IK failed for landing={tuple(landing_world)}; "
                f"success={result.success} position_error={error:.5f}m "
                f"orientation_error={math.degrees(orientation_error):.4f}deg"
            )
        self.solution = solution
        print(
            "[ARM-IK] landing_world=%s arm_control_target=%s "
            "base_translation_world_xy=%s base_share=%.2f "
            "position_error=%.5fm orientation_error=%.4fdeg solution=%s"
            % (
                tuple(round(float(v), 4) for v in landing_world),
                tuple(round(float(v), 4) for v in arm_control_target),
                tuple(round(float(v), 4) for v in self.base_translation_world_xy),
                float(self.args.base_landing_share),
                error,
                math.degrees(orientation_error),
                tuple(round(float(v), 4) for v in solution),
            ),
            flush=True,
        )

    def update(self, episode_step):
        if episode_step < self.args.arm_start_step:
            self.controller.set_automatic_base_offset_xy((0.0, 0.0))
            self.controller.set_arm_position(self.home)
            return
        progress = min(
            1.0,
            (episode_step - self.args.arm_start_step + 1)
            / float(self.args.arm_move_steps),
        )
        # Smoothstep avoids a velocity jump at the beginning/end.
        alpha = progress * progress * (3.0 - 2.0 * progress)
        self.controller.set_automatic_base_offset_xy(
            alpha * self.base_translation_world_xy
        )
        self.controller.set_arm_position(
            self.home + alpha * (self.solution - self.home)
        )
        if episode_step == self.args.arm_start_step:
            print(
                f"[ARM] movement started at episode_step={episode_step}; "
                f"duration={self.args.arm_move_steps} steps",
                flush=True,
            )


def _joint_pos_torch(articulation, arm_indices):
    """Return measured arm joint positions as a CPU float tensor (radians)."""
    joint_pos = articulation.data.joint_pos
    tensor = getattr(joint_pos, "torch", joint_pos)
    return tensor[0, arm_indices].detach().cpu().float()


def _format_joint_line(step, measured, target):
    target = target.detach().cpu().float()
    err = measured - target
    return (
        "[JOINT] step=%d\n"
        "        measured rad : %s\n"
        "        target  rad : %s\n"
        "        error   rad : %s"
        % (
            step,
            " ".join("%+.4f" % float(v) for v in measured),
            " ".join("%+.4f" % float(v) for v in target),
            " ".join("%+.4f" % float(v) for v in err),
        )
    )


def _sample_ball_throw(args, rng):
    """Sample the WAM point-to-point throw in the base_link frame.

    CatchIt's local +Y is forward. The launch point is sampled uniformly on a
    line centered ``ball_distance`` metres in front of the base, extending
    ``ball_distance_range`` metres forward and backward, with zero lateral
    offset. The landing point is sampled in the configured landing disk,
    clipped by the minimum forward and maximum absolute lateral coordinates,
    at --ball-target-height. Flight time is quantized to physics steps and
    velocity matches semi-implicit integration.
    """
    base = np.asarray(BASE_LINK_POS, dtype=np.float64).copy()
    # Calibration translates /World/CatchIt absolutely in this launcher.
    base[:2] += np.asarray(args.robot_translate[:2], dtype=np.float64) - np.array(
        [0.0, -0.115864], dtype=np.float64
    )
    base[2] += float(args.robot_translate[2])
    yaw = math.radians(float(args.robot_rotate[2]))
    forward = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    lateral = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)

    launch_forward = args.ball_distance + rng.uniform(
        -args.ball_distance_range, args.ball_distance_range
    )
    launch_lateral = 0.0
    height = rng.uniform(*args.ball_height_range)
    launch_xy = base[:2] + launch_forward * forward + launch_lateral * lateral
    launch = np.array([*launch_xy, height], dtype=np.float64)

    landing_forward = args.ball_landing_forward_min
    landing_lateral = 0.0
    for _ in range(200):
        radius = args.ball_landing_radius * math.sqrt(float(rng.random()))
        angle = rng.uniform(0.0, 2.0 * math.pi)
        candidate_forward = radius * math.cos(angle)
        candidate_lateral = radius * math.sin(angle)
        if (
            candidate_forward >= args.ball_landing_forward_min
            and abs(candidate_lateral) <= args.ball_landing_lateral_max
        ):
            landing_forward = candidate_forward
            landing_lateral = candidate_lateral
            break
    landing_xy = base[:2] + landing_forward * forward + landing_lateral * lateral
    landing = np.array(
        [*landing_xy, float(args.ball_target_height)], dtype=np.float64
    )
    actual_forward = float(np.dot(landing[:2] - base[:2], forward))
    actual_lateral = float(np.dot(landing[:2] - base[:2], lateral))
    if actual_forward <= 0.0:
        raise RuntimeError(
            f"Sampled landing is not in front of base_link: forward={actual_forward}"
        )
    if abs(actual_lateral) > args.ball_landing_lateral_max + 1e-9:
        raise RuntimeError(
            "Sampled landing exceeds lateral limit: "
            f"lateral={actual_lateral}, limit={args.ball_landing_lateral_max}"
        )

    requested_flight_time = rng.uniform(*args.ball_flight_time_range)
    flight_steps = max(1, int(round(requested_flight_time / args.physics_dt)))
    flight_time = flight_steps * float(args.physics_dt)
    velocity = (landing - launch) / flight_time
    # PhysX uses semi-implicit Euler: v[k+1]=v[k]-g*dt followed by
    # z[k+1]=z[k]+v[k+1]*dt. This v0 lands exactly at z_target after N steps.
    velocity[2] = (
        (landing[2] - launch[2]) / flight_time
        + 0.5 * 9.81 * (flight_time + float(args.physics_dt))
    )
    return launch, velocity, landing, flight_time, flight_steps, actual_forward


def _create_marker_material(stage, path, color):
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _create_camera_position_markers(stage, args, hidden):
    """Mark the three arm-base camera origins with non-colliding 2 cm spheres."""
    parent = "/World/CatchIt/base_link/arm_base"
    marker_specs = (
        ("LeftCameraMarker", args.arm_base_camera_left, (0.0, 0.35, 1.0),
         "/World/Looks/LeftCameraMarkerBlue"),
        ("RightCameraMarker", args.arm_base_camera_right, (1.0, 0.85, 0.0),
         "/World/Looks/RightCameraMarkerYellow"),
        ("UpwardCameraMarker", args.arm_base_camera_upward, (0.85, 0.0, 1.0),
         "/World/Looks/UpwardCameraMarkerPurple"),
    )
    for name, position, color, material_path in marker_specs:
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{parent}/{name}"))
        sphere.CreateRadiusAttr(0.02)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        sphere.CreateDoubleSidedAttr(True)
        sphere.AddTranslateOp().Set(
            Gf.Vec3d(*[float(value) for value in position])
        )
        if hidden:
            sphere.MakeInvisible()
        else:
            sphere.MakeVisible()
        material = _create_marker_material(stage, material_path, color)
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    state = "hidden" if hidden else "visible"
    print(f"[INFO] Camera-position markers: {state}; radius=2 cm, "
          "left=blue, right=yellow, upward=purple.", flush=True)


def _create_throw_markers(stage, hidden):
    """Create non-colliding 3 cm launch/landing disks."""
    marker_specs = (
        ("/World/ThrowMarkers/Launch", (0.0, 1.0, 0.0),
         "/World/Looks/ThrowPointGreen"),
        ("/World/ThrowMarkers/Landing", (1.0, 0.0, 0.0),
         "/World/Looks/LandingPointRed"),
    )
    paths = []
    for path, color, material_path in marker_specs:
        xform = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
        disk = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{path}/Disk"))
        disk.CreateRadiusAttr(0.03)
        disk.CreateHeightAttr(0.004)
        disk.CreateAxisAttr("Z")
        disk.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        disk.CreateDoubleSidedAttr(True)
        if hidden:
            xform.MakeInvisible()
        else:
            xform.MakeVisible()
        material = _create_marker_material(stage, material_path, color)
        UsdShade.MaterialBindingAPI.Apply(disk.GetPrim()).Bind(material)
        paths.append(path)
    state = "hidden" if hidden else "visible"
    print(f"[INFO] Throw-point markers: {state}; radius=3 cm, "
          "launch=green, landing=red.", flush=True)
    return tuple(paths)


def _update_throw_markers(stage, marker_paths, launch, landing):
    if marker_paths is None:
        return
    for path, point in zip(marker_paths, (launch, landing)):
        xform = UsdGeom.Xform(stage.GetPrimAtPath(path))
        translate_ops = [
            op for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if translate_ops:
            translate_ops[0].Set(Gf.Vec3d(*[float(v) for v in point]))


def _throw_ball(ball, args, rng):
    """Place and launch the existing CatchIt object as the tennis ball."""
    launch, velocity, landing, flight_time, flight_steps, landing_forward = _sample_ball_throw(
        args, rng
    )
    pose = torch.tensor(
        [[*launch, 0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=ball.device,
    )
    root_velocity = torch.tensor(
        [[*velocity, 0.0, 25.0, 0.0]],
        dtype=torch.float32,
        device=ball.device,
    )
    ball.write_root_link_pose_to_sim_index(root_pose=pose)
    ball.write_root_com_velocity_to_sim_index(root_velocity=root_velocity)
    ball.reset()
    print(
        "[THROW] launch=%s velocity=%s landing=%s landing_forward=%.3fm "
        "flight_time=%.3fs flight_steps=%d"
        % (
            tuple(round(float(v), 4) for v in launch),
            tuple(round(float(v), 4) for v in velocity),
            tuple(round(float(v), 4) for v in landing),
            landing_forward,
            flight_time,
            flight_steps,
        ),
        flush=True,
    )
    return launch, landing, flight_steps


def _reset_throw_episode(
    sim, articulation, controller, tennis_ball, args, rng, episode,
    stage, marker_paths, arm_motion,
):
    """Reset physics/assets and start one throw from a clean initial state."""
    sim.reset(soft=True)
    articulation.reset()
    tennis_ball.reset()
    controller.reset_episode()
    articulation.write_data_to_sim()
    if args.inference_control:
        launch, landing, flight_steps = _throw_ball(tennis_ball, args, rng)
    else:
        for attempt in range(100):
            launch, landing, flight_steps = _throw_ball(tennis_ball, args, rng)
            try:
                arm_motion.solve_landing(landing)
                break
            except RuntimeError as exc:
                if attempt == 99:
                    raise
                print(f"[ARM-IK] resampling unreachable landing: {exc}", flush=True)
    _update_throw_markers(stage, marker_paths, launch, landing)
    if not args.inference_control:
        arm_motion.ball_target_world = np.asarray(landing, dtype=np.float64)
        arm_motion.ball_target_step = int(flight_steps)
    sim.forward()
    print(f"[EPISODE] reset complete; episode={episode}", flush=True)
    return launch, landing, flight_steps


def _update_inference_controller(inference_bridge, controller, episode):
    """Hold the collection pose until the first observation is published."""
    if not inference_bridge.control_ready:
        # Match _LandingArmMotion.update() during the collection prefix. The
        # regular controller.apply() calls below perform the static pose hold.
        controller.set_automatic_base_offset_xy((0.0, 0.0))
        controller.set_home()
        return
    inference_bridge.poll_actions(episode)
    predicted_action = inference_bridge.pop_action()
    if predicted_action is not None:
        controller.apply_inference_delta(predicted_action)


def _render_inference_hold(sim, articulation, controller, app):
    """Keep the scene responsive without advancing simulated time.

    The controller and articulation writes preserve the current kinematic hold,
    while deliberately omitting ``sim.step`` and ball updates keeps every
    physical state at the observation timestamp during external inference.
    """
    controller.apply()
    articulation.write_data_to_sim()
    sim.render()
    app.update()


def _write_catch_results(output_dir, results, args, completed, successful):
    """Atomically write cumulative catch-success statistics."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "catch_radius_m": float(args.catch_radius),
        "episodes_completed": int(completed),
        "episodes_successful": int(successful),
        "success_rate": float(successful / completed) if completed else 0.0,
        "episodes": results,
    }
    path = output_dir / "catch_results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return path


def _catch_center_from_link6_position(link6_position):
    """Return the success-test center without moving the simulated Ring.

    The dataset IK targets this point relative to link6.  Success evaluation
    uses the same fixed world-axis offset, while the actual Ring body remains
    untouched in the simulation.
    """
    offset = torch.as_tensor(
        (0.0, 0.10, 0.025), dtype=link6_position.dtype,
        device=link6_position.device,
    )
    return link6_position + offset


def _run_loop(
    args, sim, articulation, arm_indices, steer_indices, drive_indices,
    arm_rad, tennis_ball, recorder, inference_bridge=None,
    observation_video=None,
):
    """Main update loop; extension hooks (drive/capture/eval) belong here."""
    from omni.timeline import get_timeline_interface

    timeline = get_timeline_interface()
    if args.static:
        timeline.pause()
        print("[INFO] Static mode: physics paused.", flush=True)
    else:
        timeline.play()
        print("[INFO] Physics playing; base and arm are held kinematically.",
              flush=True)
    print("[INFO] Entering update loop (close window or Ctrl+C to exit).", flush=True)

    dt = sim.get_physics_dt()
    print(f"[INFO] Physics dt={dt:.7f}s ({1.0 / dt:.2f} Hz).", flush=True)
    controller = _ArmHoldController(
        articulation, arm_indices, steer_indices, drive_indices, arm_rad
    )
    body_names = list(articulation.body_names)
    base_body_index = body_names.index("base_link") if "base_link" in body_names else 0
    link6_body_index = body_names.index("link6") if "link6" in body_names else None
    if link6_body_index is None:
        print("[WARN] link6 body not found; catch success will remain false.", flush=True)
    initial_base_body_pos = articulation.data.body_pos_w.torch[0, base_body_index].clone()
    initial_base_body_pose = articulation.data.body_pose_w.torch[
        0, base_body_index
    ].detach().cpu().numpy()
    controller.set_base_reference(initial_base_body_pose)
    print(
        f"[INFO] Articulation bodies={body_names}; "
        f"physical_base_index={base_body_index}; link6_index={link6_body_index}",
        flush=True,
    )
    throw_rng = np.random.default_rng(args.seed)
    stage = omni.usd.get_context().get_stage()
    marker_paths = _create_throw_markers(stage, args.hide_throw_markers)
    arm_motion = _LandingArmMotion(controller, args)
    keyboard = _KeyboardDriveInput(
        enable_viewport_keyboard=(getattr(args, "viz", "kit") != "none")
    )
    print(
        "[INFO] Base controls: W=forward S=backward A=left D=right "
        "SPACE=stop Q/ESC=quit.", flush=True,
    )
    episode = 0
    episode_step = 0
    episode_success = False
    episode_min_ring_distance = float("inf")
    episode_catch_time = None
    completed_episodes = 0
    successful_episodes = 0
    catch_results = []
    catch_results_dir = Path(args.inference_input_dir).expanduser().resolve()
    catch_results_path = _write_catch_results(
        catch_results_dir, catch_results, args, completed_episodes, successful_episodes
    )
    print(f"[INFO] Catch results will be written to {catch_results_path}", flush=True)
    launch, landing, flight_steps = _reset_throw_episode(
        sim, articulation, controller, tennis_ball, args, throw_rng, episode,
        stage, marker_paths, arm_motion,
    )
    recorder.start_episode(episode, launch, landing, flight_steps)
    throw_period_steps = (
        max(1, int(round(args.throw_period / dt))) if args.throw_period > 0.0 else 0
    )
    measured = _joint_pos_torch(articulation, arm_indices)
    print("[INFO] Initial measured joint angles (rad): %s"
          % " ".join("%+.4f" % float(v) for v in measured), flush=True)
    print(
        "[INFO] Upright articulation root target: pos=%s quat_xyzw=%s"
        % (ROBOT_ROOT_POS, ROBOT_ROOT_QUAT_XYZW),
        flush=True,
    )

    step = 0
    while simulation_app.is_running():
        if inference_bridge is not None:
            _update_inference_controller(inference_bridge, controller, episode)
            keys = set()
        else:
            keys = set(args.scripted_keys.lower()) if args.scripted_keys else keyboard.poll()
        if keyboard.quit_requested:
            print("[INFO] Keyboard quit requested.", flush=True)
            break
        # Keep the base pose fixed until the configured arm-start frame. This
        # also blocks manual/scripted base motion during the camera-stability
        # prefix; automatic landing-share motion is gated by arm_motion.update.
        base_unlocked = (
            inference_bridge.control_ready
            if inference_bridge is not None
            else episode_step >= args.arm_start_step
        )
        forward_speed = (
            args.linear_speed * (("w" in keys) - ("s" in keys))
            if base_unlocked else 0.0
        )
        yaw_rate = (
            args.turn_speed * (("a" in keys) - ("d" in keys))
            if base_unlocked else 0.0
        )
        controller.drive_base(forward_speed, yaw_rate, dt)

        # Match isaacsim_evaluate_tennis.py: after publishing an observation,
        # freeze the complete physical scene until the matching action chunk
        # arrives.  Polling above still enforces the wall-clock timeout, while
        # this branch intentionally skips sim.step(), timestamp increments,
        # ball updates, recorder frames, and episode progression.
        if inference_bridge is not None and inference_bridge.awaiting_action:
            _render_inference_hold(sim, articulation, controller, simulation_app)
            continue

        if args.static:
            # Nothing steps the scene in static mode; only keep the window alive.
            simulation_app.update()
        else:
            # Keep the free MJCF articulation root kinematic, matching
            # keyboard_drive_3dgs_scene.py's --kinematic-physics-step path.
            # Do not render the pose immediately after PhysX perturbs it: pin
            # the root again after the physics step, then render the corrected
            # pose. Rendering before the second pin produces visible jitter.
            if inference_bridge is None:
                arm_motion.update(episode_step)
            controller.update_wheel_animation(dt)
            controller.apply()
            articulation.write_data_to_sim()
            # Resolve the imported articulation around its physical base_link,
            # not around PhysX's automatically selected link root.
            sim.forward()
            articulation.update(dt)
            measured_root_pose = articulation.data.root_pose_w.torch[0].detach().cpu().numpy()
            measured_base_pose = articulation.data.body_pose_w.torch[
                0, base_body_index
            ].detach().cpu().numpy()
            controller.correct_root_for_physical_base(
                measured_root_pose, measured_base_pose
            )
            controller.apply()
            articulation.write_data_to_sim()
            sim.step(render=False)
            # Pass the real dt: update(0.0) never advances the internal
            # timestamp, so joint_pos reads after the first would return the
            # frozen cache.
            articulation.update(dt)
            # Re-apply the inference/manual desired pose after PhysX so the
            # free base cannot tilt before rendering and camera capture.
            measured_root_pose = articulation.data.root_pose_w.torch[0].detach().cpu().numpy()
            measured_base_pose = articulation.data.body_pose_w.torch[
                0, base_body_index
            ].detach().cpu().numpy()
            controller.correct_root_for_physical_base(
                measured_root_pose, measured_base_pose
            )
            controller.apply()
            articulation.write_data_to_sim()
            sim.forward()
            tennis_ball.update(dt)
            if link6_body_index is not None and args.catch_radius > 0.0:
                link6_pos = articulation.data.body_pos_w.torch[0, link6_body_index]
                ring_pos = _catch_center_from_link6_position(link6_pos)
                ball_pos = tennis_ball.data.root_pos_w.torch[0]
                ring_distance = float(torch.linalg.norm(ball_pos - ring_pos).item())
                episode_min_ring_distance = min(episode_min_ring_distance, ring_distance)
                if not episode_success and ring_distance <= args.catch_radius:
                    episode_success = True
                    episode_catch_time = (episode_step + 1) * dt
                    print(
                        "[CATCH] Episode %d: ball entered ring radius, distance=%.3f m time=%.3f s"
                        % (episode, ring_distance, episode_catch_time), flush=True,
                    )
            if (arm_motion.ball_target_step is not None
                    and episode_step + 1 == arm_motion.ball_target_step):
                ball_pos = tennis_ball.data.root_pos_w.torch[0].detach().cpu().numpy()
                target_error = ball_pos - arm_motion.ball_target_world
                print(
                    "[BALL-TARGET] step=%d actual=%s target=%s error=%s norm=%.5fm"
                    % (episode_step + 1,
                       tuple(round(float(v), 5) for v in ball_pos),
                       tuple(round(float(v), 5) for v in arm_motion.ball_target_world),
                       tuple(round(float(v), 5) for v in target_error),
                       float(np.linalg.norm(target_error))),
                    flush=True,
                )
            controller.apply()
            articulation.write_data_to_sim()
            sim.render()
            # Explicitly pump Kit after every render before reading annotators.
            # Without this, --viz kit can return RGB/depth from the previous
            # frame while state, ball position, and extrinsics are current.
            simulation_app.update()
            recorder.add_frame(
                articulation, arm_indices, base_body_index, link6_body_index,
                controller, tennis_ball
            )
            if (args.print_camera_extrinsics > 0
                    and episode_step % args.print_camera_extrinsics == 0):
                recorder.print_camera_extrinsics(episode, episode_step)
            if inference_bridge is not None:
                inference_bridge.capture(
                    episode, episode_step, articulation, arm_indices,
                    base_body_index,
                )
            if observation_video is not None and inference_bridge is not None:
                saved_input = getattr(inference_bridge, "last_input_directory", None)
                if saved_input is not None and saved_input[0] == episode:
                    observation_video.request_dirs.add(saved_input[1])
            if observation_video is not None:
                observation_video.capture(episode)
        if args.print_joints > 0 and step % args.print_joints == 0:
            measured = _joint_pos_torch(articulation, arm_indices)
            print(_format_joint_line(step, measured, controller.target[0]),
                  flush=True)
        if args.print_base > 0 and step % args.print_base == 0:
            root_actual = articulation.data.root_pose_w.torch[0].detach().cpu()
            root_lin_vel = articulation.data.root_lin_vel_w.torch[0].detach().cpu()
            root_ang_vel = articulation.data.root_ang_vel_w.torch[0].detach().cpu()
            base_actual = articulation.data.body_pos_w.torch[
                0, base_body_index
            ].detach().cpu()
            base_delta = base_actual - initial_base_body_pos.detach().cpu()
            print(
                "[BASE] step=%d keys=%s target_xy=(%.6f,%.6f) "
                "actual_root_pos=(%.6f,%.6f,%.6f) "
                "actual_base_pos=(%.6f,%.6f,%.6f) "
                "base_delta=(%+.6f,%+.6f,%+.6f) "
                "root_lin_vel=(%+.6f,%+.6f,%+.6f) "
                "root_ang_vel=(%+.6f,%+.6f,%+.6f)"
                % (step, "".join(sorted(keys)) or "-",
                   float(controller.root_pose[0, 0]), float(controller.root_pose[0, 1]),
                   *[float(v) for v in root_actual[:3]],
                   *[float(v) for v in base_actual],
                   *[float(v) for v in base_delta],
                   *[float(v) for v in root_lin_vel],
                   *[float(v) for v in root_ang_vel]),
                flush=True,
            )
        step += 1
        if throw_period_steps > 0 and step % throw_period_steps == 0:
            if observation_video is not None:
                observation_video.save_episode()
            recorder.save_episode()
            completed_episodes += 1
            successful_episodes += int(episode_success)
            catch_results.append({
                "episode": int(episode),
                "success": bool(episode_success),
                "min_ring_distance_m": (
                    float(episode_min_ring_distance)
                    if np.isfinite(episode_min_ring_distance) else None
                ),
                "catch_time_s": (
                    float(episode_catch_time)
                    if episode_catch_time is not None else None
                ),
                "duration_s": float(episode_step * dt),
                "steps": int(episode_step),
                "throw_position": [float(value) for value in launch],
                "landing_position": [float(value) for value in landing],
                "episodes_completed": int(completed_episodes),
                "successes_so_far": int(successful_episodes),
                "success_rate": float(successful_episodes / completed_episodes),
            })
            _write_catch_results(
                catch_results_dir, catch_results, args,
                completed_episodes, successful_episodes,
            )
            print(
                "[INFO] Episode %d ended: success=%s min_ring_distance=%s "
                "catch_time=%s; cumulative success rate=%.1f%%"
                % (
                    episode, episode_success,
                    "%.3f m" % episode_min_ring_distance
                    if np.isfinite(episode_min_ring_distance) else "N/A",
                    "%.3f s" % episode_catch_time
                    if episode_catch_time is not None else "N/A",
                    100.0 * successful_episodes / completed_episodes,
                ), flush=True,
            )
            if completed_episodes >= args.max_episodes:
                print(
                    f"[INFO] Reached --max-episodes={args.max_episodes}; exiting.",
                    flush=True,
                )
                break
            if recorder.complete:
                print("[INFO] Requested dataset episode count reached.", flush=True)
                break
            episode += 1
            launch, landing, flight_steps = _reset_throw_episode(
                sim, articulation, controller, tennis_ball, args, throw_rng, episode,
                stage, marker_paths, arm_motion,
            )
            recorder.start_episode(episode, launch, landing, flight_steps)
            if inference_bridge is not None:
                inference_bridge.reset_episode()
            episode_step = 0
            episode_success = False
            episode_min_ring_distance = float("inf")
            episode_catch_time = None
        else:
            episode_step += 1
        if args.max_steps > 0 and step >= args.max_steps:
            print(f"[INFO] Reached --max-steps={args.max_steps}; exiting.", flush=True)
            break
    keyboard.close()


def main():
    _validate_inputs(args_cli)
    _clear_inference_input_dir(args_cli)
    print(f"[INFO] hide_throw_markers={args_cli.hide_throw_markers}", flush=True)
    stage = _open_scene(args_cli)
    _apply_calibration(args_cli, stage)
    _apply_net_looks(stage, args_cli)
    _create_extra_arm_base_camera(stage)
    _set_arm_base_camera_positions(args_cli, stage)
    camera_render_products = _create_arm_base_camera_render_products(stage, args_cli)
    _create_camera_position_markers(stage, args_cli, args_cli.hide_throw_markers)
    # 机械臂初始位姿 (弧度制; 可由 --arm-init 覆盖)
    arm_rad = tuple(float(v) for v in args_cli.arm_init)
    _apply_arm_state(stage, arm_rad)
    sim, articulation, arm_indices, steer_indices, drive_indices, tennis_ball = _create_runtime_arm(
        stage, arm_rad, args_cli.device, args_cli.physics_dt
    )
    if args_cli.save_lerobot_dataset and not args_cli.dynamicvla_only_dataset:
        _label_segmentation_classes(stage, args_cli)
    recorder = _LeRobotRecorder(args_cli, camera_render_products, stage)
    inference_bridge = (
        _VLAInferenceBridge(args_cli, camera_render_products)
        if args_cli.inference_control else None
    )
    # AppLauncher disables this for ``--viz none`` and SimulationContext may
    # re-apply its rendering settings during construction.  Off-screen RTX
    # cameras still require dynamic PhysX poses to be copied into Fabric.
    if args_cli.enable_cameras:
        carb.settings.get_settings().set_bool(
            "/physics/fabricUpdateTransformations", True
        )
    simulation_app.update()  # one pump so the viewport/window exists
    _set_arm_target(articulation, arm_indices, arm_rad)
    _set_initial_view(args_cli, stage)
    observation_video = _ObservationVideoRecorder(args_cli)
    try:
        _run_loop(
            args_cli, sim, articulation, arm_indices, steer_indices,
            drive_indices, arm_rad, tennis_ball,
            recorder, inference_bridge, observation_video,
        )
    finally:
        observation_video.close()
        recorder.finalize()
        if inference_bridge is not None:
            inference_bridge.close()
    # Keep Replicator items alive for the full simulation loop.
    del camera_render_products


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr, flush=True)
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        if args_cli.clean_close:
            simulation_app.close(wait_for_replicator=False)
        else:
            os._exit(exit_code)
