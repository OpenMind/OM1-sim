# ruff: noqa: E402
"""
Isaac Sim robot simulation with ROS2 integration.
Supports Go2 (quadruped), G1 (humanoid), and TRON1 (bipedal wheelfoot) robots.

Default mode: warehouse or apartment scene with optional human pedestrian.
IRA mode (--ira): loads a pre-saved IRA scene with animated pedestrians, runs
  the robot alongside them, and logs pedestrian-collision events.

Control robot velocity:
  ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.3, y: 0.0}, angular: {z: 0.2}}"

Control human pedestrian (--human, default mode only):
  ros2 topic pub /cmd_vel_human geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0}, angular: {z: 0.3}}"

Examples
--------
  python run.py --robot_type go2                          # warehouse, Go2 (default)
  python run.py --robot_type g1 --environment apartment  # apartment, G1
  python run.py --human                                   # add a controllable pedestrian
  python run.py --ira                                     # IRA scene, Go2, animated pedestrians
  python run.py --ira --robot_type g1                    # IRA scene, G1
  python run.py --ira --ira_scene assets/scenes/ira_scene.usd --ira_config /path/to/config.yaml
  python run.py --realsense_tilt 0                       # realsense camera looks straight ahead
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

# Isaac Sim mutates sys.path during SimulationApp() init so cv2/utils/
# becomes importable as a bare "utils" module. Pre-load our local utils.py
# under the name "utils" so subsequent `import utils as ros_utils` and
# `from utils import ...` resolve correctly.
import importlib.util as _ilu
import os as _os
import sys as _sys

_local_utils_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "utils.py"
)
_spec = _ilu.spec_from_file_location("utils", _local_utils_path)
_mod = _ilu.module_from_spec(_spec)
_sys.modules["utils"] = _mod
_spec.loader.exec_module(_mod)
del _ilu, _os, _sys, _spec, _mod, _local_utils_path

import argparse
import json
import logging
import math
import os
import tempfile
import time
import traceback
from typing import Optional, Tuple

import carb
import numpy as np
import omni.appwindow
import omni.usd
import utils as ros_utils
import yaml
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.robot.policy.examples.controllers.config_loader import (
    get_action,
    get_observations,
    parse_env_config,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_GO2_POLICY_DIR = os.path.join(_SCRIPT_DIR, "checkpoints", "go2")
DEFAULT_G1_POLICY_DIR = os.path.join(_SCRIPT_DIR, "checkpoints", "g1")
DEFAULT_TRON1_POLICY_DIR = os.path.join(_SCRIPT_DIR, "checkpoints", "tron1")

ROBOT_GO2 = "go2"
ROBOT_G1 = "g1"
ROBOT_TRON1 = "tron1"

G1_INIT_HEIGHT = 1.05
G1_HISTORY_LENGTH = 5
TRON1_INIT_HEIGHT = 0.966
TRON1_HISTORY_LENGTH = 10
CMD_VEL_TIMEOUT = 0.5  # seconds – stop if no new /cmd_vel received

ENV_WAREHOUSE = "warehouse"
ENV_APARTMENT = "apartment"
APARTMENT_STAGE_PATH = "/World/Environment"
APARTMENT_USD_PATH = os.path.join(
    _SCRIPT_DIR, "assets", "environment", "Modern_Apartment.usdz"
)
APARTMENT_GROUND_Z = 0.22073

# IRA constants (used only with --ira)
_IRA_EXT_DIR = os.path.join(
    _SCRIPT_DIR,
    "env_isaacsim", "lib", "python3.11", "site-packages", "isaacsim",
    "extscache", "isaacsim.replicator.agent.core-0.7.28+107.3.3",
)
DEFAULT_IRA_SCENE = os.path.join(_SCRIPT_DIR, "assets", "scenes", "ira_scene.usd")
DEFAULT_COMMAND_FILE = os.path.join(_IRA_EXT_DIR, "config", "default_command.txt")
DEFAULT_CHARACTER_ROOT = "/World/Characters"
COLLISION_RADIUS = 0.8
COLLISION_COOLDOWN = 1.0

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _expand_param(value, size: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float32)
    if isinstance(value, (int, float)):
        return np.full(size, float(value), dtype=np.float32)
    arr = np.array(value, dtype=np.float32)
    if arr.size != size:
        return np.full(size, default, dtype=np.float32)
    return arr


def _resolve_command_limits(
    deploy_cfg: dict, env_cfg: dict
) -> Tuple[np.ndarray, np.ndarray]:
    def _from_cfg(cfg: dict):
        ranges = (
            cfg.get("commands", {}).get("base_velocity", {}).get("ranges")
            if cfg
            else None
        )
        if not isinstance(ranges, dict):
            return None

        def _pair(key: str, fallback):
            val = ranges.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    return float(val[0]), float(val[1])
                except Exception:
                    return fallback
            return fallback

        return np.array(
            [
                _pair("lin_vel_x", (-1.0, 1.0)),
                _pair("lin_vel_y", (-1.0, 1.0)),
                _pair("ang_vel_z", (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    pairs = _from_cfg(deploy_cfg)
    if pairs is None:
        pairs = _from_cfg(env_cfg)
    if pairs is None:
        pairs = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)

    return pairs[:, 0], pairs[:, 1]


def _resolve_usd_path(env_cfg: dict, robot_type: str = ROBOT_GO2) -> str:
    if robot_type == ROBOT_TRON1:
        local = os.path.join(_SCRIPT_DIR, "assets", "tron1", "usd", "tron1.usd")
        if os.path.isfile(local):
            logger.info("Using local TRON1 USD model: %s", local)
            return local
        usd_path = env_cfg.get("scene", {}).get("robot", {}).get("spawn", {}).get("usd_path")
        if usd_path:
            p = usd_path if os.path.isabs(usd_path) else os.path.join(_SCRIPT_DIR, usd_path)
            if os.path.isfile(p):
                return p
        logger.error("Could not find TRON1 USD model")
        return ""

    if robot_type == ROBOT_G1:
        local = os.path.join(_SCRIPT_DIR, "assets", "g1", "usd", "g1.usd")
        if os.path.isfile(local):
            logger.info("Using local G1 USD model: %s", local)
            return local
        usd_path = env_cfg.get("scene", {}).get("robot", {}).get("spawn", {}).get("usd_path")
        if usd_path:
            p = usd_path if os.path.isabs(usd_path) else os.path.join(_SCRIPT_DIR, usd_path)
            if os.path.isfile(p):
                return p
        assets_root = get_assets_root_path()
        if assets_root:
            return assets_root + "/Isaac/Robots/Unitree/G1/g1.usd"
        carb.log_error("Could not find Isaac Sim assets folder")
        return ""

    local = os.path.join(_SCRIPT_DIR, "assets", "go2", "usd", "go2.usd")
    if os.path.isfile(local):
        logger.info("Using local Go2 USD model: %s", local)
        return local
    usd_path = env_cfg.get("scene", {}).get("robot", {}).get("spawn", {}).get("usd_path")
    if usd_path:
        p = usd_path if os.path.isabs(usd_path) else os.path.join(_SCRIPT_DIR, usd_path)
        if os.path.isfile(p):
            return p
        logger.warning("USD path from env.yaml not found: %s", usd_path)
        return usd_path
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return ""
    return assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"


def _configure_ros_utils_paths(robot_root: str, robot_type: str = ROBOT_GO2) -> None:
    """Configure ROS utils prim paths based on robot type."""
    ros_utils.GO2_STAGE_PATH = robot_root

    if robot_type == ROBOT_TRON1:
        base_link = f"{robot_root}/base_Link"
        ros_utils.IMU_PRIM = f"{base_link}/imu_link"
        ros_utils.CAMERA_LINK_PRIM = f"{base_link}/camera_link"
        ros_utils.REALSENSE_DEPTH_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/realsense_depth_camera"
        ros_utils.REALSENSE_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/realsense_rgb_camera"
        ros_utils.FRONT_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/front_rgb_camera"
        ros_utils.TOP_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/top_rgb_camera"
        ros_utils.L1_LINK_PRIM = f"{base_link}/lidar_l1_link"
        ros_utils.L1_LIDAR_PRIM = f"{ros_utils.L1_LINK_PRIM}/lidar_l1_rtx"
        ros_utils.VELO_BASE_LINK_PRIM = f"{base_link}/velodyne_base_link"
        ros_utils.VELO_LASER_LINK_PRIM = f"{ros_utils.VELO_BASE_LINK_PRIM}/laser"
        ros_utils.VELO_LIDAR_PRIM = f"{ros_utils.VELO_LASER_LINK_PRIM}/velodyne_vlp16_rtx"
        return

    if robot_type == ROBOT_G1:
        base_link = f"{robot_root}/torso_link"
        ros_utils.IMU_PRIM = f"{base_link}/imu_link"
        ros_utils.CAMERA_LINK_PRIM = f"{base_link}/camera_link"
        ros_utils.REALSENSE_DEPTH_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/realsense_depth_camera"
        ros_utils.REALSENSE_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/realsense_rgb_camera"
        ros_utils.FRONT_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/front_rgb_camera"
        ros_utils.TOP_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/top_rgb_camera"
        ros_utils.L1_LINK_PRIM = f"{base_link}/lidar_l1_link"
        ros_utils.L1_LIDAR_PRIM = f"{ros_utils.L1_LINK_PRIM}/lidar_l1_rtx"
        ros_utils.VELO_BASE_LINK_PRIM = f"{base_link}/velodyne_base_link"
        ros_utils.VELO_LASER_LINK_PRIM = f"{ros_utils.VELO_BASE_LINK_PRIM}/laser"
        ros_utils.VELO_LIDAR_PRIM = f"{ros_utils.VELO_LASER_LINK_PRIM}/velodyne_vlp16_rtx"
    else:
        ros_utils.IMU_PRIM = f"{robot_root}/base/imu_link"
        ros_utils.CAMERA_LINK_PRIM = f"{robot_root}/base/camera_link"
        ros_utils.REALSENSE_DEPTH_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/realsense_depth_camera"
        ros_utils.REALSENSE_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/realsense_rgb_camera"
        ros_utils.FRONT_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/front_rgb_camera"
        ros_utils.TOP_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/top_rgb_camera"
        ros_utils.L1_LINK_PRIM = f"{robot_root}/base/lidar_l1_link"
        ros_utils.L1_LIDAR_PRIM = f"{ros_utils.L1_LINK_PRIM}/lidar_l1_rtx"
        ros_utils.VELO_BASE_LINK_PRIM = f"{robot_root}/base/velodyne_base_link"
        ros_utils.VELO_LASER_LINK_PRIM = f"{ros_utils.VELO_BASE_LINK_PRIM}/laser"
        ros_utils.VELO_LIDAR_PRIM = f"{ros_utils.VELO_LASER_LINK_PRIM}/velodyne_vlp16_rtx"


def _add_apartment_environment() -> bool:
    """Reference the Modern Apartment USDZ and set up matching ground + lighting."""
    import omni.usd
    from isaacsim.core.utils import stage as stage_utils
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    if not os.path.isfile(APARTMENT_USD_PATH):
        carb.log_error(f"Apartment USD file not found: {APARTMENT_USD_PATH}")
        return False

    usd_stage = omni.usd.get_context().get_stage()
    stage_utils.add_reference_to_stage(APARTMENT_USD_PATH, APARTMENT_STAGE_PATH)

    env_prim = usd_stage.GetPrimAtPath(APARTMENT_STAGE_PATH)
    if not env_prim or not env_prim.IsValid():
        carb.log_error(f"Failed to load apartment environment USD: {APARTMENT_USD_PATH}")
        return False

    xform = UsdGeom.Xformable(env_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.012))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))

    ground_path = "/World/GroundPlane"
    ground_xform_prim = usd_stage.DefinePrim(ground_path, "Xform")
    UsdGeom.Xformable(ground_xform_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, APARTMENT_GROUND_Z))
    plane_prim = usd_stage.DefinePrim(f"{ground_path}/CollisionPlane", "Plane")
    plane_geom = UsdGeom.Plane(plane_prim)
    plane_geom.CreateAxisAttr("Z")
    plane_geom.CreateExtentAttr([(-50, -50, 0), (50, 50, 0)])
    UsdPhysics.CollisionAPI.Apply(plane_prim)
    UsdGeom.Imageable(plane_prim).MakeInvisible()

    dome_light = UsdLux.DomeLight.Define(usd_stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(500.0)
    distant_light = UsdLux.DistantLight.Define(usd_stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(1500.0)
    UsdGeom.Xformable(distant_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    ceiling_z = APARTMENT_GROUND_Z + 2.5
    for i, (x, y) in enumerate(
        [(0.0, 0.0), (5.0, 0.0), (-3.0, 0.0), (0.0, -2.0),
         (5.0, -2.0), (-3.0, -2.0), (10.0, 0.0), (10.0, -2.0)]
    ):
        sphere_light = UsdLux.SphereLight.Define(usd_stage, f"/World/CeilingLight_{i}")
        sphere_light.CreateIntensityAttr(30000.0)
        sphere_light.CreateRadiusAttr(0.15)
        sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.9))
        UsdGeom.Xformable(sphere_light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, ceiling_z))

    logger.info("Modern Apartment environment loaded")
    return True


def _validate_policy_paths(
    policy_dir: str, robot_type: str = ROBOT_GO2
) -> Tuple[str, str, str]:
    policy_path = os.path.join(policy_dir, "exported", "policy.pt")
    env_path = os.path.join(policy_dir, "params", "env.yaml")
    deploy_path = os.path.join(policy_dir, "params", "deploy.yaml")

    missing = []
    if not os.path.isfile(policy_path):
        missing.append(policy_path)
    if robot_type != ROBOT_TRON1 and not os.path.isfile(env_path):
        missing.append(env_path)
    if robot_type == ROBOT_TRON1:
        encoder_path = os.path.join(policy_dir, "exported", "encoder.pt")
        if not os.path.isfile(encoder_path):
            missing.append(encoder_path)

    if missing:
        for path in missing:
            carb.log_error(f"Missing required policy file: {path}")
        raise FileNotFoundError("Required policy files not found.")

    if not os.path.isfile(deploy_path):
        logger.warning("deploy.yaml not found at %s; using env.yaml ranges only", deploy_path)

    return policy_path, env_path, deploy_path


# ---------------------------------------------------------------------------
# Policy controllers
# ---------------------------------------------------------------------------

class Go2VelocityPolicy(PolicyController):
    """The Unitree Go2 running a velocity tracking locomotion policy."""

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        env_path: str,
        root_path: Optional[str] = None,
        name: str = "go2",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        self.load_policy(policy_path, env_path)

        self._obs_order = [
            "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos_rel", "joint_vel_rel", "last_action",
        ]

        obs_cfg = get_observations(self.policy_env_params) or {}
        self._obs_scales = {}
        for name in self._obs_order:
            scale = obs_cfg.get(name, {}).get("scale")
            if scale is None:
                self._obs_scales[name] = 1.0
            elif isinstance(scale, (int, float)):
                self._obs_scales[name] = float(scale)
            else:
                self._obs_scales[name] = np.array(scale, dtype=np.float32)

        action_terms = get_action(self.policy_env_params) or {}
        self._action_cfg = next(iter(action_terms.values()), {})
        self._action_scale = None
        self._action_offset = None
        self._previous_action = None
        self._policy_counter = 0

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view=physics_sim_view, control_mode="position")
        dof_count = len(self.default_pos)
        self._action_scale = _expand_param(self._action_cfg.get("scale"), dof_count, 1.0)
        if self._action_cfg.get("use_default_offset", False):
            self._action_offset = np.array(self.default_pos, dtype=np.float32)
        else:
            self._action_offset = _expand_param(self._action_cfg.get("offset"), dof_count, 0.0)
        self._previous_action = np.zeros(dof_count, dtype=np.float32)
        self.action = np.zeros(dof_count, dtype=np.float32)

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()
        R_BI = quat_to_rot_matrix(q_IB).transpose()
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        joint_pos_rel = current_joint_pos - self.default_pos
        return np.concatenate([
            ang_vel_b * self._obs_scales["base_ang_vel"],
            gravity_b * self._obs_scales["projected_gravity"],
            command * self._obs_scales["velocity_commands"],
            joint_pos_rel * self._obs_scales["joint_pos_rel"],
            current_joint_vel * self._obs_scales["joint_vel_rel"],
            self._previous_action * self._obs_scales["last_action"],
        ], axis=0).astype(np.float32)

    def forward(self, dt: float, command: np.ndarray) -> None:
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = np.array(self._compute_action(obs), dtype=np.float32)
            self._previous_action = self.action.copy()
        target_pos = self._action_offset + (self._action_scale * self.action)
        self.robot.apply_action(ArticulationAction(joint_positions=target_pos))
        self._policy_counter += 1


class G1VelocityPolicy(PolicyController):
    """Unitree G1 humanoid with observation history."""

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        env_path: str,
        deploy_path: Optional[str] = None,
        root_path: Optional[str] = None,
        name: str = "g1",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        history_length: int = G1_HISTORY_LENGTH,
    ) -> None:
        import torch
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim, get_prim_at_path

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        self.robot = SingleArticulation(
            prim_path=root_path or prim_path, name=name,
            position=position, orientation=orientation,
        )

        if not deploy_path or not os.path.isfile(deploy_path):
            raise FileNotFoundError(f"deploy.yaml required for G1: {deploy_path}")
        self._deploy_cfg = _load_yaml(deploy_path)

        import io
        import omni.client
        self.policy = torch.jit.load(
            io.BytesIO(memoryview(omni.client.read_file(policy_path)[2]).tobytes())
        )

        self._decimation = int(self._deploy_cfg.get("step_dt", 0.02) / 0.005)
        if "decimation" in self._deploy_cfg:
            self._decimation = self._deploy_cfg["decimation"]
        self._history_length = history_length
        self._joint_ids_map = self._deploy_cfg.get("joint_ids_map")

        deploy_obs_cfg = self._deploy_cfg.get("observations", {})
        obs_names = [
            "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos_rel", "joint_vel_rel", "last_action",
        ]
        self._obs_scales = {}
        for n in obs_names:
            scale = deploy_obs_cfg.get(n, {}).get("scale")
            if scale is None:
                self._obs_scales[n] = 1.0
            elif isinstance(scale, (int, float)):
                self._obs_scales[n] = float(scale)
            else:
                self._obs_scales[n] = np.array(scale, dtype=np.float32)

        self._action_cfg = self._deploy_cfg.get("actions", {}).get("JointPositionAction", {})
        self._action_scale = None
        self._action_offset = None
        self._previous_action = None
        self._policy_counter = 0
        self._default_pos_sim = np.array(self._deploy_cfg.get("default_joint_pos", []), dtype=np.float32)
        self._stiffness_sdk = np.array(self._deploy_cfg.get("stiffness", []), dtype=np.float32)
        self._damping_sdk = np.array(self._deploy_cfg.get("damping", []), dtype=np.float32)
        self.default_pos = None
        self.default_vel = None
        self._obs_term_names = obs_names
        self._obs_term_histories = None

    def _sdk_to_sim(self, arr: np.ndarray) -> np.ndarray:
        if self._joint_ids_map is None or len(arr) != len(self._joint_ids_map):
            return arr
        out = np.zeros_like(arr)
        for sim_idx, sdk_idx in enumerate(self._joint_ids_map):
            out[sim_idx] = arr[sdk_idx]
        return out

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.policy(torch.from_numpy(obs).view(1, -1).float()).detach().view(-1).numpy()

    def post_reset(self) -> None:
        self.robot.post_reset()

    def initialize(self, physics_sim_view=None) -> None:
        from omni.physx import get_physx_simulation_interface
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        self.robot.get_articulation_controller().switch_control_mode("position")

        dof_count = len(self.robot.dof_names)
        if len(self._default_pos_sim) != dof_count:
            raise ValueError(
                f"deploy.yaml default_joint_pos has {len(self._default_pos_sim)} values, expected {dof_count}"
            )
        self.default_pos = self._default_pos_sim.copy()
        self.default_vel = np.zeros(dof_count, dtype=np.float32)

        if len(self._stiffness_sdk) == dof_count:
            damping = self._sdk_to_sim(self._damping_sdk) if len(self._damping_sdk) == dof_count else None
            self.robot._articulation_view.set_gains(self._sdk_to_sim(self._stiffness_sdk), damping)

        self.robot.set_joint_positions(self.default_pos)
        self.robot.set_joint_velocities(self.default_vel)

        self._action_scale = _expand_param(self._action_cfg.get("scale"), dof_count, 0.25)
        offset_val = self._action_cfg.get("offset")
        if (offset_val is not None and isinstance(offset_val, (list, np.ndarray))
                and len(offset_val) == dof_count):
            self._action_offset = np.array(offset_val, dtype=np.float32)
        else:
            self._action_offset = self.default_pos.copy()

        self._previous_action = np.zeros(dof_count, dtype=np.float32)
        self.action = np.zeros(dof_count, dtype=np.float32)

        term_sizes = {
            "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
            "joint_pos_rel": dof_count, "joint_vel_rel": dof_count, "last_action": dof_count,
        }
        self._obs_term_histories = {
            n: [np.zeros(term_sizes[n], dtype=np.float32) for _ in range(self._history_length)]
            for n in self._obs_term_names
        }

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        _, q_IB = self.robot.get_world_pose()
        R_BI = quat_to_rot_matrix(q_IB).transpose()
        ang_vel_b = np.matmul(R_BI, self.robot.get_angular_velocity())
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        current = {
            "base_ang_vel": (ang_vel_b * self._obs_scales["base_ang_vel"]).astype(np.float32),
            "projected_gravity": (gravity_b * self._obs_scales["projected_gravity"]).astype(np.float32),
            "velocity_commands": (command * self._obs_scales["velocity_commands"]).astype(np.float32),
            "joint_pos_rel": ((joint_pos - self.default_pos) * self._obs_scales["joint_pos_rel"]).astype(np.float32),
            "joint_vel_rel": (joint_vel * self._obs_scales["joint_vel_rel"]).astype(np.float32),
            "last_action": (self._previous_action * self._obs_scales["last_action"]).astype(np.float32),
        }
        for n in self._obs_term_names:
            self._obs_term_histories[n].pop(0)
            self._obs_term_histories[n].append(current[n])
        return np.concatenate([np.concatenate(self._obs_term_histories[n]) for n in self._obs_term_names])

    def forward(self, dt: float, command: np.ndarray) -> None:
        if self._policy_counter % self._decimation == 0:
            self.action = np.array(self._compute_action(self._compute_observation(command)), dtype=np.float32)
            self._previous_action = self.action.copy()
        target_pos = self._action_offset + (self._action_scale * self.action)
        self.robot.apply_action(ArticulationAction(joint_positions=target_pos))
        self._policy_counter += 1


class Tron1VelocityPolicy:
    """LimX TRON1 WheelFoot with observation history encoder."""

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        encoder_path: str,
        deploy_path: str,
        root_path: Optional[str] = None,
        name: str = "tron1",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        history_length: int = TRON1_HISTORY_LENGTH,
    ) -> None:
        import torch
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim, get_prim_at_path

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        self.robot = SingleArticulation(
            prim_path=root_path or prim_path, name=name,
            position=position, orientation=orientation,
        )

        self._deploy_cfg = _load_yaml(deploy_path)
        if not self._deploy_cfg:
            raise FileNotFoundError(f"deploy.yaml required for TRON1: {deploy_path}")

        import io
        import omni.client
        self.policy = torch.jit.load(
            io.BytesIO(memoryview(omni.client.read_file(policy_path)[2]).tobytes())
        )
        self.encoder_model = torch.jit.load(
            io.BytesIO(memoryview(omni.client.read_file(encoder_path)[2]).tobytes())
        )

        self._decimation = self._deploy_cfg.get("decimation", 4)
        self._history_length = self._deploy_cfg.get("history_length", history_length)
        self._num_leg_joints = self._deploy_cfg.get("num_leg_joints", 6)
        self._num_wheel_joints = self._deploy_cfg.get("num_wheel_joints", 2)

        deploy_obs_cfg = self._deploy_cfg.get("observations", {})
        self._obs_scales = {}
        for n in ["base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel", "last_action"]:
            scale = deploy_obs_cfg.get(n, {}).get("scale")
            if scale is None:
                self._obs_scales[n] = 1.0
            elif isinstance(scale, (int, float)):
                self._obs_scales[n] = float(scale)
            else:
                self._obs_scales[n] = np.array(scale, dtype=np.float32)

        leg_cfg = self._deploy_cfg.get("actions", {}).get("leg", {})
        wheel_cfg = self._deploy_cfg.get("actions", {}).get("wheel", {})
        self._leg_action_scale = leg_cfg.get("scale", 0.25)
        self._wheel_action_scale = wheel_cfg.get("scale", 1.0)
        self._default_pos_cfg = np.array(self._deploy_cfg.get("default_joint_pos", [0.0] * 8), dtype=np.float32)
        self._stiffness = np.array(self._deploy_cfg.get("stiffness", []), dtype=np.float32)
        self._damping = np.array(self._deploy_cfg.get("damping", []), dtype=np.float32)
        self.default_pos = None
        self._previous_action = None
        self._policy_counter = 0
        self._obs_history = None

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.policy(torch.from_numpy(obs).view(1, -1).float()).detach().view(-1).numpy()

    def _encode(self, obs_history: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.encoder_model(torch.from_numpy(obs_history).view(1, -1).float()).detach().view(-1).numpy()

    def post_reset(self) -> None:
        self.robot.post_reset()

    def initialize(self, physics_sim_view=None) -> None:
        from omni.physx import get_physx_simulation_interface
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        self.robot.get_articulation_controller().switch_control_mode("position")

        dof_count = len(self.robot.dof_names)
        if len(self._default_pos_cfg) != dof_count:
            raise ValueError(
                f"deploy.yaml default_joint_pos has {len(self._default_pos_cfg)} values, expected {dof_count}"
            )
        self.default_pos = self._default_pos_cfg.copy()
        if len(self._stiffness) == dof_count and len(self._damping) == dof_count:
            self.robot._articulation_view.set_gains(self._stiffness, self._damping)
        self.robot.set_joint_positions(self.default_pos)
        self.robot.set_joint_velocities(np.zeros(dof_count, dtype=np.float32))
        self._previous_action = np.zeros(dof_count, dtype=np.float32)
        self.action = np.zeros(dof_count, dtype=np.float32)
        obs_dim = 3 + 3 + self._num_leg_joints + dof_count + dof_count
        self._obs_history = [np.zeros(obs_dim, dtype=np.float32) for _ in range(self._history_length)]

    def _compute_observation(self, command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, q_IB = self.robot.get_world_pose()
        R_BI = quat_to_rot_matrix(q_IB).transpose()
        ang_vel_b = np.matmul(R_BI, self.robot.get_angular_velocity())
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        leg_pos_rel = joint_pos[:self._num_leg_joints] - self.default_pos[:self._num_leg_joints]
        obs = np.concatenate([
            ang_vel_b * self._obs_scales["base_ang_vel"],
            gravity_b * self._obs_scales["projected_gravity"],
            leg_pos_rel * self._obs_scales["joint_pos_rel"],
            joint_vel * self._obs_scales["joint_vel_rel"],
            self._previous_action * self._obs_scales["last_action"],
        ]).astype(np.float32)
        self._obs_history.pop(0)
        self._obs_history.append(obs.copy())
        return obs, np.concatenate(self._obs_history).astype(np.float32)

    def forward(self, dt: float, command: np.ndarray) -> None:
        if self._policy_counter % self._decimation == 0:
            obs, obs_history_flat = self._compute_observation(command)
            latent = self._encode(obs_history_flat)
            policy_input = np.concatenate([latent, obs, command]).astype(np.float32)
            self.action = np.array(self._compute_action(policy_input), dtype=np.float32)
            self._previous_action = self.action.copy()
        n_leg = self._num_leg_joints
        target_pos = self.default_pos.copy()
        target_pos[:n_leg] = self.default_pos[:n_leg] + self._leg_action_scale * self.action[:n_leg]
        target_vel = np.zeros_like(self.default_pos)
        target_vel[n_leg:] = self._wheel_action_scale * self.action[n_leg:]
        self.robot.apply_action(ArticulationAction(joint_positions=target_pos, joint_velocities=target_vel))
        self._policy_counter += 1


# ---------------------------------------------------------------------------
# Default-mode runner (warehouse / apartment, optional human pedestrian)
# ---------------------------------------------------------------------------

class RobotRosRunner(object):
    """Runner for warehouse/apartment scenes with optional human pedestrian."""

    def __init__(
        self,
        physics_dt: float,
        render_dt: float,
        policy_dir: str,
        cmd_vel_topic: str,
        vx_max: float,
        vy_max: float,
        wz_max: float,
        robot_root: str,
        cmd_vel_only: bool,
        enable_sensors: bool,
        enable_keyboard: bool,
        enable_human: bool = False,
        human_auto_walk: bool = False,
        human_cmd_topic: str = "/cmd_vel_human",
        human_pos: tuple = (2.0, 0.0, 0.0),
        human_yaw: float = 0.0,
        human_vel_max: float = 1.5,
        human_yaw_rate_max: float = 1.0,
        human_scale: float = 1.0,
        robot_type: str = ROBOT_GO2,
        environment: str = ENV_WAREHOUSE,
        realsense_tilt_deg: float = -25.0,
    ) -> None:
        self._robot_type = robot_type
        self._realsense_tilt_deg = realsense_tilt_deg
        policy_path, env_path, deploy_path = _validate_policy_paths(policy_dir, robot_type)

        env_cfg = parse_env_config(env_path) if os.path.isfile(env_path) else {}
        deploy_cfg = _load_yaml(deploy_path)
        usd_path = _resolve_usd_path(env_cfg, robot_type)

        if robot_type == ROBOT_TRON1:
            default_z = TRON1_INIT_HEIGHT
        elif robot_type == ROBOT_G1:
            default_z = G1_INIT_HEIGHT
        else:
            default_z = 0.4
        init_pos = np.array(
            env_cfg.get("scene", {}).get("robot", {}).get("init_state", {}).get("pos",
            (0.0, 0.0, default_z))
        )

        self._environment = environment
        if environment == ENV_APARTMENT:
            robot_height = TRON1_INIT_HEIGHT if robot_type == ROBOT_TRON1 else (
                G1_INIT_HEIGHT if robot_type == ROBOT_G1 else 0.45
            )
            init_pos[2] = APARTMENT_GROUND_Z + robot_height

        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self._physics_dt = physics_dt
        self._render_dt = render_dt

        if environment == ENV_APARTMENT:
            if not _add_apartment_environment():
                raise RuntimeError("Failed to load Modern Apartment environment")
            self._hide_default_ground()
        else:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                raise RuntimeError("Could not find Isaac Sim assets folder")
            prim = define_prim("/World/Warehouse", "Xform")
            prim.GetReferences().AddReference(
                assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
            )

        if not usd_path:
            raise RuntimeError(f"{robot_type.upper()} USD path could not be resolved")

        if robot_type == ROBOT_TRON1:
            encoder_path = os.path.join(policy_dir, "exported", "encoder.pt")
            self._robot = Tron1VelocityPolicy(
                prim_path=robot_root, name="Tron1", usd_path=usd_path,
                position=init_pos, policy_path=policy_path,
                encoder_path=encoder_path, deploy_path=deploy_path,
            )
        elif robot_type == ROBOT_G1:
            self._robot = G1VelocityPolicy(
                prim_path=robot_root, name="G1", usd_path=usd_path,
                position=init_pos, policy_path=policy_path,
                env_path=env_path, deploy_path=deploy_path,
            )
        else:
            self._robot = Go2VelocityPolicy(
                prim_path=robot_root, name="Go2", usd_path=usd_path,
                position=init_pos, policy_path=policy_path, env_path=env_path,
            )

        cmd_min, cmd_max = _resolve_command_limits(deploy_cfg, env_cfg)
        self._cmd_min = np.maximum(cmd_min, np.array([-vx_max, -vy_max, -wz_max], dtype=np.float32))
        self._cmd_max = np.minimum(cmd_max, np.array([vx_max, vy_max, wz_max], dtype=np.float32))

        self._cmd_vel_topic = cmd_vel_topic
        self._cmd_vel_only = cmd_vel_only
        self._enable_sensors = enable_sensors
        self._enable_keyboard = enable_keyboard
        self._vx_max = vx_max
        self._vy_max = vy_max
        self._wz_max = wz_max
        self._linear_attr = None
        self._angular_attr = None
        self._sensors = {}
        self._robot_root = robot_root
        _configure_ros_utils_paths(robot_root, robot_type)

        self._base_command = np.zeros(3, dtype=np.float32)
        self._last_cmd_vel_time: Optional[float] = None
        self._last_cmd_vel_count: int = 0
        self._cmd_vel_count_attr = None

        cmd_scale = np.maximum(np.abs(self._cmd_min), np.abs(self._cmd_max))
        vx, vy, wz = cmd_scale.tolist()
        self._input_keyboard_mapping = {
            "NUMPAD_8": [vx, 0.0, 0.0], "UP": [vx, 0.0, 0.0],
            "NUMPAD_2": [-vx, 0.0, 0.0], "DOWN": [-vx, 0.0, 0.0],
            "NUMPAD_6": [0.0, -vy, 0.0], "RIGHT": [0.0, -vy, 0.0],
            "NUMPAD_4": [0.0, vy, 0.0], "LEFT": [0.0, vy, 0.0],
            "NUMPAD_7": [0.0, 0.0, wz], "N": [0.0, 0.0, wz],
            "NUMPAD_9": [0.0, 0.0, -wz], "M": [0.0, 0.0, -wz],
        }
        self.needs_reset = False
        self.first_step = True

        self._enable_human = enable_human
        self._human_cmd_topic = human_cmd_topic
        self._human_init_pos = human_pos
        self._human_init_yaw = human_yaw
        self._human_vel_max = human_vel_max
        self._human_yaw_rate_max = human_yaw_rate_max
        self._human_scale = human_scale
        self._human_prim = None
        self._human_vel_attr = None
        self._human_yaw_rate_attr = None
        self._human_pos = [human_pos[0], human_pos[1]]
        self._human_yaw = human_yaw
        self._human_auto_walk = human_auto_walk
        self._walk_phase = 0
        self._walk_phase_time = 0.0
        self._walk_phase_limit = [4.0, 1.57]
        self._bump_threshold = 0.8
        self._last_bump_log_time = -999.0

    def _hide_default_ground(self) -> None:
        import omni.usd
        usd_stage = omni.usd.get_context().get_stage()
        for path in ["/World/defaultGroundPlane", "/World/GroundPlane/GroundPlane",
                     "/World/ground", "/World/ground/GroundPlane"]:
            prim = usd_stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
            for child in prim.GetAllChildren():
                child_imageable = UsdGeom.Imageable(child)
                if child_imageable:
                    child_imageable.MakeInvisible()

    def setup(self) -> None:
        if self._enable_keyboard:
            self._appwindow = omni.appwindow.get_default_app_window()
            if self._appwindow is not None:
                self._input = carb.input.acquire_input_interface()
                self._keyboard = self._appwindow.get_keyboard()
                self._sub_keyboard = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._sub_keyboard_event
                )
            else:
                logger.warning("No app window found; keyboard control disabled")
                self._enable_keyboard = False
        self._world.add_physics_callback("go2_ros2_step", callback_fn=self.on_physics_step)

    def setup_human(self) -> None:
        if not self._enable_human:
            return
        human_usdz_path = os.path.join(_SCRIPT_DIR, "assets", "human", "human.usdz")
        if not os.path.isfile(human_usdz_path):
            human_usdz_path = os.path.join(os.path.dirname(_SCRIPT_DIR), "human.usdz")
        if not os.path.isfile(human_usdz_path):
            logger.warning("Human model not found at %s, skipping", human_usdz_path)
            self._enable_human = False
            return
        self._human_prim = ros_utils.add_human_model(
            human_usdz_path, position=self._human_init_pos,
            rotation_yaw=self._human_init_yaw, scale=self._human_scale,
        )
        if self._human_prim:
            self._human_vel_attr, self._human_yaw_rate_attr = (
                ros_utils.setup_human_cmd_graph(self._human_cmd_topic)
            )
        else:
            logger.warning("Failed to add human model")
            self._enable_human = False

    def setup_ros(self) -> None:
        self._linear_attr, self._angular_attr, self._cmd_vel_count_attr = (
            ros_utils.setup_cmd_vel_graph(self._cmd_vel_topic)
        )
        if not self._enable_sensors:
            return

        for _ in range(10):
            self._world.step(render=True)

        render_hz = 1.0 / self._render_dt if self._render_dt else None

        if self._robot_type == ROBOT_G1:
            camera_link_pos = (0.0, 0.0, 0.75)
            lidar_l1_pos = (0.2, 0.0, 0.4)
            lidar_velo_pos = (0.15, 0.0, 0.5)
        else:
            camera_link_pos = (0.3, 0.0, 0.10)
            lidar_l1_pos = (0.3, 0.0, 0.08)
            lidar_velo_pos = (0.25, 0.0, 0.13)

        self._sensors = ros_utils.setup_sensors_delayed(
            simulation_app, render_hz=render_hz,
            camera_link_position=camera_link_pos, enable_lidar=True,
            lidar_l1_position=lidar_l1_pos, lidar_velo_position=lidar_velo_pos,
            robot_type=self._robot_type, realsense_tilt_deg=self._realsense_tilt_deg,
        )

        for _ in range(5):
            self._world.step(render=True)

        ros_utils.setup_ros_publishers(
            self._sensors, simulation_app, robot_type=self._robot_type,
            camera_link_pos=camera_link_pos, lidar_l1_pos=lidar_l1_pos,
            lidar_velo_pos=lidar_velo_pos,
        )
        ros_utils.setup_depth_camerainfo_graph(
            simulation_app,
            topic="/camera/realsense2_camera_node/depth/camera_info",
            frame_id="realsense_depth_camera",
            width=480, height=270, fx=242.479, fy=242.479, cx=242.736, cy=133.273,
        )
        ros_utils.setup_odom_publisher(simulation_app)
        ros_utils.setup_color_camera_publishers(self._sensors, simulation_app, self._robot_type)
        ros_utils.setup_color_camerainfo_graph(
            simulation_app,
            topic="/camera/realsense2_camera_node/color/camera_info",
            frame_id="realsense_depth_camera",
            width=424, height=240, fx=214.157, fy=215.537, cx=214.384, cy=118.465,
        )
        ros_utils.setup_joint_states_publisher(simulation_app, robot_type=self._robot_type)

    def _get_cmd_vel(self) -> Optional[np.ndarray]:
        if self._linear_attr is None or self._angular_attr is None:
            return None
        lin = self._linear_attr.get()
        ang = self._angular_attr.get()
        if lin is None or ang is None:
            return None
        return np.array([
            ros_utils.clamp(float(lin[0]), -self._vx_max, self._vx_max),
            ros_utils.clamp(float(lin[1]), -self._vy_max, self._vy_max),
            ros_utils.clamp(float(ang[2]), -self._wz_max, self._wz_max),
        ], dtype=np.float32)

    def _update_odom(self) -> None:
        try:
            pos_w, quat_wxyz = self._robot.robot.get_world_pose()
            lin_vel = self._robot.robot.get_linear_velocity()
            ang_vel = self._robot.robot.get_angular_velocity()
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            ros_utils.update_odom_tf(pos_w, quat_xyzw)
            ros_utils.update_odom(pos_w, quat_xyzw, lin_vel, ang_vel)
        except Exception:
            return

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._robot.initialize()
            self.first_step = False
            return
        if self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
            return

        cmd = self._base_command.copy()
        now = time.time()
        if self._cmd_vel_count_attr is not None:
            count = self._cmd_vel_count_attr.get()
            if count != self._last_cmd_vel_count:
                self._last_cmd_vel_count = count
                self._last_cmd_vel_time = now

        timed_out = (
            self._last_cmd_vel_time is None
            or (now - self._last_cmd_vel_time) > CMD_VEL_TIMEOUT
        )
        if not timed_out:
            cmd_vel = self._get_cmd_vel()
            if cmd_vel is not None:
                cmd = cmd_vel if self._cmd_vel_only else cmd + cmd_vel

        cmd = np.minimum(np.maximum(cmd, self._cmd_min), self._cmd_max)
        self._robot.forward(step_size, cmd)
        self._update_odom()

        if self._enable_human and self._human_prim is not None:
            if self._human_auto_walk:
                self._walk_phase_time += step_size
                if self._walk_phase_time >= self._walk_phase_limit[self._walk_phase]:
                    self._walk_phase = 1 - self._walk_phase
                    self._walk_phase_time = 0.0
                vel_x = 0.5 if self._walk_phase == 0 else 0.0
                yaw_rate = 0.0 if self._walk_phase == 0 else 1.0
                vel_y = 0.0
            elif self._human_vel_attr is not None:
                h_vel = self._human_vel_attr.get()
                h_yaw_rate = self._human_yaw_rate_attr.get()
                if h_vel is None or h_yaw_rate is None:
                    vel_x = vel_y = yaw_rate = 0.0
                else:
                    vel_x = ros_utils.clamp(float(h_vel[0]), -self._human_vel_max, self._human_vel_max)
                    vel_y = ros_utils.clamp(float(h_vel[1]), -self._human_vel_max, self._human_vel_max)
                    yaw_rate = ros_utils.clamp(float(h_yaw_rate[2]), -self._human_yaw_rate_max, self._human_yaw_rate_max)
            else:
                vel_x = vel_y = yaw_rate = 0.0

            new_x, new_y, new_yaw = ros_utils.integrate_human_velocity(
                self._human_pos, self._human_yaw, vel_x, vel_y, yaw_rate, step_size
            )
            self._human_pos = [new_x, new_y]
            self._human_yaw = new_yaw
            ros_utils.update_human_pose(self._human_prim, new_x, new_y, new_yaw)

            try:
                robot_pos, _ = self._robot.robot.get_world_pose()
                dx = new_x - float(robot_pos[0])
                dy = new_y - float(robot_pos[1])
                dist = math.sqrt(dx * dx + dy * dy)
                now = time.time()
                if dist < self._bump_threshold and (now - self._last_bump_log_time) > 1.0:
                    logger.warning(
                        "BUMP: human and robot within %.2fm at human=(%.2f, %.2f) robot=(%.2f, %.2f)",
                        dist, new_x, new_y, float(robot_pos[0]), float(robot_pos[1]),
                    )
                    self._last_bump_log_time = now
            except Exception:
                pass

    def run(self, real_time: bool) -> None:
        while simulation_app.is_running():
            t0 = time.time()
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
            if real_time:
                sleep = self._physics_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(
                    self._input_keyboard_mapping[event.input.name], dtype=np.float32
                )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(
                    self._input_keyboard_mapping[event.input.name], dtype=np.float32
                )
        return True


# ---------------------------------------------------------------------------
# IRA runner (--ira mode: animated pedestrians, collision tracking)
# ---------------------------------------------------------------------------

class IraRobotRunner:
    """
    Inserts a robot into an already-open IRA stage and runs the physics loop.

    The IRA scene must already be open and fully set up (navmesh baked,
    character scripts attached) before this class is instantiated — that work
    is done by _setup_ira() in main() before this object is created.
    """

    def __init__(
        self,
        physics_dt: float,
        render_dt: float,
        policy_dir: str,
        cmd_vel_topic: str,
        vx_max: float,
        vy_max: float,
        wz_max: float,
        robot_root: str,
        cmd_vel_only: bool,
        enable_sensors: bool,
        enable_keyboard: bool,
        robot_type: str = ROBOT_GO2,
        character_root: str = DEFAULT_CHARACTER_ROOT,
        collision_radius: float = COLLISION_RADIUS,
        realsense_tilt_deg: float = -25.0,
    ):
        self._robot_type = robot_type
        self._character_root = character_root
        self._collision_radius = collision_radius
        self._realsense_tilt_deg = realsense_tilt_deg

        policy_path, env_path, deploy_path = _validate_policy_paths(policy_dir, robot_type)
        env_cfg = parse_env_config(env_path) if os.path.isfile(env_path) else {}
        deploy_cfg = _load_yaml(deploy_path)
        usd_path = _resolve_usd_path(env_cfg, robot_type)

        default_z = (TRON1_INIT_HEIGHT if robot_type == ROBOT_TRON1
                     else G1_INIT_HEIGHT if robot_type == ROBOT_G1 else 0.4)
        init_pos = np.array(
            env_cfg.get("scene", {}).get("robot", {}).get("init_state", {}).get("pos",
            (0.0, 0.0, default_z))
        )

        # World() attaches to the IRA stage that SimulationManager already opened.
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self._physics_dt = physics_dt
        self._render_dt = render_dt

        if not usd_path:
            raise RuntimeError(f"{robot_type.upper()} USD path could not be resolved")

        if robot_type == ROBOT_TRON1:
            encoder_path = os.path.join(policy_dir, "exported", "encoder.pt")
            self._robot = Tron1VelocityPolicy(
                prim_path=robot_root, name="Tron1", usd_path=usd_path,
                position=init_pos, policy_path=policy_path,
                encoder_path=encoder_path, deploy_path=deploy_path,
            )
        elif robot_type == ROBOT_G1:
            self._robot = G1VelocityPolicy(
                prim_path=robot_root, name="G1", usd_path=usd_path,
                position=init_pos, policy_path=policy_path,
                env_path=env_path, deploy_path=deploy_path,
            )
        else:
            self._robot = Go2VelocityPolicy(
                prim_path=robot_root, name="Go2", usd_path=usd_path,
                position=init_pos, policy_path=policy_path, env_path=env_path,
            )

        cmd_min, cmd_max = _resolve_command_limits(deploy_cfg, env_cfg)
        self._cmd_min = np.maximum(cmd_min, np.array([-vx_max, -vy_max, -wz_max], dtype=np.float32))
        self._cmd_max = np.minimum(cmd_max, np.array([vx_max, vy_max, wz_max], dtype=np.float32))

        self._cmd_vel_topic = cmd_vel_topic
        self._cmd_vel_only = cmd_vel_only
        self._enable_sensors = enable_sensors
        self._enable_keyboard = enable_keyboard
        self._vx_max = vx_max
        self._vy_max = vy_max
        self._wz_max = wz_max
        self._linear_attr = None
        self._angular_attr = None
        self._cmd_vel_count_attr = None
        self._sensors = {}
        self._robot_root = robot_root
        _configure_ros_utils_paths(robot_root, robot_type)

        self._base_command = np.zeros(3, dtype=np.float32)
        self._last_cmd_vel_time: Optional[float] = None
        self._last_cmd_vel_count: int = 0
        self.needs_reset = False
        self.first_step = True
        self._virtual_cam_writers: dict = {}

        cmd_scale = np.maximum(np.abs(self._cmd_min), np.abs(self._cmd_max))
        vx, vy, wz = cmd_scale.tolist()
        self._input_keyboard_mapping = {
            "NUMPAD_8": [vx, 0.0, 0.0], "UP": [vx, 0.0, 0.0],
            "NUMPAD_2": [-vx, 0.0, 0.0], "DOWN": [-vx, 0.0, 0.0],
            "NUMPAD_6": [0.0, -vy, 0.0], "RIGHT": [0.0, -vy, 0.0],
            "NUMPAD_4": [0.0, vy, 0.0], "LEFT": [0.0, vy, 0.0],
            "NUMPAD_7": [0.0, 0.0, wz], "N": [0.0, 0.0, wz],
            "NUMPAD_9": [0.0, 0.0, -wz], "M": [0.0, 0.0, -wz],
        }

        self._collision_stats = {
            "total_collisions": 0,
            "collision_radius_m": collision_radius,
            "robot_type": robot_type,
            "simulation_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "events": [],
        }
        self._last_collision_time = -999.0

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _get_character_positions(self):
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(self._character_root)
        if not root_prim or not root_prim.IsValid():
            return []
        xform_cache = UsdGeom.XformCache()
        positions = []
        for child in root_prim.GetChildren():
            if not UsdGeom.Xformable(child):
                continue
            try:
                t = xform_cache.GetLocalToWorldTransform(child).ExtractTranslation()
                positions.append((float(t[0]), float(t[1])))
            except Exception:
                pass
        return positions

    def _check_collisions(self):
        try:
            robot_pos, _ = self._robot.robot.get_world_pose()
            rx, ry = float(robot_pos[0]), float(robot_pos[1])
        except Exception:
            return
        now = time.time()
        if (now - self._last_collision_time) < COLLISION_COOLDOWN:
            return
        for cx, cy in self._get_character_positions():
            dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            if dist < self._collision_radius:
                self._collision_stats["total_collisions"] += 1
                self._collision_stats["events"].append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "robot_pos": [round(rx, 3), round(ry, 3)],
                    "character_pos": [round(cx, 3), round(cy, 3)],
                    "distance_m": round(dist, 3),
                })
                self._last_collision_time = now
                logger.warning(
                    "COLLISION #%d: robot (%.2f, %.2f) within %.2fm of character (%.2f, %.2f)",
                    self._collision_stats["total_collisions"], rx, ry, dist, cx, cy,
                )
                break

    def save_collision_stats(self, path: str):
        self._collision_stats["simulation_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(path, "w") as f:
            json.dump(self._collision_stats, f, indent=2)
        logger.info("Collision stats -> %s  (total: %d)", path, self._collision_stats["total_collisions"])

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        if self._enable_keyboard:
            self._appwindow = omni.appwindow.get_default_app_window()
            if self._appwindow is not None:
                self._input = carb.input.acquire_input_interface()
                self._keyboard = self._appwindow.get_keyboard()
                self._sub_keyboard = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._sub_keyboard_event
                )
            else:
                logger.warning("No app window found; keyboard control disabled")
                self._enable_keyboard = False
        self._world.add_physics_callback("ira_robot_step", callback_fn=self.on_physics_step)

    def setup_ros(self):
        try:
            self._linear_attr, self._angular_attr, self._cmd_vel_count_attr = (
                ros_utils.setup_cmd_vel_graph(self._cmd_vel_topic)
            )
        except Exception as exc:
            logger.warning("setup_cmd_vel_graph failed: %s", exc)
            self._linear_attr = self._angular_attr = self._cmd_vel_count_attr = None
        if not self._enable_sensors:
            return

        for _ in range(10):
            self._world.step(render=True)

        render_hz = 1.0 / self._render_dt if self._render_dt else None
        if self._robot_type == ROBOT_G1:
            camera_link_pos = (0.0, 0.0, 0.75)
            lidar_l1_pos = (0.2, 0.0, 0.4)
            lidar_velo_pos = (0.15, 0.0, 0.5)
        else:
            camera_link_pos = (0.3, 0.0, 0.10)
            lidar_l1_pos = (0.3, 0.0, 0.08)
            lidar_velo_pos = (0.25, 0.0, 0.13)

        self._sensors = ros_utils.setup_sensors_delayed(
            simulation_app, render_hz=render_hz,
            camera_link_position=camera_link_pos, enable_lidar=True,
            lidar_l1_position=lidar_l1_pos, lidar_velo_position=lidar_velo_pos,
            robot_type=self._robot_type, realsense_tilt_deg=self._realsense_tilt_deg,
        )
        for _ in range(5):
            self._world.step(render=True)

        ros_utils.setup_ros_publishers(
            self._sensors, simulation_app, robot_type=self._robot_type,
            camera_link_pos=camera_link_pos, lidar_l1_pos=lidar_l1_pos,
            lidar_velo_pos=lidar_velo_pos,
        )
        ros_utils.setup_depth_camerainfo_graph(
            simulation_app,
            topic="/camera/realsense2_camera_node/depth/camera_info",
            frame_id="realsense_depth_camera",
            width=480, height=270, fx=242.479, fy=242.479, cx=242.736, cy=133.273,
        )
        ros_utils.setup_odom_publisher(simulation_app)
        # The realsense camera's prim is in go2.usd (file-backed layer), so the IRA
        # orchestrator rebuilds its render product on every reset and publishing
        # continues automatically.  The front/top camera prims are in an anonymous
        # sublayer that the orchestrator ignores, so we manage them separately with
        # _init_virtual_camera_writer / _restore_virtual_cameras.
        _front_cam = self._sensors.pop("robot_front_rgb_camera", None)
        _top_cam = self._sensors.pop("robot_top_rgb_camera", None)
        ros_utils.setup_color_camera_publishers(self._sensors, simulation_app, self._robot_type)
        if _front_cam is not None:
            self._sensors["robot_front_rgb_camera"] = _front_cam
        if _top_cam is not None:
            self._sensors["robot_top_rgb_camera"] = _top_cam
        self._init_virtual_camera_writer(
            "robot_front_rgb_camera", ros_utils.FRONT_RGB_CAMERA_PRIM, (640, 480),
            f"camera/{self._robot_type}/image_raw", f"{self._robot_type}_rgb_camera",
        )
        self._init_virtual_camera_writer(
            "robot_top_rgb_camera", ros_utils.TOP_RGB_CAMERA_PRIM, (640, 480),
            "camera/top/image_raw", f"{self._robot_type}_top_rgb_camera",
        )
        ros_utils.setup_color_camerainfo_graph(
            simulation_app,
            topic="/camera/realsense2_camera_node/color/camera_info",
            frame_id="realsense_depth_camera",
            width=424, height=240, fx=214.157, fy=215.537, cx=214.384, cy=118.465,
        )
        ros_utils.setup_joint_states_publisher(simulation_app, robot_type=self._robot_type)

    # ------------------------------------------------------------------
    # Virtual camera restore (IRA orchestrator may invalidate render products)
    # ------------------------------------------------------------------

    def _init_virtual_camera_writer(self, cam_key, prim_path, resolution, topic, frame_id):
        import omni.replicator.core as rep
        import omni.syntheticdata as syn_data
        import omni.syntheticdata._syntheticdata as sd

        cam = self._sensors.get(cam_key)
        if cam is None:
            return
        rp_path = cam.get_render_product_path()
        if not rp_path:
            logger.warning("[VirtualCam] %s has no render product at init", cam_key)
            return
        try:
            rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
            w = rep.writers.get(rv + "ROS2PublishImage")
            w.initialize(frameId=frame_id, nodeNamespace="", queueSize=10, topicName=topic)
            w.attach([rp_path])
            self._virtual_cam_writers[cam_key] = {
                "writer": w, "cam": cam, "prim_path": prim_path, "resolution": resolution,
            }
            logger.info("[VirtualCam] %s -> %s (rp: %s)", cam_key, topic, rp_path)
        except Exception as e:
            logger.warning("[VirtualCam] Failed to init %s: %s", cam_key, e)

    def _restore_virtual_cameras(self):
        import omni.usd
        import omni.replicator.core as rep

        if not self._virtual_cam_writers:
            return
        stage = omni.usd.get_context().get_stage()
        for cam_key, info in self._virtual_cam_writers.items():
            cam = info["cam"]
            rp_path = cam.get_render_product_path()
            if rp_path and stage.GetPrimAtPath(rp_path).IsValid():
                continue
            logger.info("[VirtualCam] %s render product gone — recreating", cam_key)
            try:
                rp_obj = rep.create.render_product(info["prim_path"], info["resolution"])
                if hasattr(cam, "_render_product_path"):
                    cam._render_product_path = rp_obj.path
                info["writer"].attach([rp_obj.path])
                logger.info("[VirtualCam] Restored %s (new rp: %s)", cam_key, rp_obj.path)
            except Exception as e:
                logger.warning("[VirtualCam] Failed to restore %s: %s", cam_key, e)

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def _get_cmd_vel(self):
        if self._linear_attr is None or self._angular_attr is None:
            return None
        lin = self._linear_attr.get()
        ang = self._angular_attr.get()
        if lin is None or ang is None:
            return None
        return np.array([
            ros_utils.clamp(float(lin[0]), -self._vx_max, self._vx_max),
            ros_utils.clamp(float(lin[1]), -self._vy_max, self._vy_max),
            ros_utils.clamp(float(ang[2]), -self._wz_max, self._wz_max),
        ], dtype=np.float32)

    def _update_odom(self):
        try:
            pos_w, quat_wxyz = self._robot.robot.get_world_pose()
            lin_vel = self._robot.robot.get_linear_velocity()
            ang_vel = self._robot.robot.get_angular_velocity()
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            ros_utils.update_odom_tf(pos_w, quat_xyzw)
            ros_utils.update_odom(pos_w, quat_xyzw, lin_vel, ang_vel)
        except Exception:
            return

    def on_physics_step(self, step_size):
        if self.first_step:
            self._robot.initialize()
            self.first_step = False
            return
        if self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
            return

        cmd = self._base_command.copy()
        now = time.time()
        if self._cmd_vel_count_attr is not None:
            count = self._cmd_vel_count_attr.get()
            if count != self._last_cmd_vel_count:
                self._last_cmd_vel_count = count
                self._last_cmd_vel_time = now

        timed_out = (
            self._last_cmd_vel_time is None
            or (now - self._last_cmd_vel_time) > CMD_VEL_TIMEOUT
        )
        if not timed_out:
            cmd_vel = self._get_cmd_vel()
            if cmd_vel is not None:
                cmd = cmd_vel if self._cmd_vel_only else cmd + cmd_vel

        cmd = np.clip(cmd, self._cmd_min, self._cmd_max)
        self._robot.forward(step_size, cmd)
        self._update_odom()
        self._check_collisions()

    def run(self, real_time: bool):
        _last_cam_check = 0.0
        while simulation_app.is_running():
            t0 = time.time()
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
            if self._enable_sensors and (t0 - _last_cam_check) > 3.0:
                self._restore_virtual_cameras()
                _last_cam_check = t0
            if real_time:
                sleep = self._physics_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    def _sub_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(
                    self._input_keyboard_mapping[event.input.name], dtype=np.float32
                )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(
                    self._input_keyboard_mapping[event.input.name], dtype=np.float32
                )
        return True


# ---------------------------------------------------------------------------
# IRA scene setup (runs before World() is created)
# ---------------------------------------------------------------------------

def _setup_ira(ira_scene: str, ira_config: Optional[str], command_file: str) -> None:
    """
    Open the IRA scene, bake the navmesh, and attach character behaviour scripts.
    Must complete before World() is created so World attaches to the IRA stage.
    """
    from isaacsim.replicator.agent.core.simulation import SimulationManager

    config_path = ira_config
    _tmp = None

    if config_path is None:
        config_data = {
            "isaacsim.replicator.agent": {
                "version": "0.7.0",
                "global": {"seed": 123456789, "simulation_length": 1800},
                "scene": {"asset_path": os.path.abspath(ira_scene)},
                "character": {
                    "asset_path": "", "command_file": os.path.abspath(command_file),
                    "num": 0, "filters": [],
                },
                "robot": {"command_file": "", "nova_carter_num": 0, "iw_hub_num": 0, "write_data": False},
                # No "replicator" key — prevents IRABasicWriter from being registered,
                # which would cause periodic orchestrator resets that destroy our camera
                # render products.
            }
        }
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="ira_run_")
        yaml.dump(config_data, _tmp)
        _tmp.close()
        config_path = _tmp.name
        print(f"[IRA] Generated config: {config_path}")

    sim_manager = SimulationManager()
    print(f"[IRA] Loading config: {config_path}")
    if not sim_manager.load_config_file(config_path):
        raise RuntimeError(f"SimulationManager failed to load config: {config_path}")

    setup_done = False

    def _on_done(event):
        nonlocal setup_done
        setup_done = True
        print("[IRA] Setup complete — characters ready.")

    _setup_sub = sim_manager.register_set_up_simulation_done_callback(_on_done)
    sim_manager.set_up_simulation_from_config_file()

    for _ in range(500):  # up to ~50 s — navmesh bake can take ~20 s
        simulation_app.update()
        if setup_done:
            break

    if not setup_done:
        logger.warning("IRA setup did not complete within 50 s — characters may not move")

    if _tmp is not None:
        try:
            os.unlink(_tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Robot args (shared by both modes)
    parser.add_argument("--robot_type", default=ROBOT_GO2, choices=[ROBOT_GO2, ROBOT_G1, ROBOT_TRON1])
    parser.add_argument("--policy_dir", default=None)
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel")
    parser.add_argument("--vx_max", type=float, default=1.0)
    parser.add_argument("--vy_max", type=float, default=1.0)
    parser.add_argument("--wz_max", type=float, default=1.0)
    parser.add_argument("--robot_root", default=None)
    parser.add_argument("--cmd_vel_only", action="store_true", default=False)
    parser.add_argument("--no_sensors", action="store_true")
    parser.add_argument("--no_keyboard", action="store_true")
    parser.add_argument("--real_time", action="store_true", default=False)
    parser.add_argument("--physics_dt", type=float, default=1 / 200.0)
    parser.add_argument("--render_dt", type=float, default=1 / 10.0)
    parser.add_argument(
        "--realsense_tilt", type=float, default=-25.0,
        help="RealSense camera X-axis tilt in degrees (negative = downward, default -25)",
    )
    # IRA mode
    parser.add_argument("--ira", action="store_true", help="Enable IRA scene with animated pedestrians")
    parser.add_argument("--ira_scene", default=DEFAULT_IRA_SCENE)
    parser.add_argument("--ira_config", default=None, help="IRA config.yaml (auto-generated if omitted)")
    parser.add_argument("--command_file", default=DEFAULT_COMMAND_FILE)
    parser.add_argument("--character_root", default=DEFAULT_CHARACTER_ROOT)
    parser.add_argument("--collision_radius", type=float, default=COLLISION_RADIUS)
    parser.add_argument("--collision_stats", default="collision_stats.json")
    # Default mode only
    parser.add_argument("--environment", default=ENV_WAREHOUSE, choices=[ENV_WAREHOUSE, ENV_APARTMENT])
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--human_auto_walk", action="store_true")
    parser.add_argument("--human_cmd_topic", default="/cmd_vel_human")
    parser.add_argument("--human_x", type=float, default=2.0)
    parser.add_argument("--human_y", type=float, default=0.0)
    parser.add_argument("--human_z", type=float, default=0.0)
    parser.add_argument("--human_yaw", type=float, default=0.0)
    parser.add_argument("--human_vel_max", type=float, default=1.5)
    parser.add_argument("--human_yaw_rate_max", type=float, default=1.0)
    parser.add_argument("--human_scale", type=float, default=1.0)
    args, _ = parser.parse_known_args()

    if args.policy_dir is None:
        args.policy_dir = (
            DEFAULT_TRON1_POLICY_DIR if args.robot_type == ROBOT_TRON1
            else DEFAULT_G1_POLICY_DIR if args.robot_type == ROBOT_G1
            else DEFAULT_GO2_POLICY_DIR
        )
    if args.robot_root is None:
        args.robot_root = (
            "/World/Tron1" if args.robot_type == ROBOT_TRON1
            else "/World/G1" if args.robot_type == ROBOT_G1
            else "/World/Go2"
        )

    from isaacsim.core.utils import extensions

    if args.ira:
        # ----------------------------------------------------------------
        # IRA mode
        # ----------------------------------------------------------------
        logger.info("IRA mode: %s robot in animated pedestrian scene", args.robot_type.upper())

        # Enable IRA extensions first; ROS2 bridge comes after IRA setup
        extensions.enable_extension("omni.anim.graph.core")
        extensions.enable_extension("omni.anim.navigation.core")
        extensions.enable_extension("omni.anim.people")
        extensions.enable_extension("isaacsim.replicator.agent.core")
        extensions.enable_extension("isaacsim.replicator.agent.ui")
        simulation_app.update()

        _setup_ira(args.ira_scene, args.ira_config, args.command_file)

        # Best-effort: prevent the IRA orchestrator from running.
        # _restore_virtual_cameras() in the run loop is the reliable fallback.
        try:
            import omni.replicator.core as rep
            rep.orchestrator.stop()
            rep.orchestrator.run = lambda *a, **kw: None
            logger.info("[IRA] Replicator orchestrator stopped and run() blocked")
        except Exception as _exc:
            logger.warning("[IRA] Could not block replicator orchestrator: %s", _exc)
        simulation_app.update()

        extensions.enable_extension("isaacsim.ros2.bridge")
        extensions.enable_extension("isaacsim.sensors.physics")
        extensions.enable_extension("isaacsim.sensors.rtx")
        simulation_app.update()

        runner = None
        try:
            runner = IraRobotRunner(
                physics_dt=args.physics_dt, render_dt=args.render_dt,
                policy_dir=args.policy_dir, cmd_vel_topic=args.cmd_vel_topic,
                vx_max=args.vx_max, vy_max=args.vy_max, wz_max=args.wz_max,
                robot_root=args.robot_root, cmd_vel_only=args.cmd_vel_only,
                enable_sensors=not args.no_sensors, enable_keyboard=not args.no_keyboard,
                robot_type=args.robot_type, character_root=args.character_root,
                collision_radius=args.collision_radius,
                realsense_tilt_deg=args.realsense_tilt,
            )
            simulation_app.update()
            runner._world.reset()
            simulation_app.update()
            runner.setup()
            simulation_app.update()
            runner.setup_ros()
            simulation_app.update()
            runner.run(real_time=args.real_time)
        except Exception:
            traceback.print_exc()
        finally:
            if runner is not None:
                runner.save_collision_stats(args.collision_stats)
            simulation_app.close()

    else:
        # ----------------------------------------------------------------
        # Default mode (warehouse / apartment)
        # ----------------------------------------------------------------
        logger.info("Running %s robot simulation", args.robot_type.upper())

        extensions.enable_extension("isaacsim.ros2.bridge")
        extensions.enable_extension("isaacsim.sensors.physics")
        extensions.enable_extension("isaacsim.sensors.rtx")
        simulation_app.update()

        try:
            runner = RobotRosRunner(
                physics_dt=args.physics_dt, render_dt=args.render_dt,
                policy_dir=args.policy_dir, cmd_vel_topic=args.cmd_vel_topic,
                vx_max=args.vx_max, vy_max=args.vy_max, wz_max=args.wz_max,
                robot_root=args.robot_root, cmd_vel_only=args.cmd_vel_only,
                enable_sensors=not args.no_sensors, enable_keyboard=not args.no_keyboard,
                enable_human=args.human, human_auto_walk=args.human_auto_walk,
                human_cmd_topic=args.human_cmd_topic,
                human_pos=(args.human_x, args.human_y, args.human_z),
                human_yaw=math.radians(args.human_yaw),
                human_vel_max=args.human_vel_max, human_yaw_rate_max=args.human_yaw_rate_max,
                human_scale=args.human_scale, robot_type=args.robot_type,
                environment=args.environment, realsense_tilt_deg=args.realsense_tilt,
            )
            simulation_app.update()
            runner.setup_human()
            simulation_app.update()
            runner._world.reset()
            if args.environment == ENV_APARTMENT:
                from isaacsim.core.utils.viewports import set_camera_view
                set_camera_view(eye=[2.0, -2.0, 1.5], target=[0.0, 0.0, 0.5],
                                camera_prim_path="/OmniverseKit_Persp")
            simulation_app.update()
            runner.setup()
            simulation_app.update()
            runner.setup_ros()
            simulation_app.update()
            runner.run(real_time=args.real_time)
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
