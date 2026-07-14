# ruff: noqa: E402
"""
Isaac Sim robot simulation with ROS2 integration.
Supports Go2 (quadruped), G1 (humanoid), TRON1 (bipedal wheelfoot), and
M20 (DeepRobotics wheeled quadruped) robots.

Control robot velocity:
  ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.3, y: 0.0}, angular: {z: 0.2}}"

Control human pedestrian (when --human is set):
  ros2 topic pub /cmd_vel_human geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0}, angular: {z: 0.3}}"

Examples
--------
  python run.py --robot_type go2          # Run Go2 quadruped (default)
  python run.py --robot_type g1           # Run G1 humanoid
  python run.py --robot_type tron1        # Run TRON1 bipedal wheelfoot
  python run.py --robot_type m20          # Run DeepRobotics M20 wheeled quadruped
  python run.py --human                   # Spawn a human pedestrian, controllable on /cmd_vel_human
"""

# omni.anim.people characters only animate if the animation stack is enabled at
# app startup, so peek at the env config here to decide whether to enable it.
import os as _os
import sys as _sys

from isaacsim import SimulationApp

_anim_args = []
try:
    import yaml as _yaml

    _env = "warehouse"
    for _i, _a in enumerate(_sys.argv):
        if _a == "--environment" and _i + 1 < len(_sys.argv):
            _env = _sys.argv[_i + 1]
        elif _a.startswith("--environment="):
            _env = _a.split("=", 1)[1]
    _cfg_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "configs",
        "environments",
        f"{_env}.yaml",
    )
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        if (_yaml.safe_load(_f) or {}).get("people"):
            _anim_args = [
                "--enable",
                "omni.anim.people",
                "--enable",
                "omni.anim.graph.ui",
                "--enable",
                "omni.anim.navigation.bundle",
            ]
except Exception:
    _anim_args = []

simulation_app = SimulationApp(
    {
        "renderer": "RaytracedLighting",
        "headless": False,
        "extra_args": _anim_args,
    }
)

import importlib.util as _ilu
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
for _name in (
    "utils",
    "sim_config",
    "environments",
    "human",
    "agents",
    "people",
    "apriltag",
):
    _spec = _ilu.spec_from_file_location(_name, _os.path.join(_here, _name + ".py"))
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
del _ilu, _os, _sys, _here, _name, _spec, _mod

import argparse
import logging
import math
import os
import time
from typing import Optional, Tuple

import agents
import apriltag
import carb
import environments
import human
import numpy as np
import omni.appwindow  # Contains handle to keyboard
import people
import sim_config
import utils as ros_utils
from isaacsim.core.api import World
from isaacsim.robot.policy.examples.controllers.config_loader import parse_env_config
from isaacsim.storage.native import get_assets_root_path

# Imported after SimulationApp() init (the package pulls in Isaac Sim modules).
from policies import _load_yaml, build_policy

logger = logging.getLogger(__name__)

# Simulation-wide defaults
SIM_CONFIG = sim_config.load_sim_config()


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

        def _pair(key: str, fallback: Tuple[float, float]):
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

    cmd_min = pairs[:, 0]
    cmd_max = pairs[:, 1]
    return cmd_min, cmd_max


def _resolve_usd_path(env_cfg: dict, robot_cfg: "sim_config.RobotConfig") -> str:
    """
    Resolve the robot USD: local asset -> env.yaml spawn path -> Isaac fallback.
    """
    robot_name = robot_cfg.type.upper()

    # First priority: local bundled USD model.
    if os.path.isfile(robot_cfg.usd_local_abs):
        logger.info("Using local %s USD model: %s", robot_name, robot_cfg.usd_local_abs)
        return robot_cfg.usd_local_abs

    # Second priority: env.yaml spawn usd_path (absolute or relative to script dir).
    usd_path = (
        env_cfg.get("scene", {}).get("robot", {}).get("spawn", {}).get("usd_path")
    )
    if usd_path:
        if os.path.isabs(usd_path) and os.path.isfile(usd_path):
            logger.info("Using %s USD from absolute path: %s", robot_name, usd_path)
            return usd_path
        relative_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), usd_path
        )
        if os.path.isfile(relative_path):
            logger.info(
                "Using %s USD from relative path: %s", robot_name, relative_path
            )
            return relative_path
        if robot_cfg.isaac_fallback_usd:
            logger.warning("USD path from env.yaml not found: %s", usd_path)
            return usd_path
        logger.error("Could not find %s USD model", robot_name)
        return ""

    # Fallback to Isaac Sim's bundled asset, when this robot has one.
    if robot_cfg.isaac_fallback_usd:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return ""
        fallback_path = assets_root_path + robot_cfg.isaac_fallback_usd
        logger.info(
            "Using Isaac Sim default %s USD model: %s", robot_name, fallback_path
        )
        return fallback_path

    logger.error("Could not find %s USD model", robot_name)
    return ""


def _configure_ros_utils_paths(robot_root: str, base_link_name: str) -> None:
    """Point the ros_utils sensor prim-path globals at this robot's hierarchy."""
    ros_utils.GO2_STAGE_PATH = robot_root
    base_link = f"{robot_root}/{base_link_name}"
    ros_utils.BASE_LINK_PRIM = base_link
    ros_utils.IMU_PRIM = f"{base_link}/imu_link"
    ros_utils.CAMERA_LINK_PRIM = f"{base_link}/camera_link"
    ros_utils.REALSENSE_DEPTH_CAMERA_PRIM = (
        f"{ros_utils.CAMERA_LINK_PRIM}/realsense_depth_camera"
    )
    ros_utils.REALSENSE_RGB_CAMERA_PRIM = (
        f"{ros_utils.CAMERA_LINK_PRIM}/realsense_rgb_camera"
    )
    ros_utils.FRONT_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/front_rgb_camera"
    ros_utils.TOP_RGB_CAMERA_PRIM = f"{ros_utils.CAMERA_LINK_PRIM}/top_rgb_camera"
    ros_utils.L1_LINK_PRIM = f"{base_link}/lidar_l1_link"
    ros_utils.L1_LIDAR_PRIM = f"{ros_utils.L1_LINK_PRIM}/lidar_l1_rtx"
    ros_utils.VELO_BASE_LINK_PRIM = f"{base_link}/velodyne_base_link"
    ros_utils.VELO_LASER_LINK_PRIM = f"{ros_utils.VELO_BASE_LINK_PRIM}/laser"
    ros_utils.VELO_LIDAR_PRIM = f"{ros_utils.VELO_LASER_LINK_PRIM}/velodyne_vlp16_rtx"


def _validate_policy_paths(
    policy_dir: str, robot_cfg: "sim_config.RobotConfig"
) -> Tuple[str, str, str]:
    policy_path = os.path.join(policy_dir, "exported", "policy.pt")
    env_path = os.path.join(policy_dir, "params", "env.yaml")
    deploy_path = os.path.join(policy_dir, "params", "deploy.yaml")

    missing = []
    if not os.path.isfile(policy_path):
        missing.append(policy_path)
    if robot_cfg.requires_env_yaml and not os.path.isfile(env_path):
        missing.append(env_path)

    if robot_cfg.requires_encoder:
        encoder_path = os.path.join(policy_dir, "exported", "encoder.pt")
        if not os.path.isfile(encoder_path):
            missing.append(encoder_path)

    if missing:
        for path in missing:
            carb.log_error(f"Missing required policy file: {path}")
        raise FileNotFoundError("Required policy files not found.")

    if not os.path.isfile(deploy_path):
        logger.warning(
            "deploy.yaml not found at %s; using env.yaml ranges only", deploy_path
        )

    return policy_path, env_path, deploy_path


class RobotRosRunner(object):
    """Runner class for Unitree robots (Go2/G1) and TRON1 with ROS2 integration and sensor support."""

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
        robot_cfg: "sim_config.RobotConfig" = None,
        env_config: "sim_config.EnvironmentConfig" = None,
    ) -> None:
        """
        Creates the simulation world with preset physics_dt and render_dt and creates a robot inside the warehouse.

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        robot_cfg {RobotConfig} -- Loaded robot configuration (configs/robots/).
        env_config {EnvironmentConfig} -- Loaded scene configuration.

        """
        self._robot_cfg = robot_cfg
        self._env_config = env_config
        self._robot_type = robot_cfg.type
        robot_type = robot_cfg.type
        policy_path, env_path, deploy_path = _validate_policy_paths(
            policy_dir, robot_cfg
        )

        if os.path.isfile(env_path):
            env_cfg = parse_env_config(env_path)
        else:
            env_cfg = {}
        deploy_cfg = _load_yaml(deploy_path)

        usd_path = _resolve_usd_path(env_cfg, robot_cfg)

        # Default spawn height from the robot config; env.yaml may override it.
        init_pos = np.array(
            env_cfg.get("scene", {})
            .get("robot", {})
            .get("init_state", {})
            .get("pos", (0.0, 0.0, robot_cfg.init_height))
        )

        self._environment = env_config.type
        # Raised-ground scenes (e.g. apartment) set ground_z; spawn above it.
        if env_config.ground_z:
            init_pos[2] = env_config.ground_z + robot_cfg.apartment_height

        self._world = World(
            stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt
        )
        self._physics_dt = physics_dt
        self._render_dt = render_dt

        environments.add_environment(env_config)

        if not usd_path:
            raise RuntimeError(f"{robot_type.upper()} USD path could not be resolved")

        # Create the per-robot policy (see policies/).
        self._robot = build_policy(
            robot_cfg,
            robot_root=robot_root,
            usd_path=usd_path,
            policy_path=policy_path,
            env_path=env_path,
            deploy_path=deploy_path,
            policy_dir=policy_dir,
            init_pos=init_pos,
        )

        cmd_min, cmd_max = _resolve_command_limits(deploy_cfg, env_cfg)
        args_min = np.array([-vx_max, -vy_max, -wz_max], dtype=np.float32)
        args_max = np.array([vx_max, vy_max, wz_max], dtype=np.float32)
        self._cmd_min = np.maximum(cmd_min, args_min)
        self._cmd_max = np.minimum(cmd_max, args_max)

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
        _configure_ros_utils_paths(robot_root, robot_cfg.base_link_name)

        self._base_command = np.zeros(3, dtype=np.float32)
        self._last_cmd_vel_time: Optional[float] = None
        self._last_cmd_vel_count: int = 0
        self._cmd_vel_count_attr = None

        cmd_scale = np.maximum(np.abs(self._cmd_min), np.abs(self._cmd_max))
        vx, vy, wz = cmd_scale.tolist()
        self._input_keyboard_mapping = {
            "NUMPAD_8": [vx, 0.0, 0.0],
            "UP": [vx, 0.0, 0.0],
            "NUMPAD_2": [-vx, 0.0, 0.0],
            "DOWN": [-vx, 0.0, 0.0],
            "NUMPAD_6": [0.0, -vy, 0.0],
            "RIGHT": [0.0, -vy, 0.0],
            "NUMPAD_4": [0.0, vy, 0.0],
            "LEFT": [0.0, vy, 0.0],
            "NUMPAD_7": [0.0, 0.0, wz],
            "N": [0.0, 0.0, wz],
            "NUMPAD_9": [0.0, 0.0, -wz],
            "M": [0.0, 0.0, -wz],
        }
        self.needs_reset = False
        self.first_step = True

    def setup(self) -> None:
        """
        Set up keyboard listener and add physics callback.

        """
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
        self._world.add_physics_callback(
            "go2_ros2_step", callback_fn=self.on_physics_step
        )

    def setup_ros(self) -> None:
        """Set up ROS2 nodes for command velocity and sensor publishers."""
        self._linear_attr, self._angular_attr, self._cmd_vel_count_attr = (
            ros_utils.setup_cmd_vel_graph(self._cmd_vel_topic)
        )
        if not self._enable_sensors:
            return

        for _ in range(10):
            self._world.step(render=True)

        render_hz = None
        if self._render_dt:
            render_hz = 1.0 / self._render_dt

        camera_link_pos = self._robot_cfg.camera_link_pos
        lidar_l1_pos = self._robot_cfg.lidar_l1_pos
        lidar_velo_pos = self._robot_cfg.velodyne_pos
        lidars_3d = self._robot_cfg.lidars_3d

        self._sensors = ros_utils.setup_sensors_delayed(
            simulation_app,
            render_hz=render_hz,
            camera_link_position=camera_link_pos,
            enable_lidar=self._robot_cfg.enable_lidar,
            lidar_l1_position=lidar_l1_pos,
            lidar_velo_position=lidar_velo_pos,
            robot_type=self._robot_type,
            lidars_3d=lidars_3d,
            enable_2d_lidar=self._robot_cfg.enable_2d_lidar,
        )

        # Additional render steps to initialize camera render products
        for _ in range(5):
            self._world.step(render=True)

        ros_utils.setup_ros_publishers(
            self._sensors,
            simulation_app,
            robot_type=self._robot_type,
            camera_link_pos=camera_link_pos,
            lidar_l1_pos=lidar_l1_pos,
            lidar_velo_pos=lidar_velo_pos,
            lidars_3d=lidars_3d,
            enable_2d_lidar=self._robot_cfg.enable_2d_lidar,
        )

        depth_cam = SIM_CONFIG.depth_camera
        ros_utils.setup_depth_camerainfo_graph(
            simulation_app,
            topic="/camera/realsense2_camera_node/depth/camera_info",
            frame_id="realsense_depth_camera",
            width=depth_cam.width,
            height=depth_cam.height,
            fx=depth_cam.fx,
            fy=depth_cam.fy,
            cx=depth_cam.cx,
            cy=depth_cam.cy,
        )

        ros_utils.setup_odom_publisher(simulation_app)
        ros_utils.setup_color_camera_publishers(
            self._sensors, simulation_app, self._robot_type
        )
        color_cam = SIM_CONFIG.color_camera
        ros_utils.setup_color_camerainfo_graph(
            simulation_app,
            topic="/camera/realsense2_camera_node/color/camera_info",
            frame_id="realsense_depth_camera",
            width=color_cam.width,
            height=color_cam.height,
            fx=color_cam.fx,
            fy=color_cam.fy,
            cx=color_cam.cx,
            cy=color_cam.cy,
        )

        ros_utils.setup_joint_states_publisher(
            simulation_app, articulation_link=self._robot_cfg.base_link_name
        )

    def _get_cmd_vel(self) -> Optional[np.ndarray]:
        if self._linear_attr is None or self._angular_attr is None:
            return None
        lin = self._linear_attr.get()
        ang = self._angular_attr.get()
        if lin is None or ang is None:
            return None

        vx = ros_utils.clamp(float(lin[0]), -self._vx_max, self._vx_max)
        vy = ros_utils.clamp(float(lin[1]), -self._vy_max, self._vy_max)
        wz = ros_utils.clamp(float(ang[2]), -self._wz_max, self._wz_max)
        return np.array([vx, vy, wz], dtype=np.float32)

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
        """
        Physics call back, initialize robot (first frame) and call controller forward function.

        """
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

        # Check if a new /cmd_vel message arrived via the OmniGraph counter
        now = time.time()
        if self._cmd_vel_count_attr is not None:
            count = self._cmd_vel_count_attr.get()
            if count != self._last_cmd_vel_count:
                self._last_cmd_vel_count = count
                self._last_cmd_vel_time = now

        timed_out = (
            self._last_cmd_vel_time is None
            or (now - self._last_cmd_vel_time) > SIM_CONFIG.cmd_vel_timeout
        )

        if not timed_out:
            cmd_vel = self._get_cmd_vel()
            if cmd_vel is not None:
                if self._cmd_vel_only:
                    cmd = cmd_vel
                else:
                    cmd = cmd + cmd_vel

        cmd = np.minimum(np.maximum(cmd, self._cmd_min), self._cmd_max)
        self._robot.forward(step_size, cmd)
        self._update_odom()

    def run(self, real_time: bool) -> None:
        """
        Step simulation based on rendering downtime.

        """
        while simulation_app.is_running():
            t0 = time.time()
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
            if real_time:
                sleep = self._physics_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        Keyboard subscriber callback to when kit is updated.

        """
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


def main():
    """
    Instantiate the robot runner with ROS2 + sensors.
    Supports Go2 (quadruped), G1 (humanoid), TRON1 (bipedal wheelfoot), and
    M20 (DeepRobotics wheeled quadruped) robots.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_type",
        type=str,
        default="go2",
        choices=sim_config.available_robot_types(),
        help="Robot type (one per configs/robots/*.yaml)",
    )
    parser.add_argument(
        "--policy_dir",
        default=None,
        help="Policy directory (auto-detected based on robot_type if not set)",
    )
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel")
    parser.add_argument("--vx_max", type=float, default=SIM_CONFIG.max_vx)
    parser.add_argument("--vy_max", type=float, default=SIM_CONFIG.max_vy)
    parser.add_argument("--wz_max", type=float, default=SIM_CONFIG.max_wz)
    parser.add_argument(
        "--robot_root",
        default=None,
        help="Robot prim path (auto-detected based on robot_type if not set)",
    )
    parser.add_argument("--cmd_vel_only", action="store_true", default=False)
    parser.add_argument(
        "--no_sensors", action="store_true", help="Disable sensor setup"
    )
    parser.add_argument(
        "--no_keyboard", action="store_true", help="Disable keyboard control"
    )
    parser.add_argument(
        "--scan_source",
        choices=["2d", "3d"],
        default=None,
        help="Override the robot config's /scan source: '2d' spawns the "
        "simulated RPLIDAR publishing /scan directly; '3d' skips it so the "
        "om_common cloud_to_scan node derives /scan from the 3D point cloud "
        "(launch the sensor nodes with the matching SCAN_SOURCE env var)",
    )
    parser.add_argument("--real_time", action="store_true", default=False)
    parser.add_argument("--physics_dt", type=float, default=SIM_CONFIG.physics_dt)
    parser.add_argument("--render_dt", type=float, default=SIM_CONFIG.render_dt)
    # Human pedestrian arguments
    parser.add_argument(
        "--human", action="store_true", help="Enable human pedestrian model"
    )
    parser.add_argument(
        "--human_cmd_topic",
        type=str,
        default="/cmd_vel_human",
        help="ROS2 topic for human velocity control",
    )
    parser.add_argument(
        "--human_x", type=float, default=2.0, help="Human initial X position"
    )
    parser.add_argument(
        "--human_y", type=float, default=0.0, help="Human initial Y position"
    )
    parser.add_argument(
        "--human_z", type=float, default=0.0, help="Human initial Z position"
    )
    parser.add_argument(
        "--human_yaw", type=float, default=0.0, help="Human initial yaw (degrees)"
    )
    parser.add_argument(
        "--human_vel_max",
        type=float,
        default=1.5,
        help="Maximum human velocity (m/s)",
    )
    parser.add_argument(
        "--human_yaw_rate_max",
        type=float,
        default=1.0,
        help="Maximum human yaw rate (rad/s)",
    )
    parser.add_argument(
        "--human_scale", type=float, default=1.0, help="Human model scale factor"
    )
    parser.add_argument(
        "--vehicles",
        action="store_true",
        default=False,
        help="Enable scripted waypoint vehicles (forklifts/carts) from the "
        "environment config. Disabled by default.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="warehouse",
        choices=sim_config.available_environment_types(),
        help="Scene to load (one per configs/environments/*.yaml)",
    )
    args, _ = parser.parse_known_args()

    robot_cfg = sim_config.load_robot_config(args.robot_type)
    env_config = sim_config.load_environment_config(args.environment)

    # Fill robot/env defaults from the loaded config when not given on the CLI.
    if args.policy_dir is None:
        args.policy_dir = robot_cfg.policy_dir_abs
    if args.robot_root is None:
        args.robot_root = robot_cfg.prim_root
    if args.scan_source is not None:
        robot_cfg.enable_2d_lidar = args.scan_source == "2d"
        logger.info(
            "Scan source override: %s (2D lidar %s)",
            args.scan_source,
            "enabled" if robot_cfg.enable_2d_lidar else "disabled",
        )

    logger.info("Running %s robot simulation", args.robot_type.upper())

    # Resolve assets from a local mirror instead of the remote CDN when
    # ISAAC_ASSET_ROOT is set. Must precede any get_assets_root_path() call.
    asset_root = os.environ.get("ISAAC_ASSET_ROOT")
    if asset_root:
        carb.settings.get_settings().set_string(
            "/persistent/isaac/asset_root/default", asset_root
        )
        logger.info("Using local Isaac asset root: %s", asset_root)

    from isaacsim.core.utils import extensions

    extensions.enable_extension("isaacsim.ros2.bridge")
    extensions.enable_extension("isaacsim.sensors.physics")
    extensions.enable_extension("isaacsim.sensors.rtx")
    simulation_app.update()

    try:
        runner = RobotRosRunner(
            physics_dt=args.physics_dt,
            render_dt=args.render_dt,
            policy_dir=args.policy_dir,
            cmd_vel_topic=args.cmd_vel_topic,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            robot_root=args.robot_root,
            cmd_vel_only=args.cmd_vel_only,
            enable_sensors=not args.no_sensors,
            enable_keyboard=not args.no_keyboard,
            robot_cfg=robot_cfg,
            env_config=env_config,
        )
        simulation_app.update()

        # Optional human pedestrian. Added BEFORE world.reset() to avoid PhysX
        human_runner = None
        if args.human:
            human_runner = human.HumanRosRunner(
                runner._world,
                cmd_topic=args.human_cmd_topic,
                init_pos=(args.human_x, args.human_y, args.human_z),
                init_yaw=math.radians(args.human_yaw),
                vel_max=args.human_vel_max,
                yaw_rate_max=args.human_yaw_rate_max,
                scale=args.human_scale,
            )
            human_runner.setup()
        simulation_app.update()

        # Scripted waypoint vehicles; added before world.reset() (PhysX).
        agent_cfgs = env_config.agents if args.vehicles else None
        agent_runner = agents.WaypointAgentRunner(runner._world, agent_cfgs)
        agent_runner.setup()
        simulation_app.update()

        # Animated pedestrians; added before world.reset().
        people_runner = people.AnimatedPeopleRunner(env_config.people)
        people_runner.setup()
        # Let the characters steer around the moving robot.
        people_runner.register_dynamic_obstacle(runner._robot_root)
        simulation_app.update()

        runner._world.reset()
        if env_config.camera_view:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(
                eye=list(env_config.camera_view["eye"]),
                target=list(env_config.camera_view["target"]),
                camera_prim_path="/OmniverseKit_Persp",
            )
        simulation_app.update()
        runner.setup()
        if human_runner is not None:
            human_runner.register_physics_callback()
        agent_runner.register_physics_callback(
            robot_pose_fn=lambda: runner._robot.robot.get_world_pose()[0],
        )
        people_runner.register_physics_callback(
            runner._world,
            robot_pose_fn=lambda: runner._robot.robot.get_world_pose()[0],
        )
        simulation_app.update()
        runner.setup_ros()
        simulation_app.update()
        if env_config.apriltag_dock:
            apriltag.add_apriltag_dock(env_config.apriltag_dock, env_config.ground_z)
            simulation_app.update()
        people_runner.verify(runner._world)

        # Reset for IMU
        if runner._enable_sensors:
            runner._world.reset()
            runner.first_step = True

        runner.run(real_time=args.real_time)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
