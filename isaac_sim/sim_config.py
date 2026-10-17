"""Robot and environment configuration loading for the Isaac Sim runner.

Reads the YAML files under ``configs/`` and exposes typed configs
consumed by ``run.py`` and the ROS publishing helpers in ``utils.py``.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml

Vec3 = Tuple[float, float, float]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_THIS_DIR, "configs")


def _abspath(path: str) -> str:
    """Resolve a config-relative path against the isaac_sim/ directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(_THIS_DIR, path)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@dataclass
class Lidar3DConfig:
    """One simulated 3D lidar unit (see ``lidars_3d`` in configs/robots/*.yaml)."""

    name: str
    frame_id: str
    topic: str
    position: Vec3
    rpy_deg: Vec3


@dataclass
class PtzConfig:
    """Simulated pan-tilt-zoom camera (see ``ptz`` in configs/robots/*.yaml)."""

    ns: str = "zr10"
    position: Vec3 = (0.15, 0.0, 0.20)
    resolution: Tuple[int, int] = (1280, 720)
    hfov_deg: float = 62.0
    zoom_max: float = 6.0
    zoom_rate: float = 2.0
    yaw_limits_deg: Tuple[float, float] = (-135.0, 135.0)
    pitch_limits_deg: Tuple[float, float] = (-25.0, 90.0)
    slew_rate_dps: float = 60.0
    rotate_max_dps: float = 90.0


@dataclass
class RobotConfig:
    """Per-robot runtime configuration (see configs/robots/*.yaml)."""

    type: str
    policy_dir: str
    prim_root: str
    base_link_name: str
    init_height: float
    apartment_height: float
    usd_relative: str
    isaac_fallback_usd: Optional[str]
    enable_lidar: bool
    camera_link_pos: Vec3
    lidar_l1_pos: Vec3
    velodyne_pos: Vec3
    enable_2d_lidar: bool = True
    enable_odom: bool = True
    lidars_3d: Optional[List[Lidar3DConfig]] = None
    history_length: Optional[int] = None
    requires_encoder: bool = False
    requires_env_yaml: bool = True
    pivot_assist: bool = False
    ptz: Optional[PtzConfig] = None

    @property
    def policy_dir_abs(self) -> str:
        """Absolute path to the policy checkpoint directory."""
        return _abspath(self.policy_dir)

    @property
    def usd_local_abs(self) -> str:
        """Absolute path to the bundled robot USD."""
        return _abspath(self.usd_relative)


@dataclass
class CameraIntrinsics:
    """Pinhole CameraInfo parameters for a published camera stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class SimConfig:
    """Simulation-wide defaults (see configs/sim.yaml)."""

    cmd_vel_timeout: float
    physics_dt: float
    render_dt: float
    max_vx: float
    max_vy: float
    max_wz: float
    depth_camera: CameraIntrinsics
    color_camera: CameraIntrinsics


@dataclass
class EnvironmentConfig:
    """Scene configuration (see configs/environments/*.yaml).

    ``warehouse`` uses ``usd_isaac_relative`` (resolved against Isaac Sim's
    assets root); ``apartment`` uses ``usd_relative`` plus the ground/transform/
    lighting fields.
    """

    type: str
    stage_path: str
    usd_isaac_relative: Optional[str] = None
    usd_relative: Optional[str] = None
    ground_z: float = 0.0
    transform: Optional[dict] = None
    camera_view: Optional[dict] = None
    lighting: Optional[dict] = None
    props: Optional[list] = None
    agents: Optional[list] = None
    people: Optional[dict] = None
    scale_prims: Optional[list] = None
    apriltag_dock: Optional[dict] = None
    remove_prims: Optional[list] = None
    platforms: Optional[list] = None
    gauges: Optional[dict] = None

    @property
    def usd_local_abs(self) -> Optional[str]:
        """Absolute path to the local scene USD, or None if not set."""
        return _abspath(self.usd_relative) if self.usd_relative else None


def _as_vec3(value) -> Vec3:
    x, y, z = value
    return (float(x), float(y), float(z))


def _load_lidars_3d(data: dict) -> Optional[List[Lidar3DConfig]]:
    """Parse the optional ``lidars_3d`` list from a robot yaml."""
    entries = data.get("lidars_3d")
    if not entries:
        return None
    return [
        Lidar3DConfig(
            name=str(entry["name"]),
            frame_id=str(entry["frame_id"]),
            topic=str(entry["topic"]),
            position=_as_vec3(entry["position"]),
            rpy_deg=_as_vec3(entry.get("rpy_deg", (0.0, 0.0, 0.0))),
        )
        for entry in entries
    ]


def _pair(value) -> Tuple[float, float]:
    lo, hi = value
    return (float(lo), float(hi))


def _load_ptz(data: dict) -> Optional[PtzConfig]:
    """Parse the optional ``ptz`` block from a robot yaml."""
    entry = data.get("ptz")
    if not entry or not entry.get("enabled", True):
        return None
    defaults = PtzConfig()
    width, height = entry.get("resolution", defaults.resolution)
    return PtzConfig(
        ns=str(entry.get("ns", defaults.ns)).strip("/"),
        position=_as_vec3(entry.get("position", defaults.position)),
        resolution=(int(width), int(height)),
        hfov_deg=float(entry.get("hfov_deg", defaults.hfov_deg)),
        zoom_max=float(entry.get("zoom_max", defaults.zoom_max)),
        zoom_rate=float(entry.get("zoom_rate", defaults.zoom_rate)),
        yaw_limits_deg=_pair(entry.get("yaw_limits_deg", defaults.yaw_limits_deg)),
        pitch_limits_deg=_pair(
            entry.get("pitch_limits_deg", defaults.pitch_limits_deg)
        ),
        slew_rate_dps=float(entry.get("slew_rate_dps", defaults.slew_rate_dps)),
        rotate_max_dps=float(entry.get("rotate_max_dps", defaults.rotate_max_dps)),
    )


def load_robot_config(robot_type: str) -> RobotConfig:
    """Load configs/robots/<robot_type>.yaml into a RobotConfig."""
    path = os.path.join(_CONFIG_DIR, "robots", f"{robot_type}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No robot config for '{robot_type}' at {path}")
    data = _load_yaml(path)
    sensors = data.get("sensors", {})
    return RobotConfig(
        type=data["type"],
        policy_dir=data["policy_dir"],
        prim_root=data["prim_root"],
        base_link_name=data["base_link_name"],
        init_height=float(data["init_height"]),
        apartment_height=float(data["apartment_height"]),
        usd_relative=data["usd_relative"],
        isaac_fallback_usd=data.get("isaac_fallback_usd"),
        enable_lidar=bool(data.get("enable_lidar", True)),
        enable_2d_lidar=bool(data.get("enable_2d_lidar", True)),
        enable_odom=bool(data.get("enable_odom", True)),
        camera_link_pos=_as_vec3(sensors["camera_link"]),
        lidar_l1_pos=_as_vec3(sensors["lidar_l1"]),
        velodyne_pos=_as_vec3(sensors["velodyne"]),
        lidars_3d=_load_lidars_3d(data),
        history_length=data.get("history_length"),
        requires_encoder=bool(data.get("requires_encoder", False)),
        requires_env_yaml=bool(data.get("requires_env_yaml", True)),
        pivot_assist=bool(data.get("pivot_assist", False)),
        ptz=_load_ptz(data),
    )


def load_environment_config(env_type: str) -> EnvironmentConfig:
    """Load configs/environments/<env_type>.yaml into an EnvironmentConfig."""
    path = os.path.join(_CONFIG_DIR, "environments", f"{env_type}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No environment config for '{env_type}' at {path}")
    data = _load_yaml(path)
    return EnvironmentConfig(
        type=data["type"],
        stage_path=data["stage_path"],
        usd_isaac_relative=data.get("usd_isaac_relative"),
        usd_relative=data.get("usd_relative"),
        ground_z=float(data.get("ground_z", 0.0)),
        transform=data.get("transform"),
        camera_view=data.get("camera_view"),
        lighting=data.get("lighting"),
        props=data.get("props"),
        agents=data.get("agents"),
        people=data.get("people"),
        scale_prims=data.get("scale_prims"),
        apriltag_dock=data.get("apriltag_dock"),
        remove_prims=data.get("remove_prims"),
        platforms=data.get("platforms"),
        gauges=data.get("gauges"),
    )


def _camera_intrinsics(data: dict) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
    )


def load_sim_config() -> SimConfig:
    """Load configs/sim.yaml into a SimConfig."""
    path = os.path.join(_CONFIG_DIR, "sim.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No sim config at {path}")
    data = _load_yaml(path)
    caps = data.get("max_cmd_vel", {})
    cameras = data.get("camera", {})
    return SimConfig(
        cmd_vel_timeout=float(data["cmd_vel_timeout"]),
        physics_dt=float(data["physics_dt"]),
        render_dt=float(data["render_dt"]),
        max_vx=float(caps["vx"]),
        max_vy=float(caps["vy"]),
        max_wz=float(caps["wz"]),
        depth_camera=_camera_intrinsics(cameras["depth"]),
        color_camera=_camera_intrinsics(cameras["color"]),
    )


def _available_types(subdir: str) -> List[str]:
    type_dir = os.path.join(_CONFIG_DIR, subdir)
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(type_dir) if f.endswith(".yaml")
    )


def available_robot_types() -> List[str]:
    """Robot types that have a config under configs/robots/."""
    return _available_types("robots")


def available_environment_types() -> List[str]:
    """Environment types that have a config under configs/environments/."""
    return _available_types("environments")
