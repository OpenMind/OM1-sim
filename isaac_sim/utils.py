# ruff: noqa: E402

import glob
import logging
import math
import os
import re
import subprocess
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Constants
WAREHOUSE_STAGE_PATH = "/World/Warehouse"
WAREHOUSE_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

# Human model stage path
HUMAN_STAGE_PATH = "/World/Human"

# Go2 Robot prim path
GO2_STAGE_PATH = "/World/envs/env_0/Robot"

# Sensor prim paths on Go2 base
IMU_PRIM = f"{GO2_STAGE_PATH}/base/imu_link"
CAMERA_LINK_PRIM = f"{GO2_STAGE_PATH}/base/camera_link"
REALSENSE_DEPTH_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/realsense_depth_camera"
REALSENSE_RGB_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/realsense_rgb_camera"
GO2_RGB_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/go2_rgb_camera"
L1_LINK_PRIM = f"{GO2_STAGE_PATH}/base/lidar_l1_link"
L1_LIDAR_PRIM = f"{L1_LINK_PRIM}/lidar_l1_rtx"
VELO_BASE_LINK_PRIM = f"{GO2_STAGE_PATH}/base/velodyne_base_link"
VELO_LASER_LINK_PRIM = f"{VELO_BASE_LINK_PRIM}/laser"
VELO_LIDAR_PRIM = f"{VELO_LASER_LINK_PRIM}/velodyne_vlp16_rtx"

# Odom TF handles
odom_graph_path = "/OdomActionGraph"
odom_tf_trans_attr = None
odom_tf_rot_attr = None

odom_pos_attr = None
odom_orient_attr = None
odom_lin_vel_attr = None
odom_ang_vel_attr = None

# Apartment environment constants
ENV_STAGE_PATH = "/World/Environment"
APARTMENT_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "environment",
    "Modern_Apartment.usdz",
)


def find_latest_checkpoint(log_root: str) -> str:
    """Find the latest checkpoint file in the log directory."""
    log_root = os.path.abspath(log_root)
    candidates = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
    if not candidates:
        raise RuntimeError(f"No checkpoints found under: {log_root}")
    best_it, best_path = -1, None
    for p in candidates:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) > best_it:
            best_it, best_path = int(m.group(1)), p
    if best_path is None:
        raise RuntimeError(f"No model_<iter>.pt found under: {log_root}")
    return best_path


def set_base_velocity_command(cm, cmd_tensor) -> None:
    """Set the base velocity command on the command manager."""
    if hasattr(cm, "set_command"):
        cm.set_command("base_velocity", cmd_tensor)
        return
    if hasattr(cm, "get_command"):
        cm.get_command("base_velocity")[:] = cmd_tensor
        return
    if hasattr(cm, "get_term"):
        term = cm.get_term("base_velocity")
        for attr in ("command", "_command", "commands", "_commands"):
            if hasattr(term, attr):
                getattr(term, attr)[:] = cmd_tensor
                return
    raise AttributeError("Could not set base_velocity")


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value between a lower and upper bound."""
    return max(lo, min(hi, x))


def yaw_to_quat_xyzw(yaw: float):
    """Convert yaw angle to quaternion in xyzw format."""
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def setup_cmd_vel_graph(
    topic_name: str = "/cmd_vel",
) -> Tuple[object, object, object]:
    """Set up the command velocity subscriber graph for ROS2 integration.

    Returns (linear_attr, angular_attr, msg_count_attr). msg_count_attr
    increments each time a new /cmd_vel message is received; the runner uses
    it to detect controller stalls and zero the command (see CMD_VEL_TIMEOUT).
    """
    import omni.graph.core as og
    from isaacsim.core.utils import extensions
    from isaacsim.core.utils.prims import is_prim_path_valid

    extensions.enable_extension("isaacsim.ros2.bridge")

    graph_path = "/CmdVelActionGraph"
    if not is_prim_path_valid(graph_path):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("TwistSub", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                    ("MsgCounter", "omni.graph.action.Counter"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "TwistSub.inputs:execIn"),
                    ("ROS2Context.outputs:context", "TwistSub.inputs:context"),
                    ("TwistSub.outputs:execOut", "MsgCounter.inputs:execIn"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ROS2Context.inputs:useDomainIDEnvVar", True),
                    ("TwistSub.inputs:topicName", topic_name),
                    ("TwistSub.inputs:queueSize", 10),
                ],
            },
        )
    twist_node_path = graph_path + "/TwistSub"
    counter_node_path = graph_path + "/MsgCounter"
    return (
        og.Controller.attribute(twist_node_path + ".outputs:linearVelocity"),
        og.Controller.attribute(twist_node_path + ".outputs:angularVelocity"),
        og.Controller.attribute(counter_node_path + ".outputs:count"),
    )


def add_warehouse_environment() -> bool:
    """Add the warehouse environment."""
    import carb
    import omni.usd
    from isaacsim.core.utils import nucleus, stage
    from pxr import Gf, UsdGeom

    assets_root_path = nucleus.get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder.")
        return False

    stage.add_reference_to_stage(
        assets_root_path + WAREHOUSE_USD_PATH, WAREHOUSE_STAGE_PATH
    )
    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    warehouse_prim = usd_stage.GetPrimAtPath(WAREHOUSE_STAGE_PATH)
    if not warehouse_prim or not warehouse_prim.IsValid():
        carb.log_error(f"Could not find warehouse prim at {WAREHOUSE_STAGE_PATH}")
        return False

    warehouse_xform = UsdGeom.Xformable(warehouse_prim)
    warehouse_xform.ClearXformOpOrder()
    translate_op = warehouse_xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, -0.01))
    logger.info("Warehouse environment added successfully")
    return True


def add_apartment_environment() -> bool:
    """Add an apartment environment to the stage."""
    import carb
    import omni.usd
    from isaacsim.core.utils import stage as stage_utils
    from pxr import Gf, UsdGeom

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    stage_utils.add_reference_to_stage(APARTMENT_USD_PATH, ENV_STAGE_PATH)

    env_prim = usd_stage.GetPrimAtPath(ENV_STAGE_PATH)
    if not env_prim or not env_prim.IsValid():
        carb.log_error(
            f"Failed to load apartment environment USD: {APARTMENT_USD_PATH}"
        )
        return False

    xform = UsdGeom.Xformable(env_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.22))

    xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))

    xform.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))

    logger.info("Apartment USD environment loaded successfully")
    return True


def add_human_model(
    human_usdz_path: str,
    position=(2.0, 0.0, 0.0),
    rotation_yaw: float = 0.0,
    scale: float = 1.0,
):
    """Add human model to the scene.

    Args:
        human_usdz_path: Path to the human USDZ file
        position: Initial (x, y, z) position in meters
        rotation_yaw: Initial yaw rotation in radians
        scale: Scale factor for the human model (default 1.0)

    Returns:
        The human prim if successful, None otherwise
    """
    import carb
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    if not os.path.exists(human_usdz_path):
        carb.log_error(f"Human model not found: {human_usdz_path}")
        return None

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    # Create prim and add reference with explicit path to /object
    # (the USDZ has no defaultPrim set, so we must specify the prim path)
    human_prim = usd_stage.DefinePrim(HUMAN_STAGE_PATH, "Xform")
    human_prim.GetReferences().AddReference(human_usdz_path, Sdf.Path("/object"))

    if not human_prim or not human_prim.IsValid():
        carb.log_error(f"Failed to load human model at {HUMAN_STAGE_PATH}")
        return None

    xform = UsdGeom.Xformable(human_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))
    # Rotate 90° around X to convert from Y-up to Z-up, then apply yaw
    xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, math.degrees(rotation_yaw)))
    if scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    logger.info(
        "Human model added at position %s, yaw=%s°, scale=%s",
        position,
        math.degrees(rotation_yaw),
        scale,
    )
    return human_prim


def setup_human_cmd_graph(topic_name: str = "/cmd_vel_human") -> Tuple[object, object]:
    """Setup ROS2 subscriber for human velocity control."""
    import omni.graph.core as og
    from isaacsim.core.utils import extensions
    from isaacsim.core.utils.prims import is_prim_path_valid

    extensions.enable_extension("isaacsim.ros2.bridge")

    graph_path = "/HumanCmdActionGraph"
    if not is_prim_path_valid(graph_path):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("TwistSub", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "TwistSub.inputs:execIn"),
                    ("ROS2Context.outputs:context", "TwistSub.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ROS2Context.inputs:useDomainIDEnvVar", True),
                    ("TwistSub.inputs:topicName", topic_name),
                    ("TwistSub.inputs:queueSize", 10),
                ],
            },
        )
    twist_node_path = graph_path + "/TwistSub"
    logger.info("Human command subscriber -> %s", topic_name)
    return (
        og.Controller.attribute(twist_node_path + ".outputs:linearVelocity"),
        og.Controller.attribute(twist_node_path + ".outputs:angularVelocity"),
    )


def update_human_pose(human_prim, x: float, y: float, yaw_rad: float) -> None:
    """Update human model position and rotation.

    Args:
        human_prim: The USD prim for the human model
        x: Target X position (meters)
        y: Target Y position (meters)
        yaw_rad: Target yaw rotation (radians)
    """
    from pxr import Gf, UsdGeom

    if human_prim is None or not human_prim.IsValid():
        return

    xform = UsdGeom.Xformable(human_prim)

    translate_ops = [
        op
        for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    rotate_ops = [
        op
        for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
    ]

    if translate_ops:
        current_pos = translate_ops[0].Get()
        z = current_pos[2] if current_pos else 0.0
        translate_ops[0].Set(Gf.Vec3d(x, y, z))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

    if rotate_ops:
        # Preserve the 90° X rotation for Y-up to Z-up conversion, update yaw
        rotate_ops[0].Set(Gf.Vec3f(90.0, 0.0, math.degrees(yaw_rad)))
    else:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, math.degrees(yaw_rad)))


def integrate_human_velocity(
    current_pos,
    current_yaw: float,
    vel_x: float,
    vel_y: float,
    yaw_rate: float,
    dt: float,
):
    """Integrate human velocity to get new position and orientation.

    Args:
        current_pos: Current [x, y] position
        current_yaw: Current yaw angle in radians
        vel_x: Forward velocity (m/s) in body frame
        vel_y: Lateral velocity (m/s) in body frame
        yaw_rate: Yaw rate (rad/s)
        dt: Time step (seconds)

    Returns:
        Tuple of (new_x, new_y, new_yaw)
    """
    new_yaw = current_yaw + yaw_rate * dt

    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)

    world_vx = vel_x * cos_yaw - vel_y * sin_yaw
    world_vy = vel_x * sin_yaw + vel_y * cos_yaw

    new_x = current_pos[0] + world_vx * dt
    new_y = current_pos[1] + world_vy * dt

    return (new_x, new_y, new_yaw)


def make_ground_invisible() -> None:
    """Make Isaac Lab's ground plane invisible but keep it for physics."""
    import omni.usd
    from pxr import UsdGeom

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    for path in ["/World/ground", "/World/ground/GroundPlane", "/World/GroundPlane"]:
        prim = usd_stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                logger.info("Made %s invisible", path)
            for child in prim.GetAllChildren():
                child_imageable = UsdGeom.Imageable(child)
                if child_imageable:
                    child_imageable.MakeInvisible()


def modify_env_config_for_warehouse(env_cfg, robot_pos, robot_yaw):
    """Adjust env config to match the warehouse demo setup."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "robot"):
        robot_cfg = env_cfg.scene.robot
        if hasattr(robot_cfg, "init_state"):
            init_state = robot_cfg.init_state
            if hasattr(init_state, "pos"):
                init_state.pos = robot_pos
                logger.info("Robot init pos: %s", robot_pos)
            if hasattr(init_state, "rot"):
                half_yaw = robot_yaw / 2.0
                init_state.rot = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

    if hasattr(env_cfg, "curriculum"):
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
            logger.info("Disabled terrain_levels curriculum")

    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
        logger.info("Disabled push_robot event")

    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 10000.0

    return env_cfg


def set_robot_pose(env, pos, yaw) -> bool:
    """Set the root pose for all envs when supported by the articulation."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    pos_tensor = torch.tensor([pos], device=device, dtype=torch.float32).repeat(
        num_envs, 1
    )
    half_yaw = yaw / 2.0
    quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    quat_tensor = torch.tensor([quat], device=device, dtype=torch.float32).repeat(
        num_envs, 1
    )

    unwrapped = env.unwrapped
    if hasattr(unwrapped, "scene") and hasattr(unwrapped.scene, "articulations"):
        for name, articulation in unwrapped.scene.articulations.items():
            if hasattr(articulation, "write_root_pose_to_sim"):
                pose = torch.cat([pos_tensor, quat_tensor], dim=-1)
                articulation.write_root_pose_to_sim(pose)
                logger.info("Set %s pose via write_root_pose_to_sim", name)
                return True
    return False


def ensure_link_xform(usd_stage, path: str, translation=None, rpy_rad=None):
    """Ensure a link Xform exists with the specified translation and rotation."""
    from pxr import Gf, UsdGeom

    prim = usd_stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = usd_stage.DefinePrim(path, "Xform")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rpy_rad is not None:
        roll, pitch, yaw_ = [math.degrees(v) for v in rpy_rad]
        xform.AddRotateXYZOp().Set(Gf.Vec3f(roll, pitch, yaw_))
    return prim


def setup_sensors_delayed(
    simulation_app,
    render_hz: Optional[float] = None,
    camera_link_position: Optional[Tuple[float, float, float]] = None,
    enable_lidar: bool = True,
    lidar_l1_position: Optional[Tuple[float, float, float]] = None,
    lidar_velo_position: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Setup sensors after simulation is fully running.

    Args:
        simulation_app: The Isaac Sim application instance
        render_hz: Render frequency for adjusting camera Hz
        camera_link_position: Position of camera link (x, y, z).
        enable_lidar: Whether to set up LiDAR sensors.
        lidar_l1_position: Position of L1 LiDAR link (x, y, z).
        lidar_velo_position: Position of Velodyne base link (x, y, z).

    Returns:
        Dictionary of initialized sensors
    """
    import omni.kit.commands
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import IMUSensor
    from pxr import Gf

    if camera_link_position is None:
        camera_link_position = (0.3, 0.0, 0.10)  # Default Go2 position
    if lidar_l1_position is None:
        lidar_l1_position = (0.15, 0.0, 0.15)  # Default Go2 position
    if lidar_velo_position is None:
        lidar_velo_position = (0.1, 0.0, 0.2)  # Default Go2 position

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    sensors = {
        "realsense_depth_camera": None,
        "realsense_rgb_camera": None,
        "go2_rgb_camera": None,
        "imu": None,
    }

    depth_camera_hz = 25
    rgb_camera_hz = 10
    if render_hz is not None and render_hz > 0:
        render_hz_int = int(round(render_hz))
        if render_hz_int > 0:
            divisors = [
                d for d in range(1, render_hz_int + 1) if render_hz_int % d == 0
            ]
            if divisors:
                depth_candidate = [d for d in divisors if d <= 25]
                rgb_candidate = [d for d in divisors if d <= 10]
                depth_camera_hz = (
                    max(depth_candidate) if depth_candidate else min(divisors)
                )
                rgb_camera_hz = max(rgb_candidate) if rgb_candidate else min(divisors)
                logger.info(
                    "[Sensors] Camera Hz adjusted for render_hz=%s: depth=%s, rgb=%s",
                    render_hz_int,
                    depth_camera_hz,
                    rgb_camera_hz,
                )

    try:
        ensure_link_xform(
            usd_stage,
            CAMERA_LINK_PRIM,
            translation=camera_link_position,
            rpy_rad=(math.radians(90.0), math.radians(0.0), math.radians(-90.0)),
        )

        realsense_depth_camera = Camera(
            prim_path=REALSENSE_DEPTH_CAMERA_PRIM,
            name="realsense_depth_camera",
            frequency=depth_camera_hz,
            resolution=(480, 270),
        )
        realsense_depth_camera.initialize()

        realsense_depth_cam_prim = usd_stage.GetPrimAtPath(REALSENSE_DEPTH_CAMERA_PRIM)
        if realsense_depth_cam_prim and realsense_depth_cam_prim.IsValid():
            from pxr import Gf, UsdGeom

            xformable = UsdGeom.Xformable(realsense_depth_cam_prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            # 25° downward tilt (X-axis) to bias the depth view toward the floor
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 0.0, 0.0))
            logger.info("Set realsense_depth_camera to 25° downward tilt")

        realsense_depth_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        realsense_depth_camera.add_distance_to_image_plane_to_frame()

        sensors["realsense_depth_camera"] = realsense_depth_camera
        logger.info("[Sensors] RealSense depth camera initialized with depth enabled")

        realsense_rgb_camera = Camera(
            prim_path=REALSENSE_RGB_CAMERA_PRIM,
            name="realsense_rgb_camera",
            frequency=rgb_camera_hz,
            resolution=(640, 480),
        )
        realsense_rgb_camera.initialize()

        realsense_rgb_cam_prim = usd_stage.GetPrimAtPath(REALSENSE_RGB_CAMERA_PRIM)
        if realsense_rgb_cam_prim and realsense_rgb_cam_prim.IsValid():
            from pxr import Gf, UsdGeom

            xformable = UsdGeom.Xformable(realsense_rgb_cam_prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            # 25° downward tilt (X-axis) — match the depth camera
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 0.0, 0.0))
            logger.info("Set realsense_rgb_camera to 25° downward tilt")

        realsense_rgb_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        sensors["realsense_rgb_camera"] = realsense_rgb_camera
        logger.info("[Sensors] RealSense RGB camera initialized")

        go2_rgb_camera = Camera(
            prim_path=GO2_RGB_CAMERA_PRIM,
            name="go2_rgb_camera",
            frequency=rgb_camera_hz,
            resolution=(640, 480),
        )
        go2_rgb_camera.initialize()

        go2_rgb_cam_prim = usd_stage.GetPrimAtPath(GO2_RGB_CAMERA_PRIM)
        if go2_rgb_cam_prim and go2_rgb_cam_prim.IsValid():
            from pxr import Gf, UsdGeom

            xformable = UsdGeom.Xformable(go2_rgb_cam_prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            # +25° around Y to counteract camera_link's -25° pitch so the
            # forward-facing Go2 camera stays level.
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 25.0, 0.0))
            logger.info(
                "Set go2_rgb_camera to face forward (counteracts camera_link tilt)"
            )

        go2_rgb_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        sensors["go2_rgb_camera"] = go2_rgb_camera
        logger.info("[Sensors] Go2 RGB camera initialized")
    except Exception as e:
        logger.warning("[WARN] Camera setup failed: %s", e)
        import traceback

        traceback.print_exc()

    try:
        imu_sensor = IMUSensor(
            prim_path=IMU_PRIM,
            name="imu_sensor",
            frequency=50,
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        imu_sensor.initialize()
        sensors["imu"] = imu_sensor
        logger.info("[Sensors] IMU initialized")
    except Exception as e:
        logger.warning("[WARN] IMU setup failed: %s", e)

    if enable_lidar:
        try:
            ensure_link_xform(
                usd_stage,
                L1_LINK_PRIM,
                translation=lidar_l1_position,
                rpy_rad=(0.0, 0.0, 0.0),
            )
            lidar_path = L1_LIDAR_PRIM
            if not is_prim_path_valid(lidar_path):
                result = omni.kit.commands.execute(
                    "IsaacSensorCreateRtxLidar",
                    path="lidar_l1_rtx",
                    parent=L1_LINK_PRIM,
                    config="Example_Rotary",
                    translation=(0.0, 0.0, 0.0),
                    orientation=Gf.Quatd(1, 0, 0, 0),
                )
                if result and len(result) > 1 and result[1]:
                    lidar_prim = result[1]
                    lidar_path = lidar_prim.GetPath().pathString
                    logger.info("[Sensors] L1 LiDAR created at: %s", lidar_path)
                else:
                    logger.warning("[WARN] L1 LiDAR creation returned: %s", result)
                    lidar_path = None
            if lidar_path:
                l1_rp = rep.create.render_product(
                    lidar_path, resolution=(1, 1), name="l1_lidar_rp"
                )
                pc_writer = rep.writers.get("RtxLidarROS2PublishPointCloud")
                pc_writer.initialize(
                    frameId="lidar_l1_link",
                    nodeNamespace="",
                    topicName="/unitree_lidar/points",
                    queueSize=10,
                )
                pc_writer.attach([l1_rp])
                logger.info("[Sensors] L1 LiDAR -> /unitree_lidar/points")
        except Exception as e:
            logger.warning("[WARN] L1 LiDAR setup failed: %s", e)
            import traceback

            traceback.print_exc()

        try:
            ensure_link_xform(
                usd_stage,
                VELO_BASE_LINK_PRIM,
                translation=lidar_velo_position,
                rpy_rad=(0.0, 0.0, 0.0),
            )
            ensure_link_xform(
                usd_stage,
                VELO_LASER_LINK_PRIM,
                translation=(0.0, 0.0, 0.0377),
                rpy_rad=(0.0, 0.0, 0.0),
            )
            rplidar_path = f"{VELO_LASER_LINK_PRIM}/rplidar"
            lidar_path = rplidar_path
            if not is_prim_path_valid(rplidar_path):
                result = omni.kit.commands.execute(
                    "IsaacSensorCreateRtxLidar",
                    path="rplidar",
                    parent=VELO_LASER_LINK_PRIM,
                    config="Slamtec_RPLIDAR_S2E",
                    translation=(0.0, 0.0, 0.0),
                    orientation=Gf.Quatd(1, 0, 0, 0),
                )
                if result and len(result) > 1 and result[1]:
                    lidar_prim = result[1]
                    lidar_path = lidar_prim.GetPath().pathString
                    logger.info("[Sensors] 2D LiDAR created at: %s", lidar_path)
                else:
                    logger.warning("[WARN] 2D LiDAR creation returned: %s", result)
                    lidar_path = None
            if lidar_path:
                velo_rp = rep.create.render_product(
                    lidar_path, resolution=(1, 1), name="velo_lidar_rp"
                )
                scan_writer = rep.writers.get("RtxLidarROS2PublishLaserScan")
                scan_writer.initialize(
                    frameId="laser", nodeNamespace="", topicName="/scan", queueSize=10
                )
                scan_writer.attach([velo_rp])
                logger.info("[Sensors] 2D LiDAR -> /scan")
        except Exception as e:
            logger.warning("[WARN] 2D LiDAR setup failed: %s", e)
    else:
        logger.info("LiDAR sensors skipped (enable_lidar=False)")

    simulation_app.update()
    return sensors


def setup_static_tfs(
    simulation_app,
    robot_type: str = "go2",
    camera_link_pos: Tuple[float, float, float] = (0.3, 0.0, 0.1),
    lidar_l1_pos: Tuple[float, float, float] = (0.15, 0.0, 0.15),
    lidar_velo_pos: Tuple[float, float, float] = (0.1, 0.0, 0.2),
) -> None:
    """Publish static TFs for sensor frames."""
    import omni.graph.core as og
    from isaacsim.core.utils.prims import is_prim_path_valid

    graph_path = "/StaticTFGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Static TF graph already exists")
        return

    if robot_type == "g1":
        body_frame = "torso_link"
        static_transforms = [
            ("base_link", "torso_link", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (body_frame, "lidar_l1_link", list(lidar_l1_pos), [0.0, 0.0, 0.0, 1.0]),
            (
                body_frame,
                "velodyne_base_link",
                list(lidar_velo_pos),
                [0.0, 0.0, 0.0, 1.0],
            ),
            ("velodyne_base_link", "laser", [0.0, 0.0, 0.0377], [0.0, 0.0, 0.0, 1.0]),
            (body_frame, "imu_link", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (
                body_frame,
                "camera_link",
                list(camera_link_pos),
                [0.5, -0.5, -0.5, 0.5],
            ),
            (
                "camera_link",
                "realsense_depth_camera",
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ),
            (
                "camera_link",
                "realsense_rgb_camera",
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ),
            (
                "base_link",
                "realsense_depth_camera",
                list(camera_link_pos),
                [0.5, -0.5, -0.5, 0.5],
            ),
            ("map", "odom", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ]
    else:
        body_frame = "base"
        static_transforms = [
            ("base_link", "base", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (body_frame, "lidar_l1_link", list(lidar_l1_pos), [0.0, 0.0, 0.0, 1.0]),
            (
                body_frame,
                "velodyne_base_link",
                list(lidar_velo_pos),
                [0.0, 0.0, 0.0, 1.0],
            ),
            ("velodyne_base_link", "laser", [0.0, 0.0, 0.0377], [0.0, 0.0, 0.0, 1.0]),
            (body_frame, "imu_link", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (
                body_frame,
                "camera_link",
                list(camera_link_pos),
                [0.5, -0.5, -0.5, 0.5],
            ),
            (
                "camera_link",
                "realsense_depth_camera",
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ),
            (
                "camera_link",
                "realsense_rgb_camera",
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ),
            (
                "base_link",
                "realsense_depth_camera",
                list(camera_link_pos),
                [0.5, -0.5, -0.5, 0.5],
            ),
            ("map", "odom", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ]

    create_nodes = [
        ("OnTick", "omni.graph.action.OnTick"),
        ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
    ]

    for i, _ in enumerate(static_transforms):
        create_nodes.append(
            (f"TF{i}", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
        )

    connections = []
    for i, _ in enumerate(static_transforms):
        connections.append(("OnTick.outputs:tick", f"TF{i}.inputs:execIn"))
        connections.append(("Clock.outputs:simulationTime", f"TF{i}.inputs:timeStamp"))
        connections.append(("Ctx.outputs:context", f"TF{i}.inputs:context"))

    set_values = [("Ctx.inputs:useDomainIDEnvVar", True)]

    for i, (parent, child, trans, rot) in enumerate(static_transforms):
        set_values.extend(
            [
                (f"TF{i}.inputs:parentFrameId", parent),
                (f"TF{i}.inputs:childFrameId", child),
                (f"TF{i}.inputs:topicName", "/tf_static"),
                (f"TF{i}.inputs:translation", trans),
                (f"TF{i}.inputs:rotation", rot),
                (f"TF{i}.inputs:staticPublisher", True),
                (f"TF{i}.inputs:queueSize", 10),
            ]
        )

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.CONNECT: connections,
            og.Controller.Keys.SET_VALUES: set_values,
        },
    )

    logger.info(
        "[ROS2] Static TFs published for %d transforms (staticPublisher=True)",
        len(static_transforms),
    )
    simulation_app.update()


def setup_odom_publisher(simulation_app) -> None:
    """Publish nav_msgs/Odometry on /odom topic."""
    import omni.graph.core as og
    from isaacsim.core.utils.prims import is_prim_path_valid

    global odom_pos_attr, odom_orient_attr, odom_lin_vel_attr, odom_ang_vel_attr

    graph_path = "/OdomPublisherGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Odom publisher graph already exists")
        return

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                ("OdomPub", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "OdomPub.inputs:execIn"),
                ("Clock.outputs:simulationTime", "OdomPub.inputs:timeStamp"),
                ("Ctx.outputs:context", "OdomPub.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Ctx.inputs:useDomainIDEnvVar", True),
                ("OdomPub.inputs:topicName", "/odom"),
                ("OdomPub.inputs:odomFrameId", "odom"),
                ("OdomPub.inputs:chassisFrameId", "base_link"),
                ("OdomPub.inputs:queueSize", 10),
                ("OdomPub.inputs:position", [0.0, 0.0, 0.0]),
                ("OdomPub.inputs:orientation", [0.0, 0.0, 0.0, 1.0]),
                ("OdomPub.inputs:linearVelocity", [0.0, 0.0, 0.0]),
                ("OdomPub.inputs:angularVelocity", [0.0, 0.0, 0.0]),
            ],
        },
    )

    odom_pos_attr = og.Controller.attribute(graph_path + "/OdomPub.inputs:position")
    odom_orient_attr = og.Controller.attribute(
        graph_path + "/OdomPub.inputs:orientation"
    )
    odom_lin_vel_attr = og.Controller.attribute(
        graph_path + "/OdomPub.inputs:linearVelocity"
    )
    odom_ang_vel_attr = og.Controller.attribute(
        graph_path + "/OdomPub.inputs:angularVelocity"
    )

    logger.info("[ROS2] Odometry publisher -> /odom")
    simulation_app.update()


def update_odom(pos, quat_xyzw, lin_vel, ang_vel) -> None:
    """Update the odometry message each frame."""
    global odom_pos_attr, odom_orient_attr, odom_lin_vel_attr, odom_ang_vel_attr

    if odom_pos_attr is not None:
        odom_pos_attr.set([float(pos[0]), float(pos[1]), float(pos[2])])
    if odom_orient_attr is not None:
        odom_orient_attr.set(
            [
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
                float(quat_xyzw[3]),
            ]
        )
    if odom_lin_vel_attr is not None:
        odom_lin_vel_attr.set([float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])])
    if odom_ang_vel_attr is not None:
        odom_ang_vel_attr.set([float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])])


def setup_color_camera_publishers(sensors, simulation_app) -> None:
    """Set up ROS2 publishers for color camera images."""
    import omni.replicator.core as rep
    import omni.syntheticdata as syn_data
    import omni.syntheticdata._syntheticdata as sd

    if sensors.get("realsense_depth_camera"):
        cam = sensors["realsense_depth_camera"]
        rp = cam.get_render_product_path()
        if rp:
            try:
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
                w = rep.writers.get(rv + "ROS2PublishImage")
                w.initialize(
                    frameId="realsense_depth_camera",
                    nodeNamespace="",
                    queueSize=10,
                    topicName="/camera/realsense2_camera_node/color/image_raw",
                )
                w.attach([rp])
                logger.info(
                    "[ROS2] Color camera -> /camera/realsense2_camera_node/color/image_raw"
                )

            except Exception as e:
                logger.warning("[WARN] Color camera publisher setup failed: %s", e)

    if sensors.get("go2_rgb_camera"):
        cam = sensors["go2_rgb_camera"]
        rp = cam.get_render_product_path()
        if rp:
            try:
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
                for topic in ("/camera/image_raw", "camera/go2/image_raw"):
                    w = rep.writers.get(rv + "ROS2PublishImage")
                    w.initialize(
                        frameId="go2_rgb_camera",
                        nodeNamespace="",
                        queueSize=10,
                        topicName=topic,
                    )
                    w.attach([rp])
                    logger.info("[ROS2] Go2 RGB camera -> %s", topic)

            except Exception as e:
                logger.warning("[WARN] Go2 RGB camera publisher setup failed: %s", e)


def setup_color_camerainfo_graph(
    simulation_app,
    topic="/camera/realsense2_camera_node/color/camera_info",
    frame_id="realsense_depth_camera",
    width=480,
    height=270,
    fx=242.479,
    fy=242.479,
    cx=None,
    cy=None,
) -> bool:
    """Publish CameraInfo for color camera."""
    import omni.graph.core as og
    from isaacsim.core.utils.prims import is_prim_path_valid

    graph_path = "/ColorCameraInfoGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Color CameraInfo graph already exists")
        return True

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                ("Pub", "isaacsim.ros2.bridge.ROS2PublishCameraInfo"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "Pub.inputs:execIn"),
                ("Clock.outputs:simulationTime", "Pub.inputs:timeStamp"),
                ("Ctx.outputs:context", "Pub.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Ctx.inputs:useDomainIDEnvVar", True),
                ("Pub.inputs:topicName", topic),
                ("Pub.inputs:frameId", frame_id),
                ("Pub.inputs:queueSize", 10),
                ("Pub.inputs:width", width),
                ("Pub.inputs:height", height),
                ("Pub.inputs:k", K),
                ("Pub.inputs:r", R),
                ("Pub.inputs:p", P),
                ("Pub.inputs:physicalDistortionModel", "plumb_bob"),
                (
                    "Pub.inputs:physicalDistortionCoefficients",
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        },
    )

    logger.info("[ROS2] Color CameraInfo -> %s", topic)
    simulation_app.update()
    return True


def setup_joint_states_publisher(simulation_app, robot_type: str = "go2") -> None:
    """Publish sensor_msgs/JointState on /joint_states."""
    import omni.graph.core as og
    from isaacsim.core.nodes.scripts.utils import set_target_prims
    from isaacsim.core.utils.prims import is_prim_path_valid

    graph_path = "/JointStatesGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Joint states graph already exists")
        return

    if robot_type == "g1":
        ROBOT_ARTICULATION_PATH = f"{GO2_STAGE_PATH}/torso_link"
    elif robot_type == "tron1":
        ROBOT_ARTICULATION_PATH = f"{GO2_STAGE_PATH}/base_Link"
    else:
        ROBOT_ARTICULATION_PATH = f"{GO2_STAGE_PATH}/base"

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                ("JointStatePub", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "JointStatePub.inputs:execIn"),
                ("Clock.outputs:simulationTime", "JointStatePub.inputs:timeStamp"),
                ("Ctx.outputs:context", "JointStatePub.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Ctx.inputs:useDomainIDEnvVar", True),
                ("JointStatePub.inputs:topicName", "/joint_states"),
                ("JointStatePub.inputs:queueSize", 10),
            ],
        },
    )

    set_target_prims(
        primPath=graph_path + "/JointStatePub",
        inputName="inputs:targetPrim",
        targetPrimPaths=[ROBOT_ARTICULATION_PATH],
    )

    logger.info(
        "[ROS2] Joint states publisher -> /joint_states (articulation: %s)",
        ROBOT_ARTICULATION_PATH,
    )
    simulation_app.update()


def setup_ros_publishers(
    sensors,
    simulation_app,
    robot_type: str = "go2",
    camera_link_pos: Tuple[float, float, float] = (0.3, 0.0, 0.1),
    lidar_l1_pos: Tuple[float, float, float] = (0.15, 0.0, 0.15),
    lidar_velo_pos: Tuple[float, float, float] = (0.1, 0.0, 0.2),
) -> None:
    """Setup ROS2 publishers for sensors."""
    import omni.graph.core as og
    import omni.replicator.core as rep
    import omni.syntheticdata as syn_data
    import omni.syntheticdata._syntheticdata as sd
    from isaacsim.core.utils.prims import is_prim_path_valid

    # Clock publisher
    graph_path = "/ClockGraph"
    if not is_prim_path_valid(graph_path):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Pub", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "Pub.inputs:execIn"),
                    ("Clock.outputs:simulationTime", "Pub.inputs:timeStamp"),
                ],
            },
        )
    logger.info("[ROS2] Clock publisher -> /clock")

    # IMU publisher
    if not is_prim_path_valid("/ImuGraph"):
        og.Controller.edit(
            {
                "graph_path": "/ImuGraph",
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                    ("Read", "isaacsim.sensors.physics.IsaacReadIMU"),
                    ("Pub", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "Read.inputs:execIn"),
                    ("Read.outputs:execOut", "Pub.inputs:execIn"),
                    ("Ctx.outputs:context", "Pub.inputs:context"),
                    ("Clock.outputs:simulationTime", "Pub.inputs:timeStamp"),
                    ("Read.outputs:angVel", "Pub.inputs:angularVelocity"),
                    ("Read.outputs:linAcc", "Pub.inputs:linearAcceleration"),
                    ("Read.outputs:orientation", "Pub.inputs:orientation"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Ctx.inputs:useDomainIDEnvVar", True),
                    ("Read.inputs:imuPrim", IMU_PRIM),
                    ("Read.inputs:readGravity", True),
                    ("Pub.inputs:frameId", "imu_link"),
                    ("Pub.inputs:topicName", "/imu/data"),
                    ("Pub.inputs:queueSize", 10),
                ],
            },
        )
    logger.info("[ROS2] IMU publisher -> /imu/data")

    # Camera publishers with CameraInfo
    if sensors.get("realsense_depth_camera"):
        cam = sensors["realsense_depth_camera"]
        rp = cam.get_render_product_path()
        if rp:
            try:
                # Depth Image
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.DistanceToImagePlane.name
                )
                w_rs_depth = rep.writers.get(rv + "ROS2PublishImage")
                w_rs_depth.initialize(
                    frameId="realsense_depth_camera",
                    nodeNamespace="",
                    queueSize=10,
                    topicName="/camera/realsense2_camera_node/depth/image_rect_raw",
                )
                w_rs_depth.attach([rp])
                logger.info(
                    "[ROS2] Depth camera -> /camera/realsense2_camera_node/depth/image_rect_raw"
                )

                # For easier RViz viewing
                try:
                    depth_colorized = rep.writers.get(
                        "ROS2PublishNormalized" + "DepthImage"
                    )
                    depth_colorized.initialize(
                        frameId="realsense_depth_camera",
                        nodeNamespace="",
                        queueSize=10,
                        topicName="camera/depth/image_colorized",
                    )
                    depth_colorized.attach([rp])
                    logger.info(
                        "[ROS2] Depth colorized -> camera/depth/image_colorized"
                    )
                except Exception as de:
                    logger.info("[INFO] Normalized depth writer not available: %s", de)

            except Exception as e:
                logger.warning("[WARN] Camera publisher setup failed: %s", e)
                import traceback

                traceback.print_exc()

    # Setup static TFs for sensor frames
    setup_static_tfs(
        simulation_app,
        robot_type=robot_type,
        camera_link_pos=camera_link_pos,
        lidar_l1_pos=lidar_l1_pos,
        lidar_velo_pos=lidar_velo_pos,
    )

    # Odom TF publisher (dynamic - updated each frame)
    global odom_tf_trans_attr, odom_tf_rot_attr
    if not is_prim_path_valid(odom_graph_path):
        og.Controller.edit(
            {
                "graph_path": odom_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                    ("TF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "TF.inputs:execIn"),
                    ("Clock.outputs:simulationTime", "TF.inputs:timeStamp"),
                    ("Ctx.outputs:context", "TF.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Ctx.inputs:useDomainIDEnvVar", True),
                    ("TF.inputs:parentFrameId", "odom"),
                    ("TF.inputs:childFrameId", "base_link"),
                    ("TF.inputs:topicName", "/tf"),
                ],
            },
        )
    odom_tf_trans_attr = og.Controller.attribute(
        odom_graph_path + "/TF.inputs:translation"
    )
    odom_tf_rot_attr = og.Controller.attribute(odom_graph_path + "/TF.inputs:rotation")
    logger.info("[ROS2] Odom TF -> /tf (odom->base_link)")

    simulation_app.update()


def setup_depth_camerainfo_graph(
    simulation_app,
    topic="/camera/realsense2_camera_node/depth/camera_info",
    frame_id="realsense_depth_camera",
    width=480,
    height=270,
    fx=242.479,
    fy=242.479,
    cx=None,
    cy=None,
) -> bool:
    """Publish depth CameraInfo."""
    import omni.graph.core as og
    from isaacsim.core.utils.prims import is_prim_path_valid

    graph_path = "/DepthCameraInfoGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Depth CameraInfo graph already exists")
        return True

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                ("Pub", "isaacsim.ros2.bridge.ROS2PublishCameraInfo"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "Pub.inputs:execIn"),
                ("Clock.outputs:simulationTime", "Pub.inputs:timeStamp"),
                ("Ctx.outputs:context", "Pub.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Ctx.inputs:useDomainIDEnvVar", True),
                ("Pub.inputs:topicName", topic),
                ("Pub.inputs:frameId", frame_id),
                ("Pub.inputs:queueSize", 10),
                ("Pub.inputs:width", width),
                ("Pub.inputs:height", height),
                ("Pub.inputs:k", K),
                ("Pub.inputs:r", R),
                ("Pub.inputs:p", P),
                ("Pub.inputs:physicalDistortionModel", "plumb_bob"),
                (
                    "Pub.inputs:physicalDistortionCoefficients",
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        },
    )

    logger.info(
        "Depth CameraInfo -> %s (width=%s, height=%s, fx=%s, fy=%s)",
        topic,
        width,
        height,
        fx,
        fy,
    )
    simulation_app.update()
    return True


def update_odom_tf(pos, quat_xyzw) -> None:
    """Update the odom -> base_link transform each frame."""
    if odom_tf_trans_attr is not None and odom_tf_rot_attr is not None:
        odom_tf_trans_attr.set([float(pos[0]), float(pos[1]), float(pos[2])])
        odom_tf_rot_attr.set(
            [
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
                float(quat_xyzw[3]),
            ]
        )


def find_robot_articulation_path():
    """Find the actual robot articulation path in the stage."""
    import omni.usd
    from pxr import UsdPhysics

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    logger.debug("Searching for articulation roots in stage...")

    articulations = []
    for prim in usd_stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append(prim.GetPath().pathString)
            logger.debug("  [ArticulationRoot] %s", prim.GetPath().pathString)

    common_paths = [
        "/World/envs/env_0/Robot",
        "/World/envs/env_0/Robot/base",
        "/World/Go2",
        "/World/go2",
        "/World/robot",
    ]

    logger.debug("Checking common paths:")
    for path in common_paths:
        prim = usd_stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            has_arctic = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            logger.debug("  %s: exists=True, has_articulation_api=%s", path, has_arctic)
        else:
            logger.debug("  %s: exists=False", path)

    robot_prim = usd_stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if robot_prim and robot_prim.IsValid():
        logger.debug("Children of /World/envs/env_0/Robot:")
        for child in robot_prim.GetChildren():
            has_arctic = child.HasAPI(UsdPhysics.ArticulationRootAPI)
            logger.debug(
                "  %s (articulation=%s)", child.GetPath().pathString, has_arctic
            )

    env_prim = usd_stage.GetPrimAtPath("/World/envs/env_0")
    if env_prim and env_prim.IsValid():
        logger.debug("Children of /World/envs/env_0:")
        for child in env_prim.GetChildren():
            has_arctic = child.HasAPI(UsdPhysics.ArticulationRootAPI)
            logger.debug(
                "  %s (articulation=%s)", child.GetPath().pathString, has_arctic
            )

    return articulations


# Lowstate publisher fallback: spawns lowstate_node.py as a subprocess.
# Normally the launch files start lowstate_node as a Node — these helpers
# are kept for the case where run.py is run outside ros2 launch.
_LOWSTATE_ROS2_WS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "install",
    "setup.bash",
)

_lowstate_process: Optional["subprocess.Popen"] = None


def setup_lowstate_publisher(
    ros2_ws: Optional[str] = None,
) -> None:
    """
    Launch the lowstate publisher as a subprocess under system Python + ROS2.

    Parameters
    ----------
    ros2_ws : str, optional
        Path to the ROS2 workspace install/setup.bash.
    """
    global _lowstate_process
    if _lowstate_process is not None and _lowstate_process.poll() is None:
        logger.info("[Lowstate] Already running (pid=%d)", _lowstate_process.pid)
        return

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lowstate_node.py"
    )

    # Find workspace setup.bash
    ws_setup = None
    if ros2_ws and os.path.isfile(ros2_ws):
        ws_setup = os.path.abspath(ros2_ws)
    else:
        # Search relative to this script. Look for both:
        #   <ancestor>/install/setup.bash         (OM1-sim colcon build)
        #   <ancestor>/ros2_ws/install/setup.bash (separate workspace layout)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.path.abspath(_LOWSTATE_ROS2_WS)]
        search = script_dir
        for _ in range(5):
            search = os.path.dirname(search)
            candidates.append(os.path.join(search, "install", "setup.bash"))
            candidates.append(os.path.join(search, "ros2_ws", "install", "setup.bash"))
        for c in candidates:
            if os.path.isfile(c):
                ws_setup = c
                break

    # Build the shell command
    parts = ["source /opt/ros/humble/setup.bash"]
    if ws_setup:
        parts.append(f"source {ws_setup}")
    parts.append(f"exec python3 {script_path}")
    shell_cmd = " && ".join(parts)

    # Start with a clean environment to avoid Isaac Sim's Python 3.11
    # paths leaking into the system Python 3.10 subprocess.
    clean_env = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TERM": os.environ.get("TERM", "xterm"),
    }
    # Preserve ROS_DOMAIN_ID if set
    if "ROS_DOMAIN_ID" in os.environ:
        clean_env["ROS_DOMAIN_ID"] = os.environ["ROS_DOMAIN_ID"]

    _lowstate_process = subprocess.Popen(
        ["bash", "-c", shell_cmd],
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logger.info(
        "[Lowstate] Subprocess started (pid=%d), script=%s",
        _lowstate_process.pid,
        script_path,
    )


def stop_lowstate_publisher() -> None:
    """Stop the lowstate publisher subprocess."""
    global _lowstate_process
    if _lowstate_process is None:
        return
    if _lowstate_process.poll() is None:
        _lowstate_process.terminate()
        try:
            _lowstate_process.wait(timeout=3)
        except Exception:
            _lowstate_process.kill()
    logger.info("[Lowstate] Subprocess stopped")
    _lowstate_process = None


def update_lowstate():
    """No-op: the subprocess handles its own publishing at 100Hz."""
    pass


def create_apriltag_dock(
    simulation_app,
    position: tuple = (-2.0, -2.0, 0.15),
    rotation_deg: float = 90.0,
    tag_size: float = 0.2,
) -> bool:
    """
    Create a charging dock with AprilTag texture for auto-docking.

    Parameters
    ----------
    simulation_app : SimulationApp
        The Isaac Sim application instance.
    position : tuple
        (x, y, z) position for the dock in world coordinates.
    rotation_deg : float
        Yaw rotation in degrees (facing direction).
    tag_size : float
        Size of the AprilTag panel in meters.

    Returns
    -------
    bool
        True if dock was created successfully.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    dock_path = "/World/ChargingDock"

    # Create dock xform
    dock_prim = usd_stage.DefinePrim(dock_path, "Xform")
    dock_xform = UsdGeom.Xformable(dock_prim)
    dock_xform.ClearXformOpOrder()

    # Set position
    dock_xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    # Set rotation (yaw)
    dock_xform.AddRotateZOp().Set(rotation_deg)

    # Get the AprilTag texture path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    texture_paths = [
        os.path.join(script_dir, "assets", "apriltags.png"),
        os.path.join(script_dir, "..", "go2_description", "tags", "apriltags.png"),
    ]

    texture_path = None
    for path in texture_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            texture_path = abs_path
            break

    # Create the AprilTag panel as a mesh quad with UV coordinates
    panel_path = f"{dock_path}/TagPanel"
    panel_prim = usd_stage.DefinePrim(panel_path, "Mesh")
    panel_mesh = UsdGeom.Mesh(panel_prim)

    # Match mesh aspect ratio to the source image (909x587) so tags aren't stretched
    img_w, img_h = 909, 587
    half_w = tag_size / 2.0
    half_h = half_w * (img_h / img_w)  # preserve aspect ratio

    # Define a vertical quad facing +Y direction (toward robot approach)
    points = [
        Gf.Vec3f(-half_w, 0.0, -half_h),  # bottom-left
        Gf.Vec3f(half_w, 0.0, -half_h),  # bottom-right
        Gf.Vec3f(half_w, 0.0, half_h),  # top-right
        Gf.Vec3f(-half_w, 0.0, half_h),  # top-left
    ]
    panel_mesh.GetPointsAttr().Set(points)
    panel_mesh.GetFaceVertexCountsAttr().Set([4])
    panel_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])

    # UV coordinates for texture mapping
    texcoords = UsdGeom.PrimvarsAPI(panel_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    texcoords.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )

    # Set normals
    normals = UsdGeom.PrimvarsAPI(panel_prim).CreatePrimvar(
        "normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying
    )
    normals.Set(
        [
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )

    panel_mesh.GetDoubleSidedAttr().Set(True)

    # Create material with AprilTag texture
    material_path = f"{dock_path}/AprilTagMaterial"

    try:
        if texture_path:
            # Create a simple preview surface material
            material_prim = usd_stage.DefinePrim(material_path, "Material")
            material = UsdShade.Material(material_prim)

            shader_path = f"{material_path}/Shader"
            shader_prim = usd_stage.DefinePrim(shader_path, "Shader")
            shader = UsdShade.Shader(shader_prim)
            shader.CreateIdAttr("UsdPreviewSurface")

            # Create texture sampler
            tex_path = f"{material_path}/diffuse_texture"
            tex_prim = usd_stage.DefinePrim(tex_path, "Shader")
            tex_shader = UsdShade.Shader(tex_prim)
            tex_shader.CreateIdAttr("UsdUVTexture")
            tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
            tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
            tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            # Create ST reader for UV coordinates
            st_path = f"{material_path}/st_reader"
            st_prim = usd_stage.DefinePrim(st_path, "Shader")
            st_shader = UsdShade.Shader(st_prim)
            st_shader.CreateIdAttr("UsdPrimvarReader_float2")
            st_shader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_shader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # Connect ST to texture
            tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_shader.ConnectableAPI(), "result"
            )

            # Make tag self-illuminating (unlit) so AprilTag detection works
            # regardless of scene lighting — connect texture to emissive only.
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.0, 0.0, 0.0)
            )
            shader.CreateInput(
                "emissiveColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

            # Connect shader outputs
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )

            # Bind to panel
            UsdShade.MaterialBindingAPI(panel_prim).Bind(material)
            logger.info("[Dock] Using AprilTag texture: %s", texture_path)
        else:
            logger.warning(
                "AprilTag texture not found. Creating dock without texture. "
                "Copy apriltags.png to assets/"
            )
    except Exception as e:
        logger.warning("[Dock] Material creation failed: %s", e)

    logger.info(
        "[Dock] Created charging dock at position (%.2f, %.2f, %.2f) with rotation %.1f°",
        position[0],
        position[1],
        position[2],
        rotation_deg,
    )

    simulation_app.update()
    return True


# Dock position for navigation from go2_nav_to_charger.py hardcoded goal
DOCK_POSITION = (-0.96, -2.25, 0)
DOCK_YAW_DEG = 90.0  # Facing +Y direction
