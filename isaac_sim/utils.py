# ruff: noqa: E402

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Go2 Robot prim path
GO2_STAGE_PATH = "/World/envs/env_0/Robot"

# Sensor prim paths on Go2 base
IMU_PRIM = f"{GO2_STAGE_PATH}/base/imu_link"
CAMERA_LINK_PRIM = f"{GO2_STAGE_PATH}/base/camera_link"
REALSENSE_DEPTH_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/realsense_depth_camera"
REALSENSE_RGB_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/realsense_rgb_camera"
FRONT_RGB_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/front_rgb_camera"
TOP_RGB_CAMERA_PRIM = f"{CAMERA_LINK_PRIM}/top_rgb_camera"
BASE_LINK_PRIM = f"{GO2_STAGE_PATH}/base"
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


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value between a lower and upper bound."""
    return max(lo, min(hi, x))


def quat_xyzw_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list:
    """Fixed-axis (roll, pitch, yaw) degrees -> xyzw quaternion (URDF rpy)."""
    r, p, y = (math.radians(v) / 2.0 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def setup_cmd_vel_graph(
    topic_name: str = "/cmd_vel",
) -> Tuple[object, object, object]:
    """Set up the command velocity subscriber graph for ROS2 integration.

    Returns (linear_attr, angular_attr, msg_count_attr).
    msg_count_attr increments each time a new /cmd_vel message is received.
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


FRONT_RGB_RESOLUTION = (640, 480)
FRONT_RGB_FOCAL_LENGTH_MM = 24.0
FRONT_RGB_HORIZONTAL_APERTURE_MM = 20.955

REALSENSE_PITCH_DEG = -25.0
FRONT_RGB_PITCH_DEG = 0.0
TOP_RGB_PITCH_DEG = 30.0


def camera_intrinsics_from_lens(
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    resolution: Tuple[int, int],
) -> Tuple[int, int, float, float, float, float]:
    """Pinhole intrinsics of a square-pixel USD camera: (w, h, fx, fy, cx, cy).

    The RTX renderer fits the horizontal aperture to the render product's width
    and derives the vertical extent from its aspect ratio, so fy == fx and the
    principal point is the image centre.
    """
    width, height = resolution
    fx = width * float(focal_length_mm) / float(horizontal_aperture_mm)
    return width, height, fx, fx, width / 2.0, height / 2.0


def set_camera_lens(
    usd_stage,
    path: str,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    resolution: Tuple[int, int],
) -> None:
    """Pin a camera prim's lens so its intrinsics are explicit, not inherited.

    A prim created by ``Camera(...)`` keeps the UsdGeomCamera schema fallbacks
    (50 mm focal length), so any config that hard-codes fx/fy has to guess.
    The vertical aperture is set from ``resolution`` so pixels are square, which
    is what ``camera_intrinsics_from_lens`` assumes.
    """
    prim = usd_stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        logger.info(f"[WARN] set_camera_lens: no prim at {path}")
        return
    width, height = resolution
    vertical_aperture_mm = horizontal_aperture_mm * float(height) / float(width)
    prim.GetAttribute("focalLength").Set(float(focal_length_mm))
    prim.GetAttribute("horizontalAperture").Set(float(horizontal_aperture_mm))
    prim.GetAttribute("verticalAperture").Set(float(vertical_aperture_mm))
    _, _, fx, _, cx, cy = camera_intrinsics_from_lens(
        focal_length_mm, horizontal_aperture_mm, resolution
    )
    logger.info(
        f"[Sensors] {path} lens: f={focal_length_mm} mm, "
        f"aperture={horizontal_aperture_mm}x{vertical_aperture_mm:.5f} mm "
        f"-> fx=fy={fx:.1f}, cx/cy={cx}/{cy}"
    )


def set_camera_pitch(usd_stage, path: str, pitch_deg: float, label: str) -> None:
    """Aim a camera prim up or down within camera_link's vertical plane.

    camera_link is oriented rpy (90, 0, -90), the USD camera convention: it looks
    along base +X with up = base +Z, which puts its local X axis along base -Y.
    A rotation about that local X is therefore a *pitch* about base +Y, and the
    only rotation that aims a camera up or down without also swinging it
    sideways. ``pitch_deg`` is positive up, negative down; 0 looks straight
    ahead. Rotating about the local Y axis instead would yaw the camera left,
    which is never what a front/top/down mount wants.
    """
    from pxr import Gf, UsdGeom

    prim = usd_stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        logger.info(f"[WARN] set_camera_pitch: no prim at {path}")
        return
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(float(pitch_deg), 0.0, 0.0))
    if pitch_deg > 0:
        aim = f"{pitch_deg:g} deg above horizontal"
    elif pitch_deg < 0:
        aim = f"{-pitch_deg:g} deg below horizontal"
    else:
        aim = "straight ahead"
    logger.info(f"[Sensors] Set {label} {aim}")


def _create_rtx_lidar_pointcloud(
    usd_stage,
    link_prim: str,
    sensor_name: str,
    position: Tuple[float, float, float],
    rpy_deg: Tuple[float, float, float],
    topic: str,
    frame_id: str,
    rp_name: str,
    render_hz: Optional[float] = None,
) -> bool:
    """Create an RTX OS0 lidar under ``link_prim`` publishing PointCloud2 on ``topic``."""
    import omni.kit.commands
    import omni.replicator.core as rep
    from pxr import Gf

    ensure_link_xform(
        usd_stage,
        link_prim,
        translation=position,
        rpy_rad=tuple(math.radians(v) for v in rpy_deg),
    )
    result = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=sensor_name,
        parent=link_prim,
        config="OS0",
        variant="OS0_REV6_128ch10hz512res",
        translation=(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1, 0, 0, 0),
    )
    if not (result and len(result) > 1 and result[1]):
        logger.info(f"[WARN] RTX lidar creation returned: {result}")
        return False

    lidar_prim = result[1]
    lidar_path = lidar_prim.GetPath().pathString
    logger.info(f"[Sensors] RTX lidar created at: {lidar_path}")

    if render_hz:
        scan_attr = lidar_prim.GetAttribute("omni:sensor:Core:scanRateBaseHz")
        if scan_attr and scan_attr.IsValid():
            cur_scan = scan_attr.Get()
            if cur_scan and render_hz > cur_scan:
                target_scan = float(render_hz)
                scan_attr.Set(target_scan)
                logger.info(
                    f"[Sensors] RTX lidar scan rate {cur_scan}->{target_scan} Hz "
                    "(azimuth resolution auto-reduced to fit fire time)"
                )
        else:
            logger.info(
                "[WARN] RTX lidar scanRateBaseHz attr not found; leaving stock "
                "10 Hz (cloud may flash at higher render rates)"
            )

    render_product = rep.create.render_product(
        lidar_path, resolution=(1, 1), name=rp_name
    )
    pc_writer = rep.writers.get("RtxLidarROS2PublishPointCloud")
    pc_writer.initialize(
        frameId=frame_id,
        nodeNamespace="",
        topicName=topic,
        queueSize=10,
    )
    pc_writer.attach([render_product])
    logger.info(f"[Sensors] RTX lidar -> {topic} (frame_id={frame_id})")
    return True


def setup_sensors_delayed(
    simulation_app,
    render_hz: Optional[float] = None,
    camera_link_position: Optional[Tuple[float, float, float]] = None,
    enable_lidar: bool = True,
    lidar_l1_position: Optional[Tuple[float, float, float]] = None,
    lidar_velo_position: Optional[Tuple[float, float, float]] = None,
    robot_type: str = "go2",
    lidars_3d: Optional[list] = None,
    enable_2d_lidar: bool = True,
    physics_hz: Optional[float] = None,
) -> dict:
    """Setup sensors after simulation is fully running.

    ``lidars_3d`` is an optional list of per-unit 3D lidar configs
    (``sim_config.Lidar3DConfig``); when set it replaces the single L1 lidar
    with one RTX lidar per entry (e.g. the M20's front + back units).
    ``enable_2d_lidar`` gates the simulated RPLIDAR (-> /scan); robots that
    derive /scan from their 3D clouds turn it off. ``physics_hz`` is the physics
    step rate; the IMU samples at that rate so its output is not the bottleneck
    for the LIO/LIVO stacks, which want an IMU well above the LiDAR rate.
    """
    import omni.kit.commands
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import IMUSensor
    from pxr import Gf

    usd_context = omni.usd.get_context()
    usd_stage = usd_context.get_stage()

    # Default positions for Go2
    if camera_link_position is None:
        camera_link_position = (0.3, 0.0, 0.35)
    if lidar_l1_position is None:
        lidar_l1_position = (0.3, 0.0, 0.08)
    if lidar_velo_position is None:
        lidar_velo_position = (0.25, 0.0, 0.13)

    sensors = {
        "realsense_depth_camera": None,
        "realsense_rgb_camera": None,
        "robot_front_rgb_camera": None,
        "robot_top_rgb_camera": None,
        "imu": None,
    }

    # --- Cameras ---
    try:
        # Camera link
        ensure_link_xform(
            usd_stage,
            CAMERA_LINK_PRIM,
            translation=camera_link_position,
            rpy_rad=(math.radians(90.0), math.radians(0.0), math.radians(-90.0)),
        )

        realsense_depth_camera = Camera(
            prim_path=REALSENSE_DEPTH_CAMERA_PRIM,
            name="realsense_depth_camera",
            resolution=(480, 270),
        )
        realsense_depth_camera.initialize()

        set_camera_pitch(
            usd_stage,
            REALSENSE_DEPTH_CAMERA_PRIM,
            REALSENSE_PITCH_DEG,
            "realsense_depth_camera",
        )

        realsense_depth_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        realsense_depth_camera.add_distance_to_image_plane_to_frame()

        sensors["realsense_depth_camera"] = realsense_depth_camera
        logger.info("[Sensors] RealSense depth camera initialized with depth enabled")

        realsense_rgb_camera = Camera(
            prim_path=REALSENSE_RGB_CAMERA_PRIM,
            name="realsense_rgb_camera",
            resolution=(424, 240),
        )
        realsense_rgb_camera.initialize()

        set_camera_pitch(
            usd_stage,
            REALSENSE_RGB_CAMERA_PRIM,
            REALSENSE_PITCH_DEG,
            "realsense_rgb_camera",
        )

        realsense_rgb_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        sensors["realsense_rgb_camera"] = realsense_rgb_camera
        logger.info("[Sensors] RealSense RGB camera initialized")

        robot_rgb_camera = Camera(
            prim_path=FRONT_RGB_CAMERA_PRIM,
            name=f"{robot_type}_rgb_camera",
            resolution=FRONT_RGB_RESOLUTION,
        )
        robot_rgb_camera.initialize()

        set_camera_pitch(
            usd_stage,
            FRONT_RGB_CAMERA_PRIM,
            FRONT_RGB_PITCH_DEG,
            f"{robot_type}_rgb_camera",
        )

        robot_rgb_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        set_camera_lens(
            usd_stage,
            FRONT_RGB_CAMERA_PRIM,
            focal_length_mm=FRONT_RGB_FOCAL_LENGTH_MM,
            horizontal_aperture_mm=FRONT_RGB_HORIZONTAL_APERTURE_MM,
            resolution=FRONT_RGB_RESOLUTION,
        )
        sensors["robot_front_rgb_camera"] = robot_rgb_camera
        logger.info(f"[Sensors] {robot_type.upper()} front RGB camera initialized")

        # Add top RGB camera
        robot_top_camera = Camera(
            prim_path=TOP_RGB_CAMERA_PRIM,
            name=f"{robot_type}_top_rgb_camera",
            resolution=(640, 480),
        )
        robot_top_camera.initialize()

        set_camera_pitch(
            usd_stage,
            TOP_RGB_CAMERA_PRIM,
            TOP_RGB_PITCH_DEG,
            f"{robot_type}_top_rgb_camera",
        )

        robot_top_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
        sensors["robot_top_rgb_camera"] = robot_top_camera
        logger.info(
            f"[Sensors] {robot_type.upper()} top RGB camera (fisheye) initialized"
        )
    except Exception as e:
        logger.info(f"[WARN] Camera setup failed: {e}")
        import traceback

        traceback.print_exc()

    # --- IMU ---
    try:
        imu_hz = int(round(physics_hz)) if physics_hz else 200
        imu_sensor = IMUSensor(
            prim_path=IMU_PRIM,
            name="imu_sensor",
            frequency=imu_hz,
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        imu_sensor.initialize()
        sensors["imu"] = imu_sensor
        logger.info(f"[Sensors] IMU initialized at {imu_hz} Hz")
    except Exception as e:
        logger.info(f"[WARN] IMU setup failed: {e}")

    # --- LiDARs ---
    if enable_lidar:
        if lidars_3d:
            for unit in lidars_3d:
                try:
                    _create_rtx_lidar_pointcloud(
                        usd_stage,
                        link_prim=f"{BASE_LINK_PRIM}/{unit.frame_id}",
                        sensor_name=f"lidar_{unit.name}_rtx",
                        position=unit.position,
                        rpy_deg=unit.rpy_deg,
                        topic=unit.topic,
                        frame_id=unit.frame_id,
                        rp_name=f"{unit.name}_lidar_rp",
                        render_hz=render_hz,
                    )
                except Exception as e:
                    logger.info(f"[WARN] {unit.name} LiDAR setup failed: {e}")
                    import traceback

                    traceback.print_exc()
        else:
            try:
                _create_rtx_lidar_pointcloud(
                    usd_stage,
                    link_prim=L1_LINK_PRIM,
                    sensor_name="lidar_l1_rtx",
                    position=lidar_l1_position,
                    rpy_deg=(0.0, 0.0, 0.0),
                    topic="/utlidar/cloud_raw",
                    frame_id="lidar_l1_link",
                    rp_name="l1_lidar_rp",
                    render_hz=render_hz,
                )
            except Exception as e:
                logger.info(f"[WARN] L1 LiDAR setup failed: {e}")
                import traceback

                traceback.print_exc()

    if enable_lidar and enable_2d_lidar:
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
                logger.info(f"[Sensors] 2D LiDAR created at: {lidar_path}")

                # The stock 10 Hz profile accumulates one sweep across several
                # render frames, so the scan tears during rotation (error grows
                # with range). 120 Hz completes >=1 full sweep per frame down to
                # 8.3 ms frame times; reportRate keeps 3200 points/rev.
                scan_hz = 120
                points_per_rev = 3200
                scan_rate_attr = lidar_prim.GetAttribute(
                    "omni:sensor:Core:scanRateBaseHz"
                )
                report_rate_attr = lidar_prim.GetAttribute(
                    "omni:sensor:Core:reportRateBaseHz"
                )
                if scan_rate_attr and report_rate_attr:
                    scan_rate_attr.Set(scan_hz)
                    report_rate_attr.Set(scan_hz * points_per_rev)
                    logger.info(f"[Sensors] 2D LiDAR scan rate set to {scan_hz} Hz")
                else:
                    logger.info("[WARN] 2D LiDAR scanRateBaseHz attr not found")
                velo_rp = rep.create.render_product(
                    lidar_path, resolution=(1, 1), name="velo_lidar_rp"
                )
                scan_writer = rep.writers.get("RtxLidarROS2PublishLaserScan")
                scan_writer.initialize(
                    frameId="laser", nodeNamespace="", topicName="/scan", queueSize=10
                )
                scan_writer.attach([velo_rp])
                logger.info("[Sensors] 2D LiDAR -> /scan")
            else:
                logger.info(f"[WARN] 2D LiDAR creation returned: {result}")
        except Exception as e:
            logger.info(f"[WARN] 2D LiDAR setup failed: {e}")
            import traceback

            traceback.print_exc()

    simulation_app.update()
    return sensors


def setup_static_tfs(
    simulation_app,
    camera_link_pos=(0.3, 0.0, 0.10),
    lidar_l1_pos=(0.3, 0.0, 0.08),
    velodyne_pos=(0.25, 0.0, 0.13),
    lidars_3d=None,
    enable_2d_lidar=True,
    robot_type: str = "go2",
) -> None:
    """Publish static TFs for sensor frames to complete the TF tree.

    Sensor mount positions are passed in (from the robot config) so the TF
    tree matches each robot's actual sensor placement. The camera_link
    quaternion and the velodyne laser offset are fixed sensor-internal
    geometry; each camera's own rotation comes from the *_PITCH_DEG constants,
    the same ones that orient its prim. When ``lidars_3d`` is set, one
    ``base -> frame_id`` transform is published per 3D lidar unit instead of the
    single ``lidar_l1_link``.

    Every frame gets exactly one parent. ``realsense_depth_camera`` used to be
    published under both ``camera_link`` and ``base_link``, which is not a tree;
    the ``base_link`` edge was also the un-pitched one. It is gone - the pose is
    still reachable as base_link -> base -> camera_link -> realsense_depth_camera.
    """
    import omni.graph.core as og
    from isaacsim.core.utils.prims import is_prim_path_valid

    graph_path = "/StaticTFGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Static TF graph already exists")
        return

    cam = [float(v) for v in camera_link_pos]
    l1 = [float(v) for v in lidar_l1_pos]
    velo = [float(v) for v in velodyne_pos]
    cam_quat = [0.5, -0.5, -0.5, 0.5]
    identity = [0.0, 0.0, 0.0, 1.0]
    realsense_quat = quat_xyzw_from_rpy_deg(REALSENSE_PITCH_DEG, 0.0, 0.0)

    if lidars_3d:
        lidar_tfs = [
            (
                "base",
                unit.frame_id,
                [float(v) for v in unit.position],
                quat_xyzw_from_rpy_deg(*unit.rpy_deg),
            )
            for unit in lidars_3d
        ]
    else:
        lidar_tfs = [("base", "lidar_l1_link", l1, identity)]

    if enable_2d_lidar:
        velo_tfs = [
            ("base", "velodyne_base_link", velo, identity),
            ("velodyne_base_link", "laser", [0.0, 0.0, 0.0377], identity),
        ]
    else:
        velo_tfs = []

    # Format: (parent, child, translation, rotation_xyzw)
    static_transforms = [
        ("base_link", "base", [0.0, 0.0, 0.0], identity),
        *lidar_tfs,
        *velo_tfs,
        ("base", "imu_link", [0.0, 0.0, 0.0], identity),
        ("base", "camera_link", cam, cam_quat),
        # One entry per camera prim created in setup_sensors_delayed, each with
        # the same pitch that prim was given and at the prim's own origin. The
        # front and top cameras had no frame at all, so the CameraInfo published
        # for them named a frame nothing could look up; the RealSense pair had a
        # frame but with the pitch dropped.
        (
            "camera_link",
            "realsense_depth_camera",
            [0.0, 0.0, 0.0],
            realsense_quat,
        ),
        (
            "camera_link",
            "realsense_rgb_camera",
            [0.0, 0.0, 0.0],
            realsense_quat,
        ),
        (
            "camera_link",
            f"{robot_type}_rgb_camera",
            [0.0, 0.0, 0.0],
            quat_xyzw_from_rpy_deg(FRONT_RGB_PITCH_DEG, 0.0, 0.0),
        ),
        (
            "camera_link",
            f"{robot_type}_top_rgb_camera",
            [0.0, 0.0, 0.0],
            quat_xyzw_from_rpy_deg(TOP_RGB_PITCH_DEG, 0.0, 0.0),
        ),
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
        f"[ROS2] Static TFs published for {len(static_transforms)} transforms (staticPublisher=True)"
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
                # Initialize with zeros - will be updated each frame
                ("OdomPub.inputs:position", [0.0, 0.0, 0.0]),
                ("OdomPub.inputs:orientation", [0.0, 0.0, 0.0, 1.0]),  # xyzw
                ("OdomPub.inputs:linearVelocity", [0.0, 0.0, 0.0]),
                ("OdomPub.inputs:angularVelocity", [0.0, 0.0, 0.0]),
            ],
        },
    )

    # Get attribute handles for updating each frame
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


def setup_color_camera_publishers(
    sensors, simulation_app, robot_type: str = "go2"
) -> None:
    """Set up ROS2 publishers for color camera images."""
    import omni.replicator.core as rep
    import omni.syntheticdata as syn_data
    import omni.syntheticdata._syntheticdata as sd

    if sensors.get("realsense_rgb_camera"):
        cam = sensors["realsense_rgb_camera"]
        rp = cam.get_render_product_path()
        if rp:
            try:
                # Color image on RealSense topic
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
                w = rep.writers.get(rv + "ROS2PublishImage")
                w.initialize(
                    frameId="realsense_rgb_camera",
                    nodeNamespace="",
                    queueSize=10,
                    topicName="/camera/realsense2_camera_node/color/image_isaac_sim_raw",
                )
                w.attach([rp])
                logger.info(
                    "[ROS2] Color camera -> /camera/realsense2_camera_node/color/image_isaac_sim_raw"
                )

            except Exception as e:
                logger.info(f"[WARN] Color camera publisher setup failed: {e}")

    # Robot RGB Camera
    if sensors.get("robot_front_rgb_camera"):
        cam = sensors["robot_front_rgb_camera"]
        rp = cam.get_render_product_path()
        if rp:
            try:
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
                w = rep.writers.get(rv + "ROS2PublishImage")
                topic_name = f"camera/{robot_type}/image_raw"
                w.initialize(
                    frameId=f"{robot_type}_rgb_camera",
                    nodeNamespace="",
                    queueSize=10,
                    topicName=topic_name,
                )
                w.attach([rp])
                logger.info(f"[ROS2] {robot_type.upper()} RGB camera -> {topic_name}")

            except Exception as e:
                logger.info(
                    f"[WARN] {robot_type.upper()} RGB camera publisher setup failed: {e}"
                )

    # Robot Top RGB Camera
    if sensors.get("robot_top_rgb_camera"):
        cam = sensors["robot_top_rgb_camera"]
        rp = cam.get_render_product_path()
        if rp:
            try:
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
                w = rep.writers.get(rv + "ROS2PublishImage")
                topic_name = "camera/top/image_raw"
                w.initialize(
                    frameId=f"{robot_type}_top_rgb_camera",
                    nodeNamespace="",
                    queueSize=10,
                    topicName=topic_name,
                )
                w.attach([rp])
                logger.info(
                    f"[ROS2] {robot_type.upper()} top RGB camera -> {topic_name}"
                )

            except Exception as e:
                logger.info(
                    f"[WARN] {robot_type.upper()} top RGB camera publisher setup failed: {e}"
                )


def setup_camerainfo_graph(
    simulation_app,
    graph_path: str,
    label: str,
    topic: str,
    frame_id: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> bool:
    """Publish a static sensor_msgs/CameraInfo for one rendered camera.

    The RTX camera writers publish only the image, never a CameraInfo, so every
    stream that a downstream node needs to project has to get one from here.
    The intrinsics are constant, so a single graph re-stamped each tick is
    enough. ``cx``/``cy`` default to the image centre.
    """
    import omni.graph.core as og
    from isaacsim.core.utils.prims import is_prim_path_valid

    if is_prim_path_valid(graph_path):
        logger.info(f"[ROS2] {label} CameraInfo graph already exists")
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
        f"[ROS2] {label} CameraInfo -> {topic} (frame_id={frame_id}, "
        f"{width}x{height}, fx={fx:.1f}, fy={fy:.1f}, cx={cx}, cy={cy})"
    )
    simulation_app.update()
    return True


def setup_color_camerainfo_graph(
    simulation_app,
    topic="/camera/realsense2_camera_node/color/camera_info",
    frame_id="realsense_depth_camera",
    width=424,
    height=240,
    fx=320.0,
    fy=320.0,
    cx=None,
    cy=None,
) -> bool:
    """Publish CameraInfo for the RealSense color camera."""
    return setup_camerainfo_graph(
        simulation_app,
        graph_path="/ColorCameraInfoGraph",
        label="Color",
        topic=topic,
        frame_id=frame_id,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def setup_front_rgb_camerainfo_graph(simulation_app, robot_type: str = "go2") -> bool:
    """Publish CameraInfo for the robot's front RGB camera.

    Same lens constants the prim was pinned with, so /camera/<robot>/camera_info
    always agrees with /camera/<robot>/image_raw. Nothing published this before,
    which left the topic empty for every consumer that needed to project it.
    """
    width, height, fx, fy, cx, cy = camera_intrinsics_from_lens(
        FRONT_RGB_FOCAL_LENGTH_MM,
        FRONT_RGB_HORIZONTAL_APERTURE_MM,
        FRONT_RGB_RESOLUTION,
    )
    return setup_camerainfo_graph(
        simulation_app,
        graph_path="/FrontRgbCameraInfoGraph",
        label=f"{robot_type.upper()} front RGB",
        topic=f"/camera/{robot_type}/camera_info",
        frame_id=f"{robot_type}_rgb_camera",
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def setup_joint_states_publisher(
    simulation_app, articulation_link: str = "base"
) -> None:
    """Publish sensor_msgs/JointState on /joint_states topic."""
    import omni.graph.core as og
    from isaacsim.core.nodes.scripts.utils import set_target_prims
    from isaacsim.core.utils.prims import is_prim_path_valid

    graph_path = "/JointStatesGraph"
    if is_prim_path_valid(graph_path):
        logger.info("[ROS2] Joint states graph already exists")
        return

    ROBOT_ARTICULATION_PATH = f"{GO2_STAGE_PATH}/{articulation_link}"

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
        f"[ROS2] Joint states publisher -> /joint_states (articulation: {ROBOT_ARTICULATION_PATH})"
    )
    simulation_app.update()


_IMU_TICK_CANDIDATES = (
    ("omni.physx.graph.OnPhysicsStep", "step"),
    ("omni.graph.action.OnPhysicsStep", "step"),
    ("omni.graph.action.OnTick", "tick"),
)


def _create_imu_graph(graph_path: str, imu_prim: str, topic: str, frame_id: str) -> str:
    """Build the IMU read/publish graph, ticked per physics step where possible.

    Returns the node type used as the trigger, or "" if none of the candidates
    could be instantiated.
    """
    import omni.graph.core as og
    from isaacsim.core.nodes.scripts.utils import set_target_prims
    from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

    for tick_type, tick_out in _IMU_TICK_CANDIDATES:
        try:
            og.Controller.edit(
                {
                    "graph_path": graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": (
                        og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION
                    ),
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("Tick", tick_type),
                        ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
                        ("Read", "isaacsim.sensors.physics.IsaacReadIMU"),
                        ("Pub", "isaacsim.ros2.bridge.ROS2PublishImu"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        (f"Tick.outputs:{tick_out}", "Read.inputs:execIn"),
                        (f"Tick.outputs:{tick_out}", "Pub.inputs:execIn"),
                        ("Ctx.outputs:context", "Pub.inputs:context"),
                        ("Clock.outputs:simulationTime", "Pub.inputs:timeStamp"),
                        ("Read.outputs:angVel", "Pub.inputs:angularVelocity"),
                        ("Read.outputs:linAcc", "Pub.inputs:linearAcceleration"),
                        ("Read.outputs:orientation", "Pub.inputs:orientation"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("Ctx.inputs:useDomainIDEnvVar", True),
                        ("Read.inputs:readGravity", True),
                        ("Read.inputs:useLatestData", True),
                        ("Pub.inputs:frameId", frame_id),
                        ("Pub.inputs:topicName", topic),
                        ("Pub.inputs:queueSize", 10),
                    ],
                },
            )
        except Exception as e:
            logger.info(f"[WARN] IMU graph trigger {tick_type} unavailable: {e}")
            if is_prim_path_valid(graph_path):
                delete_prim(graph_path)
            continue
        set_target_prims(
            primPath=f"{graph_path}/Read",
            inputName="inputs:imuPrim",
            targetPrimPaths=[imu_prim],
        )
        return tick_type
    logger.info("[WARN] IMU graph could not be created; /imu will be silent")
    return ""


def setup_ros_publishers(
    sensors,
    simulation_app,
    robot_type: str = "go2",
    camera_link_pos: Optional[Tuple[float, float, float]] = None,
    lidar_l1_pos: Optional[Tuple[float, float, float]] = None,
    lidar_velo_pos: Optional[Tuple[float, float, float]] = None,
    lidars_3d: Optional[list] = None,
    enable_2d_lidar: bool = True,
    enable_odom: bool = True,
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

    if not is_prim_path_valid("/ImuGraph"):
        tick_type = _create_imu_graph("/ImuGraph", IMU_PRIM, "/imu", "imu_link")
        rate = "physics rate" if "PhysicsStep" in tick_type else "render rate"
        logger.info(
            f"[ROS2] IMU publisher -> /imu at {rate} via {tick_type or 'nothing'} "
            f"(imu prim: {IMU_PRIM})"
        )
    else:
        logger.info(f"[ROS2] IMU publisher -> /imu (imu prim: {IMU_PRIM})")

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
                    topicName="/camera/realsense2_camera_node/depth/image_rect_isaac_sim_raw",
                )
                w_rs_depth.attach([rp])
                logger.info(
                    "[ROS2] Depth camera -> /camera/realsense2_camera_node/depth/image_rect_isaac_sim_raw"
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
                    logger.info(f"[INFO] Normalized depth writer not available: {de}")

            except Exception as e:
                logger.info(f"[WARN] Camera publisher setup failed: {e}")
                import traceback

                traceback.print_exc()

    # Setup static TFs for sensor frames (mount positions from the robot config)
    setup_static_tfs(
        simulation_app,
        camera_link_pos=camera_link_pos or (0.3, 0.0, 0.10),
        lidar_l1_pos=lidar_l1_pos or (0.3, 0.0, 0.08),
        velodyne_pos=lidar_velo_pos or (0.25, 0.0, 0.13),
        lidars_3d=lidars_3d,
        robot_type=robot_type,
        enable_2d_lidar=enable_2d_lidar,
    )

    # Odom TF publisher (dynamic - updated each frame).
    global odom_tf_trans_attr, odom_tf_rot_attr
    if not enable_odom:
        logger.info("[ROS2] Odom TF disabled (enable_odom=false)")
        simulation_app.update()
        return

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
    fx=320.0,
    fy=320.0,
    cx=None,
    cy=None,
) -> bool:
    """Publish CameraInfo for the RealSense depth camera."""
    return setup_camerainfo_graph(
        simulation_app,
        graph_path="/DepthCameraInfoGraph",
        label="Depth",
        topic=topic,
        frame_id=frame_id,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


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
