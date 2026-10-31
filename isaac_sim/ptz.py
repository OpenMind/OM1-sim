"""
Simulated pan-tilt-zoom head: a visible zoom camera and, optionally, a thermal
camera with its own fixed lens, turning together on one gimbal.
"""

import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

HORIZONTAL_APERTURE_MM = 20.955

OPTICAL_QUAT_XYZW = (-0.5, 0.5, -0.5, 0.5)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _approach(current: float, target: float, max_step: float) -> float:
    """Move current toward target by at most max_step."""
    if abs(target - current) <= max_step:
        return target
    return current + math.copysign(max_step, target - current)


def focal_length_mm(hfov_deg: float, zoom: float) -> float:
    """Focal length that gives hfov_deg at 1x, scaled by the zoom level."""
    base = HORIZONTAL_APERTURE_MM / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return base * zoom


def gimbal_quat_xyzw(yaw_deg: float, pitch_deg: float) -> Tuple[float, ...]:
    """Body-frame rotation of a yaw-then-pitch gimbal (yaw + right, pitch + up)."""
    hy = -math.radians(yaw_deg) / 2.0
    hp = -math.radians(pitch_deg) / 2.0
    cy, sy = math.cos(hy), math.sin(hy)
    cp, sp = math.cos(hp), math.sin(hp)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


class PtzState:
    """Kinematic pan/tilt/zoom model, mirroring the SIYI command semantics."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        self._target: Optional[Tuple[float, float]] = None
        self._rate = (0.0, 0.0)
        self._zoom_target: Optional[float] = None
        self._zoom_dir = 0

    def center(self) -> None:
        """Slew back to straight ahead."""
        self.set_angle(0.0, 0.0)

    def set_angle(self, yaw: float, pitch: float) -> None:
        """Slew to an absolute yaw/pitch, clamped to the mount limits."""
        self._rate = (0.0, 0.0)
        self._target = (
            _clamp(float(yaw), *self.cfg.yaw_limits_deg),
            _clamp(float(pitch), *self.cfg.pitch_limits_deg),
        )

    def rotate(self, yaw_speed: float, pitch_speed: float) -> None:
        """Turn at the given speeds (-100..100); 0, 0 holds position."""
        scale = self.cfg.rotate_max_dps / 100.0
        self._target = None
        self._rate = (
            _clamp(float(yaw_speed), -100.0, 100.0) * scale,
            -_clamp(float(pitch_speed), -100.0, 100.0) * scale,
        )

    def set_zoom(self, level: float) -> None:
        """Zoom to an absolute level, clamped to 1..zoom_max."""
        self._zoom_dir = 0
        self._zoom_target = _clamp(float(level), 1.0, self.cfg.zoom_max)

    def zoom_step(self, direction: int) -> None:
        """Zoom in (1), out (-1) or stop (0)."""
        self._zoom_target = None
        self._zoom_dir = int(_clamp(int(direction), -1, 1))

    def update(self, dt: float) -> bool:
        """Advance the model by dt seconds; return True if anything moved."""
        before = (self.yaw, self.pitch, self.zoom)
        if self._target is not None:
            step = self.cfg.slew_rate_dps * dt
            self.yaw = _approach(self.yaw, self._target[0], step)
            self.pitch = _approach(self.pitch, self._target[1], step)
            if (self.yaw, self.pitch) == self._target:
                self._target = None
        elif self._rate != (0.0, 0.0):
            self.yaw = _clamp(self.yaw + self._rate[0] * dt, *self.cfg.yaw_limits_deg)
            self.pitch = _clamp(
                self.pitch + self._rate[1] * dt, *self.cfg.pitch_limits_deg
            )

        zoom_step = self.cfg.zoom_rate * dt
        if self._zoom_target is not None:
            self.zoom = _approach(self.zoom, self._zoom_target, zoom_step)
            if self.zoom == self._zoom_target:
                self._zoom_target = None
        elif self._zoom_dir:
            self.zoom = _clamp(
                self.zoom + self._zoom_dir * zoom_step, 1.0, self.cfg.zoom_max
            )
        return (self.yaw, self.pitch, self.zoom) != before


class SimPtzCamera:
    """The PTZ camera prims, publishers and ROS command handling."""

    def __init__(
        self, cfg, base_link_prim: str, tf_parent: str = "base", heat=None
    ) -> None:
        self.cfg = cfg
        self.heat = heat
        self.state = PtzState(cfg)
        ns = cfg.ns
        self.gimbal_prim = f"{base_link_prim}/{ns}_gimbal"
        self.camera_link_prim = f"{self.gimbal_prim}/camera_link"
        self.camera_prim = f"{self.camera_link_prim}/{ns}_camera"
        self.tf_parent = tf_parent
        self.gimbal_frame = f"{ns}_gimbal"
        self.optical_frame = f"{ns}_camera_optical_frame"
        self.camerainfo_graph = f"/Ptz_{ns}_CameraInfoGraph"
        self.ros_graph = f"/Ptz_{ns}_RosGraph"
        self.thermal_link_prim = f"{self.gimbal_prim}/thermal_link"
        self.thermal_camera_prim = f"{self.thermal_link_prim}/{ns}_thermal_camera"
        self.thermal_optical_frame = f"{ns}_thermal_optical_frame"
        self.thermal_camerainfo_graph = f"/Ptz_{ns}_ThermalCameraInfoGraph"

        self._camera = None
        self._thermal_camera = None
        self._thermal = None
        self._yaw_op = None
        self._pitch_op = None
        self._focal_attr = None
        self._k_attr = None
        self._p_attr = None
        self._applied_zoom = None
        self._commands = {}
        self._attitude_attrs = None
        self._zoom_level_attr = None
        self._gimbal_tf_attr = None

    def setup(self) -> None:
        """Build the gimbal Xforms and the camera prim under base_link."""
        import omni.usd
        import utils as ros_utils
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        gimbal = stage.DefinePrim(self.gimbal_prim, "Xform")
        xf = UsdGeom.Xformable(gimbal)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.cfg.position))
        self._yaw_op = xf.AddRotateZOp()
        self._pitch_op = xf.AddRotateYOp()
        self._yaw_op.Set(0.0)
        self._pitch_op.Set(0.0)

        ros_utils.ensure_link_xform(
            stage,
            self.camera_link_prim,
            translation=(0.0, 0.0, 0.0),
            rpy_rad=(math.radians(90.0), 0.0, math.radians(-90.0)),
        )

        self._camera = Camera(
            prim_path=self.camera_prim,
            name=f"{self.cfg.ns}_camera",
            resolution=tuple(self.cfg.resolution),
        )
        self._camera.initialize()
        cam_xf = UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_prim))
        cam_xf.ClearXformOpOrder()
        self._camera.set_clipping_range(near_distance=0.05, far_distance=200.0)
        ros_utils.set_camera_lens(
            stage,
            self.camera_prim,
            focal_length_mm=focal_length_mm(self.cfg.hfov_deg, 1.0),
            horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
            resolution=tuple(self.cfg.resolution),
        )
        self._focal_attr = stage.GetPrimAtPath(self.camera_prim).GetAttribute(
            "focalLength"
        )
        self._applied_zoom = 1.0
        logger.info(
            "[PTZ] %s camera at %s, %dx%d, hfov %.0f deg at 1x, zoom up to %.0fx",
            self.cfg.ns,
            self.cfg.position,
            self.cfg.resolution[0],
            self.cfg.resolution[1],
            self.cfg.hfov_deg,
            self.cfg.zoom_max,
        )
        if self.cfg.thermal is not None:
            self._setup_thermal_camera(stage)

    def _thermal_lens(self):
        """The thermal camera's lens: its own, or the visible one's at 1x."""
        from sim_config import ThermalLensConfig

        lens = self.cfg.thermal_lens
        if lens is None:
            lens = ThermalLensConfig(
                resolution=tuple(self.cfg.resolution), hfov_deg=self.cfg.hfov_deg
            )
        return lens

    def _setup_thermal_camera(self, stage) -> None:
        """Build the thermal camera prim on the gimbal, beside the visible lens."""
        import utils as ros_utils
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        lens = self._thermal_lens()
        ros_utils.ensure_link_xform(
            stage,
            self.thermal_link_prim,
            translation=tuple(lens.offset),
            rpy_rad=(math.radians(90.0), 0.0, math.radians(-90.0)),
        )
        self._thermal_camera = Camera(
            prim_path=self.thermal_camera_prim,
            name=f"{self.cfg.ns}_thermal_camera",
            resolution=tuple(lens.resolution),
        )
        self._thermal_camera.initialize()
        cam_xf = UsdGeom.Xformable(stage.GetPrimAtPath(self.thermal_camera_prim))
        cam_xf.ClearXformOpOrder()
        self._thermal_camera.set_clipping_range(near_distance=0.05, far_distance=200.0)
        ros_utils.set_camera_lens(
            stage,
            self.thermal_camera_prim,
            focal_length_mm=focal_length_mm(lens.hfov_deg, 1.0),
            horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
            resolution=tuple(lens.resolution),
        )
        logger.info(
            "[PTZ] %s thermal camera, %dx%d, hfov %.1f deg, %s from the visible lens",
            self.cfg.ns,
            lens.resolution[0],
            lens.resolution[1],
            lens.hfov_deg,
            tuple(lens.offset),
        )

    def setup_ros(self, simulation_app) -> None:
        """Attach the image writer, the CameraInfo graph and the gimbal graph."""
        import omni.graph.core as og
        import omni.replicator.core as rep
        import omni.syntheticdata as syn_data
        import omni.syntheticdata._syntheticdata as sd
        import utils as ros_utils

        if self._camera is None:
            return
        ns = self.cfg.ns
        rp = self._camera.get_render_product_path()
        if rp:
            try:
                rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
                writer = rep.writers.get(rv + "ROS2PublishImage")
                writer.initialize(
                    frameId=self.optical_frame,
                    nodeNamespace="",
                    queueSize=10,
                    topicName=f"/{ns}/image_raw",
                )
                writer.attach([rp])
                logger.info("[PTZ] image -> /%s/image_raw", ns)
            except Exception as e:
                logger.warning("[PTZ] image publisher setup failed: %s", e)
        if self._thermal_camera is not None:
            thermal_rp = self._thermal_camera.get_render_product_path()
            if thermal_rp:
                self._setup_thermal(simulation_app, thermal_rp)

        width, height, fx, fy, cx, cy = ros_utils.camera_intrinsics_from_lens(
            focal_length_mm(self.cfg.hfov_deg, 1.0),
            HORIZONTAL_APERTURE_MM,
            tuple(self.cfg.resolution),
        )
        ros_utils.setup_camerainfo_graph(
            simulation_app,
            graph_path=self.camerainfo_graph,
            label=f"{ns} PTZ",
            topic=f"/{ns}/camera_info",
            frame_id=self.optical_frame,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        try:
            self._k_attr = og.Controller.attribute(
                f"{self.camerainfo_graph}/Pub.inputs:k"
            )
            self._p_attr = og.Controller.attribute(
                f"{self.camerainfo_graph}/Pub.inputs:p"
            )
        except Exception as e:
            logger.warning("[PTZ] CameraInfo will not follow zoom: %s", e)

        try:
            self._setup_gimbal_graph(simulation_app)
        except Exception as e:
            self._commands = {}
            logger.warning("[PTZ] gimbal ROS graph setup failed: %s", e)

    def _setup_thermal(self, simulation_app, render_product: str) -> None:
        """Publish the thermal camera's image, temperatures and camera_info."""
        import thermal
        import utils as ros_utils

        lens = self._thermal_lens()
        ns = self.cfg.ns
        try:
            self._thermal = thermal.SimThermalCamera(
                self.cfg.thermal,
                self.heat if self.heat is not None else thermal.HeatScene(),
                render_product,
                tuple(lens.resolution),
                topic=f"/{ns}/{self.cfg.thermal.topic}",
                frame_id=self.thermal_optical_frame,
                mode_topic=f"/{ns}/thermal/mode",
                mode_state_topic=f"/{ns}/thermal/mode_state",
                focal_px=self._thermal_focal_px,
                camera_to_world=self._thermal_camera_to_world,
                graph_prefix=f"/Ptz_{self.cfg.ns}_",
                temperature_topic=(
                    f"/{self.cfg.ns}/{self.cfg.thermal.temperature_topic}"
                    if self.cfg.thermal.temperature_topic
                    else None
                ),
            )
            self._thermal.setup_ros(simulation_app)
        except Exception as e:
            self._thermal = None
            logger.warning("[PTZ] thermal view setup failed: %s", e)

        width, height, fx, fy, cx, cy = ros_utils.camera_intrinsics_from_lens(
            focal_length_mm(lens.hfov_deg, 1.0),
            HORIZONTAL_APERTURE_MM,
            tuple(lens.resolution),
        )
        ros_utils.setup_camerainfo_graph(
            simulation_app,
            graph_path=self.thermal_camerainfo_graph,
            label=f"{ns} thermal",
            topic=f"/{ns}/thermal/camera_info",
            frame_id=self.thermal_optical_frame,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    def _thermal_focal_px(self) -> float:
        """The thermal lens's horizontal focal length in pixels; it is fixed."""
        lens = self._thermal_lens()
        focal = focal_length_mm(lens.hfov_deg, 1.0)
        return focal / HORIZONTAL_APERTURE_MM * lens.resolution[0]

    def _thermal_camera_to_world(self):
        """The thermal camera's current 4x4 world transform (USD row vectors)."""
        import numpy as np
        import omni.usd
        from pxr import Usd, UsdGeom

        prim = (
            omni.usd.get_context().get_stage().GetPrimAtPath(self.thermal_camera_prim)
        )
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        return np.array(matrix, dtype=np.float64)

    def _setup_gimbal_graph(self, simulation_app) -> None:
        import omni.graph.core as og
        import utils as ros_utils

        g = f"/{self.cfg.ns}/gimbal"
        ros_utils.setup_ros2_topic_graph(
            simulation_app,
            self.ros_graph,
            subscribers={
                "Center": (f"{g}/center", "std_msgs/msg/Bool"),
                "Angle": (f"{g}/angle", "geometry_msgs/msg/Vector3"),
                "Rotate": (f"{g}/rotate", "geometry_msgs/msg/Vector3"),
                "Zoom": (f"{g}/zoom", "std_msgs/msg/Float32"),
                "ZoomStep": (f"{g}/zoom_step", "std_msgs/msg/Int8"),
            },
            publishers={
                "Attitude": (f"{g}/attitude", "geometry_msgs/msg/Vector3"),
                "ZoomLevel": (f"{g}/zoom_level", "std_msgs/msg/Float32"),
            },
            transforms={
                "GimbalTf": {
                    "parent": self.tf_parent,
                    "child": self.gimbal_frame,
                    "translation": self.cfg.position,
                    "rotation": gimbal_quat_xyzw(0.0, 0.0),
                },
                "OpticalTf": {
                    "parent": self.gimbal_frame,
                    "child": self.optical_frame,
                    "translation": (0.0, 0.0, 0.0),
                    "rotation": OPTICAL_QUAT_XYZW,
                    "static": True,
                },
                **self._thermal_tf(),
            },
        )

        def attr(node, name):
            return og.Controller.attribute(f"{self.ros_graph}/{node}.{name}")

        def out(node, field):
            return attr(node, f"outputs:{field}")

        state = self.state
        center = out("Center", "data")
        angle_x, angle_y = out("Angle", "x"), out("Angle", "y")
        rotate_x, rotate_y = out("Rotate", "x"), out("Rotate", "y")
        zoom = out("Zoom", "data")
        zoom_step = out("ZoomStep", "data")
        handlers = {
            "Center": lambda: center.get() and state.center(),
            "Angle": lambda: state.set_angle(angle_x.get(), angle_y.get()),
            "Rotate": lambda: state.rotate(rotate_x.get(), rotate_y.get()),
            "Zoom": lambda: state.set_zoom(zoom.get()),
            "ZoomStep": lambda: state.zoom_step(zoom_step.get()),
        }
        for node, handler in handlers.items():
            count = attr(f"{node}Count", "outputs:count")
            self._commands[node] = [count, count.get(), handler]

        self._attitude_attrs = (
            attr("Attitude", "inputs:x"),
            attr("Attitude", "inputs:y"),
        )
        self._zoom_level_attr = attr("ZoomLevel", "inputs:data")
        self._gimbal_tf_attr = attr("GimbalTf", "inputs:rotation")
        self._publish_state()
        logger.info("[PTZ] serving %s/*", g)

    def _thermal_tf(self) -> dict:
        """The thermal optical frame's static TF, if the head has one."""
        if self.cfg.thermal is None:
            return {}
        return {
            "ThermalOpticalTf": {
                "parent": self.gimbal_frame,
                "child": self.thermal_optical_frame,
                "translation": tuple(self._thermal_lens().offset),
                "rotation": OPTICAL_QUAT_XYZW,
                "static": True,
            }
        }

    def register_physics_callback(self, world) -> None:
        """Step the gimbal and thermal view (call after world.reset())."""
        world.add_physics_callback(
            f"sim_ptz_{self.cfg.ns}_step", callback_fn=self.on_physics_step
        )
        if self._thermal is not None:
            self._thermal.register_render_callback(world)

    def on_physics_step(self, step_size: float) -> None:
        """Apply new commands, advance the gimbal and update its published state."""
        if self._yaw_op is None:
            return
        for entry in self._commands.values():
            count = entry[0].get()
            if count != entry[1]:
                entry[1] = count
                entry[2]()
        if self.state.update(step_size):
            self._yaw_op.Set(float(-self.state.yaw))
            self._pitch_op.Set(float(-self.state.pitch))
            if self.state.zoom != self._applied_zoom:
                self._apply_zoom(self.state.zoom)
            self._publish_state()

    def _apply_zoom(self, zoom: float) -> None:
        import utils as ros_utils

        focal = focal_length_mm(self.cfg.hfov_deg, zoom)
        self._focal_attr.Set(float(focal))
        self._applied_zoom = zoom
        if self._k_attr is None:
            return
        _, _, fx, fy, cx, cy = ros_utils.camera_intrinsics_from_lens(
            focal, HORIZONTAL_APERTURE_MM, tuple(self.cfg.resolution)
        )
        try:
            self._k_attr.set([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])
            self._p_attr.set([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        except Exception as e:
            logger.debug("[PTZ] CameraInfo update failed: %s", e)

    def _publish_state(self) -> None:
        """Set the attitude, zoom level and gimbal TF the graph publishes."""
        if self._attitude_attrs is None:
            return
        self._attitude_attrs[0].set(round(self.state.yaw, 1))
        self._attitude_attrs[1].set(round(self.state.pitch, 1))
        self._zoom_level_attr.set(round(self.state.zoom, 1))
        self._gimbal_tf_attr.set(
            list(gimbal_quat_xyzw(self.state.yaw, self.state.pitch))
        )
