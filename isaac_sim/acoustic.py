import logging
import math

from ptz import HORIZONTAL_APERTURE_MM, focal_length_mm

logger = logging.getLogger(__name__)


class SimAcousticCamera:
    """The acoustic imager's camera prim and image publisher."""

    def __init__(self, cfg, base_link_prim: str) -> None:
        self.cfg = cfg
        self.link_prim = f"{base_link_prim}/acoustic_link"
        self.camera_prim = f"{self.link_prim}/acoustic_camera"
        self.optical_frame = "acoustic_optical_frame"
        self._camera = None

    def setup(self) -> None:
        """Build the camera prim under base_link, looking along the body's x axis."""
        import omni.usd
        import utils as ros_utils
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        ros_utils.ensure_link_xform(
            stage,
            self.link_prim,
            translation=tuple(self.cfg.position),
            rpy_rad=(math.radians(90.0), 0.0, math.radians(-90.0)),
        )
        self._camera = Camera(
            prim_path=self.camera_prim,
            name="acoustic_camera",
            resolution=tuple(self.cfg.resolution),
        )
        self._camera.initialize()
        UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_prim)).ClearXformOpOrder()
        self._camera.set_clipping_range(near_distance=0.05, far_distance=200.0)
        ros_utils.set_camera_lens(
            stage,
            self.camera_prim,
            focal_length_mm=focal_length_mm(self.cfg.hfov_deg, 1.0),
            horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
            resolution=tuple(self.cfg.resolution),
        )
        logger.info(
            "[Acoustic] camera at %s, %dx%d, hfov %.0f deg",
            self.cfg.position,
            self.cfg.resolution[0],
            self.cfg.resolution[1],
            self.cfg.hfov_deg,
        )

    def setup_ros(self) -> None:
        """Publish the camera's RGB image on the configured topic."""
        import omni.replicator.core as rep
        import omni.syntheticdata as syn_data
        import omni.syntheticdata._syntheticdata as sd

        if self._camera is None:
            return
        rp = self._camera.get_render_product_path()
        if not rp:
            logger.warning("[Acoustic] camera has no render product")
            return
        try:
            rv = syn_data.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.Rgb.name
            )
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=self.optical_frame,
                nodeNamespace="",
                queueSize=10,
                topicName=self.cfg.topic,
            )
            writer.attach([rp])
            logger.info("[Acoustic] image -> %s", self.cfg.topic)
        except Exception as e:
            logger.warning("[Acoustic] image publisher setup failed: %s", e)
