"""Human pedestrian actor: a velocity-controlled USD model driven over ROS2.

Imported after SimulationApp() init. Independent of the robot runner: it owns
its own model, /cmd_vel_human subscriber, and physics callback on the shared World.
"""

import logging
import math
import os
from typing import Tuple

import utils as ros_utils

logger = logging.getLogger(__name__)

HUMAN_STAGE_PATH = "/World/Human"


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

    Returns
    -------
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

    Returns
    -------
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


class HumanRosRunner:
    """Spawns a human model and steers it from a Twist topic each physics step."""

    def __init__(
        self,
        world,
        cmd_topic: str = "/cmd_vel_human",
        init_pos: tuple = (2.0, 0.0, 0.0),
        init_yaw: float = 0.0,
        vel_max: float = 1.5,
        yaw_rate_max: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        self._world = world
        self._cmd_topic = cmd_topic
        self._init_pos = init_pos
        self._init_yaw = init_yaw
        self._vel_max = vel_max
        self._yaw_rate_max = yaw_rate_max
        self._scale = scale

        self._prim = None
        self._vel_attr = None
        self._yaw_rate_attr = None
        self._pos = [init_pos[0], init_pos[1]]
        self._yaw = init_yaw
        self.enabled = False

    def setup(self) -> bool:
        """Add the human model and /cmd_vel_human subscriber. Must run before
        world.reset() (PhysX). Returns True if the human was added.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        human_usdz_path = os.path.join(script_dir, "assets", "human", "human.usdz")

        # Fallback to project root if not found in standalone assets
        if not os.path.isfile(human_usdz_path):
            project_root = os.path.dirname(script_dir)
            human_usdz_path = os.path.join(project_root, "human.usdz")

        if not os.path.isfile(human_usdz_path):
            logger.warning(
                "Human model not found at %s, skipping human setup", human_usdz_path
            )
            return False

        self._prim = add_human_model(
            human_usdz_path,
            position=self._init_pos,
            rotation_yaw=self._init_yaw,
            scale=self._scale,
        )
        if not self._prim:
            logger.warning("Failed to add human model")
            return False

        self._vel_attr, self._yaw_rate_attr = setup_human_cmd_graph(self._cmd_topic)
        logger.info(
            "Human model initialized at (%s, %s)", self._init_pos[0], self._init_pos[1]
        )
        self.enabled = True
        return True

    def register_physics_callback(self) -> None:
        """Drive the human each physics step (call after world.reset())."""
        if self.enabled:
            self._world.add_physics_callback("human_step", callback_fn=self.update)

    def update(self, step_size) -> None:
        """Read /cmd_vel_human, integrate body-frame velocity, move the prim."""
        if self._prim is None or self._vel_attr is None:
            return
        h_vel = self._vel_attr.get()
        h_yaw_rate = self._yaw_rate_attr.get()
        if h_vel is None or h_yaw_rate is None:
            return

        vel_x = ros_utils.clamp(float(h_vel[0]), -self._vel_max, self._vel_max)
        vel_y = ros_utils.clamp(float(h_vel[1]), -self._vel_max, self._vel_max)
        yaw_rate = ros_utils.clamp(
            float(h_yaw_rate[2]), -self._yaw_rate_max, self._yaw_rate_max
        )

        new_x, new_y, new_yaw = integrate_human_velocity(
            self._pos, self._yaw, vel_x, vel_y, yaw_rate, step_size
        )
        self._pos = [new_x, new_y]
        self._yaw = new_yaw
        update_human_pose(self._prim, new_x, new_y, new_yaw)
