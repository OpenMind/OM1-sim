"""Scripted waypoint agents (forklifts, carts) for the environment configs.

Each agent references a USD model and patrols a list of (x, y) waypoints
kinematically. Authored rigid bodies are pinned kinematic so the robot collides
with them instead of pushing them. Configured under ``agents:`` in
configs/environments/.
"""

import logging
import math
import os

logger = logging.getLogger(__name__)

AGENTS_ROOT = "/World/Agents"


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WaypointAgent:
    """One USD model patrolling a waypoint list at fixed speed."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.name = cfg.get("name", "Agent")
        self.waypoints = [(float(x), float(y)) for x, y in cfg.get("waypoints", [])]
        self.speed = float(cfg.get("speed", 1.0))
        self.turn_rate = float(cfg.get("turn_rate", 1.0))
        self.loop = bool(cfg.get("loop", True))
        self.pause = float(cfg.get("pause", 0.0))
        self.arrive_tol = float(cfg.get("arrive_tol", 0.3))
        self.z = float(cfg.get("z", 0.0))
        # Yaw correction when the model's front is not +x.
        self.yaw_offset_deg = float(cfg.get("yaw_offset_deg", 0.0))

        start = cfg.get("start", self.waypoints[0] if self.waypoints else (0.0, 0.0))
        self.pos = [float(start[0]), float(start[1])]
        self.yaw = math.radians(float(cfg.get("start_yaw_deg", 0.0)))
        self.idx = 0
        self.direction = 1
        self.wait_left = 0.0

        self._model_rot = tuple(
            float(v) for v in cfg.get("model_rotate_xyz", (0.0, 0.0, 0.0))
        )
        self._translate_op = None
        self._rotate_op = None

    def spawn(self) -> bool:
        """Reference the model USD and place it at the start pose."""
        import omni.usd
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        if not self.waypoints:
            logger.error("Agent '%s' has no waypoints; skipping", self.name)
            return False

        if self.cfg.get("usd_isaac_relative"):
            assets_root = get_assets_root_path()
            if assets_root is None:
                logger.error("No Isaac assets root; cannot spawn agent '%s'", self.name)
                return False
            usd_path = assets_root + self.cfg["usd_isaac_relative"]
        elif self.cfg.get("usd_relative"):
            usd_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.cfg["usd_relative"]
            )
            if not os.path.isfile(usd_path):
                logger.error("Agent '%s' model not found: %s", self.name, usd_path)
                return False
        else:
            logger.error("Agent '%s' has no model USD configured", self.name)
            return False

        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(f"{AGENTS_ROOT}/{self.name}", "Xform")
        # reference_prim targets a specific prim for files without a defaultPrim.
        if self.cfg.get("reference_prim"):
            prim.GetReferences().AddReference(
                usd_path, Sdf.Path(self.cfg["reference_prim"])
            )
        else:
            prim.GetReferences().AddReference(usd_path)
        if not prim or not prim.IsValid():
            logger.error("Failed to load agent '%s' from %s", self.name, usd_path)
            return False

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        self._translate_op = xform.AddTranslateOp()
        self._rotate_op = xform.AddRotateXYZOp()
        scale = float(self.cfg.get("scale", 1.0))
        if scale != 1.0:
            xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

        # Pin authored rigid bodies kinematic (the agent is animated, not
        # physics-driven); add one if the asset has none (e.g. the forklift),
        # else it would be a moving static collider, which PhysX handles badly.
        has_rigid_body = False
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateKinematicEnabledAttr(True)
                has_rigid_body = True
        if not has_rigid_body:
            body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            body_api.CreateKinematicEnabledAttr(True)

        self._write_pose()
        logger.info(
            "Agent '%s' spawned at (%.2f, %.2f), %d waypoints",
            self.name,
            self.pos[0],
            self.pos[1],
            len(self.waypoints),
        )
        return True

    def _write_pose(self) -> None:
        from pxr import Gf

        rx, ry, rz = self._model_rot
        yaw_deg = math.degrees(self.yaw) + self.yaw_offset_deg
        self._translate_op.Set(Gf.Vec3d(self.pos[0], self.pos[1], self.z))
        self._rotate_op.Set(Gf.Vec3f(rx, ry, rz + yaw_deg))

    def update(self, dt: float) -> None:
        """Advance the patrol by dt: pause, turn toward, or drive."""
        if self._translate_op is None:
            return
        if self.wait_left > 0.0:
            self.wait_left -= dt
            return

        target_x, target_y = self.waypoints[self.idx]
        dx = target_x - self.pos[0]
        dy = target_y - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist < self.arrive_tol:
            self._advance()
            return

        desired = math.atan2(dy, dx)
        err = _wrap_angle(desired - self.yaw)
        max_turn = self.turn_rate * dt
        self.yaw = _wrap_angle(self.yaw + max(-max_turn, min(max_turn, err)))

        # Drive only when roughly facing the waypoint, like a vehicle.
        if abs(err) < math.radians(45.0):
            step = min(self.speed * dt, dist)
            self.pos[0] += step * math.cos(self.yaw)
            self.pos[1] += step * math.sin(self.yaw)

        self._write_pose()

    def _advance(self) -> None:
        self.wait_left = self.pause
        if len(self.waypoints) < 2:
            return
        if self.loop:
            self.idx = (self.idx + 1) % len(self.waypoints)
        else:
            nxt = self.idx + self.direction
            if nxt < 0 or nxt >= len(self.waypoints):
                self.direction *= -1
                nxt = self.idx + self.direction
            self.idx = nxt


class WaypointAgentRunner:
    """Spawns the env config's ``agents`` and steps them each physics tick."""

    def __init__(self, world, agent_cfgs) -> None:
        self._world = world
        self._agents = [WaypointAgent(cfg) for cfg in agent_cfgs or []]
        self.enabled = False

    def setup(self) -> bool:
        """Spawn all agents. Must run before world.reset() (PhysX)."""
        spawned = [agent for agent in self._agents if agent.spawn()]
        failed = len(self._agents) - len(spawned)
        if failed:
            logger.warning("%d agent(s) failed to spawn", failed)
        self._agents = spawned
        self.enabled = bool(spawned)
        return self.enabled

    def register_physics_callback(self) -> None:
        """Drive the agents each physics step (call after world.reset())."""
        if self.enabled:
            self._world.add_physics_callback(
                "waypoint_agents_step", callback_fn=self.update
            )

    def update(self, step_size) -> None:
        """Advance every agent's patrol by one physics step."""
        for agent in self._agents:
            agent.update(step_size)
