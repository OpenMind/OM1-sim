"""Animated pedestrians driven by the omni.anim.people extension.

Spawns skeletal-animated characters that follow a command script and, with
dynamic avoidance, dodge each other and the robot. Each also gets an invisible
kinematic capsule collider so the robot physically collides with it while the
RTX sensors still see the animated mesh. Configured under ``people:`` in
configs/environments/.
"""

import logging
import math
import os
import tempfile

logger = logging.getLogger(__name__)

CHARACTERS_ROOT = "/World/Characters"
COLLIDERS_ROOT = "/World/CharacterColliders"
BIPED_SETUP_NAME = "Biped_Setup"
BIPED_SETUP_USD = "/Isaac/People/Characters/Biped_Setup.usd"

DEFAULT_COLLIDER_RADIUS = 0.3
DEFAULT_COLLIDER_HEIGHT = 1.8


class AnimatedPeopleRunner:
    """Spawns omni.anim.people characters from the env config's ``people``."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg or {}
        self._characters = self._cfg.get("characters") or []
        self._skel_root_paths = {}
        self._colliders = {}
        self.enabled = False

        self._stop_distance = float(self._cfg.get("stop_distance", 1.2))
        self._robot_pose_fn = None
        self._frozen = {}
        self._idled = set()

        self._commands_by_name = {
            c["name"]: [f"{c['name']} {cmd}" for cmd in (c.get("commands") or [])]
            for c in self._characters
            if c.get("name")
        }

    def setup(self) -> bool:
        """
        Enable the extension, configure it, and spawn the characters.
        """
        if not self._characters:
            return False

        import carb
        import omni.kit.app
        import omni.usd
        from isaacsim.core.utils import extensions
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Usd, UsdGeom, UsdSkel

        extensions.enable_extension("omni.anim.people")
        if self._cfg.get("navmesh_enabled", False):
            extensions.enable_extension("omni.anim.navigation.bundle")

        assets_root = get_assets_root_path()
        if assets_root is None:
            logger.error("No Isaac assets root; cannot spawn animated people")
            return False

        behavior_script = (
            omni.kit.app.get_app()
            .get_extension_manager()
            .get_extension_path_by_module("omni.anim.people")
            + "/omni/anim/people/scripts/character_behavior.py"
        )

        settings = carb.settings.get_settings()
        settings.set_string(
            "/exts/omni.anim.people/command_settings/command_file_path",
            self._write_command_file(),
        )
        settings.set_string(
            "/exts/omni.anim.people/command_settings/number_of_loop",
            str(self._cfg.get("loop", "inf")),
        )
        settings.set_bool(
            "/exts/omni.anim.people/navigation_settings/navmesh_enabled",
            bool(self._cfg.get("navmesh_enabled", False)),
        )
        settings.set_bool(
            "/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled",
            bool(self._cfg.get("dynamic_avoidance_enabled", True)),
        )
        settings.set_string(
            "/persistent/exts/omni.anim.people/character_prim_path",
            CHARACTERS_ROOT,
        )
        settings.set_string(
            "/persistent/exts/omni.anim.people/asset_settings/character_assets_path",
            assets_root + "/Isaac/People/Characters",
        )
        settings.set_string(
            "/persistent/exts/omni.anim.people/behavior_script_settings/behavior_script_path",
            behavior_script,
        )

        biped_prim = create_prim(f"{CHARACTERS_ROOT}/{BIPED_SETUP_NAME}", "Xform")
        biped_prim.GetReferences().AddReference(assets_root + BIPED_SETUP_USD)
        UsdGeom.Imageable(biped_prim).MakeInvisible()
        app = omni.kit.app.get_app()

        def _find_anim_graph():
            return next(
                (
                    prim
                    for prim in Usd.PrimRange(biped_prim)
                    if prim.GetTypeName() == "AnimationGraph"
                ),
                None,
            )

        anim_graph = _find_anim_graph()
        for _ in range(600):  # ~10s budget at 60Hz app updates
            if anim_graph is not None:
                break
            app.update()
            anim_graph = _find_anim_graph()

        if anim_graph is None:
            logger.error(
                "No AnimationGraph found in %s after waiting for load", BIPED_SETUP_USD
            )
            return False

        spawned = 0
        for char in self._characters:
            name = char["name"]
            if char.get("usd_isaac_relative"):
                usd_path = assets_root + char["usd_isaac_relative"]
            elif char.get("usd_path"):
                usd_path = char["usd_path"]
            else:
                logger.error("Character '%s' has no model USD; skipping", name)
                continue

            x, y, z = (float(v) for v in char.get("position", (0.0, 0.0, 0.0)))
            yaw = math.radians(float(char.get("yaw_deg", 0.0)))
            prim = create_prim(
                f"{CHARACTERS_ROOT}/{name}",
                "Xform",
                position=(x, y, z),
                orientation=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
            )
            prim.GetReferences().AddReference(usd_path)

            import AnimGraphSchema
            import OmniScriptingSchema

            skel_roots = [p for p in Usd.PrimRange(prim) if p.IsA(UsdSkel.Root)]
            if not skel_roots:
                logger.error("Character '%s' has no SkelRoot; skipping", name)
                continue
            for skel_root in skel_roots:
                graph_api = AnimGraphSchema.AnimationGraphAPI.Apply(skel_root)
                graph_api.GetAnimationGraphRel().SetTargets([anim_graph.GetPrimPath()])
                OmniScriptingSchema.OmniScriptingAPI.Apply(skel_root)
                skel_root.GetAttribute("omni:scripting:scripts").Set([behavior_script])
            self._skel_root_paths[name] = str(skel_roots[0].GetPrimPath())
            self._add_collider(name, str(skel_roots[0].GetPrimPath()), (x, y))
            spawned += 1
            logger.info("Animated character '%s' spawned at (%.2f, %.2f)", name, x, y)

        self.enabled = spawned > 0
        return self.enabled

    def _collider_cfg(self, char: dict) -> dict:
        """Merge the global ``collider`` defaults with a per-character override."""
        cfg = dict(self._cfg.get("collider") or {})
        cfg.update(char.get("collider") or {})
        return cfg

    def _add_collider(self, name: str, skel_path: str, xy) -> None:
        """Create an invisible kinematic capsule that follows this character."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        char = next((c for c in self._characters if c.get("name") == name), {})
        col = self._collider_cfg(char)
        if not col.get("enabled", True):
            return

        radius = float(col.get("radius", DEFAULT_COLLIDER_RADIUS))
        total_height = float(col.get("height", DEFAULT_COLLIDER_HEIGHT))
        cyl_height = max(total_height - 2.0 * radius, 0.0)
        center_z = total_height / 2.0

        stage = omni.usd.get_context().get_stage()
        capsule = UsdGeom.Capsule.Define(stage, f"{COLLIDERS_ROOT}/{name}")
        capsule.CreateAxisAttr("Z")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(cyl_height)
        prim = capsule.GetPrim()

        translate_op = UsdGeom.Xformable(prim).AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), center_z))

        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).MakeInvisible()

        self._colliders[name] = (skel_path, translate_op, center_z)
        logger.info(
            "Collider capsule for '%s' (r=%.2f, h=%.2f)", name, radius, total_height
        )

    def register_dynamic_obstacle(self, prim_path: str) -> bool:
        """Attach the dynamic_obstacle script so characters steer around a prim.

        Call after setup() and before world.reset().
        """
        if not self.enabled:
            return False
        import omni.kit.app
        import omni.usd
        import OmniScriptingSchema

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            logger.warning("Dynamic-obstacle prim not found: %s", prim_path)
            return False
        if not self._cfg.get("dynamic_avoidance_enabled", True):
            logger.warning(
                "dynamic_avoidance_enabled is false; robot will not be avoided"
            )
            return False

        script = (
            omni.kit.app.get_app()
            .get_extension_manager()
            .get_extension_path_by_module("omni.anim.people")
            + "/omni/anim/people/scripts/dynamic_obstacle.py"
        )
        OmniScriptingSchema.OmniScriptingAPI.Apply(prim)
        prim.GetAttribute("omni:scripting:scripts").Set([script])
        logger.info("Robot registered as dynamic obstacle for people: %s", prim_path)
        return True

    def register_physics_callback(self, world, robot_pose_fn=None) -> None:
        """Make each capsule follow its character every physics step.

        ``robot_pose_fn`` is an optional callable returning the robot base's
        live world position (x, y, z); it drives the stop-when-near freeze.
        """
        self._robot_pose_fn = robot_pose_fn
        if self._colliders:
            world.add_physics_callback(
                "character_colliders_step", callback_fn=self._update_colliders
            )

    def _robot_xy(self):
        """Return the robot base's live world (x, y), or None if unavailable."""
        if self._robot_pose_fn is None or self._stop_distance <= 0.0:
            return None
        try:
            pos = self._robot_pose_fn()
        except Exception:  # articulation not ready yet, etc.
            return None
        if pos is None:
            return None
        return (float(pos[0]), float(pos[1]))

    def _update_colliders(self, step_size) -> None:
        import carb
        import omni.anim.graph.core as ag
        from pxr import Gf

        robot_xy = self._robot_xy()

        for name, (skel_path, translate_op, center_z) in self._colliders.items():
            character = ag.get_character(skel_path)
            if character is None:
                continue
            pos = carb.Float3(0.0, 0.0, 0.0)
            rot = carb.Float4(0.0, 0.0, 0.0, 1.0)
            character.get_world_transform(pos, rot)

            if robot_xy is not None:
                pos = self._apply_stop_when_near(name, character, pos, rot, robot_xy)

            translate_op.Set(Gf.Vec3d(float(pos.x), float(pos.y), center_z))

    def _apply_stop_when_near(self, name, character, pos, rot, robot_xy):
        """
        Freeze ``character`` while the robot is near; return the pose to use.
        """
        import carb

        dx = pos.x - robot_xy[0]
        dy = pos.y - robot_xy[1]
        dist_sq = dx * dx + dy * dy
        held = self._frozen.get(name)
        threshold = self._stop_distance * (1.2 if held is not None else 1.0)

        if dist_sq > threshold * threshold:
            if held is not None:
                self._frozen.pop(name, None)
                if name in self._idled:
                    self._resume_walking(name)
                    self._idled.discard(name)
            return pos

        if held is None:
            held = (
                carb.Float3(pos.x, pos.y, pos.z),
                carb.Float4(rot.x, rot.y, rot.z, rot.w),
            )
            self._frozen[name] = held

        character.set_world_transform(held[0], held[1])
        if name not in self._idled and self._request_idle(name):
            self._idled.add(name)

        return held[0]

    def _request_idle(self, name) -> bool:
        """
        Switch a character to a long Idle command so its legs stop moving.
        """
        try:
            from omni.anim.people.scripts.utils import Utils

            if Utils.fetch_target_character_instance_by_name(name) is None:
                return False
            Utils.runtime_inject_command(
                character_name=name,
                command_list=[f"{name} Idle 1000000"],
                force_inject=True,
                set_status=False,
            )
            return True
        except Exception:
            logger.exception("Failed to idle character '%s'", name)
            return False

    def _resume_walking(self, name) -> None:
        """Restore a character's patrol commands after the robot leaves."""
        commands = self._commands_by_name.get(name)
        if not commands:
            return
        try:
            from omni.anim.people.scripts.utils import Utils

            Utils.runtime_inject_command(
                character_name=name,
                command_list=commands,
                force_inject=True,
                set_status=False,
            )
        except Exception:
            logger.exception("Failed to resume character '%s'", name)

    def verify(self, world) -> None:
        """Warn if the anim-graph runtime didn't pick up a character (won't move)."""
        if not self.enabled:
            return
        import omni.anim.graph.core as ag

        for _ in range(3):
            world.step(render=True)
        for name, skel_path in self._skel_root_paths.items():
            if ag.get_character(skel_path) is None:
                logger.warning("Character '%s' not registered; it will not move", name)
            else:
                logger.info("Character '%s' active (%s)", name, skel_path)

    def _write_command_file(self) -> str:
        """Write the per-character command scripts to one file and return it."""
        lines = []
        for char in self._characters:
            for command in char.get("commands", []):
                lines.append(f"{char['name']} {command}")
        path = os.path.join(tempfile.gettempdir(), "om1_people_commands.txt")
        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
        logger.info("People command file: %s (%d commands)", path, len(lines))
        return path
