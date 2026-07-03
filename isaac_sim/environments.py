"""Scene/environment loading for the Isaac Sim runner.

Imported after SimulationApp() init; Isaac Sim / USD modules are imported
lazily inside the functions so this module is safe to preload.
"""

import logging
import os

logger = logging.getLogger(__name__)


def add_environment(env_cfg) -> None:
    """Load the scene for env_cfg.type into the current stage (configs/environments/)."""
    if env_cfg.type == "apartment":
        if not _add_apartment(env_cfg):
            raise RuntimeError("Failed to load Modern Apartment environment")
        _hide_default_ground()
    else:
        _add_warehouse(env_cfg)
    _add_props(env_cfg)
    _scale_scene_prims(env_cfg)
    _remove_scene_prims(env_cfg)


def _add_props(env_cfg) -> None:
    """Spawn the static prop USDs listed under ``props`` in the env config."""
    from isaacsim.core.utils.prims import define_prim
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    assets_root_path = None
    for prop in env_cfg.props or []:
        name = prop.get("name", "Prop")
        usd_isaac_relative = prop.get("usd_isaac_relative")
        if not usd_isaac_relative:
            logger.error("Prop '%s' has no usd_isaac_relative; skipping", name)
            continue
        if assets_root_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                raise RuntimeError("Could not find Isaac Sim assets folder")
        usd_path = assets_root_path + usd_isaac_relative

        dynamic = bool(prop.get("dynamic", False))
        for i, pos in enumerate(prop.get("positions", [])):
            prim = define_prim(f"/World/{name}_{i}", "Xform")
            prim.GetReferences().AddReference(usd_path)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
            if dynamic:
                _make_dynamic(prim, mass=float(prop.get("mass", 1.0)))
            else:
                # Drop joints first, else PhysX rejects joints between the
                # kinematic-pinned (static) links of articulated props.
                _neutralize_articulation(prim)
                for child in Usd.PrimRange(prim):
                    if child.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(child).CreateKinematicEnabledAttr(True)
        logger.info(
            "Spawned %d x %s from %s", len(prop.get("positions", [])), name, usd_path
        )


def _scale_scene_prims(env_cfg) -> None:
    """Enlarge baked scene prims listed under ``scale_prims`` in the env config."""
    specs = getattr(env_cfg, "scale_prims", None)
    if not specs:
        return

    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(env_cfg.stage_path)
    if not root or not root.IsValid():
        logger.warning("scale_prims: stage_path %s not found", env_cfg.stage_path)
        return

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    def _apply(prim, svec):
        xf = UsdGeom.Xformable(prim)
        scale_op = next(
            (
                op
                for op in xf.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )
        if scale_op is not None:
            cur = scale_op.Get() or Gf.Vec3f(1.0, 1.0, 1.0)
            scale_op.Set(Gf.Vec3f(cur[0] * svec[0], cur[1] * svec[1], cur[2] * svec[2]))
        else:
            xf.AddScaleOp().Set(Gf.Vec3f(*svec))

    total = 0
    for spec in specs:
        sc = spec.get("scale", 1.0)
        svec = (
            (float(sc),) * 3
            if isinstance(sc, (int, float))
            else tuple(float(v) for v in sc)
        )
        paths = spec.get("paths")
        if paths:
            for p in paths:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    _apply(prim, svec)
                    total += 1
                else:
                    logger.warning("scale_prims: prim not found: %s", p)
            continue

        region = spec.get("region")  # [xmin, xmax, ymin, ymax]
        match = [m.lower() for m in (spec.get("match") or [])]
        matched = 0
        for prim in root.GetChildren():  # baked clutter sits at the scene root
            nm = prim.GetName().lower()
            if match and not any(m in nm for m in match):
                continue
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
            if region and not (
                region[0] <= cx <= region[1] and region[2] <= cy <= region[3]
            ):
                continue
            _apply(prim, svec)
            matched += 1
        logger.info(
            "scale_prims: scaled %d prims in region=%s match=%s by %s",
            matched,
            region,
            match or "*",
            svec,
        )
        total += matched

    logger.info("scale_prims: %d prims scaled total", total)


def _remove_scene_prims(env_cfg) -> None:
    """Remove baked scene prims listed under ``remove_prims`` in the env config."""
    specs = getattr(env_cfg, "remove_prims", None)
    if not specs:
        return

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(env_cfg.stage_path)
    if not root or not root.IsValid():
        logger.warning("remove_prims: stage_path %s not found", env_cfg.stage_path)
        return

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    total = 0
    for spec in specs:
        paths = spec.get("paths")
        if paths:
            for p in paths:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    prim.SetActive(False)
                    total += 1
                else:
                    logger.warning("remove_prims: prim not found: %s", p)
            continue

        region = spec.get("region")  # [xmin, xmax, ymin, ymax]
        match = [m.lower() for m in (spec.get("match") or [])]
        removed = 0
        for prim in root.GetChildren():  # baked clutter sits at the scene root
            nm = prim.GetName().lower()
            if match and not any(m in nm for m in match):
                continue
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
            if region and not (
                region[0] <= cx <= region[1] and region[2] <= cy <= region[3]
            ):
                continue
            prim.SetActive(False)
            removed += 1
        logger.info(
            "remove_prims: removed %d prims in region=%s match=%s",
            removed,
            region,
            match or "*",
        )
        total += removed

    logger.info("remove_prims: %d prims removed total", total)


def _neutralize_articulation(prim) -> None:
    """
    Deactivate joints / articulation on a prop meant to be static clutter.
    """
    from pxr import Usd, UsdPhysics

    for child in Usd.PrimRange(prim):
        if child.IsA(UsdPhysics.Joint):
            child.SetActive(False)
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            child.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _make_dynamic(prim, mass: float) -> None:
    """Give a prop a dynamic rigid body so the robot can push/grasp it."""
    from pxr import Usd, UsdGeom, UsdPhysics

    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(child).CreateKinematicEnabledAttr(False)
            return

    # No rigid body authored: add one with convex collision on the meshes.
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh) and not child.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(child)
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(
                "convexHull"
            )


def _add_warehouse(env_cfg) -> None:
    """Reference Isaac Sim's bundled warehouse asset."""
    from isaacsim.core.utils.prims import define_prim
    from isaacsim.storage.native import get_assets_root_path

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find Isaac Sim assets folder")
    prim = define_prim(env_cfg.stage_path, "Xform")
    prim.GetReferences().AddReference(assets_root_path + env_cfg.usd_isaac_relative)


def _add_apartment(env_cfg) -> bool:
    """Reference the Modern Apartment USDZ and set up matching ground + lighting."""
    import carb
    import omni.usd
    from isaacsim.core.utils import stage as stage_utils
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    usd_path = env_cfg.usd_local_abs
    if not usd_path or not os.path.isfile(usd_path):
        carb.log_error(f"Apartment USD file not found: {usd_path}")
        return False

    usd_stage = omni.usd.get_context().get_stage()

    stage_utils.add_reference_to_stage(usd_path, env_cfg.stage_path)

    env_prim = usd_stage.GetPrimAtPath(env_cfg.stage_path)
    if not env_prim or not env_prim.IsValid():
        carb.log_error(f"Failed to load apartment environment USD: {usd_path}")
        return False

    tf = env_cfg.transform or {}
    xform = UsdGeom.Xformable(env_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*tf.get("translate", (0.0, 0.0, -0.012))))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*tf.get("rotate_xyz", (90.0, 0.0, 0.0))))
    xform.AddScaleOp().Set(Gf.Vec3f(*tf.get("scale", (0.01, 0.01, 0.01))))

    ground_z = env_cfg.ground_z
    ground_path = "/World/GroundPlane"
    ground_xform_prim = usd_stage.DefinePrim(ground_path, "Xform")
    ground_xform = UsdGeom.Xformable(ground_xform_prim)
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ground_z))
    plane_prim = usd_stage.DefinePrim(f"{ground_path}/CollisionPlane", "Plane")
    plane_geom = UsdGeom.Plane(plane_prim)
    plane_geom.CreateAxisAttr("Z")
    plane_geom.CreateExtentAttr([(-50, -50, 0), (50, 50, 0)])
    UsdPhysics.CollisionAPI.Apply(plane_prim)
    UsdGeom.Imageable(plane_prim).MakeInvisible()

    light = env_cfg.lighting or {}
    dome_light = UsdLux.DomeLight.Define(usd_stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(light.get("dome_intensity", 500.0))

    distant_light = UsdLux.DistantLight.Define(usd_stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(light.get("distant_intensity", 1500.0))
    distant_xform = UsdGeom.Xformable(distant_light.GetPrim())
    distant_xform.AddRotateXYZOp().Set(
        Gf.Vec3f(*light.get("distant_rotate_xyz", (-45.0, 30.0, 0.0)))
    )

    ceiling = light.get("ceiling", {})
    ceiling_z = ground_z + ceiling.get("height_above_ground", 2.5)
    for i, (x, y) in enumerate(ceiling.get("positions", [])):
        sphere_light = UsdLux.SphereLight.Define(usd_stage, f"/World/CeilingLight_{i}")
        sphere_light.CreateIntensityAttr(ceiling.get("intensity", 30000.0))
        sphere_light.CreateRadiusAttr(ceiling.get("radius", 0.15))
        sphere_light.CreateColorAttr(Gf.Vec3f(*ceiling.get("color", (1.0, 0.95, 0.9))))
        light_xform = UsdGeom.Xformable(sphere_light.GetPrim())
        light_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, ceiling_z))

    logger.info("Modern Apartment environment loaded")
    return True


def _hide_default_ground() -> None:
    """Hide any default ground planes World may have spawned."""
    import omni.usd
    from pxr import UsdGeom

    usd_stage = omni.usd.get_context().get_stage()
    ground_paths = [
        "/World/defaultGroundPlane",
        "/World/GroundPlane/GroundPlane",
        "/World/ground",
        "/World/ground/GroundPlane",
    ]
    for path in ground_paths:
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
