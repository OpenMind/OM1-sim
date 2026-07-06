"""
AprilTag charging dock for the Isaac Sim runner.
"""

import logging
import os

logger = logging.getLogger(__name__)

DOCK_PATH = "/World/ChargingDock"
IMG_W, IMG_H = 909, 587
TAG_STANDOFF = 0.01


def _texture_path() -> str:
    """Return the apriltags.png path (bundled asset, else the go2 description)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "assets", "tags", "apriltags.png"),
        os.path.join(
            here,
            "..",
            "unitree",
            "go2_gazebo_sim",
            "go2_description",
            "tags",
            "apriltags.png",
        ),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return ""


def add_apriltag_dock(dock_cfg: dict, ground_z: float = 0.0) -> bool:
    """
    Add a charging dock to the current stage, with an AprilTag panel.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    pos = dock_cfg.get("position", (-1.0, -2.25))
    yaw_deg = float(dock_cfg.get("yaw_deg", 90.0))
    tag_size = float(dock_cfg.get("tag_size", 0.2))
    z_offset = float(dock_cfg.get("z_offset", 0.35))
    z = float(pos[2]) if len(pos) > 2 else ground_z

    stage = omni.usd.get_context().get_stage()
    dock_xform = UsdGeom.Xformable(stage.DefinePrim(DOCK_PATH, "Xform"))
    dock_xform.ClearXformOpOrder()
    dock_xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), z))
    dock_xform.AddRotateZOp().Set(yaw_deg)

    half_w = tag_size / 2.0
    half_h = half_w * (IMG_H / IMG_W)
    panel_prim = stage.DefinePrim(f"{DOCK_PATH}/TagPanel", "Mesh")
    panel = UsdGeom.Mesh(panel_prim)
    panel.GetPointsAttr().Set(
        [
            Gf.Vec3f(-half_w, -TAG_STANDOFF, z_offset - half_h),
            Gf.Vec3f(half_w, -TAG_STANDOFF, z_offset - half_h),
            Gf.Vec3f(half_w, -TAG_STANDOFF, z_offset + half_h),
            Gf.Vec3f(-half_w, -TAG_STANDOFF, z_offset + half_h),
        ]
    )
    panel.GetFaceVertexCountsAttr().Set([4])
    panel.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    panel.GetDoubleSidedAttr().Set(True)
    UsdGeom.PrimvarsAPI(panel_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    texture = _texture_path()
    if not texture:
        logger.warning("AprilTag texture not found; dock created without tag image")
    else:
        _bind_tag_material(stage, panel_prim, texture)

    _add_dock_body(stage, dock_cfg, tag_size, half_h, z_offset)

    logger.info(
        "Charging dock at (%.2f, %.2f, %.2f), yaw %.0f, tag %.2f m",
        pos[0],
        pos[1],
        z,
        yaw_deg,
        tag_size,
    )
    return True


def _add_dock_body(
    stage, dock_cfg: dict, tag_size: float, tag_half_h: float, z_offset: float
) -> None:
    """
    Add a simple base and riser behind the AprilTag panel, with a shared material.
    """
    from pxr import UsdShade

    margin = float(dock_cfg.get("panel_margin", 0.05))
    panel_thickness = float(dock_cfg.get("panel_thickness", 0.03))
    base_height = float(dock_cfg.get("base_height", 0.05))
    base_front = float(dock_cfg.get("base_depth", 0.18))
    panel_width = max(float(dock_cfg.get("dock_width", 0.0)), tag_size + 2 * margin)
    base_width = panel_width + 0.05

    panel_z_bottom = base_height
    panel_z_top = max(base_height, z_offset + tag_half_h + margin)

    material = _dock_body_material(
        stage, dock_cfg.get("dock_color", (0.12, 0.12, 0.13))
    )

    riser = _add_box(
        stage,
        f"{DOCK_PATH}/Riser",
        center=(0.0, panel_thickness / 2.0, (panel_z_bottom + panel_z_top) / 2.0),
        size=(panel_width, panel_thickness, panel_z_top - panel_z_bottom),
    )
    UsdShade.MaterialBindingAPI(riser).Bind(material)

    base = _add_box(
        stage,
        f"{DOCK_PATH}/Base",
        center=(0.0, (panel_thickness - base_front) / 2.0, base_height / 2.0),
        size=(base_width, base_front + panel_thickness, base_height),
    )
    UsdShade.MaterialBindingAPI(base).Bind(material)


def _add_box(stage, path: str, center, size):
    """Define a UsdGeom.Cube prim scaled/translated to (center, size)."""
    from pxr import Gf, UsdGeom

    cube_prim = stage.DefinePrim(path, "Cube")
    cube = UsdGeom.Cube(cube_prim)
    cube.CreateSizeAttr(2.0)
    xf = UsdGeom.Xformable(cube_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))
    return cube_prim


def _dock_body_material(stage, color):
    """Plain lit plastic/metal material shared by the dock's base and riser."""
    from pxr import Gf, Sdf, UsdShade

    mat_path = f"{DOCK_PATH}/DockBodyMaterial"
    material = UsdShade.Material(stage.DefinePrim(mat_path, "Material"))
    shader = UsdShade.Shader(stage.DefinePrim(f"{mat_path}/Shader", "Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_tag_material(stage, panel_prim, texture_path: str) -> None:
    """Bind an unlit (emissive) AprilTag texture so detection ignores lighting."""
    from pxr import Gf, Sdf, UsdShade

    mat_path = f"{DOCK_PATH}/AprilTagMaterial"
    material = UsdShade.Material(stage.DefinePrim(mat_path, "Material"))

    shader = UsdShade.Shader(stage.DefinePrim(f"{mat_path}/Shader", "Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")

    st = UsdShade.Shader(stage.DefinePrim(f"{mat_path}/st_reader", "Shader"))
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader(stage.DefinePrim(f"{mat_path}/diffuse_texture", "Shader"))
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st.ConnectableAPI(), "result"
    )
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(panel_prim).Bind(material)
