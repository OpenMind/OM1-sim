"""
Procedural analog pressure gauges for inspection scenes.
"""

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

GAUGES_ROOT = "/World/Gauges"
SWEEP_DEG = 270.0
NEEDLE_OVERTRAVEL_DEG = 4.0
TEXTURE_PX = 1024
NEEDLE_PERIOD_S = 0.1

DEFAULTS = {
    "diameter": 0.16,
    "depth": 0.05,
    "range": [0.0, 1.6],
    "units": "MPa",
    "divisions": 8,
    "minor": 5,
    "red_from": None,
    "mount": "post",
    "post_base_z": 0.0,
    "post_radius": 0.025,
    "yaw_deg": 0.0,
}


@dataclass
class GaugeSpec:
    """One gauge, with every default applied."""

    id: str
    position: Tuple[float, float, float]
    yaw_deg: float
    diameter: float
    depth: float
    range: Tuple[float, float]
    units: str
    value: float
    divisions: int
    minor: int
    red_from: Optional[float]
    label: str
    mount: str
    post_base_z: float
    post_radius: float
    drift_amplitude: float = 0.0
    drift_period_s: float = 60.0
    drift_phase_deg: float = 0.0

    def value_at(self, t: float) -> float:
        """The reading at simulation time t (seconds)."""
        if not self.drift_amplitude:
            return self.value
        phase = math.radians(self.drift_phase_deg)
        return self.value + self.drift_amplitude * math.sin(
            2.0 * math.pi * t / self.drift_period_s + phase
        )


@dataclass
class PipeSpec:
    """An axis-aligned pipe segment."""

    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    radius: float = 0.06
    color: Tuple[float, float, float] = (0.55, 0.12, 0.10)


@dataclass
class GaugeLayout:
    """Every gauge and pipe in a scene."""

    gauges: List[GaugeSpec] = field(default_factory=list)
    pipes: List[PipeSpec] = field(default_factory=list)


def _vec3(value, what: str) -> Tuple[float, float, float]:
    try:
        x, y, z = value
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be [x, y, z], got {value!r}")
    return (float(x), float(y), float(z))


def parse_layout(cfg: Optional[dict]) -> GaugeLayout:
    """Turn the ``gauges:`` config block into specs, applying defaults."""
    if not cfg:
        return GaugeLayout()
    defaults = dict(DEFAULTS)
    defaults.update(cfg.get("defaults") or {})
    layout = GaugeLayout()
    seen = set()
    for raw in cfg.get("items") or []:
        item = dict(defaults)
        item.update(raw)
        gid = str(item.get("id") or "").strip()
        if not gid:
            raise ValueError(f"gauge without an id: {raw!r}")
        if gid in seen:
            raise ValueError(f"duplicate gauge id {gid!r}")
        seen.add(gid)
        lo, hi = (float(v) for v in item["range"])
        if hi <= lo:
            raise ValueError(f"gauge {gid}: range max must exceed min")
        value = item.get("value")
        drift = item.get("drift") or {}
        red_from = item.get("red_from")
        layout.gauges.append(
            GaugeSpec(
                id=gid,
                position=_vec3(item.get("position"), f"gauge {gid} position"),
                yaw_deg=float(item["yaw_deg"]),
                diameter=float(item["diameter"]),
                depth=float(item["depth"]),
                range=(lo, hi),
                units=str(item["units"]),
                value=float(value) if value is not None else (lo + hi) / 2.0,
                divisions=max(1, int(item["divisions"])),
                minor=max(1, int(item["minor"])),
                red_from=float(red_from) if red_from is not None else None,
                label=str(item.get("label") or gid),
                mount=str(item["mount"]),
                post_base_z=float(item["post_base_z"]),
                post_radius=float(item["post_radius"]),
                drift_amplitude=float(drift.get("amplitude", 0.0)),
                drift_period_s=float(drift.get("period_s", 60.0)),
                drift_phase_deg=float(drift.get("phase_deg", 0.0)),
            )
        )
    for raw in cfg.get("pipes") or []:
        start = _vec3(raw.get("start"), "pipe start")
        end = _vec3(raw.get("end"), "pipe end")
        if sum(1 for a, b in zip(start, end) if abs(a - b) > 1e-6) != 1:
            raise ValueError(f"pipes must run along one axis: {raw!r}")
        layout.pipes.append(
            PipeSpec(
                start=start,
                end=end,
                radius=float(raw.get("radius", 0.06)),
                color=tuple(float(c) for c in raw.get("color", (0.55, 0.12, 0.10))),
            )
        )
    return layout


def value_to_angle_deg(value: float, lo: float, hi: float) -> float:
    """Needle angle, clockwise from 12 o'clock as seen facing the dial."""
    frac = (value - lo) / (hi - lo)
    angle = -SWEEP_DEG / 2.0 + frac * SWEEP_DEG
    limit = SWEEP_DEG / 2.0 + NEEDLE_OVERTRAVEL_DEG
    return max(-limit, min(limit, angle))


def scale_labels(lo: float, hi: float, divisions: int) -> List[str]:
    """Numbers printed at each major tick, without trailing zeros."""
    step = (hi - lo) / divisions
    decimals = 0
    while decimals < 3 and abs(round(step, decimals) - step) > 1e-9:
        decimals += 1
    labels = []
    for i in range(divisions + 1):
        labels.append(f"{round(lo + i * step, decimals) + 0.0:.{decimals}f}")
    return labels


def _font(size: int):
    from PIL import ImageFont

    for name in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text(draw, xy, text, font, fill) -> None:
    try:
        draw.text(xy, text, font=font, fill=fill, anchor="mm")
    except (ValueError, TypeError):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (xy[0] - (right - left) / 2.0, xy[1] - (bottom - top) / 2.0),
            text,
            font=font,
            fill=fill,
        )


def render_dial(spec: GaugeSpec, path: str, size: int = TEXTURE_PX) -> str:
    """Draw the dial face for spec to a PNG at path and return the path."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (246, 245, 238))
    draw = ImageDraw.Draw(img)
    c = size / 2.0
    r = size / 2.0

    def polar(theta_deg, radius):
        t = math.radians(theta_deg)
        return (c + radius * math.sin(t), c - radius * math.cos(t))

    lo, hi = spec.range
    draw.ellipse(
        [c - 0.97 * r, c - 0.97 * r, c + 0.97 * r, c + 0.97 * r],
        outline=(20, 20, 20),
        width=max(2, int(0.025 * size)),
    )

    if spec.red_from is not None and spec.red_from < hi:
        a0 = value_to_angle_deg(max(spec.red_from, lo), lo, hi)
        a1 = value_to_angle_deg(hi, lo, hi)
        band = 0.80 * r
        width = int(0.07 * r)
        draw.arc(
            [c - band, c - band, c + band, c + band],
            start=a0 - 90.0,
            end=a1 - 90.0,
            fill=(200, 30, 30),
            width=width,
        )

    ink = (15, 15, 15)
    major_w = max(2, int(0.012 * size))
    minor_w = max(1, int(0.005 * size))
    steps = spec.divisions * spec.minor
    for i in range(steps + 1):
        theta = -SWEEP_DEG / 2.0 + SWEEP_DEG * i / steps
        major = i % spec.minor == 0
        inner = 0.74 * r if major else 0.80 * r
        draw.line(
            [polar(theta, inner), polar(theta, 0.87 * r)],
            fill=ink,
            width=major_w if major else minor_w,
        )

    num_font = _font(int(0.085 * size))
    for i, text in enumerate(scale_labels(lo, hi, spec.divisions)):
        theta = -SWEEP_DEG / 2.0 + SWEEP_DEG * i / spec.divisions
        _text(draw, polar(theta, 0.60 * r), text, num_font, ink)

    _text(draw, (c, c + 0.42 * r), spec.units, _font(int(0.10 * size)), ink)
    _text(draw, (c, c - 0.30 * r), spec.label, _font(int(0.065 * size)), (60, 60, 60))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def texture_dir() -> str:
    """Directory the generated dial textures are written to."""
    return os.path.join(tempfile.gettempdir(), "om1_sim_gauges")


def _material(
    stage, path, color=(0.5, 0.5, 0.5), texture=None, roughness=0.5, metallic=0.0
):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if texture:
        reader = UsdShade.Shader.Define(stage, f"{path}/St")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, f"{path}/Dial")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _cylinder(stage, path, axis, radius, height, translate, mat, collide=False):
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*translate))
    _bind(cyl.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def _disc_mesh(stage, path, radius, x, mat, segments=72):
    """A disc in the local YZ plane facing +X, UV-mapped to a square texture."""
    from pxr import Gf, Sdf, UsdGeom

    points = [Gf.Vec3f(x, 0.0, 0.0)]
    st = [Gf.Vec2f(0.5, 0.5)]
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        sa, ca = math.sin(a), math.cos(a)
        points.append(Gf.Vec3f(x, radius * sa, radius * ca))
        st.append(Gf.Vec2f(0.5 + 0.5 * sa, 0.5 + 0.5 * ca))
    indices = []
    for i in range(segments):
        indices += [0, 1 + (i + 1) % segments, 1 + i]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * segments)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr([Gf.Vec3f(1.0, 0.0, 0.0)] * len(points))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(st)
    _bind(mesh.GetPrim(), mat)
    return mesh


def _needle_mesh(stage, path, length, tail, width, mat):
    """A tapered needle in the YZ plane, pointing +Z, pivoting at the origin."""
    from pxr import Gf, UsdGeom

    w = width / 2.0
    points = [
        Gf.Vec3f(0.0, w, -tail),
        Gf.Vec3f(0.0, 0.2 * w, length),
        Gf.Vec3f(0.0, -0.2 * w, length),
        Gf.Vec3f(0.0, -w, -tail),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), mat)
    return mesh


class GaugeRunner:
    """Builds the gauges and pipes, moves needles and publishes the truth."""

    def __init__(self, cfg: Optional[dict]) -> None:
        self.layout = parse_layout(cfg)
        self._needle_ops = {}
        self._world = None
        self.ros_graph = "/GaugesRosGraph"
        self._set_count = None
        self._set_seen = None
        self._set_data = None
        self._truth_attr = None
        self._since_needles = 0.0
        self._overrides = {}
        self._cleared = set()

    @property
    def enabled(self) -> bool:
        """True if the scene has any gauges or pipes."""
        return bool(self.layout.gauges or self.layout.pipes)

    def setup(self) -> bool:
        """Add every gauge and pipe to the stage. Must run before world.reset()."""
        if not self.enabled:
            return False
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, GAUGES_ROOT)
        mats = f"{GAUGES_ROOT}/Materials"
        UsdGeom.Scope.Define(stage, mats)
        housing_mat = _material(
            stage, f"{mats}/Housing", (0.62, 0.63, 0.65), roughness=0.3, metallic=0.9
        )
        post_mat = _material(stage, f"{mats}/Post", (0.25, 0.27, 0.30), roughness=0.6)
        needle_mat = _material(stage, f"{mats}/Needle", (0.75, 0.05, 0.05))
        hub_mat = _material(stage, f"{mats}/Hub", (0.05, 0.05, 0.05), roughness=0.3)

        for gauge in self.layout.gauges:
            safe = "".join(ch if ch.isalnum() else "_" for ch in gauge.id)
            root = f"{GAUGES_ROOT}/{safe}"
            xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, root))
            xf.AddTranslateOp().Set(Gf.Vec3d(*gauge.position))
            xf.AddRotateZOp().Set(float(gauge.yaw_deg))

            radius = gauge.diameter / 2.0
            rim = radius * 1.12
            _cylinder(
                stage,
                f"{root}/Housing",
                "X",
                rim,
                gauge.depth,
                (-gauge.depth / 2.0, 0.0, 0.0),
                housing_mat,
                collide=True,
            )
            texture = render_dial(gauge, os.path.join(texture_dir(), f"{safe}.png"))
            face_mat = _material(stage, f"{mats}/Face_{safe}", texture=texture)
            _disc_mesh(stage, f"{root}/Face", radius, 0.002, face_mat)

            pivot = UsdGeom.Xform.Define(stage, f"{root}/NeedlePivot")
            pivot_xf = UsdGeom.Xformable(pivot)
            pivot_xf.AddTranslateOp().Set(Gf.Vec3d(0.006, 0.0, 0.0))
            op = pivot_xf.AddRotateXOp()
            self._needle_ops[gauge.id] = op
            self._set_needle(gauge, gauge.value)
            _needle_mesh(
                stage,
                f"{root}/NeedlePivot/Needle",
                length=0.82 * radius,
                tail=0.18 * radius,
                width=0.07 * radius,
                mat=needle_mat,
            )
            _cylinder(
                stage,
                f"{root}/Hub",
                "X",
                0.07 * radius,
                0.006,
                (0.008, 0.0, 0.0),
                hub_mat,
            )

            if gauge.mount == "post":
                top = gauge.position[2]
                height = top - gauge.post_base_z
                if height > 0.0:
                    _cylinder(
                        stage,
                        f"{root}/Post",
                        "Z",
                        gauge.post_radius,
                        height,
                        (-gauge.depth - gauge.post_radius, 0.0, -height / 2.0),
                        post_mat,
                        collide=True,
                    )

        for i, pipe in enumerate(self.layout.pipes):
            axis = next(
                n for n, a, b in zip("XYZ", pipe.start, pipe.end) if abs(a - b) > 1e-6
            )
            length = max(abs(a - b) for a, b in zip(pipe.start, pipe.end))
            centre = tuple((a + b) / 2.0 for a, b in zip(pipe.start, pipe.end))
            mat = _material(stage, f"{mats}/Pipe_{i}", pipe.color, roughness=0.4)
            _cylinder(
                stage,
                f"{GAUGES_ROOT}/Pipe_{i}",
                axis,
                pipe.radius,
                length,
                centre,
                mat,
                collide=True,
            )

        logger.info(
            "[Gauges] built %d gauges and %d pipes (dial textures in %s)",
            len(self.layout.gauges),
            len(self.layout.pipes),
            texture_dir(),
        )

    def _set_needle(self, gauge: GaugeSpec, value: float) -> None:
        angle = value_to_angle_deg(value, *gauge.range)
        self._needle_ops[gauge.id].Set(float(-angle))

    def _value(self, gauge: GaugeSpec, t: float) -> float:
        if gauge.id in self._overrides:
            return self._overrides[gauge.id]
        return gauge.value_at(t)

    def setup_ros(self, simulation_app) -> None:
        """Build the graph that publishes the truth and takes overrides."""
        if not self.layout.gauges:
            return
        import omni.graph.core as og
        import utils as ros_utils

        try:
            ros_utils.setup_ros2_topic_graph(
                simulation_app,
                self.ros_graph,
                subscribers={"Set": ("/sim/gauges/set", "std_msgs/msg/String")},
                publishers={"Truth": ("/sim/gauges", "std_msgs/msg/String")},
            )
            self._set_count = og.Controller.attribute(
                f"{self.ros_graph}/SetCount.outputs:count"
            )
            self._set_seen = self._set_count.get()
            self._set_data = og.Controller.attribute(
                f"{self.ros_graph}/Set.outputs:data"
            )
            self._truth_attr = og.Controller.attribute(
                f"{self.ros_graph}/Truth.inputs:data"
            )
        except Exception as e:
            self._set_count = self._truth_attr = None
            logger.warning("[Gauges] ROS graph setup failed: %s", e)
            return
        logger.info("[Gauges] truth on /sim/gauges, overrides on /sim/gauges/set")

    def _on_set(self, data: str) -> None:
        try:
            req = json.loads(data)
            gid = str(req["id"])
            if gid not in self._needle_ops:
                raise KeyError(gid)
            if req.get("value") is None:
                if self._overrides.pop(gid, None) is not None:
                    self._cleared.add(gid)
            else:
                self._overrides[gid] = float(req["value"])
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("[Gauges] bad /sim/gauges/set request %r: %s", data, e)

    def register_physics_callback(self, world) -> None:
        """Move needles and refresh the truth each physics step (after reset)."""
        if not self.layout.gauges:
            return
        self._world = world
        world.add_physics_callback("sim_gauges_step", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size: float) -> None:
        """Apply overrides, then update needles and the truth every NEEDLE_PERIOD_S."""
        if self._set_count is not None:
            count = self._set_count.get()
            if count != self._set_seen:
                self._set_seen = count
                self._on_set(self._set_data.get())
        self._since_needles += step_size
        if self._since_needles < NEEDLE_PERIOD_S:
            return
        self._since_needles = 0.0
        t = self._world.current_time if self._world is not None else 0.0
        for gauge in self.layout.gauges:
            if (
                gauge.drift_amplitude
                or gauge.id in self._overrides
                or gauge.id in self._cleared
            ):
                self._set_needle(gauge, self._value(gauge, t))
        self._cleared.clear()
        if self._truth_attr is not None:
            self._truth_attr.set(json.dumps(self.truth(t)))

    def truth(self, t: float) -> dict:
        """The true reading of every gauge at simulation time t."""
        return {
            "sim_time": round(t, 3),
            "gauges": [
                {
                    "id": g.id,
                    "value": round(self._value(g, t), 4),
                    "units": g.units,
                    "range": list(g.range),
                    "position": list(g.position),
                    "yaw_deg": g.yaw_deg,
                }
                for g in self.layout.gauges
            ],
        }
