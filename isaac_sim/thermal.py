"""
Simulated thermal camera on the PTZ head.
"""

import fnmatch
import logging
import math
import zlib
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

MODES = ("fixed", "adaptive")

PALETTE_STOPS = {
    "rainbow": [
        (0.0, (16, 0, 48)),
        (0.12, (48, 0, 140)),
        (0.27, (0, 40, 230)),
        (0.42, (0, 170, 255)),
        (0.55, (0, 230, 140)),
        (0.67, (120, 240, 0)),
        (0.78, (255, 230, 0)),
        (0.89, (255, 120, 0)),
        (1.0, (220, 0, 0)),
    ],
    "ironbow": [
        (0.0, (0, 0, 0)),
        (0.2, (40, 0, 110)),
        (0.4, (150, 0, 150)),
        (0.6, (230, 60, 30)),
        (0.8, (255, 170, 0)),
        (1.0, (255, 255, 230)),
    ],
    "whitehot": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "blackhot": [(0.0, (255, 255, 255)), (1.0, (0, 0, 0))],
}

MIN_ADAPTIVE_SPAN_C = 4.0
ADAPTIVE_EDGES_C = np.linspace(-40.0, 200.0, 961)
ADAPTIVE_PLATEAU = 0.03
ADAPTIVE_LINEAR_SHARE = 0.25
STRATIFICATION_REF_Z = 1.5
MAX_DEPTH_M = 1000.0
REFLECTION_BLUR_PX = 12.0

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceField:
    """
    Temperature across one prim's surface, evaluated per pixel.
    """

    temperature_c: float
    emissivity: float = 0.95
    origin: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 0.0, 0.0)
    gradient_c_per_m: float = 0.0
    top_c: float = 0.0
    radius: float = 0.0
    decay_m: float = 0.0
    radial_decay: bool = False
    mottle: float = 0.0
    mottle_m: float = 0.12
    length: float = 0.0
    start_boost_c: float = 0.0
    end_boost_c: float = 0.0
    boost_m: float = 0.15


Heat = Union[float, SurfaceField]


@dataclass
class ThermalConfig:
    """Look of the thermal image (see ``ptz.thermal`` in configs/robots/*.yaml)."""

    topic: str = "thermal/image_raw"
    rate_hz: float = 9.0
    palette: str = "rainbow"
    mode: str = "fixed"
    range_c: Tuple[float, float] = (16.0, 34.0)
    adapt_s: float = 3.0
    sensor_scale: int = 2
    detail_c: float = 4.0
    detail_px: float = 2.0
    edge_falloff: float = 0.5
    blur_px: float = 0.7
    noise_c: float = 0.12
    pattern_noise_c: float = 0.08
    patch_c: float = 1.5
    patch_m: float = 1.5
    stratification_c_per_m: float = 0.3
    glow_gain: float = 0.08
    glow_px: float = 5.0
    plume_gain: float = 0.06
    plume_px: float = 10.0
    glow_above_c: float = 4.0
    knee: Optional[Tuple[float, float]] = None
    reflection: float = 0.5
    temperature_topic: str = "thermal/temperature"


@dataclass
class HeatScene:
    """Temperatures of the prims in a scene (the environment's ``thermal:`` block)."""

    ambient_c: float = 21.0
    spread_c: float = 2.0
    part_spread: float = 0.1
    sources: List[Tuple[str, float, Optional[float]]] = field(default_factory=list)
    exact: Dict[str, Heat] = field(default_factory=dict)
    emissivity: float = 0.95
    stable_range_c: Optional[Tuple[float, float]] = None
    stable_knee: Optional[Tuple[float, float]] = None
    _cache: Dict[str, Heat] = field(default_factory=dict, repr=False)

    def heat(self, prim_path: str) -> Heat:
        """A prim's temperature, or its SurfaceField if it has one."""
        found = self._cache.get(prim_path)
        if found is None:
            found = self._resolve(prim_path)
            self._cache[prim_path] = found
        return found

    def temperature(self, prim_path: str) -> float:
        """Temperature of a prim (a SurfaceField's at its origin)."""
        found = self.heat(prim_path)
        return found.temperature_c if isinstance(found, SurfaceField) else found

    def _resolve(self, prim_path: str) -> Heat:
        probe = prim_path
        while probe:
            if probe in self.exact:
                return self.exact[probe]
            probe = probe.rsplit("/", 1)[0] if "/" in probe else ""
        unit = 2.0 * zlib.crc32(prim_path.encode()) / 0xFFFFFFFF - 1.0
        for pattern, temp, emissivity in self.sources:
            if fnmatch.fnmatchcase(prim_path, pattern) or fnmatch.fnmatchcase(
                prim_path, pattern.rstrip("/") + "/*"
            ):
                temp = temp + (temp - self.ambient_c) * self.part_spread * unit
                if emissivity is None:
                    return temp
                return SurfaceField(temp, emissivity=emissivity)
        return self.ambient_c + unit * self.spread_c


def _positive(entry: dict, key: str, default: float) -> float:
    value = float(entry.get(key, default))
    if value <= 0.0:
        raise ValueError(f"thermal {key} must be positive, got {value}")
    return value


def _non_negative(entry: dict, key: str, default: float) -> float:
    return max(0.0, float(entry.get(key, default)))


def parse_config(entry: Optional[dict]) -> Optional[ThermalConfig]:
    """Parse the optional ``thermal`` block of a robot's ``ptz`` config."""
    if not entry or not entry.get("enabled", True):
        return None
    defaults = ThermalConfig()
    palette = str(entry.get("palette", defaults.palette)).lower()
    if palette not in PALETTE_STOPS:
        raise ValueError(
            f"thermal palette must be one of {sorted(PALETTE_STOPS)}, got {palette!r}"
        )
    mode = str(entry.get("mode", defaults.mode)).lower()
    if mode not in MODES:
        raise ValueError(f"thermal mode must be one of {MODES}, got {mode!r}")
    lo, hi = (float(v) for v in entry.get("range_c", defaults.range_c))
    if hi <= lo:
        raise ValueError(f"thermal range_c max must exceed min, got {[lo, hi]}")
    sensor_scale = int(entry.get("sensor_scale", defaults.sensor_scale))
    if sensor_scale < 1:
        raise ValueError(f"thermal sensor_scale must be at least 1, got {sensor_scale}")
    return ThermalConfig(
        topic=str(entry.get("topic", defaults.topic)).strip("/"),
        rate_hz=_positive(entry, "rate_hz", defaults.rate_hz),
        palette=palette,
        mode=mode,
        range_c=(lo, hi),
        adapt_s=_positive(entry, "adapt_s", defaults.adapt_s),
        sensor_scale=sensor_scale,
        detail_c=_non_negative(entry, "detail_c", defaults.detail_c),
        detail_px=_positive(entry, "detail_px", defaults.detail_px),
        edge_falloff=min(
            1.0, _non_negative(entry, "edge_falloff", defaults.edge_falloff)
        ),
        blur_px=_non_negative(entry, "blur_px", defaults.blur_px),
        noise_c=_non_negative(entry, "noise_c", defaults.noise_c),
        pattern_noise_c=_non_negative(
            entry, "pattern_noise_c", defaults.pattern_noise_c
        ),
        patch_c=_non_negative(entry, "patch_c", defaults.patch_c),
        patch_m=_positive(entry, "patch_m", defaults.patch_m),
        stratification_c_per_m=float(
            entry.get("stratification_c_per_m", defaults.stratification_c_per_m)
        ),
        glow_gain=_non_negative(entry, "glow_gain", defaults.glow_gain),
        glow_px=_positive(entry, "glow_px", defaults.glow_px),
        plume_gain=_non_negative(entry, "plume_gain", defaults.plume_gain),
        plume_px=_positive(entry, "plume_px", defaults.plume_px),
        glow_above_c=_non_negative(entry, "glow_above_c", defaults.glow_above_c),
        knee=_knee(entry.get("knee"), (lo, hi)),
        reflection=min(1.0, _non_negative(entry, "reflection", defaults.reflection)),
        temperature_topic=str(
            entry.get("temperature_topic", defaults.temperature_topic) or ""
        ).strip("/"),
    )


def _range(value, what: str) -> Tuple[float, float]:
    lo, hi = (float(v) for v in value)
    if hi <= lo:
        raise ValueError(f"{what} max must exceed min, got {[lo, hi]}")
    return lo, hi


def _knee(value, range_c: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Validate a (temperature, palette position) knee inside range_c."""
    if value is None:
        return None
    temp, level = (float(v) for v in value)
    if not range_c[0] < temp < range_c[1] or not 0.0 < level < 1.0:
        raise ValueError(
            f"thermal knee must sit inside range_c {list(range_c)} at a palette "
            f"position between 0 and 1, got {[temp, level]}"
        )
    return temp, level


def parse_heat(cfg: Optional[dict]) -> HeatScene:
    """Turn an environment's ``thermal:`` block into a HeatScene."""
    cfg = cfg or {}
    scene = HeatScene(
        ambient_c=float(cfg.get("ambient_c", 21.0)),
        spread_c=abs(float(cfg.get("spread_c", 2.0))),
        part_spread=abs(float(cfg.get("part_spread", 0.1))),
        emissivity=min(1.0, max(0.01, float(cfg.get("emissivity", 0.95)))),
    )
    if cfg.get("stable_range_c") is not None:
        scene.stable_range_c = _range(cfg["stable_range_c"], "stable_range_c")
        scene.stable_knee = _knee(cfg.get("stable_knee"), scene.stable_range_c)
    for raw in cfg.get("sources") or []:
        pattern = str(raw.get("prim") or "").strip()
        if not pattern.startswith("/"):
            raise ValueError(f"thermal source needs an absolute prim path: {raw!r}")
        if "temperature_c" not in raw:
            raise ValueError(f"thermal source {pattern} has no temperature_c")
        emissivity = raw.get("emissivity")
        if emissivity is not None:
            emissivity = min(1.0, max(0.01, float(emissivity)))
        scene.sources.append((pattern, float(raw["temperature_c"]), emissivity))
    return scene


def palette_lut(name: str) -> np.ndarray:
    """256 x 3 uint8 lookup table for a named palette."""
    stops = PALETTE_STOPS[name]
    xs = np.array([p for p, _ in stops])
    rgb = np.array([c for _, c in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(t, xs, rgb[:, ch]) for ch in range(3)], axis=1).astype(
        np.uint8
    )


def surface_image(
    ids: np.ndarray,
    id_to_path: dict,
    scene: HeatScene,
    depth: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    camera_to_world: Optional[np.ndarray] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel true temperature (Celsius) and emissivity from an instance-id
    image.
    """
    ids = np.asarray(ids)
    if stride > 1 and ids.ndim == 2 and min(ids.shape) >= stride:
        h, w = ids.shape
        sub = depth[::stride, ::stride] if depth is not None else None
        temps, emissivity = surface_image(
            ids[::stride, ::stride],
            id_to_path,
            scene,
            sub,
            fx / stride if fx else fx,
            camera_to_world,
        )
        return _grow(temps, stride, h, w), _grow(emissivity, stride, h, w)
    if ids.size == 0:
        zeros = np.zeros(ids.shape, dtype=np.float32)
        return zeros, zeros + scene.emissivity
    heats = {}
    for key, path in id_to_path.items():
        try:
            heats[int(key)] = scene.heat(str(path))
        except (TypeError, ValueError):
            continue
    uniq, inverse = _compact(ids)
    temps = np.full(len(uniq), scene.ambient_c, dtype=np.float32)
    emissivity = np.full(len(uniq), scene.emissivity, dtype=np.float32)
    fields = {}
    for k, i in enumerate(uniq):
        found = heats.get(int(i))
        if isinstance(found, SurfaceField):
            fields[k] = found
            temps[k] = found.temperature_c
            emissivity[k] = found.emissivity
        elif found is not None:
            temps[k] = found
    temp_image = temps[inverse]
    emissivity_image = emissivity[inverse]
    if fields and depth is not None and fx and camera_to_world is not None:
        _apply_fields(
            temp_image,
            emissivity_image,
            inverse,
            fields,
            depth,
            fx,
            camera_to_world,
            scene.ambient_c,
        )
    return temp_image, emissivity_image


def _compact(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The distinct ids on screen, and each pixel's index into them."""
    max_id = int(ids.max())
    if max_id < 1 << 22:
        present = np.zeros(max_id + 1, dtype=bool)
        present[ids] = True
        uniq = np.flatnonzero(present)
        remap = np.zeros(max_id + 1, dtype=np.int64)
        remap[uniq] = np.arange(len(uniq))
        return uniq, remap[ids]
    uniq, inverse = np.unique(ids, return_inverse=True)
    return uniq, inverse.reshape(ids.shape)


def _grow(img: np.ndarray, s: int, h: int, w: int) -> np.ndarray:
    """Repeat each pixel s x s times and crop to h x w."""
    return np.repeat(np.repeat(img, s, 0), s, 1)[:h, :w]


def temperature_image(
    ids: np.ndarray, id_to_path: dict, scene: HeatScene
) -> np.ndarray:
    """Per-pixel temperature (float32, Celsius) from an instance-id image."""
    return surface_image(ids, id_to_path, scene)[0]


def _apply_fields(
    temps: np.ndarray,
    emissivity: np.ndarray,
    index: np.ndarray,
    fields: Dict[int, SurfaceField],
    depth: np.ndarray,
    fx: float,
    camera_to_world: np.ndarray,
    ambient_c: float,
) -> None:
    """Overwrite the pixels of SurfaceField prims with their field values."""
    is_field = np.zeros(int(index.max()) + 1, dtype=bool)
    is_field[list(fields)] = True
    rows, cols = np.nonzero(is_field[index])
    z = depth[rows, cols]
    ok = np.isfinite(z) & (z > 0.0) & (z < MAX_DEPTH_M)
    rows, cols, z = rows[ok], cols[ok], z[ok].astype(np.float32)
    if rows.size == 0:
        return
    points = _pixel_points(rows, cols, z, fx, depth.shape, camera_to_world)
    keys = sorted(fields)
    slot_of = np.zeros(int(index.max()) + 1, dtype=np.int64)
    slot_of[keys] = np.arange(len(keys))
    slot = slot_of[index[rows, cols]]

    def column(get):
        return np.array([get(fields[k]) for k in keys], dtype=np.float32)[slot]

    origin = np.stack([column(lambda f, i=i: f.origin[i]) for i in range(3)], -1)
    axis = np.stack([column(lambda f, i=i: f.axis[i]) for i in range(3)], -1)
    d = points - origin
    s = np.sum(d * axis, axis=-1)
    rise = column(lambda f: f.temperature_c - ambient_c) + s * column(
        lambda f: f.gradient_c_per_m
    )
    radius = column(lambda f: f.radius)
    with np.errstate(divide="ignore", invalid="ignore"):
        height = np.where(radius > 0.0, np.clip(d[:, 2] / radius, -1.0, 1.0), 0.0)
    rise = rise + column(lambda f: f.top_c) * height
    start_boost = column(lambda f: f.start_boost_c)
    end_boost = column(lambda f: f.end_boost_c)
    if start_boost.any() or end_boost.any():
        boost_m = np.maximum(column(lambda f: f.boost_m), 1e-3)
        to_end = column(lambda f: f.length) - s
        rise = (
            rise
            + start_boost * np.exp(-np.maximum(s, 0.0) / boost_m)
            + end_boost * np.exp(-np.maximum(to_end, 0.0) / boost_m)
        )
    decay = column(lambda f: f.decay_m)
    distance = np.where(
        column(lambda f: float(f.radial_decay)) > 0.5,
        np.linalg.norm(d, axis=-1),
        np.maximum(s, 0.0),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        fade = np.where(decay > 0.0, np.exp(-distance / decay), 1.0)
    rise = rise * fade
    temps[rows, cols] = ambient_c + rise

    mottle = column(lambda f: f.mottle)
    if mottle.any():
        scaled = points / column(lambda f: f.mottle_m)[:, None]
        noise = 0.65 * value_noise(scaled) + 0.35 * value_noise(scaled * 3.3 + 7.1)
        base = column(lambda f: f.emissivity)
        emissivity[rows, cols] = np.clip(base + mottle * noise, 0.03, 0.98)


def _gaussian_blur(
    img: np.ndarray, sigma: float, sigma_y: Optional[float] = None
) -> np.ndarray:
    """
    Separable Gaussian blur with edge padding (numpy only).
    """
    for axis, sig in ((0, sigma if sigma_y is None else sigma_y), (1, sigma)):
        if sig <= 0.0:
            continue
        radius = max(1, int(math.ceil(3.0 * sig)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sig) ** 2)
        kernel /= kernel.sum()
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(img, pad, mode="edge")
        out = np.zeros_like(img)
        n = img.shape[axis]
        for i, w in enumerate(kernel):
            out += w * (padded[i : i + n] if axis == 0 else padded[:, i : i + n])
        img = out
    return img


def _plume(excess: np.ndarray, length_px: float) -> np.ndarray:
    """
    Heat carried upward from hot pixels.
    """
    keep = math.exp(-1.0 / max(length_px, 1e-3))
    out = excess.astype(np.float32).copy()
    for row in range(out.shape[0] - 2, -1, -1):
        np.maximum(out[row], keep * out[row + 1], out=out[row])
    # Keep only what rose, and soften it sideways.
    return _gaussian_blur(np.maximum(out - excess, 0.0), 1.0)


def _downsample(img: np.ndarray, s: int) -> np.ndarray:
    """Average s x s blocks (a sensor pixel integrates its patch of scene)."""
    if s == 1:
        return img.astype(np.float32)
    h, w = img.shape[0] // s * s, img.shape[1] // s * s
    blocks = img[:h, :w].astype(np.float32).reshape(h // s, s, w // s, s)
    return blocks.mean(axis=(1, 3))


def _pick(img: np.ndarray, s: int) -> np.ndarray:
    """
    Every s-th pixel, cropped like _downsample.
    """
    h, w = img.shape[0] // s * s, img.shape[1] // s * s
    return img[:h:s, :w:s]


def _upsample(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of a 2-D image to h x w."""
    if img.shape == (h, w):
        return img
    sh, sw = img.shape

    def axis(n_out, n_in):
        pos = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n_in - 1)
        return lo, hi, (pos - lo).astype(np.float32)

    y0, y1, wy = axis(h, sh)
    x0, x1, wx = axis(w, sw)
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bottom = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy)[:, None] + bottom * wy[:, None]


def facing(depth: np.ndarray, fx: float) -> np.ndarray:
    """
    How directly each pixel's surface faces the camera: 1 head-on, 0 edge-on.
    """
    z = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan).astype(np.float32)
    h, w = z.shape
    u = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0)[None, :]
    v = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0)[:, None]
    points = np.stack([u * z / fx, v * z / fx, z], axis=-1)
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    normal = np.cross(du, dv)
    dot = np.abs(np.sum(normal * points, axis=-1))
    norms = np.linalg.norm(normal, axis=-1) * np.linalg.norm(points, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / norms
    return np.clip(np.nan_to_num(cos, nan=1.0), 0.0, 1.0)


def world_points(
    depth: np.ndarray, fx: float, camera_to_world: np.ndarray
) -> np.ndarray:
    """
    World position (HxWx3) of each pixel, NaN where there is nothing.
    """
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < MAX_DEPTH_M)
    z = np.where(valid, depth, np.nan).astype(np.float32)
    h, w = z.shape
    rows, cols = np.mgrid[0:h, 0:w]
    points = _pixel_points(rows, cols, z, fx, (h, w), camera_to_world)
    return points.reshape(h, w, 3)


def _pixel_points(rows, cols, z, fx, shape, camera_to_world) -> np.ndarray:
    """World positions of the given pixels at depths z."""
    h, w = shape
    u = cols.astype(np.float32) - (w - 1) / 2.0
    v = rows.astype(np.float32) - (h - 1) / 2.0
    camera = np.stack([u * z / fx, -v * z / fx, -z, np.ones_like(z)], axis=-1)
    return (camera @ np.asarray(camera_to_world, dtype=np.float32))[..., :3]


def _lattice(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """A fixed pseudo-random value in [-1, 1] per integer lattice point."""
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
    h = (h & 0xFFFFFFFF).astype(np.uint32)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h.astype(np.float32) / np.float32(2147483647.5) - 1.0


def value_noise(points: np.ndarray) -> np.ndarray:
    """Smooth noise in [-1, 1] over (..., 3) points, varying over ~1 unit."""
    base = np.floor(points)
    frac = points - base
    smooth = frac * frac * (3.0 - 2.0 * frac)
    base = base.astype(np.int64)
    out = np.zeros(points.shape[:-1], dtype=np.float32)
    for dx in (0, 1):
        wx = smooth[..., 0] if dx else 1.0 - smooth[..., 0]
        for dy in (0, 1):
            wy = smooth[..., 1] if dy else 1.0 - smooth[..., 1]
            for dz in (0, 1):
                wz = smooth[..., 2] if dz else 1.0 - smooth[..., 2]
                corner = _lattice(
                    base[..., 0] + dx, base[..., 1] + dy, base[..., 2] + dz
                )
                out += corner * wx * wy * wz
    return out


def surface_offset(world: np.ndarray, cfg: ThermalConfig) -> np.ndarray:
    """
    Temperature offset fixed to the world: warm and cool patches on every
    surface, plus warmer air higher up. Zero where there is nothing.
    """
    scaled = np.nan_to_num(world, nan=0.0) / cfg.patch_m
    noise = 0.7 * value_noise(scaled) + 0.3 * value_noise(scaled * 2.7 + 17.3)
    offset = cfg.patch_c * noise + cfg.stratification_c_per_m * (
        world[..., 2] - STRATIFICATION_REF_Z
    )
    return np.nan_to_num(offset, nan=0.0).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Visible-light brightness in 0..1 from an RGB(A) uint8 image."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb[..., :3].astype(np.float32) @ weights) / 255.0


class ThermalImager:
    """Turns temperature maps into thermal camera frames, keeping its state."""

    def __init__(self, cfg: ThermalConfig, ambient_c: float, seed: int = 0) -> None:
        self.cfg = cfg
        self.ambient_c = ambient_c
        self.mode = cfg.mode
        self.lut = palette_lut(cfg.palette)
        self._rng = np.random.default_rng(seed)
        self._pattern_seed = seed + 1
        self._pattern = None
        self._cdf = None
        self.celsius: Optional[np.ndarray] = None

    def set_mode(self, mode: str) -> bool:
        """Switch between "fixed" and "adaptive"; False if mode is unknown."""
        mode = str(mode).strip().lower()
        if mode not in MODES:
            return False
        if mode != self.mode:
            self.mode = mode
            self._cdf = None
        return True

    def render(
        self,
        temps: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        fx: Optional[float] = None,
        dt: Optional[float] = None,
        camera_to_world: Optional[np.ndarray] = None,
        emissivity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        One HxWx3 uint8 frame from a true temperature map (Celsius).
        """
        cfg = self.cfg
        h, w = temps.shape
        s = max(1, min(cfg.sensor_scale, h, w))
        scene = _pick(temps, s).astype(np.float32)
        if depth is None or depth.shape != (h, w) or not fx:
            depth = None
        else:
            depth = np.nan_to_num(
                _pick(depth, s).astype(np.float32), nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M
            )

        if depth is not None and camera_to_world is not None:
            if cfg.patch_c or cfg.stratification_c_per_m:
                coarse = _downsample(depth, 2) if min(depth.shape) >= 2 else depth
                ratio = coarse.shape[1] / w
                world = world_points(coarse, fx * ratio, camera_to_world)
                offset = surface_offset(world, cfg)
                scene = scene + _upsample(offset, *scene.shape)

        eff = None
        if emissivity is not None and emissivity.shape == (h, w):
            eff = _pick(emissivity, s).astype(np.float32)
        if depth is not None and cfg.edge_falloff:
            cos = facing(depth, fx * depth.shape[1] / w)
            weight = 1.0 - cfg.edge_falloff * (1.0 - cos)
            eff = weight if eff is None else eff * weight
        if eff is not None:
            reflected = self._surroundings(scene)
            scene = eff * scene + (1.0 - eff) * reflected

        if cfg.glow_gain or cfg.plume_gain:
            excess = np.maximum(scene - self.ambient_c - cfg.glow_above_c, 0.0)
            if excess.any():
                coarse = _downsample(excess, 2) if min(excess.shape) >= 2 else excess
                heat = np.zeros_like(coarse)
                if cfg.glow_gain:
                    heat += cfg.glow_gain * _gaussian_blur(coarse, cfg.glow_px / 2.0)
                if cfg.plume_gain:
                    heat += cfg.plume_gain * _plume(coarse, cfg.plume_px / 2.0)
                scene = scene + _upsample(heat, *scene.shape)

        scene = _gaussian_blur(scene, cfg.blur_px)
        scene = scene + self._fixed_pattern(scene.shape)
        if cfg.noise_c:
            scene = scene + self._rng.normal(0.0, cfg.noise_c, scene.shape).astype(
                np.float32
            )
        self.celsius = _upsample(scene, h, w).astype(np.float32)

        if rgb is not None and cfg.detail_c and rgb.shape[:2] == (h, w):
            luma = luminance(_pick(rgb, s))
            scene = scene + cfg.detail_c * (luma - _gaussian_blur(luma, cfg.detail_px))

        level = _upsample(self._level(scene, dt), h, w)
        return self.lut[np.clip(level * 255.0 + 0.5, 0, 255).astype(np.uint8)]

    def _surroundings(self, scene: np.ndarray) -> np.ndarray:
        """Temperature a reflective surface mirrors: ambient, warmed by what's near."""
        if not self.cfg.reflection:
            return np.full_like(scene, self.ambient_c)
        coarse = _downsample(scene, 2) if min(scene.shape) >= 2 else scene
        sigma = REFLECTION_BLUR_PX / 2.0
        blurred = _upsample(
            _gaussian_blur(coarse, sigma, sigma_y=2.5 * sigma), *scene.shape
        )
        return self.ambient_c + self.cfg.reflection * (blurred - self.ambient_c)

    def _fixed_pattern(self, shape: Tuple[int, int]) -> np.ndarray:
        """Column stripes and per-pixel offsets that stay put, like a real sensor's."""
        if self._pattern is None or self._pattern.shape != shape:
            rng = np.random.default_rng(self._pattern_seed)
            sigma = self.cfg.pattern_noise_c
            columns = rng.normal(0.0, sigma, (1, shape[1]))
            pixels = rng.normal(0.0, sigma / 2.0, shape)
            self._pattern = (columns + pixels).astype(np.float32)
        return self._pattern

    def _level(self, scene: np.ndarray, dt: Optional[float]) -> np.ndarray:
        """Palette position (0..1) of each pixel."""
        if self.mode == "fixed":
            lo, hi = self.cfg.range_c
            if self.cfg.knee is None:
                return (scene - lo) / (hi - lo)
            knee_t, knee_level = self.cfg.knee
            return np.interp(scene, (lo, knee_t, hi), (0.0, knee_level, 1.0)).astype(
                np.float32
            )

        sample = scene[::2, ::2].ravel()
        edges = ADAPTIVE_EDGES_C
        hist, _ = np.histogram(sample, bins=edges)
        hist = np.minimum(hist, max(1.0, ADAPTIVE_PLATEAU * sample.size))
        cdf = np.concatenate([[0.0], np.cumsum(hist)]) / max(1.0, hist.sum())
        lo, hi = np.percentile(sample, (0.5, 99.5))
        if hi - lo < MIN_ADAPTIVE_SPAN_C:
            mid = (lo + hi) / 2.0
            lo, hi = mid - MIN_ADAPTIVE_SPAN_C / 2.0, mid + MIN_ADAPTIVE_SPAN_C / 2.0
        linear = np.clip((edges - lo) / (hi - lo), 0.0, 1.0)
        target = (1.0 - ADAPTIVE_LINEAR_SHARE) * cdf + ADAPTIVE_LINEAR_SHARE * linear
        if self._cdf is None or dt is None:
            self._cdf = target
        else:
            alpha = 1.0 - math.exp(-max(0.0, dt) / self.cfg.adapt_s)
            self._cdf = self._cdf + alpha * (target - self._cdf)
        return np.interp(scene, edges, self._cdf).astype(np.float32)


class SimThermalCamera:
    """Thermal view of the PTZ render product, published as an rgb8 Image."""

    def __init__(
        self,
        cfg: ThermalConfig,
        scene: HeatScene,
        render_product: str,
        resolution: Tuple[int, int],
        topic: str,
        frame_id: str,
        mode_topic: str,
        mode_state_topic: str,
        focal_px: Callable[[], float],
        camera_to_world: Optional[Callable[[], Optional[np.ndarray]]] = None,
        temperature_topic: Optional[str] = None,
        graph_prefix: str = "/Ptz",
    ) -> None:
        if scene.stable_range_c is not None:
            cfg = replace(cfg, range_c=scene.stable_range_c, knee=scene.stable_knee)
        self.cfg = cfg
        self.scene = scene
        self.temperature_topic = temperature_topic
        self.render_product = render_product
        self.width, self.height = (int(v) for v in resolution)
        self.topic = topic
        self.frame_id = frame_id
        self.mode_topic = mode_topic
        self.mode_state_topic = mode_state_topic
        self.focal_px = focal_px
        self.camera_to_world = camera_to_world
        self.graph = f"{graph_prefix}ThermalGraph"
        self.mode_graph = f"{graph_prefix}ThermalModeGraph"
        self.callback_name = f"{graph_prefix.strip('/')}thermal".lower()
        self.imager = ThermalImager(cfg, scene.ambient_c)
        self._seg = None
        self._rgb = None
        self._depth = None
        self._data_attr = None
        self._celsius_attr = None
        self._impulse_attr = None
        self._mode_cmd = None
        self._mode_state_attr = None
        self._world = None
        self._last_t = None

    def setup_ros(self, simulation_app) -> None:
        """Attach the annotators and build the image and mode graphs."""
        import omni.graph.core as og
        import omni.replicator.core as rep
        import utils as ros_utils

        self._seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
        self._seg.attach([self.render_product])
        if self.cfg.detail_c:
            self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb.attach([self.render_product])
        self._depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._depth.attach([self.render_product])

        nodes = [
            ("Impulse", "omni.graph.action.OnImpulseEvent"),
            ("Clock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Ctx", "isaacsim.ros2.bridge.ROS2Context"),
        ]
        connections = []
        values = [("Ctx.inputs:useDomainIDEnvVar", True)]
        publishers = [("Pub", self.topic, "rgb8")]
        if self.temperature_topic:
            publishers.append(("TempPub", self.temperature_topic, "32FC1"))
        for node, topic, encoding in publishers:
            nodes.append((node, "isaacsim.ros2.bridge.ROS2PublishImage"))
            connections += [
                ("Impulse.outputs:execOut", f"{node}.inputs:execIn"),
                ("Ctx.outputs:context", f"{node}.inputs:context"),
                ("Clock.outputs:simulationTime", f"{node}.inputs:timeStamp"),
            ]
            values += [
                (f"{node}.inputs:topicName", topic),
                (f"{node}.inputs:frameId", self.frame_id),
                (f"{node}.inputs:encoding", encoding),
                (f"{node}.inputs:width", self.width),
                (f"{node}.inputs:height", self.height),
                (f"{node}.inputs:queueSize", 10),
            ]
        og.Controller.edit(
            {
                "graph_path": self.graph,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: nodes,
                og.Controller.Keys.CONNECT: connections,
                og.Controller.Keys.SET_VALUES: values,
            },
        )
        self._data_attr = og.Controller.attribute(f"{self.graph}/Pub.inputs:data")
        if self.temperature_topic:
            self._celsius_attr = og.Controller.attribute(
                f"{self.graph}/TempPub.inputs:data"
            )
        self._impulse_attr = og.Controller.attribute(
            f"{self.graph}/Impulse.state:enableImpulse"
        )

        try:
            ros_utils.setup_ros2_topic_graph(
                simulation_app,
                self.mode_graph,
                subscribers={"Mode": (self.mode_topic, "std_msgs/msg/String")},
                publishers={
                    "ModeState": (self.mode_state_topic, "std_msgs/msg/String")
                },
            )
            count = og.Controller.attribute(
                f"{self.mode_graph}/ModeCount.outputs:count"
            )
            data = og.Controller.attribute(f"{self.mode_graph}/Mode.outputs:data")
            self._mode_cmd = [count, count.get(), data]
            self._mode_state_attr = og.Controller.attribute(
                f"{self.mode_graph}/ModeState.inputs:data"
            )
            self._mode_state_attr.set(self.imager.mode)
        except Exception as e:
            self._mode_cmd = None
            logger.warning("[Thermal] mode switching unavailable: %s", e)

        simulation_app.update()
        logger.info(
            "[Thermal] %s palette, %s mode at %.0f Hz -> %s, %s "
            "(%d heat sources, %d exact)",
            self.cfg.palette,
            self.imager.mode,
            self.cfg.rate_hz,
            self.topic,
            self.temperature_topic or "no temperature topic",
            len(self.scene.sources),
            len(self.scene.exact),
        )

    def register_render_callback(self, world) -> None:
        """Produce a frame after renders, at most rate_hz in sim time."""
        self._world = world
        world.add_render_callback(self.callback_name, self.on_render)

    def on_render(self, _event) -> None:
        """Apply a pending mode switch, then build and publish a frame if due."""
        if self._data_attr is None or self._world is None:
            return
        self._poll_mode()
        t = self._world.current_time
        if self._last_t is not None and t - self._last_t < 1.0 / self.cfg.rate_hz:
            return
        dt = None if self._last_t is None else t - self._last_t
        try:
            frame = self.frame(dt)
        except Exception as e:
            logger.warning("[Thermal] frame failed: %s", e)
            self._last_t = t
            return
        if frame is None:
            return
        self._last_t = t
        self._data_attr.set(frame.reshape(-1))
        if self._celsius_attr is not None and self.imager.celsius is not None:
            celsius = np.ascontiguousarray(self.imager.celsius, dtype=np.float32)
            self._celsius_attr.set(celsius.view(np.uint8).reshape(-1))
        self._impulse_attr.set(True)

    def _poll_mode(self) -> None:
        if self._mode_cmd is None:
            return
        count_attr, seen, data_attr = self._mode_cmd
        count = count_attr.get()
        if count == seen:
            return
        self._mode_cmd[1] = count
        requested = data_attr.get()
        if self.imager.set_mode(requested):
            self._mode_state_attr.set(self.imager.mode)
            logger.info("[Thermal] mode -> %s", self.imager.mode)
        else:
            logger.warning(
                "[Thermal] ignoring unknown mode %r (one of %s)", requested, MODES
            )

    def frame(self, dt: Optional[float] = None) -> Optional[np.ndarray]:
        """The current thermal frame, or None until the annotators have data."""
        seg = self._seg.get_data()
        ids = seg.get("data") if isinstance(seg, dict) else None
        if ids is None or ids.shape[:2] != (self.height, self.width):
            return None
        info = seg.get("info") or {}
        rgb = self._rgb.get_data() if self._rgb is not None else None
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
            rgb = None
        depth = self._depth.get_data() if self._depth is not None else None
        if isinstance(depth, np.ndarray) and depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if not isinstance(depth, np.ndarray) or depth.ndim != 2:
            depth = None
        pose = self.camera_to_world() if self.camera_to_world else None
        fx = self.focal_px()
        temps, emissivity = surface_image(
            ids,
            info.get("idToLabels") or {},
            self.scene,
            depth,
            fx,
            pose,
            stride=self.cfg.sensor_scale,
        )
        return self.imager.render(temps, rgb, depth, fx, dt, pose, emissivity)
