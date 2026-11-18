"""
Procedural process pipework for thermal camera scenes.
"""

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from thermal import SurfaceField

logger = logging.getLogger(__name__)

PIPEWORK_ROOT = "/World/Pipework"

Vec3 = Tuple[float, float, float]

FLANGE_SHARE = 0.9
BOLT_SHARE = 0.8
VALVE_BODY_SHARE = 0.95
SUPPORT_CONTACT_SHARE = 0.8
CLOSED_SHARE = 0.15
TOP_SHARE = 0.04
JACKET_SHARE = 0.1
JACKET_END_SHARE = 0.55
JACKET_SOAK_M = 0.2

FINISHES = {"painted": 0.92, "rusty": 0.8, "galvanized": 0.28, "bare": 0.15}
BOLT_EMISSIVITY = 0.55
STEM_EMISSIVITY = 0.2
VALVE_EMISSIVITY = 0.9
WHEEL_EMISSIVITY = 0.9
JACKETS = {
    "aluminum": (0.15, (0.78, 0.79, 0.80)),
    "painted": (0.9, (0.86, 0.84, 0.78)),
}
INSULATION_KINDS = ("none", *JACKETS)
BAND_EVERY_M = 1.0
BAND_SHARE = 0.3
BAND_EMISSIVITY = 0.35
FINISH_MOTTLE = {"painted": 0.08, "rusty": 0.2, "galvanized": 0.12, "bare": 0.06}
JACKET_MOTTLE = {"aluminum": 0.07, "painted": 0.06}
MATERIAL_MOTTLE = {
    "valve": 0.06,
    "wheel": 0.05,
    "bolt": 0.08,
    "support": 0.1,
    "band": 0.05,
    "trap": 0.1,
}

COLLIDE_BELOW_M = 2.0

DEFAULTS = {
    "radius": 0.08,
    "temperature_c": 60.0,
    "drop_c_per_m": 0.5,
    "color": [0.55, 0.12, 0.10],
    "finish": "painted",
    "insulation": "none",
    "flange_every_m": 2.5,
    "supports": "floor",
    "support_every_m": 3.0,
    "ceiling_z": 8.5,
    "bend": 2.5,
}
SUPPORT_KINDS = ("floor", "ceiling", "none")


TRAP_OUTLET_SHARE = 0.45
TRAP_EMISSIVITY = 0.85


@dataclass
class TrapSpec:
    """A steam trap at distance ``at`` (metres) along its run's polyline."""

    at: float


@dataclass
class ValveSpec:
    """A gate valve at distance ``at`` (metres) along its run's polyline."""

    at: float
    closed: bool = False


@dataclass
class RunSpec:
    """One pipe run, with every default applied."""

    name: str
    path: List[Vec3]
    radius: float
    temperature_c: float
    drop_c_per_m: float
    color: Tuple[float, float, float]
    flange_every_m: float
    supports: str
    support_every_m: float
    ceiling_z: float
    bend: float
    valves: List[ValveSpec] = field(default_factory=list)
    traps: List[TrapSpec] = field(default_factory=list)
    finish: str = "painted"
    insulation: str = "none"
    service: str = ""

    @property
    def bend_radius(self) -> float:
        """Centre-line radius of the elbows."""
        return self.bend * self.radius

    @property
    def trap_length(self) -> float:
        """Pipe length a steam trap takes up, union to union."""
        return 2.6 * self.radius

    @property
    def valve_length(self) -> float:
        """Pipe length a valve takes up, flange face to flange face."""
        return 3.2 * self.radius

    @property
    def jacket_radius(self) -> float:
        """Outer radius of the insulation jacket."""
        return 1.5 * self.radius

    @property
    def emissivity(self) -> float:
        """Emissivity of the pipe's own surface."""
        return FINISHES[self.finish]


@dataclass
class Part:
    """
    One prim of a run.
    """

    name: str
    shape: str
    params: dict
    surface: SurfaceField
    material: str
    collide: bool = False
    along: float = 0.0

    @property
    def temperature_c(self) -> float:
        """Surface temperature at the field's origin."""
        return self.surface.temperature_c

    @property
    def emissivity(self) -> float:
        """Emissivity of the part's surface."""
        return self.surface.emissivity


@dataclass
class Leg:
    """A straight stretch of a run's polyline."""

    start: Vec3
    axis: int
    sign: float
    length: float
    along: float

    def point(self, t: float) -> Vec3:
        """The point t metres along this leg."""
        p = list(self.start)
        p[self.axis] += self.sign * t
        return (p[0], p[1], p[2])

    @property
    def direction(self) -> Vec3:
        """Unit vector the leg runs along."""
        d = [0.0, 0.0, 0.0]
        d[self.axis] = self.sign
        return (d[0], d[1], d[2])

    @property
    def horizontal(self) -> bool:
        """True if the leg runs along X or Y."""
        return self.axis != 2


def _vec3(value, what: str) -> Vec3:
    try:
        x, y, z = value
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be [x, y, z], got {value!r}")
    return (float(x), float(y), float(z))


def legs_of(path: List[Vec3]) -> List[Leg]:
    """Split a polyline into axis-aligned legs, checking it is one."""
    legs = []
    along = 0.0
    for i, (a, b) in enumerate(zip(path, path[1:])):
        moving = [k for k in range(3) if abs(a[k] - b[k]) > 1e-6]
        if len(moving) != 1:
            raise ValueError(f"leg {i} ({a} -> {b}) must run along exactly one axis")
        k = moving[0]
        length = abs(b[k] - a[k])
        legs.append(Leg(a, k, math.copysign(1.0, b[k] - a[k]), length, along))
        along += length
    for i, (p, q) in enumerate(zip(legs, legs[1:])):
        if p.axis == q.axis:
            raise ValueError(f"legs {i} and {i + 1} are collinear; merge or turn")
    return legs


def _choice(value, options, what: str) -> str:
    value = str(value).lower()
    if value not in options:
        raise ValueError(f"{what} must be one of {tuple(options)}, got {value!r}")
    return value


def parse_layout(cfg: Optional[dict]) -> List[RunSpec]:
    """Turn the ``pipework:`` config block into run specs, applying defaults."""
    if not cfg:
        return []
    defaults = dict(DEFAULTS)
    defaults.update(cfg.get("defaults") or {})
    runs = []
    seen = set()
    for raw in cfg.get("runs") or []:
        item = dict(defaults)
        item.update(raw)
        name = str(item.get("name") or "").strip()
        if not name or not all(ch.isalnum() or ch == "_" for ch in name):
            raise ValueError(f"pipe run needs an alphanumeric name: {raw!r}")
        if name in seen:
            raise ValueError(f"duplicate pipe run {name!r}")
        seen.add(name)
        path = [_vec3(p, f"run {name} point") for p in item.get("path") or []]
        if len(path) < 2:
            raise ValueError(f"run {name} needs at least two points")
        run = RunSpec(
            name=name,
            path=path,
            radius=float(item["radius"]),
            temperature_c=float(item["temperature_c"]),
            drop_c_per_m=float(item["drop_c_per_m"]),
            color=tuple(float(c) for c in item["color"]),
            flange_every_m=float(item["flange_every_m"]),
            supports=_choice(item["supports"], SUPPORT_KINDS, f"run {name} supports"),
            support_every_m=float(item["support_every_m"]),
            ceiling_z=float(item["ceiling_z"]),
            bend=float(item["bend"]),
            valves=[
                ValveSpec(at=float(v["at"]), closed=bool(v.get("closed", False)))
                for v in item.get("valves") or []
            ],
            traps=[TrapSpec(at=float(t["at"])) for t in item.get("traps") or []],
            service=str(item.get("service") or ""),
            finish=_choice(item["finish"], FINISHES, f"run {name} finish"),
            insulation=_choice(
                item["insulation"], INSULATION_KINDS, f"run {name} insulation"
            ),
        )
        if run.radius <= 0.0 or run.flange_every_m <= 0.0 or run.bend < 1.0:
            raise ValueError(f"run {name}: radius, flange_every_m and bend look wrong")
        _check_run(run)
        runs.append(run)
    return runs


def _check_run(run: RunSpec) -> None:
    legs = legs_of(run.path)
    r_bend = run.bend_radius
    for i, leg in enumerate(legs):
        trim = r_bend * ((i > 0) + (i < len(legs) - 1))
        if leg.length < trim + 0.05:
            raise ValueError(
                f"run {run.name}: leg {i} is {leg.length:.2f} m, too short for "
                f"its elbows ({trim:.2f} m)"
            )
    fittings = [("valve", v.at, run.valve_length) for v in run.valves] + [
        ("trap", t.at, run.trap_length) for t in run.traps
    ]
    for kind, at, length in fittings:
        leg, t = _locate(legs, at)
        if leg is None:
            raise ValueError(f"run {run.name}: {kind} at {at} m is off the run")
        if not leg.horizontal:
            raise ValueError(f"run {run.name}: {kind} at {at} m is on a vertical leg")
        i = legs.index(leg)
        lo = (r_bend if i > 0 else 0.0) + length / 2.0
        hi = leg.length - (r_bend if i < len(legs) - 1 else 0.0) - length / 2.0
        if not lo <= t <= hi:
            raise ValueError(
                f"run {run.name}: {kind} at {at} m is too close to a bend or end"
            )
    spans = sorted((at - length / 2.0, at + length / 2.0) for _, at, length in fittings)
    for (_, a_hi), (b_lo, _) in zip(spans, spans[1:]):
        if b_lo < a_hi + 0.3:
            raise ValueError(f"run {run.name}: fittings at {a_hi:.2f} m overlap")


def _locate(legs: List[Leg], along: float) -> Tuple[Optional[Leg], float]:
    for leg in legs:
        if leg.along <= along <= leg.along + leg.length:
            return leg, along - leg.along
    return None, 0.0


class _Thermal:
    """Fluid temperature along a run."""

    def __init__(self, run: RunSpec, ambient_c: float) -> None:
        self.run = run
        self.ambient_c = ambient_c
        self.closed_at = sorted(v.at for v in run.valves if v.closed)
        self.trap_at = sorted(t.at for t in run.traps)

    def rise(self, along: float) -> float:
        """Fluid temperature above ambient at a distance along the run."""
        temp = self.run.temperature_c - self.run.drop_c_per_m * along
        rise = temp - self.ambient_c
        # Cooling stops at ambient, from either side.
        if (self.run.temperature_c - self.ambient_c) * rise < 0.0:
            rise = 0.0
        if self.closed_at and along > self.closed_at[0]:
            rise *= CLOSED_SHARE
        for at in self.trap_at:
            if along > at:
                rise *= TRAP_OUTLET_SHARE
        return rise

    def at(self, along: float, share: float = 1.0) -> float:
        return self.ambient_c + share * self.rise(along)


def _axis_name(k: int) -> str:
    return "XYZ"[k]


def _fade(surface: SurfaceField, distance: float, ambient_c: float) -> float:
    """Temperature a decaying field reaches at a distance from its origin."""
    rise = surface.temperature_c - ambient_c
    return ambient_c + rise * math.exp(-distance / surface.decay_m)


def _section_surface(
    run: RunSpec,
    leg: Leg,
    lo: float,
    hi: float,
    heat: _Thermal,
    share: float,
    emissivity: float,
) -> SurfaceField:
    """A pipe (or jacket) section from lo to hi along a leg."""
    start = heat.at(leg.along + lo, share)
    end = heat.at(leg.along + hi, share)
    rise = start - heat.ambient_c
    return SurfaceField(
        start,
        emissivity=emissivity,
        origin=leg.point(lo),
        axis=leg.direction,
        gradient_c_per_m=(end - start) / max(hi - lo, 1e-6),
        top_c=TOP_SHARE * rise if leg.horizontal else 0.0,
        radius=run.radius,
    )


def _jacket_surface(
    run: RunSpec,
    leg: Leg,
    lo: float,
    hi: float,
    heat: _Thermal,
    emissivity: float,
    soak_start: bool,
    soak_end: bool,
) -> SurfaceField:
    """An insulation jacket section, warming towards any bare fitting it meets."""
    surface = _section_surface(run, leg, lo, hi, heat, JACKET_SHARE, emissivity)
    extra = JACKET_END_SHARE - JACKET_SHARE

    def boost(t):
        return extra * heat.rise(leg.along + t)

    return replace(
        surface,
        length=hi - lo,
        start_boost_c=boost(lo) if soak_start else 0.0,
        end_boost_c=boost(hi) if soak_end else 0.0,
        boost_m=JACKET_SOAK_M,
    )


def build_run(run: RunSpec, ambient_c: float) -> List[Part]:
    """
    Every part of a run, positioned in world space, with its surface field.
    """
    legs = legs_of(run.path)
    heat = _Thermal(run, ambient_c)
    r, r_bend = run.radius, run.bend_radius
    insulated = run.insulation != "none"
    jacket_emissivity = JACKETS[run.insulation][0] if insulated else 0.0
    parts: List[Part] = []

    for j in range(1, len(legs)):
        before, after = legs[j - 1], legs[j]
        corner = after.start
        a = before.direction
        b = after.direction
        centre = tuple(corner[k] - a[k] * r_bend + b[k] * r_bend for k in range(3))
        share = JACKET_SHARE if insulated else 1.0
        parts.append(
            Part(
                f"Elbow{j}",
                "elbow",
                {
                    "centre": centre,
                    "u": tuple(-x for x in b),
                    "v": tuple(a),
                    "bend_radius": r_bend,
                    "radius": run.jacket_radius if insulated else r,
                },
                SurfaceField(
                    heat.at(after.along, share),
                    emissivity=jacket_emissivity if insulated else run.emissivity,
                ),
                "jacket" if insulated else "pipe",
            )
        )

    support_n = 0
    for i, leg in enumerate(legs):
        t0 = r_bend if i > 0 else 0.0
        t1 = leg.length - (r_bend if i < len(legs) - 1 else 0.0)
        valves = sorted(
            (v.at - leg.along, v)
            for v in run.valves
            if leg.along <= v.at <= leg.along + leg.length
        )
        traps = sorted(
            t.at - leg.along
            for t in run.traps
            if leg.along <= t.at <= leg.along + leg.length
        )
        blocked = [
            (t - run.valve_length / 2.0, t + run.valve_length / 2.0) for t, _ in valves
        ] + [(t - run.trap_length / 2.0, t + run.trap_length / 2.0) for t in traps]

        joints = []
        t = t0 + run.flange_every_m
        while t < t1 - 0.3:
            if not any(lo - 0.3 < t < hi + 0.3 for lo, hi in blocked):
                joints.append(t)
            t += run.flange_every_m

        breaks = set(joints) | {x for span in blocked for x in span}
        cuts = sorted({t0, t1, *joints, *(x for span in blocked for x in span)})
        k = 0
        for lo, hi in zip(cuts, cuts[1:]):
            mid = (lo + hi) / 2.0
            if hi - lo < 1e-3 or any(a <= mid <= b for a, b in blocked):
                continue
            parts.append(
                Part(
                    f"Leg{i}_Pipe{k}",
                    "cylinder",
                    {
                        "axis": _axis_name(leg.axis),
                        "radius": r,
                        "height": hi - lo,
                        "centre": leg.point(mid),
                    },
                    _section_surface(run, leg, lo, hi, heat, 1.0, run.emissivity),
                    "pipe",
                    collide=_low(leg, lo, hi, r),
                )
            )
            if insulated:
                j_lo = lo + (r if lo in breaks else 0.0)
                j_hi = hi - (r if hi in breaks else 0.0)
                if j_hi - j_lo > 0.05:
                    parts.append(
                        Part(
                            f"Leg{i}_Jacket{k}",
                            "cylinder",
                            {
                                "axis": _axis_name(leg.axis),
                                "radius": run.jacket_radius,
                                "height": j_hi - j_lo,
                                "centre": leg.point((j_lo + j_hi) / 2.0),
                            },
                            _jacket_surface(
                                run,
                                leg,
                                j_lo,
                                j_hi,
                                heat,
                                jacket_emissivity,
                                soak_start=lo in breaks,
                                soak_end=hi in breaks,
                            ),
                            "jacket",
                            collide=_low(leg, j_lo, j_hi, run.jacket_radius),
                        )
                    )
                    parts += _bands(f"Leg{i}_Jacket{k}", leg, j_lo, j_hi, run, heat)
            k += 1

        for n, t in enumerate(joints):
            parts += _flange(
                f"Leg{i}_Flange{n}",
                leg,
                t,
                r,
                SurfaceField(
                    heat.at(leg.along + t, FLANGE_SHARE), emissivity=run.emissivity
                ),
            )
            for m in range(6):
                parts.append(_bolt(f"Leg{i}_Flange{n}_Bolt{m}", leg, t, r, m, heat))

        for n, (t, v) in enumerate(valves):
            parts += _valve(f"Valve{i}_{n}", leg, t, run, heat)

        for n, t in enumerate(traps):
            parts += _trap(f"Trap{i}_{n}", leg, t, run, heat)

        if leg.horizontal and run.supports != "none":
            t = t0 + 0.5
            while t < t1 - 0.4:
                if not any(
                    lo - 0.4 < t < hi + 0.4
                    for lo, hi in blocked + [(j - 0.15, j + 0.15) for j in joints]
                ):
                    parts.append(_support(f"Support{support_n}", leg, t, run, heat))
                    support_n += 1
                t += run.support_every_m
    parts = [
        replace(_mottled(part, run), along=_along_of(part.params, legs))
        for part in parts
    ]
    return parts


def _along_of(params: dict, legs: List[Leg]) -> float:
    """Distance along the polyline of the point nearest a part's centre."""
    centre = params["centre"]
    best = (float("inf"), 0.0)
    for leg in legs:
        offset = (centre[leg.axis] - leg.start[leg.axis]) * leg.sign
        t = min(max(offset, 0.0), leg.length)
        point = leg.point(t)
        dist = sum((a - b) ** 2 for a, b in zip(point, centre))
        best = min(best, (dist, leg.along + t))
    return best[1]


def _mottled(part: Part, run: RunSpec) -> Part:
    """Give a part the emissivity patchiness of its finish or material."""
    if part.material == "pipe":
        amount = FINISH_MOTTLE[run.finish]
    elif part.material == "jacket":
        amount = JACKET_MOTTLE[run.insulation]
    else:
        amount = MATERIAL_MOTTLE.get(part.material, 0.0)
    return replace(part, surface=replace(part.surface, mottle=amount))


def _bands(
    name: str, leg: Leg, lo: float, hi: float, run: RunSpec, heat: _Thermal
) -> List[Part]:
    """Strapped seams along a jacket section, a little warmer than the jacket."""
    bands = []
    t = lo + BAND_EVERY_M / 2.0
    n = 0
    while t < hi - 0.05:
        bands.append(
            Part(
                f"{name}_Band{n}",
                "cylinder",
                {
                    "axis": _axis_name(leg.axis),
                    "radius": run.jacket_radius * 1.04,
                    "height": 0.025,
                    "centre": leg.point(t),
                },
                SurfaceField(
                    heat.at(leg.along + t, BAND_SHARE), emissivity=BAND_EMISSIVITY
                ),
                "band",
            )
        )
        n += 1
        t += BAND_EVERY_M
    return bands


def _low(leg: Leg, lo: float, hi: float, r: float) -> bool:
    bottom = min(leg.point(lo)[2], leg.point(hi)[2]) - r
    return bottom < COLLIDE_BELOW_M


def _flange(
    name: str, leg: Leg, t: float, r: float, surface: SurfaceField
) -> List[Part]:
    return [
        Part(
            name,
            "cylinder",
            {
                "axis": _axis_name(leg.axis),
                "radius": 1.7 * r,
                "height": 0.35 * r,
                "centre": leg.point(t),
            },
            surface,
            "pipe",
            collide=_low(leg, t, t, 1.7 * r),
        )
    ]


def _bolt(name: str, leg: Leg, t: float, r: float, m: int, heat: _Thermal) -> Part:
    angle = 2.0 * math.pi * (m + 0.5) / 6.0
    across = [k for k in range(3) if k != leg.axis]
    centre = list(leg.point(t))
    centre[across[0]] += 1.42 * r * math.cos(angle)
    centre[across[1]] += 1.42 * r * math.sin(angle)
    return Part(
        name,
        "cylinder",
        {
            "axis": _axis_name(leg.axis),
            "radius": max(0.006, 0.11 * r),
            "height": 0.6 * r,
            "centre": tuple(centre),
        },
        SurfaceField(heat.at(leg.along + t, BOLT_SHARE), emissivity=BOLT_EMISSIVITY),
        "bolt",
    )


def _valve(name: str, leg: Leg, t: float, run: RunSpec, heat: _Thermal) -> List[Part]:
    r = run.radius
    amb = heat.ambient_c
    axis = _axis_name(leg.axis)
    x, y, z = leg.point(t)
    upstream = leg.along + t - 0.01
    body_temp = heat.at(upstream, VALVE_BODY_SHARE)
    half = run.valve_length / 2.0
    bonnet_h = 2.6 * r
    bonnet_base = z + 1.3 * r
    stem_h = max(0.08, 2.4 * r)
    stem_base = bonnet_base + bonnet_h
    wheel_z = stem_base + stem_h
    major = max(0.12, 2.6 * r)
    minor = max(0.012, 0.16 * r)
    spoke_r = max(0.008, 0.09 * r)

    bonnet = SurfaceField(
        body_temp,
        emissivity=VALVE_EMISSIVITY,
        origin=(x, y, bonnet_base),
        axis=(0.0, 0.0, 1.0),
        decay_m=4.0 * r,
    )
    stem = SurfaceField(
        _fade(bonnet, bonnet_h, amb),
        emissivity=STEM_EMISSIVITY,
        origin=(x, y, stem_base),
        axis=(0.0, 0.0, 1.0),
        decay_m=max(0.08, 4.0 * r),
    )
    wheel = SurfaceField(
        _fade(stem, stem_h, amb),
        emissivity=WHEEL_EMISSIVITY,
        origin=(x, y, wheel_z),
        decay_m=1.5 * major,
        radial_decay=True,
    )
    parts = [
        Part(
            f"{name}/Body",
            "cylinder",
            {"axis": axis, "radius": 1.45 * r, "height": 2.2 * r, "centre": (x, y, z)},
            SurfaceField(body_temp, emissivity=VALVE_EMISSIVITY),
            "valve",
            collide=z - 1.45 * r < COLLIDE_BELOW_M,
        ),
        Part(
            f"{name}/Bonnet",
            "cylinder",
            {
                "axis": "Z",
                "radius": 0.95 * r,
                "height": bonnet_h,
                "centre": (x, y, bonnet_base + bonnet_h / 2.0),
            },
            bonnet,
            "valve",
        ),
        Part(
            f"{name}/Stem",
            "cylinder",
            {
                "axis": "Z",
                "radius": max(0.006, 0.12 * r),
                "height": stem_h,
                "centre": (x, y, stem_base + stem_h / 2.0),
            },
            stem,
            "bolt",
        ),
        Part(
            f"{name}/Wheel",
            "torus",
            {"centre": (x, y, wheel_z), "axis": "Z", "major": major, "minor": minor},
            wheel,
            "wheel",
        ),
        Part(
            f"{name}/Hub",
            "cylinder",
            {
                "axis": "Z",
                "radius": max(0.02, 0.3 * r),
                "height": max(0.02, 0.3 * r),
                "centre": (x, y, wheel_z),
            },
            wheel,
            "wheel",
        ),
    ]
    for spoke in ("X", "Y"):
        parts.append(
            Part(
                f"{name}/Spoke{spoke}",
                "cylinder",
                {
                    "axis": spoke,
                    "radius": spoke_r,
                    "height": 2.0 * major,
                    "centre": (x, y, wheel_z),
                },
                wheel,
                "wheel",
            )
        )
    for end, sign in (("A", -1.0), ("B", 1.0)):
        flange_t = t + sign * (half - 0.2 * r)
        side = upstream if sign < 0 else leg.along + t + 0.01
        parts += _flange(
            f"{name}/Flange{end}",
            leg,
            flange_t,
            1.1 * r,
            SurfaceField(heat.at(side, FLANGE_SHARE), emissivity=VALVE_EMISSIVITY),
        )
    return parts


def _trap(name: str, leg: Leg, t: float, run: RunSpec, heat: _Thermal) -> List[Part]:
    """An inline steam trap: body, condensate bowl below, and two unions."""
    r = run.radius
    axis = _axis_name(leg.axis)
    x, y, z = leg.point(t)
    inlet = leg.along + t - run.trap_length / 2.0 - 0.01
    outlet = leg.along + t + run.trap_length / 2.0 + 0.01
    body_temp = heat.at(inlet, 0.92)
    bowl_h = 1.4 * r
    bowl = SurfaceField(
        body_temp,
        emissivity=TRAP_EMISSIVITY,
        origin=(x, y, z - 1.2 * r),
        axis=(0.0, 0.0, -1.0),
        decay_m=3.0 * r,
    )
    parts = [
        Part(
            f"{name}/Body",
            "cylinder",
            {"axis": axis, "radius": 1.35 * r, "height": 1.8 * r, "centre": (x, y, z)},
            SurfaceField(body_temp, emissivity=TRAP_EMISSIVITY),
            "trap",
            collide=z - 1.35 * r < COLLIDE_BELOW_M,
        ),
        Part(
            f"{name}/Bowl",
            "cylinder",
            {
                "axis": "Z",
                "radius": 0.9 * r,
                "height": bowl_h,
                "centre": (x, y, z - 1.2 * r - bowl_h / 2.0),
            },
            bowl,
            "trap",
        ),
    ]
    half = run.trap_length / 2.0 - 0.2 * r
    for end, sign, along in (("Inlet", -1.0, inlet), ("Outlet", 1.0, outlet)):
        parts.append(
            Part(
                f"{name}/{end}",
                "cylinder",
                {
                    "axis": axis,
                    "radius": 1.2 * r,
                    "height": 0.4 * r,
                    "centre": leg.point(t + sign * half),
                },
                SurfaceField(heat.at(along, 0.95), emissivity=TRAP_EMISSIVITY),
                "trap",
            )
        )
    return parts


def _support(name: str, leg: Leg, t: float, run: RunSpec, heat: _Thermal) -> Part:
    x, y, z = leg.point(t)
    contact = heat.at(leg.along + t, SUPPORT_CONTACT_SHARE)
    emissivity = FINISHES["galvanized"]
    if run.supports == "floor":
        top = z - run.radius
        return Part(
            name,
            "cylinder",
            {"axis": "Z", "radius": 0.03, "height": top, "centre": (x, y, top / 2.0)},
            SurfaceField(
                contact,
                emissivity=emissivity,
                origin=(x, y, top),
                axis=(0.0, 0.0, -1.0),
                decay_m=0.35,
            ),
            "support",
            collide=True,
        )
    bottom = z + run.radius
    height = max(0.05, run.ceiling_z - bottom)
    return Part(
        name,
        "cylinder",
        {
            "axis": "Z",
            "radius": 0.008,
            "height": height,
            "centre": (x, y, bottom + height / 2.0),
        },
        SurfaceField(
            contact,
            emissivity=emissivity,
            origin=(x, y, bottom),
            axis=(0.0, 0.0, 1.0),
            decay_m=0.25,
        ),
        "support",
    )


def heat_sources(runs: List[RunSpec], ambient_c: float) -> Dict[str, SurfaceField]:
    """Exact prim path -> surface field for every part of every run."""
    return {
        f"{PIPEWORK_ROOT}/{run.name}/{part.name}": part.surface
        for run in runs
        for part in build_run(run, ambient_c)
    }


class PipeworkRunner:
    """Builds the pipe runs into the stage."""

    def __init__(self, cfg: Optional[dict], ambient_c: float = 21.0) -> None:
        self.runs = parse_layout(cfg)
        self.ambient_c = ambient_c

    @property
    def enabled(self) -> bool:
        """True if the scene has any pipe runs."""
        return bool(self.runs)

    def setup(self) -> bool:
        """Add every run to the stage. Must run before world.reset()."""
        if not self.enabled:
            return False
        import omni.usd
        from gauges import _material
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, PIPEWORK_ROOT)
        mats = f"{PIPEWORK_ROOT}/Materials"
        UsdGeom.Scope.Define(stage, mats)
        shared = {
            "valve": _material(
                stage, f"{mats}/Valve", (0.16, 0.30, 0.20), roughness=0.5
            ),
            "bolt": _material(
                stage, f"{mats}/Bolt", (0.12, 0.12, 0.13), roughness=0.4, metallic=0.8
            ),
            "wheel": _material(
                stage, f"{mats}/Wheel", (0.72, 0.08, 0.06), roughness=0.4
            ),
            "band": _material(
                stage,
                f"{mats}/Band",
                (0.70, 0.71, 0.72),
                roughness=0.3,
                metallic=0.9,
            ),
            "trap": _material(
                stage, f"{mats}/Trap", (0.22, 0.22, 0.24), roughness=0.6, metallic=0.5
            ),
            "support": _material(
                stage,
                f"{mats}/Support",
                (0.42, 0.43, 0.45),
                roughness=0.6,
                metallic=0.6,
            ),
        }
        count = 0
        for run in self.runs:
            root = f"{PIPEWORK_ROOT}/{run.name}"
            UsdGeom.Xform.Define(stage, root)
            materials = dict(shared)
            materials["pipe"] = _material(
                stage, f"{mats}/Pipe_{run.name}", run.color, roughness=0.45
            )
            if run.insulation != "none":
                color = JACKETS[run.insulation][1]
                shiny = run.insulation == "aluminum"
                materials["jacket"] = _material(
                    stage,
                    f"{mats}/Jacket_{run.name}",
                    color,
                    roughness=0.35 if shiny else 0.7,
                    metallic=0.9 if shiny else 0.0,
                )
            for part in build_run(run, self.ambient_c):
                _build(stage, f"{root}/{part.name}", part, materials[part.material])
                count += 1
        logger.info("[Pipework] built %d runs, %d parts", len(self.runs), count)
        return True


def _build(stage, path: str, part: Part, mat) -> None:
    from gauges import _bind, _cylinder
    from pxr import UsdGeom

    parent = path.rsplit("/", 1)[0]
    if not stage.GetPrimAtPath(parent):
        UsdGeom.Xform.Define(stage, parent)
    p = part.params
    if part.shape == "cylinder":
        _cylinder(
            stage,
            path,
            p["axis"],
            p["radius"],
            p["height"],
            p["centre"],
            mat,
            collide=part.collide,
        )
        return
    if part.shape == "elbow":
        mesh = _sweep_mesh(
            stage,
            path,
            p["centre"],
            p["u"],
            p["v"],
            p["bend_radius"],
            p["radius"],
            math.pi / 2.0,
        )
    else:
        u, v = {
            "Z": ((1, 0, 0), (0, 1, 0)),
            "X": ((0, 1, 0), (0, 0, 1)),
            "Y": ((0, 0, 1), (1, 0, 0)),
        }[p["axis"]]
        mesh = _sweep_mesh(
            stage, path, p["centre"], u, v, p["major"], p["minor"], 2.0 * math.pi
        )
    _bind(mesh.GetPrim(), mat)


def _sweep_mesh(stage, path, centre, u, v, major, minor, sweep, rings=None, sides=16):
    """A tube swept round an arc from centre + major*u towards centre + major*v."""
    from pxr import Gf, UsdGeom

    rings = rings or max(6, int(round(24 * sweep / math.pi)))
    closed = sweep >= 2.0 * math.pi - 1e-6
    w = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    n_rings = rings if closed else rings + 1
    points, normals = [], []
    for i in range(n_rings):
        theta = sweep * i / rings
        radial = [u[k] * math.cos(theta) + v[k] * math.sin(theta) for k in range(3)]
        spine = [centre[k] + major * radial[k] for k in range(3)]
        for j in range(sides):
            phi = 2.0 * math.pi * j / sides
            n = [radial[k] * math.cos(phi) + w[k] * math.sin(phi) for k in range(3)]
            points.append(Gf.Vec3f(*(spine[k] + minor * n[k] for k in range(3))))
            normals.append(Gf.Vec3f(*n))
    counts, indices = [], []
    for i in range(rings):
        a, b = i, (i + 1) % n_rings
        for j in range(sides):
            k = (j + 1) % sides
            counts.append(4)
            indices += [a * sides + j, b * sides + j, b * sides + k, a * sides + k]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    return mesh
