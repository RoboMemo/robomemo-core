# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Procedural USD generation for screw/bolt objects.

Creates simplified screw/bolt geometry with:
- Cylindrical shank
- Head (hex, Phillips cross-slot, flathead, Torx star recess)
- Optional threading grooves (visual only — simplified cylinder for physics)

Screws can be spawned at specific mount points and are set up as rigid bodies
with appropriate mass for physics simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

try:
    import isaacsim  # noqa: F401
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Enums & Configuration
# ---------------------------------------------------------------------------

class ScrewType(Enum):
    """Supported screw head types."""
    HEX = "hex"
    PHILLIPS = "phillips"
    FLATHEAD = "flathead"
    TORX = "torx"


class MetricSize(Enum):
    """Common metric screw sizes (diameter in mm)."""
    M3 = 3.0
    M4 = 4.0
    M5 = 5.0
    M6 = 6.0
    M8 = 8.0
    M10 = 10.0
    M12 = 12.0


@dataclass
class ScrewBoltCfg:
    """Configuration for a single screw/bolt.

    All linear dimensions in meters unless noted.
    """

    screw_type: ScrewType = ScrewType.HEX
    """Head type."""

    metric_size: MetricSize = MetricSize.M6
    """Metric thread size (determines shank diameter)."""

    shank_length: float = 0.025
    """Length of the threaded shank, meters."""

    head_height: float | None = None
    """Height of the screw head. If *None*, derived from metric_size."""

    thread_pitch: float | None = None
    """Thread pitch in meters. If *None*, uses ISO standard for the size."""

    # Appearance
    color: Sequence[float] = (0.75, 0.75, 0.78, 1.0)
    """RGBA color (stainless steel default)."""

    # Physics
    density: float = 7850.0
    """Steel density, kg/m³."""
    friction: float = 0.5
    """Surface friction coefficient."""

    # Driving parameters (for task)
    target_turns: float = 8.0
    """Number of full turns to fully tighten."""
    target_torque_nm: float = 10.0
    """Target tightening torque, N·m."""

    def __post_init__(self) -> None:
        """Derive defaults from metric size."""
        d_mm = self.metric_size.value
        d_m = d_mm / 1000.0

        if self.head_height is None:
            # Roughly 0.7 × diameter for hex bolts
            self.head_height = d_m * 0.7

        if self.thread_pitch is None:
            # ISO coarse pitch approximation
            _iso_pitch = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75}
            self.thread_pitch = _iso_pitch.get(int(d_mm), 1.0) / 1000.0

    @property
    def shank_radius(self) -> float:
        """Shank radius in meters."""
        return (self.metric_size.value / 1000.0) / 2.0

    @property
    def head_radius(self) -> float:
        """Head outer radius (across flats for hex → across corners)."""
        d = self.metric_size.value / 1000.0
        if self.screw_type == ScrewType.HEX:
            return d * 0.95  # across-corners ≈ 1.9 × diameter / 2
        elif self.screw_type == ScrewType.FLATHEAD:
            return d * 1.0
        else:
            return d * 0.85


# ---------------------------------------------------------------------------
# Spawner
# ---------------------------------------------------------------------------

def _create_screw_material(
    stage: Usd.Stage, path: str, color: Sequence[float]
) -> UsdShade.Material:
    """Create a metallic PBR material for the screw."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(color[0], color[1], color[2])
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.95)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def spawn_screw_bolt(
    prim_path: str,
    cfg: ScrewBoltCfg | None = None,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    orientation_euler_deg: Sequence[float] = (0.0, 0.0, 0.0),
    stage: Usd.Stage | None = None,
    as_rigid_body: bool = True,
) -> str:
    """Spawn a screw/bolt at the given position.

    The screw is oriented with the shank pointing along −Z (into the surface)
    and the head at the top.

    Args:
        prim_path: USD prim path for the screw root.
        cfg: Screw configuration.
        position: World position ``(x, y, z)``.
        orientation_euler_deg: Euler rotation ``(rx, ry, rz)`` in degrees.
        stage: USD stage (current stage if *None*).
        as_rigid_body: If True, set up as a dynamic rigid body.

    Returns:
        The prim path of the spawned screw root.
    """
    if cfg is None:
        cfg = ScrewBoltCfg()

    if stage is None:
        from isaaclab.sim import SimulationContext
        stage = SimulationContext.instance().stage

    # Root Xform ----------------------------------------------------------
    root = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(root.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    if any(v != 0 for v in orientation_euler_deg):
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*orientation_euler_deg))

    # Material
    mat = _create_screw_material(stage, f"{prim_path}/mat_screw", cfg.color)

    # Shank (cylinder) ----------------------------------------------------
    shank = UsdGeom.Cylinder.Define(stage, f"{prim_path}/shank")
    shank.CreateRadiusAttr(cfg.shank_radius)
    shank.CreateHeightAttr(cfg.shank_length)
    shank.CreateAxisAttr("Z")

    shank_xf = UsdGeom.Xformable(shank.GetPrim())
    shank_xf.ClearXformOpOrder()
    shank_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -cfg.shank_length / 2.0))

    UsdPhysics.CollisionAPI.Apply(shank.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(shank.GetPrim()).Bind(mat)

    # Head (cylinder for simplicity) --------------------------------------
    head = UsdGeom.Cylinder.Define(stage, f"{prim_path}/head")
    head.CreateRadiusAttr(cfg.head_radius)
    head.CreateHeightAttr(cfg.head_height)
    head.CreateAxisAttr("Z")

    head_xf = UsdGeom.Xformable(head.GetPrim())
    head_xf.ClearXformOpOrder()
    head_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, cfg.head_height / 2.0))

    UsdPhysics.CollisionAPI.Apply(head.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(head.GetPrim()).Bind(mat)

    # Drive slot (visual marker on top of head) ---------------------------
    _add_drive_slot(stage, f"{prim_path}/head/slot", cfg)

    # Threading grooves (visual only — thin cylinders along shank) --------
    if cfg.thread_pitch is not None:
        num_grooves = int(cfg.shank_length / cfg.thread_pitch)
        for g in range(min(num_grooves, 20)):  # cap visual complexity
            gz = -g * cfg.thread_pitch
            groove = UsdGeom.Cylinder.Define(
                stage, f"{prim_path}/shank/groove_{g}"
            )
            groove.CreateRadiusAttr(cfg.shank_radius * 1.08)
            groove.CreateHeightAttr(cfg.thread_pitch * 0.3)
            groove.CreateAxisAttr("Z")
            gxf = UsdGeom.Xformable(groove.GetPrim())
            gxf.ClearXformOpOrder()
            gxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, gz))

    # Rigid body physics --------------------------------------------------
    if as_rigid_body:
        rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
        # Approximate mass: shank + head cylinders
        shank_vol = math.pi * cfg.shank_radius**2 * cfg.shank_length
        head_vol = math.pi * cfg.head_radius**2 * cfg.head_height
        mass = cfg.density * (shank_vol + head_vol)
        mass_api.CreateMassAttr(mass)

    return prim_path


def _add_drive_slot(stage: Usd.Stage, path: str, cfg: ScrewBoltCfg) -> None:
    """Add a visual drive slot on the screw head (Phillips cross / hex recess / etc.)."""
    head_top_z = cfg.head_height
    slot_depth = cfg.head_height * 0.3

    if cfg.screw_type == ScrewType.PHILLIPS:
        # Cross-slot: two thin boxes at 90°
        for i, rot in enumerate([0.0, 90.0]):
            slot = UsdGeom.Cube.Define(stage, f"{path}/cross_{i}")
            slot.CreateSizeAttr(1.0)
            sxf = UsdGeom.Xformable(slot.GetPrim())
            sxf.ClearXformOpOrder()
            sxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, head_top_z - slot_depth / 2))
            sxf.AddRotateZOp().Set(rot)
            sxf.AddScaleOp().Set(
                Gf.Vec3f(cfg.head_radius * 1.5, cfg.shank_radius * 0.3, slot_depth)
            )

    elif cfg.screw_type == ScrewType.HEX:
        # Hex recess: a smaller cylinder
        recess = UsdGeom.Cylinder.Define(stage, f"{path}/hex_recess")
        recess.CreateRadiusAttr(cfg.shank_radius * 0.85)
        recess.CreateHeightAttr(slot_depth)
        recess.CreateAxisAttr("Z")
        rxf = UsdGeom.Xformable(recess.GetPrim())
        rxf.ClearXformOpOrder()
        rxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, head_top_z - slot_depth / 2))

    elif cfg.screw_type == ScrewType.FLATHEAD:
        # Single slot
        slot = UsdGeom.Cube.Define(stage, f"{path}/flat_slot")
        slot.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(slot.GetPrim())
        sxf.ClearXformOpOrder()
        sxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, head_top_z - slot_depth / 2))
        sxf.AddScaleOp().Set(
            Gf.Vec3f(cfg.head_radius * 1.5, cfg.shank_radius * 0.25, slot_depth)
        )

    elif cfg.screw_type == ScrewType.TORX:
        # Star recess approximated by 6-pointed arrangement of small cylinders
        for i in range(6):
            angle = math.radians(i * 60)
            tx = cfg.shank_radius * 0.4 * math.cos(angle)
            ty = cfg.shank_radius * 0.4 * math.sin(angle)
            point = UsdGeom.Cylinder.Define(stage, f"{path}/torx_{i}")
            point.CreateRadiusAttr(cfg.shank_radius * 0.2)
            point.CreateHeightAttr(slot_depth)
            point.CreateAxisAttr("Z")
            pxf = UsdGeom.Xformable(point.GetPrim())
            pxf.ClearXformOpOrder()
            pxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, head_top_z - slot_depth / 2))


# ---------------------------------------------------------------------------
# Batch spawning
# ---------------------------------------------------------------------------

def spawn_screws_at_holes(
    screw_holes: list[dict],
    cfg: ScrewBoltCfg | None = None,
    parent_path: str = "/World/Screws",
    stage: Usd.Stage | None = None,
) -> list[str]:
    """Spawn screws at all mounting holes returned by ``spawn_solar_panel_rack``.

    Args:
        screw_holes: List of hole dicts with ``position`` and ``normal`` keys.
        cfg: Screw configuration (same for all screws).
        parent_path: USD parent path for the screw prims.
        stage: USD stage.

    Returns:
        List of prim paths for all spawned screws.
    """
    if cfg is None:
        cfg = ScrewBoltCfg()

    paths = []
    for i, hole in enumerate(screw_holes):
        pos = hole["position"]
        normal = hole["normal"]

        # Compute orientation so shank aligns with −normal
        # Default screw shank is along −Z; rotate to align with −normal
        nx, ny, nz = normal
        pitch_deg = -math.degrees(math.asin(ny))
        yaw_deg = math.degrees(math.atan2(nx, nz))

        path = f"{parent_path}/screw_{i:03d}"
        spawn_screw_bolt(
            prim_path=path,
            cfg=cfg,
            position=pos,
            orientation_euler_deg=(pitch_deg, yaw_deg, 0.0),
            stage=stage,
        )
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for st in ScrewType:
        for ms in [MetricSize.M6, MetricSize.M8]:
            cfg = ScrewBoltCfg(screw_type=st, metric_size=ms)
            print(
                f"{st.value:10s} {ms.name}: shank_r={cfg.shank_radius*1000:.1f}mm, "
                f"head_r={cfg.head_radius*1000:.1f}mm, "
                f"head_h={cfg.head_height*1000:.1f}mm, "
                f"pitch={cfg.thread_pitch*1000:.2f}mm"
            )
