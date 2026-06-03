# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Procedural USD generation for solar panel mounting rack structure.

This module creates a simplified solar panel rack assembly in USD format using
Isaac Lab's sim utilities. The rack consists of:
- Vertical support posts (steel tubes)
- Horizontal cross-beams
- Mounting brackets with screw holes
- Solar panel surface (flat rectangular body)

The geometry is parameterized for domain randomization (size, spacing, material).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    import isaacsim  # noqa: F401
    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    # Allow import for documentation / offline tooling
    sim_utils = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SolarPanelRackCfg:
    """Configuration for the solar panel rack geometry.

    All dimensions in meters. The rack is oriented with the panel face
    tilted at ``panel_tilt_deg`` from horizontal, facing south (−Y).
    """

    # Panel dimensions
    panel_width: float = 2.0
    """Width of the solar panel (X-axis), meters."""
    panel_height: float = 1.0
    """Height of the solar panel (Z-axis when tilt=0), meters."""
    panel_thickness: float = 0.035
    """Thickness of the panel glass/cell sandwich, meters."""
    panel_tilt_deg: float = 30.0
    """Tilt angle of the panel from horizontal, degrees."""

    # Support structure
    post_height: float = 1.5
    """Height of the vertical support posts, meters."""
    post_radius: float = 0.04
    """Radius of steel tube posts, meters."""
    post_spacing: float = 1.8
    """Distance between left and right posts, meters."""
    crossbeam_radius: float = 0.025
    """Radius of horizontal cross-beams, meters."""

    # Mounting brackets
    num_brackets_per_side: int = 3
    """Number of mounting brackets along each long edge."""
    bracket_width: float = 0.06
    """Width of the L-bracket, meters."""
    bracket_thickness: float = 0.005
    """Sheet metal thickness, meters."""
    bracket_height: float = 0.04
    """Height of the bracket flange, meters."""

    # Screw holes
    screw_hole_radius: float = 0.004
    """Radius of the screw mounting hole, meters."""
    screws_per_bracket: int = 2
    """Number of screws per bracket."""

    # Material appearance
    frame_color: Sequence[float] = (0.6, 0.6, 0.65, 1.0)
    """RGBA color for the steel frame."""
    panel_color: Sequence[float] = (0.05, 0.05, 0.25, 1.0)
    """RGBA color for the solar panel surface (dark blue)."""
    bracket_color: Sequence[float] = (0.7, 0.7, 0.72, 1.0)
    """RGBA color for the brackets (galvanized steel)."""

    # Physics
    density: float = 7800.0
    """Density of steel frame, kg/m³."""
    panel_density: float = 2500.0
    """Effective density of the panel, kg/m³."""
    is_rigid: bool = True
    """Whether the rack is a rigid body (vs. static collider)."""


# ---------------------------------------------------------------------------
# Spawner
# ---------------------------------------------------------------------------

def _create_material(
    stage: Usd.Stage,
    path: str,
    color: Sequence[float],
    metallic: float = 0.8,
    roughness: float = 0.4,
) -> UsdShade.Material:
    """Create a simple PBR material on *stage* at *path*."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(color[0], color[1], color[2])
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _add_cylinder(
    stage: Usd.Stage,
    path: str,
    radius: float,
    height: float,
    position: Sequence[float],
    orientation_euler_deg: Sequence[float] = (0.0, 0.0, 0.0),
    material: UsdShade.Material | None = None,
) -> UsdGeom.Cylinder:
    """Add a cylinder prim with optional material binding."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr("Z")

    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if any(v != 0 for v in orientation_euler_deg):
        rx, ry, rz = orientation_euler_deg
        xform.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))

    # Collision
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)

    return cyl


def _add_box(
    stage: Usd.Stage,
    path: str,
    size: Sequence[float],
    position: Sequence[float],
    orientation_euler_deg: Sequence[float] = (0.0, 0.0, 0.0),
    material: UsdShade.Material | None = None,
) -> UsdGeom.Cube:
    """Add a box (scaled cube) prim."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if any(v != 0 for v in orientation_euler_deg):
        rx, ry, rz = orientation_euler_deg
        xform.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))
    xform.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    return cube


def spawn_solar_panel_rack(
    prim_path: str,
    cfg: SolarPanelRackCfg | None = None,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    stage: Usd.Stage | None = None,
) -> list[dict]:
    """Spawn a solar panel rack assembly and return screw-hole world positions.

    Args:
        prim_path: Root prim path in the USD stage (e.g. ``/World/SolarRack``).
        cfg: Rack configuration. Uses defaults if *None*.
        translation: World-space translation ``(x, y, z)`` for the rack origin.
        stage: USD stage. If *None*, uses the current stage from ``sim_utils``.

    Returns:
        List of dicts ``{"position": (x,y,z), "normal": (nx,ny,nz), "bracket_idx": int}``
        describing each screw-hole location in world frame.
    """
    if cfg is None:
        cfg = SolarPanelRackCfg()
    if stage is None:
        stage = sim_utils.SimulationContext.instance().stage  # type: ignore[union-attr]

    tx, ty, tz = translation
    tilt_rad = math.radians(cfg.panel_tilt_deg)

    # Root Xform ---------------------------------------------------------
    root = UsdGeom.Xform.Define(stage, prim_path)
    root_xf = UsdGeom.Xformable(root.GetPrim())
    root_xf.ClearXformOpOrder()
    root_xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

    # Materials -----------------------------------------------------------
    mat_frame = _create_material(stage, f"{prim_path}/mat_frame", cfg.frame_color)
    mat_panel = _create_material(
        stage, f"{prim_path}/mat_panel", cfg.panel_color, metallic=0.1, roughness=0.2
    )
    mat_bracket = _create_material(stage, f"{prim_path}/mat_bracket", cfg.bracket_color)

    # Vertical posts ------------------------------------------------------
    half_spacing = cfg.post_spacing / 2.0
    for i, x_offset in enumerate([-half_spacing, half_spacing]):
        _add_cylinder(
            stage,
            f"{prim_path}/post_{i}",
            cfg.post_radius,
            cfg.post_height,
            position=(x_offset, 0.0, cfg.post_height / 2.0),
            material=mat_frame,
        )

    # Horizontal cross-beam at the top ------------------------------------
    beam_len = cfg.post_spacing + 2 * cfg.post_radius
    _add_cylinder(
        stage,
        f"{prim_path}/crossbeam_top",
        cfg.crossbeam_radius,
        beam_len,
        position=(0.0, 0.0, cfg.post_height),
        orientation_euler_deg=(0.0, 90.0, 0.0),
        material=mat_frame,
    )

    # Lower cross-beam (mid-height) ---------------------------------------
    _add_cylinder(
        stage,
        f"{prim_path}/crossbeam_mid",
        cfg.crossbeam_radius,
        beam_len,
        position=(0.0, 0.0, cfg.post_height * 0.5),
        orientation_euler_deg=(0.0, 90.0, 0.0),
        material=mat_frame,
    )

    # Solar panel (tilted box) --------------------------------------------
    panel_center_z = cfg.post_height + cfg.panel_height / 2.0 * math.sin(tilt_rad)
    panel_center_y = -cfg.panel_height / 2.0 * math.cos(tilt_rad)
    _add_box(
        stage,
        f"{prim_path}/panel",
        size=(cfg.panel_width, cfg.panel_thickness, cfg.panel_height),
        position=(0.0, panel_center_y, panel_center_z),
        orientation_euler_deg=(cfg.panel_tilt_deg, 0.0, 0.0),
        material=mat_panel,
    )

    # Mounting brackets & screw holes ------------------------------------
    screw_holes: list[dict] = []
    bracket_positions_x = np.linspace(
        -cfg.post_spacing / 2.0 + 0.15,
        cfg.post_spacing / 2.0 - 0.15,
        cfg.num_brackets_per_side,
    )

    for side_idx, z_frac in enumerate([0.0, 1.0]):  # bottom & top edges of panel
        for b_idx, bx in enumerate(bracket_positions_x):
            # Position along panel edge (in panel local frame, then rotate)
            local_z = z_frac * cfg.panel_height - cfg.panel_height / 2.0
            world_y = panel_center_y + local_z * (-math.cos(tilt_rad))
            world_z = panel_center_z + local_z * math.sin(tilt_rad)

            bracket_path = f"{prim_path}/bracket_s{side_idx}_b{b_idx}"
            _add_box(
                stage,
                bracket_path,
                size=(cfg.bracket_width, cfg.bracket_height, cfg.bracket_thickness),
                position=(bx, world_y, world_z),
                orientation_euler_deg=(cfg.panel_tilt_deg, 0.0, 0.0),
                material=mat_bracket,
            )

            # Screw holes (small cylinder markers for visual reference)
            for s_idx in range(cfg.screws_per_bracket):
                sx_offset = (s_idx - (cfg.screws_per_bracket - 1) / 2.0) * 0.02
                hole_path = f"{bracket_path}/screw_hole_{s_idx}"
                _add_cylinder(
                    stage,
                    hole_path,
                    cfg.screw_hole_radius,
                    cfg.bracket_thickness * 1.5,
                    position=(
                        bx + sx_offset,
                        world_y,
                        world_z,
                    ),
                    orientation_euler_deg=(cfg.panel_tilt_deg, 0.0, 0.0),
                )

                # Normal points outward from bracket face
                normal = (
                    0.0,
                    -math.sin(tilt_rad),
                    math.cos(tilt_rad),
                )
                screw_holes.append(
                    {
                        "position": (tx + bx + sx_offset, ty + world_y, tz + world_z),
                        "normal": normal,
                        "bracket_idx": side_idx * cfg.num_brackets_per_side + b_idx,
                        "screw_idx": s_idx,
                        "prim_path": hole_path,
                    }
                )

    return screw_holes


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = SolarPanelRackCfg()
    print(f"SolarPanelRackCfg: panel {cfg.panel_width}×{cfg.panel_height}m, "
          f"tilt {cfg.panel_tilt_deg}°, {cfg.num_brackets_per_side} brackets/side, "
          f"{cfg.screws_per_bracket} screws/bracket")
    total_screws = 2 * cfg.num_brackets_per_side * cfg.screws_per_bracket
    print(f"Total screw holes: {total_screws}")
