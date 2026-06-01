# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Procedural desert terrain generation for the solar panel installation scene.

Creates a ground plane with:
- Height-field variation (gentle dunes / rocky patches)
- Sand-colored PBR material with roughness
- Optional scattered rocks and debris
- Configurable size for different scene scales

Uses Isaac Lab's terrain generation utilities where available,
falling back to simple mesh-based height fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    import isaacsim  # noqa: F401
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DesertTerrainCfg:
    """Configuration for desert terrain generation."""

    # Dimensions
    terrain_size: tuple[float, float] = (20.0, 20.0)
    """Terrain extent in meters (X, Y)."""
    resolution: int = 64
    """Number of vertices per axis for the height field."""
    max_height: float = 0.3
    """Maximum height variation (dune amplitude), meters."""

    # Height field parameters
    num_octaves: int = 4
    """Number of Perlin noise octaves for terrain height."""
    persistence: float = 0.5
    """Amplitude decay per octave."""
    lacunarity: float = 2.0
    """Frequency multiplier per octave."""
    base_frequency: float = 0.15
    """Base noise frequency (cycles per meter)."""
    seed: int = 42
    """Random seed for reproducibility."""

    # Material
    sand_color: Sequence[float] = (0.82, 0.72, 0.55, 1.0)
    """RGBA color for desert sand."""
    sand_roughness: float = 0.9
    """PBR roughness (sand is very rough)."""
    sand_metallic: float = 0.0
    """PBR metallic (sand is non-metallic)."""

    # Rocks
    num_rocks: int = 15
    """Number of scattered rocks."""
    rock_size_range: tuple[float, float] = (0.05, 0.25)
    """Min/max rock radius, meters."""
    rock_color: Sequence[float] = (0.55, 0.50, 0.45, 1.0)
    """RGBA color for rocks."""
    rock_zone_radius: float = 8.0
    """Radius around origin within which rocks are scattered."""

    # Physics
    friction: float = 0.8
    """Ground friction coefficient."""
    restitution: float = 0.1
    """Ground restitution (bounciness)."""


# ---------------------------------------------------------------------------
# Height field generation
# ---------------------------------------------------------------------------

def _perlin_noise_2d(
    shape: tuple[int, int],
    frequency: float,
    seed: int = 0,
) -> np.ndarray:
    """Generate 2D value noise (simple gradient noise approximation).

    Returns an array of shape *shape* with values in ``[-1, 1]``.
    """
    rng = np.random.RandomState(seed)
    rows, cols = shape

    # Grid of random gradients
    gx = int(math.ceil(cols * frequency)) + 2
    gy = int(math.ceil(rows * frequency)) + 2
    gradients = rng.randn(gy, gx, 2).astype(np.float32)
    gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8

    # Sample coordinates
    y_coords = np.linspace(0, (rows - 1) * frequency, rows)
    x_coords = np.linspace(0, (cols - 1) * frequency, cols)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Integer and fractional parts
    xi = np.floor(xx).astype(int)
    yi = np.floor(yy).astype(int)
    xf = xx - xi
    yf = yy - yi

    # Smoothstep
    def fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    u = fade(xf)
    v = fade(yf)

    # Dot products at corners
    def dot_grid(iy: np.ndarray, ix: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
        iy_c = np.clip(iy, 0, gy - 1)
        ix_c = np.clip(ix, 0, gx - 1)
        g = gradients[iy_c, ix_c]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot_grid(yi, xi, yf, xf)
    n10 = dot_grid(yi, xi + 1, yf, xf - 1)
    n01 = dot_grid(yi + 1, xi, yf - 1, xf)
    n11 = dot_grid(yi + 1, xi + 1, yf - 1, xf - 1)

    # Bilinear interpolation
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    result = nx0 * (1 - v) + nx1 * v

    return result


def generate_height_field(cfg: DesertTerrainCfg) -> np.ndarray:
    """Generate a 2D height field using layered noise.

    Returns:
        Height field array of shape ``(resolution, resolution)``
        with values in ``[0, max_height]``.
    """
    shape = (cfg.resolution, cfg.resolution)
    height = np.zeros(shape, dtype=np.float32)

    amplitude = 1.0
    frequency = cfg.base_frequency

    for octave in range(cfg.num_octaves):
        noise = _perlin_noise_2d(shape, frequency, seed=cfg.seed + octave)
        height += amplitude * noise
        amplitude *= cfg.persistence
        frequency *= cfg.lacunarity

    # Normalize to [0, max_height]
    h_min, h_max = height.min(), height.max()
    if h_max > h_min:
        height = (height - h_min) / (h_max - h_min) * cfg.max_height
    else:
        height[:] = 0.0

    return height


# ---------------------------------------------------------------------------
# Spawner
# ---------------------------------------------------------------------------

def spawn_desert_terrain(
    prim_path: str,
    cfg: DesertTerrainCfg | None = None,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    stage: Usd.Stage | None = None,
) -> str:
    """Spawn a desert terrain mesh with scattered rocks.

    Args:
        prim_path: Root prim path (e.g. ``/World/Terrain``).
        cfg: Terrain configuration.
        translation: World offset.
        stage: USD stage.

    Returns:
        The prim path of the terrain root.
    """
    if cfg is None:
        cfg = DesertTerrainCfg()
    if stage is None:
        from isaaclab.sim import SimulationContext
        stage = SimulationContext.instance().stage

    tx, ty, tz = translation

    # Root Xform
    root = UsdGeom.Xform.Define(stage, prim_path)
    rxf = UsdGeom.Xformable(root.GetPrim())
    rxf.ClearXformOpOrder()
    rxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

    # --- Sand material ---
    mat_path = f"{prim_path}/mat_sand"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(cfg.sand_color[0], cfg.sand_color[1], cfg.sand_color[2])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(cfg.sand_roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(cfg.sand_metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # --- Height field mesh ---
    height_field = generate_height_field(cfg)
    mesh_path = f"{prim_path}/ground_mesh"
    _create_height_mesh(stage, mesh_path, height_field, cfg, mat)

    # --- Scattered rocks ---
    _spawn_rocks(stage, f"{prim_path}/rocks", height_field, cfg)

    return prim_path


def _create_height_mesh(
    stage: Usd.Stage,
    path: str,
    height_field: np.ndarray,
    cfg: DesertTerrainCfg,
    material: UsdShade.Material,
) -> None:
    """Create a UsdGeom.Mesh from a height field array."""
    res = cfg.resolution
    sx, sy = cfg.terrain_size

    # Vertices
    points = []
    for j in range(res):
        for i in range(res):
            x = (i / (res - 1) - 0.5) * sx
            y = (j / (res - 1) - 0.5) * sy
            z = float(height_field[j, i])
            points.append(Gf.Vec3f(x, y, z))

    # Face indices (quads → 2 triangles each)
    face_vertex_indices = []
    face_vertex_counts = []
    for j in range(res - 1):
        for i in range(res - 1):
            v00 = j * res + i
            v10 = j * res + (i + 1)
            v01 = (j + 1) * res + i
            v11 = (j + 1) * res + (i + 1)
            # Triangle 1
            face_vertex_indices.extend([v00, v10, v11])
            face_vertex_counts.append(3)
            # Triangle 2
            face_vertex_indices.extend([v00, v11, v01])
            face_vertex_counts.append(3)

    # UV coordinates
    uvs = []
    for j in range(res):
        for i in range(res):
            uvs.append(Gf.Vec2f(i / (res - 1), j / (res - 1)))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))

    # Set UVs
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    uv_pv = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv_pv.Set(Vt.Vec2fArray(uvs))

    # Collision
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("meshSimplification")

    # Material
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _spawn_rocks(
    stage: Usd.Stage,
    parent_path: str,
    height_field: np.ndarray,
    cfg: DesertTerrainCfg,
) -> None:
    """Scatter rock meshes (spheres with random deformation) on the terrain."""
    rng = np.random.RandomState(cfg.seed + 100)
    res = cfg.resolution
    sx, sy = cfg.terrain_size

    # Rock material
    mat = UsdShade.Material.Define(stage, f"{parent_path}/mat_rock")
    shader = UsdShade.Shader.Define(stage, f"{parent_path}/mat_rock/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(cfg.rock_color[0], cfg.rock_color[1], cfg.rock_color[2])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Xform parent
    UsdGeom.Xform.Define(stage, parent_path)

    for i in range(cfg.num_rocks):
        # Random position within rock zone
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(2.0, cfg.rock_zone_radius)
        rx = dist * math.cos(angle)
        ry = dist * math.sin(angle)

        # Sample height field
        ix = int(((rx / sx) + 0.5) * (res - 1))
        iy = int(((ry / sy) + 0.5) * (res - 1))
        ix = max(0, min(res - 1, ix))
        iy = max(0, min(res - 1, iy))
        rz = float(height_field[iy, ix])

        # Random size
        radius = rng.uniform(cfg.rock_size_range[0], cfg.rock_size_range[1])

        # Deformed sphere (random scale per axis)
        rock_path = f"{parent_path}/rock_{i:03d}"
        sphere = UsdGeom.Sphere.Define(stage, rock_path)
        sphere.CreateRadiusAttr(radius)

        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz + radius * 0.5))
        # Random axis scaling for natural look
        scale_x = rng.uniform(0.7, 1.3)
        scale_y = rng.uniform(0.7, 1.3)
        scale_z = rng.uniform(0.5, 0.9)
        xf.AddScaleOp().Set(Gf.Vec3f(scale_x, scale_y, scale_z))

        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = DesertTerrainCfg()
    hf = generate_height_field(cfg)
    print(f"Desert terrain config: {cfg.terrain_size[0]}×{cfg.terrain_size[1]}m, "
          f"resolution={cfg.resolution}, max_height={cfg.max_height}m")
    print(f"Height field: shape={hf.shape}, range=[{hf.min():.3f}, {hf.max():.3f}]")
    print(f"Rocks: {cfg.num_rocks}, size range={cfg.rock_size_range}")
