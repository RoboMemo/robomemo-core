# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Screw driving skill generalization across screw types, sizes, and surfaces.

Provides a config-driven framework to generalize the base solar-panel screw
driving skill to diverse fastener types.  Uses curriculum learning to
progressively increase task difficulty and domain randomization to cover
the generalization dimensions.

Generalization dimensions:
    - **Screw type**: hex, Phillips, flathead, Torx, Robertson
    - **Screw size**: M3 – M12
    - **Surface material**: metal, wood, plastic, composite
    - **Surface orientation**: horizontal, 30° tilt, 45° tilt, vertical, inverted
    - **Environment**: day, night, dusty, rainy

Usage:
    python -m roboforce_skills.screw_skill_generalization --show_curriculum
    python -m roboforce_skills.screw_skill_generalization --generate_configs
    python -m roboforce_skills.screw_skill_generalization --evaluate_report /path/to/results
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screw & Surface Taxonomy
# ---------------------------------------------------------------------------

@dataclass
class ScrewTypeDef:
    """Definition of a screw / fastener type."""

    name: str
    """Human-readable name."""
    head_type: str
    """Head geometry: ``hex``, ``phillips``, ``flathead``, ``torx``, ``robertson``."""
    drive_profile: str
    """Tool engagement profile (same as head_type for most cases)."""
    typical_sizes: list[str] = field(default_factory=list)
    """Supported metric sizes (e.g. ``M4``, ``M6``)."""
    difficulty: float = 1.0
    """Relative difficulty multiplier (1.0 = baseline solar-panel hex)."""
    requires_alignment_precision_mm: float = 2.0
    """How precisely the driver must align to the screw head (mm)."""


# Screw type catalogue
SCREW_TYPES: dict[str, ScrewTypeDef] = {
    "hex": ScrewTypeDef(
        name="Hex Socket Cap",
        head_type="hex",
        drive_profile="hex",
        typical_sizes=["M3", "M4", "M5", "M6", "M8", "M10", "M12"],
        difficulty=1.0,
        requires_alignment_precision_mm=2.0,
    ),
    "phillips": ScrewTypeDef(
        name="Phillips",
        head_type="phillips",
        drive_profile="phillips",
        typical_sizes=["M3", "M4", "M5", "M6"],
        difficulty=1.2,
        requires_alignment_precision_mm=1.5,
    ),
    "flathead": ScrewTypeDef(
        name="Flathead / Slotted",
        head_type="flathead",
        drive_profile="flathead",
        typical_sizes=["M3", "M4", "M5", "M6"],
        difficulty=1.5,
        requires_alignment_precision_mm=1.0,
    ),
    "torx": ScrewTypeDef(
        name="Torx",
        head_type="torx",
        drive_profile="torx",
        typical_sizes=["M4", "M5", "M6", "M8"],
        difficulty=1.1,
        requires_alignment_precision_mm=1.8,
    ),
    "robertson": ScrewTypeDef(
        name="Robertson (Square)",
        head_type="robertson",
        drive_profile="robertson",
        typical_sizes=["M4", "M5", "M6"],
        difficulty=1.3,
        requires_alignment_precision_mm=1.6,
    ),
}


@dataclass
class ScrewSizeDef:
    """Physical parameters for a metric screw size."""

    size_name: str
    """Metric designation, e.g. ``M6``."""
    shaft_diameter_mm: float
    head_diameter_mm: float
    thread_pitch_mm: float
    typical_length_mm: float
    drive_torque_nm: float
    """Typical tightening torque (N·m)."""
    weight_g: float


SCREW_SIZES: dict[str, ScrewSizeDef] = {
    "M3": ScrewSizeDef("M3", 3.0, 5.5, 0.5, 12.0, 1.2, 1.2),
    "M4": ScrewSizeDef("M4", 4.0, 7.0, 0.7, 16.0, 2.9, 2.8),
    "M5": ScrewSizeDef("M5", 5.0, 8.5, 0.8, 20.0, 5.7, 5.5),
    "M6": ScrewSizeDef("M6", 6.0, 10.0, 1.0, 25.0, 9.9, 9.6),
    "M8": ScrewSizeDef("M8", 8.0, 13.0, 1.25, 30.0, 24.0, 22.0),
    "M10": ScrewSizeDef("M10", 10.0, 16.0, 1.5, 35.0, 47.0, 43.0),
    "M12": ScrewSizeDef("M12", 12.0, 18.0, 1.75, 40.0, 82.0, 75.0),
}


@dataclass
class SurfaceMaterialDef:
    """Physical parameters for a target surface material."""

    name: str
    friction_coefficient: float
    """Static friction coefficient between screw and surface."""
    hardness_relative: float
    """Relative hardness (1.0 = mild steel)."""
    requires_pilot_hole: bool
    deformation_risk: float
    """Risk of surface deformation (0–1)."""


SURFACE_MATERIALS: dict[str, SurfaceMaterialDef] = {
    "metal": SurfaceMaterialDef("Metal (mild steel)", 0.6, 1.0, False, 0.05),
    "aluminum": SurfaceMaterialDef("Aluminum alloy", 0.5, 0.6, False, 0.15),
    "wood": SurfaceMaterialDef("Softwood (pine)", 0.4, 0.2, True, 0.4),
    "hardwood": SurfaceMaterialDef("Hardwood (oak)", 0.45, 0.4, True, 0.2),
    "plastic": SurfaceMaterialDef("ABS Plastic", 0.35, 0.15, True, 0.6),
    "composite": SurfaceMaterialDef("Fiberglass composite", 0.5, 0.5, True, 0.25),
}


@dataclass
class SurfaceOrientationDef:
    """Surface orientation relative to gravity."""

    name: str
    tilt_degrees: float
    """Angle from horizontal (0 = flat table, 90 = vertical wall)."""
    gravity_compensation: float
    """Extra difficulty from gravity (0–1)."""


SURFACE_ORIENTATIONS: dict[str, SurfaceOrientationDef] = {
    "horizontal": SurfaceOrientationDef("Horizontal (flat)", 0.0, 0.0),
    "tilt_30": SurfaceOrientationDef("30° tilt (solar panel)", 30.0, 0.2),
    "tilt_45": SurfaceOrientationDef("45° tilt", 45.0, 0.4),
    "vertical": SurfaceOrientationDef("Vertical (wall mount)", 90.0, 0.7),
    "inverted": SurfaceOrientationDef("Inverted (overhead)", 180.0, 1.0),
}


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationCfg:
    """Domain randomization parameters per generalization dimension."""

    # Screw position jitter
    position_jitter_mm: float = 5.0
    """Uniform random offset on screw position (mm)."""

    # Screw orientation jitter
    orientation_jitter_deg: float = 3.0
    """Uniform random tilt on screw axis (degrees)."""

    # Lighting
    light_intensity_range: tuple[float, float] = (0.3, 1.5)
    """Relative light intensity range."""
    light_color_temp_range: tuple[int, int] = (3000, 7000)
    """Color temperature range (Kelvin)."""

    # Camera
    camera_fov_jitter_deg: float = 2.0
    camera_position_jitter_mm: float = 10.0

    # Material appearance
    texture_randomization: bool = True
    albedo_range: tuple[float, float] = (0.2, 0.9)
    roughness_range: tuple[float, float] = (0.1, 0.8)

    # Physics
    friction_multiplier_range: tuple[float, float] = (0.8, 1.2)
    """Multiplier on surface friction coefficient."""
    gravity_jitter: float = 0.01
    """Random perturbation on gravity vector magnitude (m/s²)."""

    # Distractors
    num_distractor_objects: tuple[int, int] = (0, 5)
    """Range of random distractor objects in the scene."""


# ---------------------------------------------------------------------------
# Curriculum Learning
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStageDef:
    """A single stage in the curriculum learning progression."""

    stage_id: int
    name: str
    description: str

    # Enabled generalization dimensions
    screw_types: list[str]
    screw_sizes: list[str]
    surface_materials: list[str]
    surface_orientations: list[str]

    # Domain randomization level (scales the DR config)
    dr_scale: float = 1.0
    """Multiplier on domain randomization ranges (0.0 = no DR, 1.0 = full DR)."""

    # Training
    num_episodes: int = 1000
    """Minimum episodes to collect for this stage."""
    success_threshold: float = 0.85
    """Success rate threshold to advance to the next stage."""
    min_training_steps: int = 10_000
    """Minimum fine-tuning steps for this stage."""

    # Evaluation
    eval_episodes: int = 100
    """Number of evaluation episodes per condition."""


def build_default_curriculum() -> list[CurriculumStageDef]:
    """Build the default 5-stage curriculum for screw skill generalization.

    Stage 1: Baseline — single screw type/size on the original solar panel setup.
    Stage 2: Size variation — same type, vary M3–M8.
    Stage 3: Type variation — add Phillips and Torx.
    Stage 4: Surface variation — introduce wood, plastic, orientation changes.
    Stage 5: Full generalization — all types × sizes × surfaces × orientations + DR.

    Returns:
        Ordered list of curriculum stages.
    """
    return [
        CurriculumStageDef(
            stage_id=1,
            name="Baseline",
            description="Single hex M6 screw on 30° metal solar panel (original task)",
            screw_types=["hex"],
            screw_sizes=["M6"],
            surface_materials=["metal"],
            surface_orientations=["tilt_30"],
            dr_scale=0.2,
            num_episodes=500,
            success_threshold=0.90,
            min_training_steps=5_000,
            eval_episodes=100,
        ),
        CurriculumStageDef(
            stage_id=2,
            name="Size variation",
            description="Hex screws across M3–M8 on metal solar panel",
            screw_types=["hex"],
            screw_sizes=["M3", "M4", "M5", "M6", "M8"],
            surface_materials=["metal"],
            surface_orientations=["tilt_30"],
            dr_scale=0.4,
            num_episodes=1000,
            success_threshold=0.85,
            min_training_steps=15_000,
            eval_episodes=100,
        ),
        CurriculumStageDef(
            stage_id=3,
            name="Type variation",
            description="Hex + Phillips + Torx screws, M4–M6, metal surface",
            screw_types=["hex", "phillips", "torx"],
            screw_sizes=["M4", "M5", "M6"],
            surface_materials=["metal", "aluminum"],
            surface_orientations=["tilt_30"],
            dr_scale=0.6,
            num_episodes=2000,
            success_threshold=0.80,
            min_training_steps=25_000,
            eval_episodes=100,
        ),
        CurriculumStageDef(
            stage_id=4,
            name="Surface & orientation variation",
            description="3 screw types × multiple surfaces × tilts",
            screw_types=["hex", "phillips", "torx", "flathead"],
            screw_sizes=["M4", "M5", "M6", "M8"],
            surface_materials=["metal", "aluminum", "wood", "plastic"],
            surface_orientations=["horizontal", "tilt_30", "tilt_45"],
            dr_scale=0.8,
            num_episodes=3000,
            success_threshold=0.75,
            min_training_steps=40_000,
            eval_episodes=100,
        ),
        CurriculumStageDef(
            stage_id=5,
            name="Full generalization",
            description="All screw types × all sizes × all surfaces × all orientations + full DR",
            screw_types=list(SCREW_TYPES.keys()),
            screw_sizes=list(SCREW_SIZES.keys()),
            surface_materials=list(SURFACE_MATERIALS.keys()),
            surface_orientations=list(SURFACE_ORIENTATIONS.keys()),
            dr_scale=1.0,
            num_episodes=5000,
            success_threshold=0.70,
            min_training_steps=80_000,
            eval_episodes=50,
        ),
    ]


# ---------------------------------------------------------------------------
# Generalization Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneralizationCfg:
    """Top-level configuration for screw skill generalization."""

    # Curriculum
    curriculum: list[CurriculumStageDef] = field(default_factory=build_default_curriculum)
    auto_advance: bool = True
    """Automatically advance curriculum stages when success_threshold is met."""

    # Domain randomization
    domain_randomization: DomainRandomizationCfg = field(
        default_factory=DomainRandomizationCfg
    )

    # Policy
    policy_type: str = "gr00t"
    """VLA policy to fine-tune: ``gr00t`` or ``openpi``."""
    base_checkpoint: str = "checkpoints/gr00t_screw_driving/best"
    """Path to the base checkpoint from initial screw-driving training."""

    # Output
    output_dir: str = "checkpoints/generalization"
    dataset_dir: str = "datasets/generalization"

    # Evaluation
    eval_after_each_stage: bool = True
    save_eval_videos: bool = True

    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Task Condition Sampler
# ---------------------------------------------------------------------------

@dataclass
class TaskCondition:
    """A specific task configuration sampled for one episode."""

    screw_type: str
    screw_size: str
    surface_material: str
    surface_orientation: str

    # Sampled domain randomization values
    position_offset_mm: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    orientation_offset_deg: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    light_intensity: float = 1.0
    light_color_temp_k: int = 5500
    friction_multiplier: float = 1.0

    def difficulty_score(self) -> float:
        """Compute a composite difficulty score for this condition."""
        screw = SCREW_TYPES.get(self.screw_type)
        size = SCREW_SIZES.get(self.screw_size)
        surface = SURFACE_MATERIALS.get(self.surface_material)
        orient = SURFACE_ORIENTATIONS.get(self.surface_orientation)

        score = 1.0
        if screw:
            score *= screw.difficulty
        if size:
            # Larger screws require more torque → harder
            score *= size.drive_torque_nm / 10.0
        if surface:
            score *= (1.0 + surface.deformation_risk)
        if orient:
            score *= (1.0 + orient.gravity_compensation)
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "screw_type": self.screw_type,
            "screw_size": self.screw_size,
            "surface_material": self.surface_material,
            "surface_orientation": self.surface_orientation,
            "difficulty_score": self.difficulty_score(),
            "light_intensity": self.light_intensity,
            "light_color_temp_k": self.light_color_temp_k,
            "friction_multiplier": self.friction_multiplier,
        }


class TaskConditionSampler:
    """Sample task conditions for a given curriculum stage."""

    def __init__(
        self,
        stage: CurriculumStageDef,
        dr_cfg: DomainRandomizationCfg,
        seed: int = 42,
    ):
        self.stage = stage
        self.dr_cfg = dr_cfg
        self.rng = np.random.default_rng(seed)

    def sample(self) -> TaskCondition:
        """Sample a single task condition for this curriculum stage.

        Returns:
            A :class:`TaskCondition` with randomized parameters.
        """
        screw_type = self.rng.choice(self.stage.screw_types)
        screw_size = self.rng.choice(self.stage.screw_sizes)
        surface_mat = self.rng.choice(self.stage.surface_materials)
        surface_ori = self.rng.choice(self.stage.surface_orientations)

        dr_s = self.stage.dr_scale

        pos_off = self.rng.uniform(
            -self.dr_cfg.position_jitter_mm * dr_s,
            self.dr_cfg.position_jitter_mm * dr_s,
            3,
        ).astype(np.float32)

        ori_off = self.rng.uniform(
            -self.dr_cfg.orientation_jitter_deg * dr_s,
            self.dr_cfg.orientation_jitter_deg * dr_s,
            3,
        ).astype(np.float32)

        lo, hi = self.dr_cfg.light_intensity_range
        light_int = self.rng.uniform(
            1.0 - (1.0 - lo) * dr_s,
            1.0 + (hi - 1.0) * dr_s,
        )

        clo, chi = self.dr_cfg.light_color_temp_range
        mid = (clo + chi) / 2
        light_ct = int(self.rng.uniform(
            mid - (mid - clo) * dr_s,
            mid + (chi - mid) * dr_s,
        ))

        flo, fhi = self.dr_cfg.friction_multiplier_range
        friction = self.rng.uniform(
            1.0 - (1.0 - flo) * dr_s,
            1.0 + (fhi - 1.0) * dr_s,
        )

        return TaskCondition(
            screw_type=screw_type,
            screw_size=screw_size,
            surface_material=surface_mat,
            surface_orientation=surface_ori,
            position_offset_mm=pos_off,
            orientation_offset_deg=ori_off,
            light_intensity=float(light_int),
            light_color_temp_k=light_ct,
            friction_multiplier=float(friction),
        )

    def sample_evaluation_grid(self) -> list[TaskCondition]:
        """Sample a structured evaluation grid for this stage.

        Generates one condition per combination of the stage's enabled
        dimensions (no domain randomization — deterministic evaluation).

        Returns:
            List of task conditions covering the evaluation grid.
        """
        conditions = []
        combos = list(itertools.product(
            self.stage.screw_types,
            self.stage.screw_sizes,
            self.stage.surface_materials,
            self.stage.surface_orientations,
        ))
        for st, ss, sm, so in combos:
            conditions.append(TaskCondition(
                screw_type=st,
                screw_size=ss,
                surface_material=sm,
                surface_orientation=so,
            ))
        return conditions


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

@dataclass
class DimensionMetrics:
    """Evaluation metrics broken down by a single generalization dimension."""

    dimension_name: str
    """Name of the dimension (e.g. ``screw_type``)."""
    per_value: dict[str, dict[str, float]] = field(default_factory=dict)
    """Metrics per value within the dimension.

    Example::

        {"hex": {"success_rate": 0.92, "mean_time_s": 8.3, ...}, ...}
    """

    def add_result(
        self,
        value: str,
        success: bool,
        completion_time_s: float,
        peak_force_n: float,
        collisions: int,
    ) -> None:
        """Record a single evaluation result."""
        if value not in self.per_value:
            self.per_value[value] = {
                "total": 0,
                "successes": 0,
                "success_rate": 0.0,
                "total_time_s": 0.0,
                "mean_time_s": 0.0,
                "max_force_n": 0.0,
                "total_collisions": 0,
            }
        v = self.per_value[value]
        v["total"] += 1
        if success:
            v["successes"] += 1
        v["success_rate"] = v["successes"] / v["total"]
        v["total_time_s"] += completion_time_s
        v["mean_time_s"] = v["total_time_s"] / v["total"]
        v["max_force_n"] = max(v["max_force_n"], peak_force_n)
        v["total_collisions"] += collisions

    def summary(self) -> dict[str, dict[str, float]]:
        return dict(self.per_value)


@dataclass
class GeneralizationEvalReport:
    """Full evaluation report across all generalization dimensions."""

    stage_id: int
    stage_name: str
    overall_success_rate: float = 0.0
    overall_mean_time_s: float = 0.0
    total_episodes: int = 0

    by_screw_type: DimensionMetrics = field(
        default_factory=lambda: DimensionMetrics("screw_type")
    )
    by_screw_size: DimensionMetrics = field(
        default_factory=lambda: DimensionMetrics("screw_size")
    )
    by_surface_material: DimensionMetrics = field(
        default_factory=lambda: DimensionMetrics("surface_material")
    )
    by_surface_orientation: DimensionMetrics = field(
        default_factory=lambda: DimensionMetrics("surface_orientation")
    )

    def record(
        self,
        condition: TaskCondition,
        success: bool,
        completion_time_s: float,
        peak_force_n: float,
        collisions: int,
    ) -> None:
        """Record one evaluation episode result."""
        self.total_episodes += 1

        for dim, value in [
            (self.by_screw_type, condition.screw_type),
            (self.by_screw_size, condition.screw_size),
            (self.by_surface_material, condition.surface_material),
            (self.by_surface_orientation, condition.surface_orientation),
        ]:
            dim.add_result(value, success, completion_time_s, peak_force_n, collisions)

        # Recompute overall
        all_success = sum(
            v["successes"] for v in self.by_screw_type.per_value.values()
        )
        all_total = sum(
            v["total"] for v in self.by_screw_type.per_value.values()
        )
        self.overall_success_rate = all_success / max(all_total, 1)

        all_time = sum(
            v["total_time_s"] for v in self.by_screw_type.per_value.values()
        )
        self.overall_mean_time_s = all_time / max(all_total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "overall_success_rate": self.overall_success_rate,
            "overall_mean_time_s": self.overall_mean_time_s,
            "total_episodes": self.total_episodes,
            "by_screw_type": self.by_screw_type.summary(),
            "by_screw_size": self.by_screw_size.summary(),
            "by_surface_material": self.by_surface_material.summary(),
            "by_surface_orientation": self.by_surface_orientation.summary(),
        }

    def to_markdown(self) -> str:
        """Render the evaluation report as Markdown."""
        lines = [
            f"# Generalization Evaluation — Stage {self.stage_id}: {self.stage_name}\n",
            f"**Total episodes:** {self.total_episodes}  ",
            f"**Overall success rate:** {self.overall_success_rate:.1%}  ",
            f"**Overall mean time:** {self.overall_mean_time_s:.2f}s\n",
        ]

        for dim_name, dim_metrics in [
            ("Screw Type", self.by_screw_type),
            ("Screw Size", self.by_screw_size),
            ("Surface Material", self.by_surface_material),
            ("Surface Orientation", self.by_surface_orientation),
        ]:
            lines.append(f"## By {dim_name}\n")
            lines.append(
                "| Value | Success Rate | Mean Time (s) | Max Force (N) | Collisions |"
            )
            lines.append(
                "|-------|-------------|---------------|---------------|------------|"
            )
            for val, m in sorted(dim_metrics.per_value.items()):
                lines.append(
                    f"| {val} | {m['success_rate']:.1%} | {m['mean_time_s']:.2f} "
                    f"| {m['max_force_n']:.1f} | {m['total_collisions']} |"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Curriculum Runner (offline / dry-run)
# ---------------------------------------------------------------------------

class CurriculumRunner:
    """Orchestrate the curriculum learning pipeline.

    In production this drives Isaac Sim data collection and VLA fine-tuning.
    Here we provide the planning and configuration generation logic.
    """

    def __init__(self, cfg: GeneralizationCfg | None = None):
        self.cfg = cfg or GeneralizationCfg()

    def print_curriculum(self) -> None:
        """Print a human-readable summary of the curriculum stages."""
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║        RoboForce Screw Skill Generalization Curriculum         ║")
        print("╠══════════════════════════════════════════════════════════════════╣")

        for stage in self.cfg.curriculum:
            combos = (
                len(stage.screw_types)
                * len(stage.screw_sizes)
                * len(stage.surface_materials)
                * len(stage.surface_orientations)
            )
            print(f"║                                                                  ║")
            print(f"║  Stage {stage.stage_id}: {stage.name:<54s}  ║")
            print(f"║    {stage.description:<60s}  ║")
            print(f"║    Types: {', '.join(stage.screw_types):<52s}  ║")
            print(f"║    Sizes: {', '.join(stage.screw_sizes):<52s}  ║")
            print(f"║    Surfaces: {', '.join(stage.surface_materials):<49s}  ║")
            print(f"║    Orientations: {', '.join(stage.surface_orientations):<45s}  ║")
            print(f"║    Combinations: {combos:<6d}  DR scale: {stage.dr_scale:.1f}"
                  f"  Threshold: {stage.success_threshold:.0%}{'':>9s}  ║")
            print(f"║    Episodes: {stage.num_episodes:<6d}  Steps: {stage.min_training_steps:<8d}"
                  f"{'':>22s}  ║")
            print(f"║{'':>66s}║")

        total_episodes = sum(s.num_episodes for s in self.cfg.curriculum)
        total_steps = sum(s.min_training_steps for s in self.cfg.curriculum)
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Total: {total_episodes} episodes, {total_steps} training steps"
              f"{'':>25s}  ║")
        print(f"║  Policy: {self.cfg.policy_type:<55s}  ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    def generate_stage_configs(self, output_dir: str | Path | None = None) -> list[str]:
        """Generate per-stage configuration files.

        Args:
            output_dir: Directory to write configs. Defaults to ``cfg.output_dir``.

        Returns:
            List of generated config file paths.
        """
        out = Path(output_dir or self.cfg.output_dir) / "stage_configs"
        out.mkdir(parents=True, exist_ok=True)

        paths = []
        for stage in self.cfg.curriculum:
            sampler = TaskConditionSampler(
                stage, self.cfg.domain_randomization, self.cfg.seed
            )
            eval_grid = sampler.sample_evaluation_grid()

            config = {
                "stage_id": stage.stage_id,
                "name": stage.name,
                "description": stage.description,
                "screw_types": stage.screw_types,
                "screw_sizes": stage.screw_sizes,
                "surface_materials": stage.surface_materials,
                "surface_orientations": stage.surface_orientations,
                "dr_scale": stage.dr_scale,
                "num_episodes": stage.num_episodes,
                "success_threshold": stage.success_threshold,
                "min_training_steps": stage.min_training_steps,
                "eval_episodes": stage.eval_episodes,
                "eval_grid_size": len(eval_grid),
                "policy_type": self.cfg.policy_type,
                "base_checkpoint": self.cfg.base_checkpoint,
                "domain_randomization": {
                    "position_jitter_mm": self.cfg.domain_randomization.position_jitter_mm * stage.dr_scale,
                    "orientation_jitter_deg": self.cfg.domain_randomization.orientation_jitter_deg * stage.dr_scale,
                    "friction_multiplier_range": list(self.cfg.domain_randomization.friction_multiplier_range),
                    "num_distractors": list(self.cfg.domain_randomization.num_distractor_objects),
                },
            }

            path = out / f"stage_{stage.stage_id}_{stage.name.lower().replace(' ', '_')}.json"
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
            paths.append(str(path))
            print(f"  Saved: {path}")

        return paths

    def estimate_compute(self) -> dict[str, Any]:
        """Estimate compute requirements for the full curriculum.

        Returns:
            Dict with time and GPU-hour estimates.
        """
        estimates = []
        for stage in self.cfg.curriculum:
            combos = (
                len(stage.screw_types)
                * len(stage.screw_sizes)
                * len(stage.surface_materials)
                * len(stage.surface_orientations)
            )

            # Rough estimates (based on typical VLA fine-tuning times)
            data_collection_hours = stage.num_episodes * 0.005  # ~18s per episode in sim
            training_hours = stage.min_training_steps * 0.001 / 60  # ~1ms per step
            eval_hours = combos * stage.eval_episodes * 0.003 / 60

            estimates.append({
                "stage": stage.stage_id,
                "name": stage.name,
                "combinations": combos,
                "data_collection_hours": round(data_collection_hours, 1),
                "training_hours": round(training_hours, 1),
                "eval_hours": round(eval_hours, 1),
                "total_hours": round(data_collection_hours + training_hours + eval_hours, 1),
            })

        total_hours = sum(e["total_hours"] for e in estimates)
        return {
            "stages": estimates,
            "total_gpu_hours": round(total_hours, 1),
            "estimated_wall_time_hours": round(total_hours * 1.2, 1),  # 20% overhead
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — Screw Skill Generalization"
    )
    parser.add_argument(
        "--show_curriculum", action="store_true",
        help="Display the curriculum learning stages",
    )
    parser.add_argument(
        "--generate_configs", action="store_true",
        help="Generate per-stage configuration files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for configs (default: from GeneralizationCfg)",
    )
    parser.add_argument(
        "--estimate_compute", action="store_true",
        help="Print compute estimates for the full curriculum",
    )
    parser.add_argument(
        "--evaluate_report", type=str, default=None,
        help="Path to evaluation results JSON — render as Markdown report",
    )
    parser.add_argument(
        "--policy", type=str, default="gr00t",
        choices=["gr00t", "openpi"],
        help="VLA policy type",
    )

    args = parser.parse_args()

    cfg = GeneralizationCfg(policy_type=args.policy)
    runner = CurriculumRunner(cfg)

    if args.show_curriculum:
        runner.print_curriculum()

    if args.generate_configs:
        print("Generating stage configs...")
        runner.generate_stage_configs(args.output_dir)

    if args.estimate_compute:
        est = runner.estimate_compute()
        print("\nCompute Estimates:")
        print(json.dumps(est, indent=2))

    if args.evaluate_report:
        path = Path(args.evaluate_report)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            report = GeneralizationEvalReport(
                stage_id=data.get("stage_id", 0),
                stage_name=data.get("stage_name", "unknown"),
            )
            # Re-render from saved data
            print(json.dumps(data, indent=2))
        else:
            print(f"File not found: {path}")

    if not any([
        args.show_curriculum, args.generate_configs,
        args.estimate_compute, args.evaluate_report,
    ]):
        runner.print_curriculum()
        est = runner.estimate_compute()
        print("Compute Estimates:")
        print(json.dumps(est, indent=2))


if __name__ == "__main__":
    main()
