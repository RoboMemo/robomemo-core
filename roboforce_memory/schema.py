# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Pydantic data models for RoboMemo episodic memory entries.

Defines the core schema for storing screw driving experiences including
observations, action chunks, force feedback, and outcome metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ScrewProperties(BaseModel):
    """Physical properties of the screw being installed."""

    screw_type: str = Field(
        default="hex",
        description="Screw head type: hex, phillips, flathead, torx, robertson.",
    )
    size: str = Field(
        default="M4",
        description="Metric screw size: M3, M4, M5, M6, M8, M10, M12.",
    )
    material: str = Field(
        default="steel",
        description="Screw material: steel, stainless, brass, aluminum.",
    )
    length_mm: float = Field(
        default=16.0,
        description="Screw shaft length in mm.",
    )
    thread_pitch_mm: float = Field(
        default=0.7,
        description="Thread pitch in mm.",
    )
    target_torque_nm: float = Field(
        default=2.5,
        description="Target tightening torque in N-m.",
    )


class ForceFeedbackData(BaseModel):
    """Force/torque sensor readings from wrist and end-effector tip."""

    wrist_ft_wrench: list[float] = Field(
        default_factory=lambda: [0.0] * 6,
        description="Wrist F/T wrench [Fx, Fy, Fz, Tx, Ty, Tz] in N and N-m.",
    )
    ee_tip_ft_wrench: list[float] = Field(
        default_factory=lambda: [0.0] * 6,
        description="EE tip F/T wrench [Fx, Fy, Fz, Tx, Ty, Tz] in N and N-m.",
    )
    peak_force_n: float = Field(
        default=0.0,
        description="Peak force magnitude during the action (N).",
    )
    peak_torque_nm: float = Field(
        default=0.0,
        description="Peak torque magnitude during the action (N-m).",
    )
    contact_detected: bool = Field(
        default=False,
        description="Whether screw-surface contact was detected.",
    )
    slip_detected: bool = Field(
        default=False,
        description="Whether lateral slip was detected.",
    )


class ObservationData(BaseModel):
    """Observation data at the time of memory creation."""

    image_refs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Camera name → image file path. "
            "Keys: head_left, head_right, wrist."
        ),
    )
    state: list[float] = Field(
        default_factory=list,
        description="Proprioceptive state vector (joints + EE pose + F/T).",
    )
    language_instruction: str = Field(
        default="",
        description="Natural language task instruction.",
    )
    ee_position: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="End-effector position [x, y, z] in meters.",
    )
    ee_orientation: list[float] = Field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0],
        description="End-effector orientation quaternion [w, x, y, z].",
    )


class ScrewMemory(BaseModel):
    """A single episodic memory entry for screw driving experience.

    Each memory captures one action segment (typically one subtask phase)
    with full observation context, action chunk, force feedback, and outcome.
    """

    id: str = Field(
        description="Unique memory identifier (UUID).",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this memory was created.",
    )
    session_id: str = Field(
        default="",
        description="Collection session identifier.",
    )
    environment_id: str = Field(
        default="",
        description="Environment / scene identifier.",
    )

    # Core data
    observation: ObservationData = Field(
        default_factory=ObservationData,
        description="Observation at memory creation time.",
    )
    action_chunk: list[list[float]] = Field(
        default_factory=list,
        description="50-step action chunk, each step is 8D.",
    )
    subtask_label: str = Field(
        default="",
        description=(
            "Current subtask: approach_screw, align_driver, insert_screw, "
            "tighten_clockwise, verify_torque."
        ),
    )
    force_feedback: ForceFeedbackData = Field(
        default_factory=ForceFeedbackData,
        description="F/T sensor data from wrist and EE tip.",
    )

    # Outcome
    success: bool = Field(
        default=False,
        description="Whether this action segment succeeded.",
    )
    reward: float = Field(
        default=0.0,
        description="Scalar reward for this segment.",
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description=(
            "If failed: alignment_fail, force_exceeded, slip, timeout, etc."
        ),
    )

    # Screw context
    screw_properties: ScrewProperties = Field(
        default_factory=ScrewProperties,
        description="Physical properties of the screw.",
    )

    # Retrieval support
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Dense embedding vector for semantic search.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Auto-generated text summary of this memory.",
    )
