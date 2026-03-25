# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Auto-summarization for screw driving memory entries.

Compresses trajectory data into structured text summaries for human
readability and for use as context in chain-of-thought prompting.

Example output:
    "M4 hex screw at 15deg tilt, 2.5Nm torque, 8 turns, success in 12.3s"
"""

from __future__ import annotations

from roboforce_memory.schema import ScrewMemory


class MemorySummarizer:
    """Generate concise text summaries of ScrewMemory entries."""

    def summarize(self, memory: ScrewMemory) -> str:
        """Produce a one-line structured summary of a memory.

        Args:
            memory: The memory to summarize.

        Returns:
            Human-readable summary string.
        """
        screw = memory.screw_properties
        ft = memory.force_feedback
        outcome = "success" if memory.success else "fail"

        # Duration estimate from action chunk length (at 50Hz)
        num_steps = len(memory.action_chunk)
        duration_s = num_steps * 0.02  # 50Hz control

        parts = [
            f"{screw.size} {screw.screw_type} screw",
        ]

        if ft.peak_torque_nm > 0:
            parts.append(f"{ft.peak_torque_nm:.1f}Nm peak torque")

        if ft.peak_force_n > 0:
            parts.append(f"{ft.peak_force_n:.1f}N peak force")

        parts.append(f"{outcome} in {duration_s:.1f}s")

        if memory.subtask_label:
            parts.append(f"phase={memory.subtask_label}")

        if memory.failure_reason:
            parts.append(f"reason={memory.failure_reason}")

        if ft.contact_detected:
            parts.append("contact_detected")
        if ft.slip_detected:
            parts.append("slip_detected")

        return ", ".join(parts)

    def summarize_batch(self, memories: list[ScrewMemory]) -> str:
        """Summarize a batch of memories into a multi-line report.

        Args:
            memories: List of memories.

        Returns:
            Multi-line summary.
        """
        if not memories:
            return "No memories."

        lines = [f"Batch summary ({len(memories)} memories):"]
        success_count = sum(1 for m in memories if m.success)
        lines.append(
            f"  Success rate: {success_count}/{len(memories)} "
            f"({success_count / len(memories):.0%})"
        )

        # Group by subtask
        by_subtask: dict[str, list[ScrewMemory]] = {}
        for m in memories:
            by_subtask.setdefault(m.subtask_label or "unknown", []).append(m)

        for subtask, mems in sorted(by_subtask.items()):
            sr = sum(1 for m in mems if m.success) / len(mems)
            avg_torque = sum(
                m.force_feedback.peak_torque_nm for m in mems
            ) / len(mems)
            lines.append(
                f"  {subtask}: {len(mems)} entries, "
                f"success={sr:.0%}, avg_torque={avg_torque:.2f}Nm"
            )

        return "\n".join(lines)
