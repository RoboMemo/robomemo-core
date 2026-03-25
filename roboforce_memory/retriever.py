# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Smart retrieval with semantic similarity, time-decay, and filtering.

Combines dense embedding similarity with temporal recency weighting to
surface the most relevant past experiences for the current observation.

Usage:
    from roboforce_memory.retriever import MemoryRetriever
    from roboforce_memory.store import MemoryStore

    store = MemoryStore()
    retriever = MemoryRetriever(store)
    results = retriever.retrieve(
        query_embedding=current_embedding,
        top_k=5,
        success_only=True,
        screw_type="hex",
    )
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from roboforce_memory.schema import ScrewMemory
from roboforce_memory.store import MemoryStore


class MemoryRetriever:
    """Smart memory retrieval with similarity search and time-decay weighting.

    Args:
        store: The memory store to query.
        time_decay_hours: Half-life for time-decay weighting (hours).
            Memories older than this get half the temporal weight.
        similarity_weight: Weight for embedding similarity (0-1).
        time_weight: Weight for temporal recency (0-1).
    """

    def __init__(
        self,
        store: MemoryStore,
        time_decay_hours: float = 24.0,
        similarity_weight: float = 0.7,
        time_weight: float = 0.3,
    ):
        self.store = store
        self.time_decay_hours = time_decay_hours
        self.similarity_weight = similarity_weight
        self.time_weight = time_weight

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        success_only: bool = False,
        failure_only: bool = False,
        screw_type: str | None = None,
        screw_size: str | None = None,
        subtask_label: str | None = None,
        environment_id: str | None = None,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> list[tuple[ScrewMemory, float]]:
        """Retrieve top-K memories by combined similarity and recency score.

        Args:
            query_embedding: Dense embedding of the current observation.
            top_k: Number of results to return.
            success_only: Only return successful memories.
            failure_only: Only return failed memories.
            screw_type: Filter by screw head type.
            screw_size: Filter by metric screw size.
            subtask_label: Filter by subtask label.
            environment_id: Filter by environment.
            session_id: Filter by session.
            now: Current time for time-decay (defaults to datetime.now()).

        Returns:
            List of (memory, combined_score) tuples, sorted by score descending.
        """
        if now is None:
            now = datetime.now()

        # Build filters
        filters: dict[str, Any] = {}
        if success_only:
            filters["success"] = True
        if failure_only:
            filters["success"] = False
        if screw_type:
            filters["screw_type"] = screw_type
        if screw_size:
            filters["screw_size"] = screw_size
        if subtask_label:
            filters["subtask_label"] = subtask_label
        if environment_id:
            filters["environment_id"] = environment_id
        if session_id:
            filters["session_id"] = session_id

        # Fetch more than needed so we can re-rank with time decay
        fetch_k = min(top_k * 3, top_k + 50)
        raw_results = self.store.search(
            query_embedding, top_k=fetch_k, filters=filters
        )

        if not raw_results:
            return []

        # Re-rank with time-decay weighting
        scored: list[tuple[ScrewMemory, float]] = []
        for memory, sim_score in raw_results:
            time_score = self._time_decay_score(memory.timestamp, now)
            combined = (
                self.similarity_weight * sim_score
                + self.time_weight * time_score
            )
            scored.append((memory, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_similar_failures(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[tuple[ScrewMemory, float]]:
        """Retrieve similar past failures for avoidance learning.

        Useful for conditioning the policy to avoid repeating mistakes.

        Args:
            query_embedding: Current observation embedding.
            top_k: Number of failure memories to return.

        Returns:
            List of (memory, score) tuples for failed attempts.
        """
        return self.retrieve(
            query_embedding, top_k=top_k, failure_only=True
        )

    def retrieve_successful_demonstrations(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        screw_type: str | None = None,
    ) -> list[tuple[ScrewMemory, float]]:
        """Retrieve successful demonstrations for the current situation.

        Args:
            query_embedding: Current observation embedding.
            top_k: Number of success memories to return.
            screw_type: Optional screw type filter.

        Returns:
            List of (memory, score) tuples for successful attempts.
        """
        return self.retrieve(
            query_embedding,
            top_k=top_k,
            success_only=True,
            screw_type=screw_type,
        )

    def _time_decay_score(
        self,
        timestamp: datetime,
        now: datetime,
    ) -> float:
        """Compute exponential time-decay score.

        Uses half-life formula: score = exp(-ln(2) * age_hours / half_life).

        Args:
            timestamp: Memory creation time.
            now: Current time.

        Returns:
            Score in [0, 1], where 1 = now, 0.5 = one half-life ago.
        """
        age_seconds = (now - timestamp).total_seconds()
        age_hours = max(age_seconds / 3600.0, 0.0)
        decay = math.exp(-math.log(2) * age_hours / self.time_decay_hours)
        return decay
