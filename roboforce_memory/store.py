# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Pluggable memory store with Qdrant vector DB and in-memory fallback.

Provides CRUD operations for ScrewMemory entries with semantic similarity
search. Falls back to numpy-based cosine similarity when Qdrant is not
installed, enabling development and testing without external dependencies.

P99 retrieval target: < 200ms.

Usage:
    from roboforce_memory.store import MemoryStore

    store = MemoryStore()  # auto-detects backend
    store.create(memory)
    results = store.search(query_embedding, top_k=5)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from roboforce_memory.schema import ScrewMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Backend
# ---------------------------------------------------------------------------

class MemoryBackend(ABC):
    """Abstract interface for memory storage backends."""

    @abstractmethod
    def insert(self, memory: ScrewMemory) -> str:
        """Insert a memory entry. Returns the memory ID."""

    @abstractmethod
    def insert_batch(self, memories: list[ScrewMemory]) -> list[str]:
        """Insert multiple memories. Returns list of IDs."""

    @abstractmethod
    def get(self, memory_id: str) -> ScrewMemory | None:
        """Retrieve a single memory by ID."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ScrewMemory, float]]:
        """Search by embedding similarity. Returns (memory, score) pairs."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a single memory. Returns True if found and deleted."""

    @abstractmethod
    def delete_by_session(self, session_id: str) -> int:
        """Delete all memories for a session (GDPR-ready). Returns count."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored memories."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return store statistics."""


# ---------------------------------------------------------------------------
# In-Memory Numpy Backend (fallback)
# ---------------------------------------------------------------------------

class InMemoryBackend(MemoryBackend):
    """In-memory backend using numpy cosine similarity.

    Suitable for development, testing, and small-scale usage (< 100k memories).
    No external dependencies required.
    """

    def __init__(self) -> None:
        self._memories: dict[str, ScrewMemory] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    def insert(self, memory: ScrewMemory) -> str:
        self._memories[memory.id] = memory
        if memory.embedding is not None:
            self._embeddings[memory.id] = np.array(
                memory.embedding, dtype=np.float32
            )
        return memory.id

    def insert_batch(self, memories: list[ScrewMemory]) -> list[str]:
        return [self.insert(m) for m in memories]

    def get(self, memory_id: str) -> ScrewMemory | None:
        return self._memories.get(memory_id)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ScrewMemory, float]]:
        if not self._embeddings:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        query = query / query_norm

        # Filter candidates
        candidates = self._memories
        if filters:
            candidates = self._apply_filters(candidates, filters)

        # Compute cosine similarities
        scored: list[tuple[str, float]] = []
        for mid, memory in candidates.items():
            if mid not in self._embeddings:
                continue
            emb = self._embeddings[mid]
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-8:
                continue
            sim = float(np.dot(query, emb / emb_norm))
            scored.append((mid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for mid, score in scored[:top_k]:
            results.append((self._memories[mid], score))
        return results

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._embeddings.pop(memory_id, None)
            return True
        return False

    def delete_by_session(self, session_id: str) -> int:
        to_delete = [
            mid for mid, m in self._memories.items()
            if m.session_id == session_id
        ]
        for mid in to_delete:
            self.delete(mid)
        return len(to_delete)

    def count(self) -> int:
        return len(self._memories)

    def stats(self) -> dict[str, Any]:
        sessions = set(m.session_id for m in self._memories.values())
        success_count = sum(1 for m in self._memories.values() if m.success)
        total = len(self._memories)
        return {
            "backend": "in_memory",
            "total_memories": total,
            "total_sessions": len(sessions),
            "success_rate": success_count / total if total > 0 else 0.0,
            "with_embeddings": len(self._embeddings),
        }

    @staticmethod
    def _apply_filters(
        candidates: dict[str, ScrewMemory],
        filters: dict[str, Any],
    ) -> dict[str, ScrewMemory]:
        """Apply simple key-value filters on memory fields."""
        result = {}
        for mid, memory in candidates.items():
            match = True
            for key, value in filters.items():
                if key == "success" and memory.success != value:
                    match = False
                elif key == "subtask_label" and memory.subtask_label != value:
                    match = False
                elif key == "screw_type" and memory.screw_properties.screw_type != value:
                    match = False
                elif key == "screw_size" and memory.screw_properties.size != value:
                    match = False
                elif key == "environment_id" and memory.environment_id != value:
                    match = False
                elif key == "session_id" and memory.session_id != value:
                    match = False
            if match:
                result[mid] = memory
        return result


# ---------------------------------------------------------------------------
# Qdrant Backend (optional)
# ---------------------------------------------------------------------------

class QdrantBackend(MemoryBackend):
    """Qdrant vector database backend for production-scale memory storage.

    Requires: ``pip install qdrant-client``
    """

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        collection_name: str = "roboforce_memories",
        embedding_dim: int = 768,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                PointStruct,
                VectorParams,
            )
        except ImportError:
            raise ImportError(
                "Qdrant backend requires qdrant-client: "
                "pip install qdrant-client"
            )

        self._client = QdrantClient(host=url, port=port)
        self._collection = collection_name
        self._embedding_dim = embedding_dim
        self._PointStruct = PointStruct

        # Ensure collection exists
        collections = [
            c.name for c in self._client.get_collections().collections
        ]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def insert(self, memory: ScrewMemory) -> str:
        embedding = memory.embedding or [0.0] * self._embedding_dim
        payload = json.loads(memory.model_dump_json())

        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._PointStruct(
                    id=memory.id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        return memory.id

    def insert_batch(self, memories: list[ScrewMemory]) -> list[str]:
        points = []
        for memory in memories:
            embedding = memory.embedding or [0.0] * self._embedding_dim
            payload = json.loads(memory.model_dump_json())
            points.append(
                self._PointStruct(
                    id=memory.id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        return [m.id for m in memories]

    def get(self, memory_id: str) -> ScrewMemory | None:
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
            with_payload=True,
        )
        if not results:
            return None
        return ScrewMemory(**results[0].payload)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ScrewMemory, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            (ScrewMemory(**hit.payload), hit.score)
            for hit in results
        ]

    def delete(self, memory_id: str) -> bool:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[memory_id]),
        )
        return True

    def delete_by_session(self, session_id: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Count first
        count_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        )
        count = self._client.count(
            collection_name=self._collection,
            count_filter=count_filter,
        ).count

        self._client.delete(
            collection_name=self._collection,
            points_selector=count_filter,
        )
        return count

    def count(self) -> int:
        return self._client.count(
            collection_name=self._collection,
        ).count

    def stats(self) -> dict[str, Any]:
        info = self._client.get_collection(self._collection)
        return {
            "backend": "qdrant",
            "collection": self._collection,
            "total_memories": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }


# ---------------------------------------------------------------------------
# MemoryStore — High-Level API
# ---------------------------------------------------------------------------

class MemoryStore:
    """High-level memory store API with pluggable backend.

    Automatically falls back to in-memory numpy if Qdrant is unavailable.

    Args:
        backend: Explicit backend ("qdrant", "in_memory", or "auto").
        qdrant_url: Qdrant server host.
        qdrant_port: Qdrant server port.
        collection_name: Qdrant collection name.
        embedding_dim: Embedding vector dimension.
        persistence_path: Path for JSON-based persistence (in-memory backend).
    """

    def __init__(
        self,
        backend: str = "auto",
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "roboforce_memories",
        embedding_dim: int = 768,
        persistence_path: str | None = None,
    ):
        self._persistence_path = (
            Path(persistence_path) if persistence_path else None
        )
        self._embedding_dim = embedding_dim

        if backend == "auto":
            self._backend = self._auto_detect_backend(
                qdrant_url, qdrant_port, collection_name, embedding_dim
            )
        elif backend == "qdrant":
            self._backend = QdrantBackend(
                qdrant_url, qdrant_port, collection_name, embedding_dim
            )
        else:
            self._backend = InMemoryBackend()

        # Load persisted data if using in-memory backend
        if isinstance(self._backend, InMemoryBackend) and self._persistence_path:
            self._load_persisted()

    @staticmethod
    def _auto_detect_backend(
        url: str, port: int, collection: str, dim: int,
    ) -> MemoryBackend:
        """Try Qdrant first, fall back to in-memory."""
        try:
            return QdrantBackend(url, port, collection, dim)
        except Exception:
            logger.info(
                "Qdrant not available, using in-memory backend. "
                "Install qdrant-client and start Qdrant for production use."
            )
            return InMemoryBackend()

    # -- CRUD --

    def create(self, memory: ScrewMemory) -> str:
        """Store a new memory entry.

        Args:
            memory: The memory to store.

        Returns:
            The memory ID.
        """
        if not memory.id:
            memory.id = str(uuid.uuid4())
        result = self._backend.insert(memory)
        self._maybe_persist()
        return result

    def create_batch(self, memories: list[ScrewMemory]) -> list[str]:
        """Store multiple memory entries.

        Args:
            memories: List of memories to store.

        Returns:
            List of memory IDs.
        """
        for m in memories:
            if not m.id:
                m.id = str(uuid.uuid4())
        result = self._backend.insert_batch(memories)
        self._maybe_persist()
        return result

    def get(self, memory_id: str) -> ScrewMemory | None:
        """Retrieve a memory by ID."""
        return self._backend.get(memory_id)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ScrewMemory, float]]:
        """Search memories by semantic similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filters: Optional key-value filters (success, screw_type, etc.).

        Returns:
            List of (memory, similarity_score) tuples.
        """
        t0 = time.monotonic()
        results = self._backend.search(query_embedding, top_k, filters)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 200:
            logger.warning(
                f"Search took {elapsed_ms:.0f}ms (target P99 < 200ms)"
            )
        return results

    def delete(self, memory_id: str) -> bool:
        """Delete a single memory."""
        result = self._backend.delete(memory_id)
        self._maybe_persist()
        return result

    def delete_by_session(self, session_id: str) -> int:
        """Delete all memories for a session (GDPR-ready).

        Args:
            session_id: The session whose memories to delete.

        Returns:
            Number of deleted memories.
        """
        count = self._backend.delete_by_session(session_id)
        self._maybe_persist()
        return count

    def count(self) -> int:
        """Return total number of stored memories."""
        return self._backend.count()

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        return self._backend.stats()

    # -- Persistence (in-memory backend only) --

    def _maybe_persist(self) -> None:
        """Persist in-memory data to disk if configured."""
        if not self._persistence_path:
            return
        if not isinstance(self._backend, InMemoryBackend):
            return
        self._save_persisted()

    def _save_persisted(self) -> None:
        """Save all in-memory data to a JSON file."""
        if not isinstance(self._backend, InMemoryBackend):
            return
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            json.loads(m.model_dump_json())
            for m in self._backend._memories.values()
        ]
        with open(self._persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_persisted(self) -> None:
        """Load persisted data from a JSON file."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        with open(self._persistence_path) as f:
            data = json.load(f)
        for entry in data:
            memory = ScrewMemory(**entry)
            self._backend.insert(memory)
        logger.info(f"Loaded {len(data)} memories from {self._persistence_path}")

    def ingest_from_dataset(
        self,
        dataset_path: str,
        session_id: str | None = None,
        environment_id: str = "sim",
    ) -> int:
        """Bulk ingest memories from a roboforce_v2 or lerobot_v3 dataset.

        Args:
            dataset_path: Path to the dataset directory.
            session_id: Session ID to assign (auto-generated if None).
            environment_id: Environment identifier.

        Returns:
            Number of memories ingested.
        """
        from roboforce_memory.summarizer import MemorySummarizer

        path = Path(dataset_path)
        if session_id is None:
            session_id = str(uuid.uuid4())

        summarizer = MemorySummarizer()
        count = 0

        # Try episode directories first
        ep_dirs = sorted(
            d for d in path.iterdir()
            if d.is_dir() and d.name.startswith("ep_")
        )

        if not ep_dirs:
            # Try data/ subdirectory (V3 format)
            data_dir = path / "data"
            if data_dir.exists():
                ep_dirs = sorted(data_dir.glob("episode_*.jsonl"))

        if not ep_dirs:
            # Fall back to flat file
            flat = path / "all_episodes.jsonl"
            if flat.exists():
                count = self._ingest_flat(
                    flat, session_id, environment_id, summarizer
                )
                return count

            logger.warning(f"No episodes found in {dataset_path}")
            return 0

        for ep_dir in ep_dirs:
            if ep_dir.is_dir():
                obs_path = ep_dir / "observations.jsonl"
            else:
                obs_path = ep_dir  # JSONL file directly

            if not obs_path.exists():
                continue

            frames = []
            with open(obs_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        frames.append(json.loads(line))

            if not frames:
                continue

            memory = self._frames_to_memory(
                frames, session_id, environment_id, summarizer
            )
            self.create(memory)
            count += 1

        logger.info(f"Ingested {count} memories from {dataset_path}")
        return count

    def _ingest_flat(
        self,
        flat_path: Path,
        session_id: str,
        environment_id: str,
        summarizer: Any,
    ) -> int:
        """Ingest from a flat all_episodes.jsonl file."""
        episodes: dict[int, list[dict]] = {}
        with open(flat_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                ep_idx = frame.get("episode_index", 0)
                episodes.setdefault(ep_idx, []).append(frame)

        count = 0
        for ep_idx in sorted(episodes.keys()):
            frames = episodes[ep_idx]
            memory = self._frames_to_memory(
                frames, session_id, environment_id, summarizer
            )
            self.create(memory)
            count += 1

        return count

    @staticmethod
    def _frames_to_memory(
        frames: list[dict],
        session_id: str,
        environment_id: str,
        summarizer: Any,
    ) -> ScrewMemory:
        """Convert a list of frames into a single ScrewMemory entry."""
        first = frames[0]
        last = frames[-1]

        # Determine success and phase
        success = last.get("done", False) and last.get("phase") == "DONE"
        final_phase = last.get("phase", "APPROACH")

        # Collect actions (up to 50 steps)
        actions = [f.get("action", [0.0] * 8) for f in frames[:50]]

        # F/T data from last frame
        wrist_ft = last.get("observation.ft.wrist_ft.wrench", [0.0] * 6)
        ee_ft = last.get("observation.ft.ee_tip_ft.wrench", [0.0] * 6)

        # Peak forces across all frames
        peak_force = 0.0
        peak_torque = 0.0
        for f in frames:
            w = f.get("observation.ft.wrist_ft.wrench", [0.0] * 6)
            force_mag = (w[0] ** 2 + w[1] ** 2 + w[2] ** 2) ** 0.5
            torque_mag = (w[3] ** 2 + w[4] ** 2 + w[5] ** 2) ** 0.5
            peak_force = max(peak_force, force_mag)
            peak_torque = max(peak_torque, torque_mag)

        # Build subtask label from most common phase
        from collections import Counter
        phase_counts = Counter(f.get("phase", "APPROACH") for f in frames)
        dominant_phase = phase_counts.most_common(1)[0][0]

        from roboforce_skills.pi05_data_converter import PHASE_TO_SUBTASK
        subtask = PHASE_TO_SUBTASK.get(dominant_phase, "approach_screw")

        # Determine failure reason
        failure_reason = None
        if not success:
            if peak_force > 50.0:
                failure_reason = "force_exceeded"
            elif final_phase in ("APPROACH", "ALIGN"):
                failure_reason = "alignment_fail"
            else:
                failure_reason = "timeout"

        # Total reward
        total_reward = sum(f.get("reward", 0.0) for f in frames)

        memory = ScrewMemory(
            id=str(uuid.uuid4()),
            session_id=session_id,
            environment_id=environment_id,
            observation=ObservationData(
                state=first.get("observation.state", []),
                language_instruction=(
                    "Pick up the screw and drive it into the solar panel "
                    "mounting bracket"
                ),
            ),
            action_chunk=actions,
            subtask_label=subtask,
            force_feedback=ForceFeedbackData(
                wrist_ft_wrench=wrist_ft,
                ee_tip_ft_wrench=ee_ft,
                peak_force_n=peak_force,
                peak_torque_nm=peak_torque,
            ),
            success=success,
            reward=total_reward,
            failure_reason=failure_reason,
            screw_properties=ScrewProperties(),
        )

        # Auto-summarize
        memory.summary = summarizer.summarize(memory)

        return memory


# Avoid circular import at module level
from roboforce_memory.schema import ObservationData  # noqa: E402
