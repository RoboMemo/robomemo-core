# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""ROS 2 integration stubs for RoboMemo memory events.

Provides publisher/subscriber interfaces and topic definitions for
real-time memory creation and querying during robot operation.

NOTE: This module provides interface stubs. Full ROS 2 integration
requires rclpy and a running ROS 2 environment.

Topics:
    /roboforce/memory/create    (roboforce_msgs/MemoryCreate)
    /roboforce/memory/query     (roboforce_msgs/MemoryQuery)
    /roboforce/memory/result    (roboforce_msgs/MemoryResult)
    /roboforce/memory/event     (roboforce_msgs/MemoryEvent)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from roboforce_memory.schema import ScrewMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topic Definitions
# ---------------------------------------------------------------------------

@dataclass
class TopicDef:
    """ROS 2 topic definition."""

    name: str
    """Fully qualified topic name."""
    msg_type: str
    """Message type (e.g. roboforce_msgs/MemoryCreate)."""
    qos_depth: int = 10
    """QoS history depth."""


MEMORY_TOPICS = {
    "create": TopicDef(
        name="/roboforce/memory/create",
        msg_type="roboforce_msgs/MemoryCreate",
    ),
    "query": TopicDef(
        name="/roboforce/memory/query",
        msg_type="roboforce_msgs/MemoryQuery",
    ),
    "result": TopicDef(
        name="/roboforce/memory/result",
        msg_type="roboforce_msgs/MemoryResult",
        qos_depth=5,
    ),
    "event": TopicDef(
        name="/roboforce/memory/event",
        msg_type="roboforce_msgs/MemoryEvent",
        qos_depth=20,
    ),
}


# ---------------------------------------------------------------------------
# Message Stubs
# ---------------------------------------------------------------------------

@dataclass
class MemoryCreateMsg:
    """Message for creating a new memory entry via ROS 2."""

    memory_json: str = ""
    """JSON-serialized ScrewMemory."""
    session_id: str = ""
    environment_id: str = ""


@dataclass
class MemoryQueryMsg:
    """Message for querying memories via ROS 2."""

    query_embedding: list[float] = field(default_factory=list)
    """Dense embedding vector for similarity search."""
    top_k: int = 5
    filters_json: str = "{}"
    """JSON-serialized filter dict."""
    request_id: str = ""
    """Unique request ID for matching responses."""


@dataclass
class MemoryResultMsg:
    """Message containing memory query results."""

    request_id: str = ""
    results_json: str = "[]"
    """JSON array of {memory, score} objects."""
    num_results: int = 0


@dataclass
class MemoryEventMsg:
    """Memory lifecycle event notification."""

    event_type: str = ""
    """Event: created, deleted, ingested, search_completed."""
    memory_id: str = ""
    session_id: str = ""
    details: str = ""


# ---------------------------------------------------------------------------
# ROS 2 Bridge (Stub)
# ---------------------------------------------------------------------------

class ROS2MemoryBridge:
    """ROS 2 publisher/subscriber bridge for memory operations.

    This is a stub implementation that logs operations. Replace with
    actual rclpy node when deploying on a ROS 2 system.

    Usage:
        bridge = ROS2MemoryBridge(store)
        bridge.start()
        # ... robot runs, memories flow ...
        bridge.stop()
    """

    def __init__(
        self,
        store: Any | None = None,
        node_name: str = "robomemo_bridge",
    ):
        self._store = store
        self._node_name = node_name
        self._running = False
        self._callbacks: dict[str, list[Callable]] = {
            "create": [],
            "query": [],
            "result": [],
            "event": [],
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the ROS 2 bridge node.

        In production, this creates an rclpy node with publishers and
        subscribers for all memory topics.
        """
        logger.info(
            f"ROS2MemoryBridge '{self._node_name}' starting (stub mode)"
        )
        self._running = True

        # In production:
        # rclpy.init()
        # self._node = rclpy.create_node(self._node_name)
        # self._create_sub = self._node.create_subscription(...)
        # self._query_sub = self._node.create_subscription(...)
        # self._result_pub = self._node.create_publisher(...)
        # self._event_pub = self._node.create_publisher(...)

        logger.info("ROS 2 bridge started (stub — no actual ROS 2 connection)")

    def stop(self) -> None:
        """Stop the ROS 2 bridge node."""
        self._running = False
        logger.info(f"ROS2MemoryBridge '{self._node_name}' stopped")

    def publish_create(self, memory: ScrewMemory) -> None:
        """Publish a memory creation event.

        Args:
            memory: The newly created memory.
        """
        msg = MemoryCreateMsg(
            memory_json=memory.model_dump_json(),
            session_id=memory.session_id,
            environment_id=memory.environment_id,
        )
        logger.debug(f"[pub] memory/create: id={memory.id}")
        self._dispatch("create", msg)

    def publish_query(self, query: MemoryQueryMsg) -> None:
        """Publish a memory query request.

        Args:
            query: The query message.
        """
        logger.debug(
            f"[pub] memory/query: request_id={query.request_id}, "
            f"top_k={query.top_k}"
        )
        self._dispatch("query", query)

    def publish_result(self, result: MemoryResultMsg) -> None:
        """Publish query results.

        Args:
            result: The result message.
        """
        logger.debug(
            f"[pub] memory/result: request_id={result.request_id}, "
            f"n={result.num_results}"
        )
        self._dispatch("result", result)

    def publish_event(self, event: MemoryEventMsg) -> None:
        """Publish a memory lifecycle event.

        Args:
            event: The event message.
        """
        logger.debug(
            f"[pub] memory/event: type={event.event_type}, "
            f"id={event.memory_id}"
        )
        self._dispatch("event", event)

    def on_create(self, callback: Callable[[MemoryCreateMsg], None]) -> None:
        """Register a callback for memory creation messages."""
        self._callbacks["create"].append(callback)

    def on_query(self, callback: Callable[[MemoryQueryMsg], None]) -> None:
        """Register a callback for memory query messages."""
        self._callbacks["query"].append(callback)

    def on_result(self, callback: Callable[[MemoryResultMsg], None]) -> None:
        """Register a callback for memory result messages."""
        self._callbacks["result"].append(callback)

    def on_event(self, callback: Callable[[MemoryEventMsg], None]) -> None:
        """Register a callback for memory event messages."""
        self._callbacks["event"].append(callback)

    def _dispatch(self, topic: str, msg: Any) -> None:
        """Dispatch message to registered callbacks."""
        for cb in self._callbacks.get(topic, []):
            try:
                cb(msg)
            except Exception:
                logger.exception(f"Callback error on topic '{topic}'")
