# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Command-line interface for RoboMemo memory operations.

Usage:
    python -m roboforce_memory.cli ingest --dataset /path/to/dataset
    python -m roboforce_memory.cli search --query "M4 screw alignment"
    python -m roboforce_memory.cli stats
    python -m roboforce_memory.cli delete --session <session_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from roboforce_memory.store import MemoryStore
from roboforce_memory.summarizer import MemorySummarizer

logger = logging.getLogger(__name__)


def _get_store(args: argparse.Namespace) -> MemoryStore:
    """Create a MemoryStore from CLI args."""
    return MemoryStore(
        backend=getattr(args, "backend", "auto"),
        persistence_path=getattr(args, "persistence_path", None),
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Bulk ingest memories from a collected dataset."""
    store = _get_store(args)
    count = store.ingest_from_dataset(
        dataset_path=args.dataset,
        session_id=args.session_id,
        environment_id=args.environment_id or "sim",
    )
    print(f"Ingested {count} memories from {args.dataset}")
    if args.persistence_path:
        print(f"Persisted to {args.persistence_path}")


def cmd_search(args: argparse.Namespace) -> None:
    """Search memories by text query.

    Uses a simple hash-based pseudo-embedding for the query text. In
    production, replace with a real text encoder (e.g. SentenceTransformers).
    """
    store = _get_store(args)

    # Pseudo-embedding from query text (deterministic hash-based)
    rng = np.random.default_rng(hash(args.query) % (2**32))
    query_embedding = rng.standard_normal(768).tolist()

    filters = {}
    if args.success_only:
        filters["success"] = True
    if args.screw_type:
        filters["screw_type"] = args.screw_type

    from roboforce_memory.retriever import MemoryRetriever
    retriever = MemoryRetriever(store)
    results = retriever.retrieve(
        query_embedding=query_embedding,
        top_k=args.top_k,
        success_only=args.success_only,
        screw_type=args.screw_type,
    )

    if not results:
        print("No memories found.")
        return

    summarizer = MemorySummarizer()
    print(f"Search results for: \"{args.query}\" (top {args.top_k})")
    print("-" * 60)
    for i, (memory, score) in enumerate(results):
        summary = memory.summary or summarizer.summarize(memory)
        print(f"  {i+1}. [score={score:.3f}] {summary}")
        print(f"     id={memory.id}, session={memory.session_id}")
    print()


def cmd_stats(args: argparse.Namespace) -> None:
    """Show memory store statistics."""
    store = _get_store(args)
    stats = store.stats()
    print("Memory Store Statistics")
    print("=" * 40)
    print(json.dumps(stats, indent=2))


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete memories by session ID."""
    store = _get_store(args)

    if not args.session:
        print("Error: --session is required for delete")
        sys.exit(1)

    if not args.force:
        confirm = input(
            f"Delete all memories for session '{args.session}'? [y/N] "
        )
        if confirm.lower() != "y":
            print("Aborted.")
            return

    count = store.delete_by_session(args.session)
    print(f"Deleted {count} memories for session '{args.session}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m roboforce_memory.cli",
        description="RoboMemo — Memory operations CLI",
    )
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "qdrant", "in_memory"],
        help="Storage backend (default: auto)",
    )
    parser.add_argument(
        "--persistence_path", type=str, default=None,
        help="JSON file path for in-memory backend persistence",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Bulk ingest from dataset")
    p_ingest.add_argument(
        "--dataset", type=str, required=True,
        help="Path to dataset directory",
    )
    p_ingest.add_argument(
        "--session_id", type=str, default=None,
        help="Session ID to assign (auto-generated if omitted)",
    )
    p_ingest.add_argument(
        "--environment_id", type=str, default="sim",
        help="Environment identifier (default: sim)",
    )

    # search
    p_search = subparsers.add_parser("search", help="Search memories")
    p_search.add_argument(
        "--query", type=str, required=True,
        help="Search query text",
    )
    p_search.add_argument(
        "--top_k", type=int, default=5,
        help="Number of results (default: 5)",
    )
    p_search.add_argument(
        "--success_only", action="store_true",
        help="Only return successful memories",
    )
    p_search.add_argument(
        "--screw_type", type=str, default=None,
        help="Filter by screw type (hex, phillips, etc.)",
    )

    # stats
    subparsers.add_parser("stats", help="Show memory statistics")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete memories by session")
    p_delete.add_argument(
        "--session", type=str, required=True,
        help="Session ID to delete",
    )
    p_delete.add_argument(
        "--force", "-f", action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "ingest": cmd_ingest,
        "search": cmd_search,
        "stats": cmd_stats,
        "delete": cmd_delete,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
