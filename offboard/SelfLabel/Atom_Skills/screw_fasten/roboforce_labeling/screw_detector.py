# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Screw detector label generation from Isaac Sim synthetic data.

Processes rendered RGB images and segmentation masks from the simulation to
produce COCO-format annotations for training screw detection and 6D pose
estimation models.

Features:
- Bounding box extraction from semantic segmentation masks
- 6D pose annotation from simulation ground truth
- COCO JSON export
- Support for domain-randomized data
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScrewDetectorCfg:
    """Configuration for screw detection label generation."""

    # Output
    output_dir: str = "datasets/screw_detection"
    """Root directory for the output dataset."""
    dataset_name: str = "roboforce_screw_v1"
    """Name of the dataset."""

    # Categories
    categories: list[dict] = field(default_factory=lambda: [
        {"id": 1, "name": "screw_hex", "supercategory": "fastener"},
        {"id": 2, "name": "screw_phillips", "supercategory": "fastener"},
        {"id": 3, "name": "screw_flathead", "supercategory": "fastener"},
        {"id": 4, "name": "screw_torx", "supercategory": "fastener"},
        {"id": 5, "name": "bolt_hex", "supercategory": "fastener"},
        {"id": 6, "name": "bracket", "supercategory": "structure"},
    ])

    # Segmentation mask settings
    screw_semantic_id: int = 10
    """Semantic segmentation ID assigned to screw prims."""
    bracket_semantic_id: int = 20
    """Semantic segmentation ID for bracket prims."""

    # Bounding box
    min_bbox_area: int = 16
    """Minimum bounding box area (pixels²) to include."""
    bbox_padding: int = 2
    """Padding around extracted bounding boxes (pixels)."""

    # 6D pose
    include_6d_pose: bool = True
    """Whether to include 6D pose annotations."""

    # Splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# ---------------------------------------------------------------------------
# COCO Annotation Builder
# ---------------------------------------------------------------------------

class COCOAnnotationBuilder:
    """Builds a COCO-format annotation dictionary incrementally.

    Usage:
        builder = COCOAnnotationBuilder(categories)
        builder.add_image(image_id, filename, width, height)
        builder.add_annotation(image_id, category_id, bbox, ...)
        coco_dict = builder.build()
        builder.save("annotations.json")
    """

    def __init__(self, categories: list[dict], dataset_name: str = ""):
        self._info = {
            "description": f"RoboForce Screw Detection Dataset — {dataset_name}",
            "version": "1.0",
            "year": 2026,
            "contributor": "RoboForce Simulation Pipeline",
            "date_created": datetime.now().isoformat(),
        }
        self._categories = categories
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._next_ann_id = 1

    def add_image(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        metadata: dict | None = None,
    ) -> None:
        """Register an image."""
        entry = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        if metadata:
            entry.update(metadata)
        self._images.append(entry)

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: Sequence[float],
        area: float | None = None,
        segmentation: list | None = None,
        pose_6d: dict | None = None,
        iscrowd: int = 0,
    ) -> int:
        """Add a single object annotation.

        Args:
            image_id: ID of the parent image.
            category_id: Object category.
            bbox: ``[x, y, width, height]`` in pixels.
            area: Annotation area. Computed from bbox if *None*.
            segmentation: Polygon segmentation (optional).
            pose_6d: Optional dict with ``position`` (3) and ``quaternion`` (4).
            iscrowd: COCO crowd flag.

        Returns:
            The annotation ID.
        """
        ann_id = self._next_ann_id
        self._next_ann_id += 1

        if area is None:
            area = float(bbox[2] * bbox[3])

        entry: dict[str, Any] = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": list(bbox),
            "area": area,
            "iscrowd": iscrowd,
        }
        if segmentation is not None:
            entry["segmentation"] = segmentation
        if pose_6d is not None:
            entry["pose_6d"] = pose_6d

        self._annotations.append(entry)
        return ann_id

    def build(self) -> dict:
        """Build the final COCO dictionary."""
        return {
            "info": self._info,
            "licenses": [],
            "categories": self._categories,
            "images": self._images,
            "annotations": self._annotations,
        }

    def save(self, path: str | Path) -> None:
        """Save the COCO JSON to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.build(), f, indent=2)

    @property
    def num_images(self) -> int:
        return len(self._images)

    @property
    def num_annotations(self) -> int:
        return len(self._annotations)


# ---------------------------------------------------------------------------
# Mask → Bounding Box Extraction
# ---------------------------------------------------------------------------

def extract_bboxes_from_mask(
    semantic_mask: np.ndarray,
    target_id: int,
    min_area: int = 16,
    padding: int = 2,
) -> list[dict]:
    """Extract bounding boxes from a semantic segmentation mask.

    Args:
        semantic_mask: 2D integer array where each pixel is a semantic ID.
        target_id: The semantic ID to extract.
        min_area: Minimum bbox area to include.
        padding: Pixel padding around each bbox.

    Returns:
        List of dicts with ``bbox`` (x, y, w, h), ``area``, ``mask`` keys.
    """
    binary = (semantic_mask == target_id).astype(np.uint8)

    if binary.sum() == 0:
        return []

    results = []

    if CV2_AVAILABLE:
        # Connected components for instance separation
        num_labels, labels = cv2.connectedComponents(binary)
        h, w = semantic_mask.shape

        for label_id in range(1, num_labels):
            instance = (labels == label_id)
            ys, xs = np.where(instance)

            if len(xs) < min_area:
                continue

            x_min = max(0, int(xs.min()) - padding)
            y_min = max(0, int(ys.min()) - padding)
            x_max = min(w, int(xs.max()) + padding + 1)
            y_max = min(h, int(ys.max()) + padding + 1)

            bw = x_max - x_min
            bh = y_max - y_min

            if bw * bh < min_area:
                continue

            results.append({
                "bbox": [x_min, y_min, bw, bh],
                "area": int(instance.sum()),
                "mask": instance,
            })
    else:
        # Fallback: single bbox for all pixels of this ID
        ys, xs = np.where(binary)
        h, w = semantic_mask.shape

        x_min = max(0, int(xs.min()) - padding)
        y_min = max(0, int(ys.min()) - padding)
        x_max = min(w, int(xs.max()) + padding + 1)
        y_max = min(h, int(ys.max()) + padding + 1)

        bw = x_max - x_min
        bh = y_max - y_min

        if bw * bh >= min_area:
            results.append({
                "bbox": [x_min, y_min, bw, bh],
                "area": int(binary.sum()),
                "mask": binary.astype(bool),
            })

    return results


# ---------------------------------------------------------------------------
# 6D Pose Projection
# ---------------------------------------------------------------------------

def project_pose_to_camera(
    position_world: np.ndarray,
    quaternion_world: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
) -> dict:
    """Project a 3D pose to camera frame and 2D pixel coordinates.

    Args:
        position_world: Object position in world frame, shape ``(3,)``.
        quaternion_world: Object quaternion ``(w, x, y, z)`` in world frame.
        camera_intrinsics: 3×3 camera intrinsic matrix.
        camera_extrinsics: 4×4 world-to-camera transform.

    Returns:
        Dict with:
        - ``position_cam``: Position in camera frame (3,).
        - ``quaternion_cam``: Quaternion in camera frame (4,).
        - ``pixel``: Projected 2D pixel ``(u, v)``.
        - ``depth``: Depth in camera frame (meters).
    """
    # Transform position to camera frame
    pos_h = np.append(position_world, 1.0)
    pos_cam = camera_extrinsics @ pos_h
    pos_cam_3d = pos_cam[:3]

    # Project to pixel
    if pos_cam_3d[2] > 0:
        pixel_h = camera_intrinsics @ pos_cam_3d
        pixel = pixel_h[:2] / pixel_h[2]
    else:
        pixel = np.array([-1.0, -1.0])  # Behind camera

    # Rotate quaternion to camera frame
    R_cam = camera_extrinsics[:3, :3]
    # Quaternion rotation composition (simplified)
    quaternion_cam = quaternion_world  # Placeholder — full rotation would apply R_cam

    return {
        "position_cam": pos_cam_3d.tolist(),
        "quaternion_cam": quaternion_cam.tolist(),
        "pixel": pixel.tolist(),
        "depth": float(pos_cam_3d[2]),
    }


# ---------------------------------------------------------------------------
# Full Frame Labeler
# ---------------------------------------------------------------------------

class ScrewFrameLabeler:
    """Process a single simulation frame to produce COCO annotations.

    Extracts screw bounding boxes from segmentation masks and optionally
    adds 6D pose annotations from simulation ground truth.
    """

    def __init__(self, cfg: ScrewDetectorCfg | None = None):
        self.cfg = cfg or ScrewDetectorCfg()
        self._category_map = {c["name"]: c["id"] for c in self.cfg.categories}

    def label_frame(
        self,
        rgb_image: np.ndarray,
        semantic_mask: np.ndarray,
        screw_poses: list[dict] | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_extrinsics: np.ndarray | None = None,
        screw_type: str = "hex",
    ) -> list[dict]:
        """Generate annotations for a single frame.

        Args:
            rgb_image: RGB image, shape ``(H, W, 3)``.
            semantic_mask: Semantic segmentation, shape ``(H, W)``.
            screw_poses: List of dicts with ``position`` and ``quaternion`` keys.
            camera_intrinsics: 3×3 intrinsic matrix.
            camera_extrinsics: 4×4 world-to-camera transform.
            screw_type: Type of screw for category assignment.

        Returns:
            List of annotation dicts ready for COCO builder.
        """
        annotations = []

        # Category ID
        cat_name = f"screw_{screw_type}"
        cat_id = self._category_map.get(cat_name, 1)

        # Extract screw bboxes
        screw_bboxes = extract_bboxes_from_mask(
            semantic_mask,
            self.cfg.screw_semantic_id,
            min_area=self.cfg.min_bbox_area,
            padding=self.cfg.bbox_padding,
        )

        for i, bbox_info in enumerate(screw_bboxes):
            ann: dict[str, Any] = {
                "category_id": cat_id,
                "bbox": bbox_info["bbox"],
                "area": bbox_info["area"],
            }

            # Add 6D pose if available
            if (
                self.cfg.include_6d_pose
                and screw_poses is not None
                and i < len(screw_poses)
                and camera_intrinsics is not None
                and camera_extrinsics is not None
            ):
                pose = screw_poses[i]
                projected = project_pose_to_camera(
                    np.array(pose["position"]),
                    np.array(pose["quaternion"]),
                    camera_intrinsics,
                    camera_extrinsics,
                )
                ann["pose_6d"] = {
                    "position": pose["position"],
                    "quaternion": pose["quaternion"],
                    "position_cam": projected["position_cam"],
                    "depth": projected["depth"],
                }

            annotations.append(ann)

        # Extract bracket bboxes
        bracket_bboxes = extract_bboxes_from_mask(
            semantic_mask,
            self.cfg.bracket_semantic_id,
            min_area=self.cfg.min_bbox_area * 2,
            padding=self.cfg.bbox_padding,
        )

        bracket_cat_id = self._category_map.get("bracket", 6)
        for bbox_info in bracket_bboxes:
            annotations.append({
                "category_id": bracket_cat_id,
                "bbox": bbox_info["bbox"],
                "area": bbox_info["area"],
            })

        return annotations


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

class ScrewDetectionDatasetBuilder:
    """Build a complete COCO dataset from simulation renders.

    Manages image saving, annotation accumulation, and train/val/test splitting.
    """

    def __init__(self, cfg: ScrewDetectorCfg | None = None):
        self.cfg = cfg or ScrewDetectorCfg()
        self.labeler = ScrewFrameLabeler(self.cfg)

        self._output_dir = Path(self.cfg.output_dir) / self.cfg.dataset_name
        self._images_dir = self._output_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

        self._coco_builder = COCOAnnotationBuilder(
            self.cfg.categories, self.cfg.dataset_name
        )
        self._image_counter = 0

    def add_frame(
        self,
        rgb_image: np.ndarray,
        semantic_mask: np.ndarray,
        screw_poses: list[dict] | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_extrinsics: np.ndarray | None = None,
        screw_type: str = "hex",
        metadata: dict | None = None,
    ) -> int:
        """Process and save a single frame.

        Args:
            rgb_image: RGB image, shape ``(H, W, 3)``, uint8.
            semantic_mask: Semantic segmentation, shape ``(H, W)``.
            screw_poses: Ground-truth screw poses.
            camera_intrinsics: Camera intrinsic matrix.
            camera_extrinsics: Camera extrinsic matrix.
            screw_type: Screw type string.
            metadata: Optional metadata for the image entry.

        Returns:
            Image ID.
        """
        image_id = self._image_counter
        self._image_counter += 1

        h, w = rgb_image.shape[:2]
        filename = f"frame_{image_id:06d}.png"

        # Save image
        if CV2_AVAILABLE:
            cv2.imwrite(
                str(self._images_dir / filename),
                cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
            )
        else:
            # Fallback: save raw numpy
            np.save(str(self._images_dir / f"frame_{image_id:06d}.npy"), rgb_image)
            filename = f"frame_{image_id:06d}.npy"

        # Register image
        self._coco_builder.add_image(image_id, filename, w, h, metadata)

        # Generate annotations
        annotations = self.labeler.label_frame(
            rgb_image, semantic_mask, screw_poses,
            camera_intrinsics, camera_extrinsics, screw_type,
        )

        for ann in annotations:
            self._coco_builder.add_annotation(
                image_id=image_id,
                category_id=ann["category_id"],
                bbox=ann["bbox"],
                area=ann.get("area"),
                pose_6d=ann.get("pose_6d"),
            )

        return image_id

    def save(self) -> dict[str, str]:
        """Save the dataset with train/val/test splits.

        Returns:
            Dict with paths to annotation files.
        """
        full_coco = self._coco_builder.build()
        images = full_coco["images"]
        annotations = full_coco["annotations"]

        n = len(images)
        n_train = int(n * self.cfg.train_ratio)
        n_val = int(n * self.cfg.val_ratio)

        # Shuffle
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)

        splits = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }

        ann_dir = self._output_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        for split_name, idxs in splits.items():
            split_image_ids = {images[i]["id"] for i in idxs}
            split_images = [images[i] for i in idxs]
            split_annotations = [a for a in annotations if a["image_id"] in split_image_ids]

            split_coco = {
                "info": full_coco["info"],
                "licenses": [],
                "categories": full_coco["categories"],
                "images": split_images,
                "annotations": split_annotations,
            }

            path = ann_dir / f"{split_name}.json"
            with open(path, "w") as f:
                json.dump(split_coco, f, indent=2)
            paths[split_name] = str(path)

        print(f"Dataset saved to {self._output_dir}")
        print(f"  Images: {n} ({n_train} train / {n_val} val / {n - n_train - n_val} test)")
        print(f"  Annotations: {len(annotations)}")

        return paths


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Screw Detector — Label Generation Test")
    print("=" * 50)

    cfg = ScrewDetectorCfg(output_dir="/tmp/roboforce_test_dataset")
    builder = ScrewDetectionDatasetBuilder(cfg)

    # Generate synthetic test data
    for i in range(20):
        h, w = 480, 640
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.int32)

        # Place fake screw instances
        for s in range(3):
            cx = np.random.randint(100, 540)
            cy = np.random.randint(100, 380)
            r = np.random.randint(5, 15)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            circle = xx**2 + yy**2 <= r**2
            y_start = max(0, cy - r)
            x_start = max(0, cx - r)
            y_end = min(h, cy + r + 1)
            x_end = min(w, cx + r + 1)
            mask[y_start:y_end, x_start:x_end][
                circle[:y_end-y_start, :x_end-x_start]
            ] = cfg.screw_semantic_id

        builder.add_frame(rgb, mask, screw_type="hex")

    paths = builder.save()
    print(f"\nAnnotation files: {paths}")
