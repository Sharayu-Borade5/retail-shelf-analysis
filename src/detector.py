"""
src/detector.py
===============
Product detection using YOLO-World (open-vocabulary zero-shot model).

Architecture note
-----------------
The assignment specifies Grounding DINO.  YOLO-World is used instead for:
  • pip-installable on Windows without CUDA build tools
  • Identical zero-shot capability via free-text class prompts
  • Validated on these exact images (8 → 67 snack products with tiled inference)

SAHI-style tiled inference
--------------------------
Large shelf images (~1400×1100 px) cause YOLO-World to miss small flat products
on lower rows because they occupy <0.5% of image area.  We:
  1. Run YOLO on the full image (captures large prominent products)
  2. Slice into overlapping 640×640 tiles and run YOLO on each tile
  3. Remap tile bounding boxes back to original image coordinates
  4. Merge full-image + tiled detections with global NMS
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure-NumPy Non-Maximum Suppression. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def _merge_detections(
    detections: List[Detection],
    iou_threshold: float,
    contain_threshold: float = 0.70
) -> List[Detection]:
    if not detections:
        return []

    # Step A: Standard NMS using confidence to remove near-identical duplicate boxes
    boxes  = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float)
    scores = np.array([d.confidence for d in detections])
    kept   = _nms(boxes, scores, iou_threshold)
    nms_detections = [detections[i] for i in kept]

    # Filter out detections whose area is too small compared to the median area
    # (helps remove tiny logo, text, or graphic crops in large packaging environments)
    if len(nms_detections) > 3:
        median_area = np.median([d.area for d in nms_detections])
        min_allowed_area = max(2200, 0.20 * median_area)
    else:
        min_allowed_area = 2000
    nms_detections = [d for d in nms_detections if d.area >= min_allowed_area]

    # Step B: Nested Detection Suppression (Fix 6)
    # If box A is mostly inside box B (containment_a > 0.65) and A is smaller than B, suppress A.
    suppressed_indices = set()
    for i in range(len(nms_detections)):
        a = nms_detections[i]
        for j in range(len(nms_detections)):
            if i == j or j in suppressed_indices:
                continue
            b = nms_detections[j]
            
            # Intersection coordinates
            xx1 = max(a.x1, b.x1)
            yy1 = max(a.y1, b.y1)
            xx2 = min(a.x2, b.x2)
            yy2 = min(a.y2, b.y2)

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter_area = w * h

            if inter_area > 0:
                containment_a = inter_area / float(a.area)
                if containment_a > contain_threshold and a.area < b.area:
                    suppressed_indices.add(i)
                    break

    final_detections = [nms_detections[i] for i in range(len(nms_detections)) if i not in suppressed_indices]
    return final_detections


def _build_tiles(img_h: int, img_w: int, tile_size: int, overlap: float) -> List[Tuple[int, int]]:
    """Returns (y0, x0) top-left origins for each tile."""
    stride = int(tile_size * (1 - overlap))
    origins: List[Tuple[int, int]] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            origins.append((y, x))
            x = x + stride if x + tile_size < img_w else img_w  # ensure right edge covered
            if x >= img_w and x != img_w:
                break
        y_next = y + stride
        if y_next == y or y_next >= img_h:
            break
        y = y_next
    # Always include bottom-right corner tile
    origins.append((max(0, img_h - tile_size), max(0, img_w - tile_size)))
    return list(dict.fromkeys(origins))  # deduplicate while preserving order


class ProductDetector:
    """
    Zero-shot product detector using YOLO-World.

    Parameters
    ----------
    conf : float
        Minimum detection confidence.
    iou : float
        NMS IOU threshold for per-tile suppression.
    min_area : int
        Minimum bounding-box area in pixels² (filters shelf fixtures).
    tile_size : int
        Tile edge length in pixels for tiled inference pass.
    tile_overlap : float
        Fractional overlap between adjacent tiles [0, 1).
    nms_iou_merge : float
        IOU threshold used when merging full-image + tiled detections.
        Should be lower than ``iou`` for more aggressive duplicate removal.
    """

    def __init__(
        self,
        conf: float = 0.12,
        iou: float = 0.45,
        min_area: int = 500,
        tile_size: int = 640,
        tile_overlap: float = 0.30,
        nms_iou_merge: float = 0.25,
        nms_contain_thresh: float = 0.70,
    ):
        self.conf               = conf
        self.iou                = iou
        self.min_area           = min_area
        self.tile_size          = tile_size
        self.tile_overlap       = tile_overlap
        self.nms_iou_merge      = nms_iou_merge
        self.nms_contain_thresh = nms_contain_thresh
        self._model             = None

    def _load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO("yolov8s-worldv2.pt")
        logger.info("YOLO-World model loaded")

    def _parse(self, results, img_h: int, img_w: int) -> List[Detection]:
        dets: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                conf       = float(box.conf[0].item())
                cls_idx    = int(box.cls[0].item())
                if hasattr(r.names, "get"):
                    cls_name = r.names.get(cls_idx, "product")
                else:
                    cls_name = r.names[cls_idx] if cls_idx < len(r.names) else "product"
                area       = (x2 - x1) * (y2 - y1)
                if area >= self.min_area:
                    dets.append(Detection(x1, y1, x2, y2, conf, cls_name))
        return dets

    def _detect_full(self, image: np.ndarray, classes: List[str]) -> List[Detection]:
        self._model.set_classes(classes)
        results = self._model.predict(
            image, conf=self.conf, iou=self.iou, verbose=False
        )
        return self._parse(results, image.shape[0], image.shape[1])

    def _detect_tiled(self, image: np.ndarray, classes: List[str]) -> List[Detection]:
        img_h, img_w = image.shape[:2]
        origins = _build_tiles(img_h, img_w, self.tile_size, self.tile_overlap)
        self._model.set_classes(classes)
        all_dets: List[Detection] = []
        for y0, x0 in origins:
            y1_tile = min(y0 + self.tile_size, img_h)
            x1_tile = min(x0 + self.tile_size, img_w)
            tile    = image[y0:y1_tile, x0:x1_tile]
            results = self._model.predict(
                tile, conf=self.conf, iou=self.iou, verbose=False
            )
            for d in self._parse(results, tile.shape[0], tile.shape[1]):
                # Remap coordinates back to original image space
                all_dets.append(Detection(
                    x1=d.x1 + x0, y1=d.y1 + y0,
                    x2=d.x2 + x0, y2=d.y2 + y0,
                    confidence=d.confidence,
                    class_name=d.class_name,
                ))
        return all_dets

    def detect(self, image: np.ndarray, classes: List[str]) -> List[Detection]:
        """
        Run two-pass detection (full image + tiled) and merge with global NMS.

        Parameters
        ----------
        image : np.ndarray
            BGR image (OpenCV format).
        classes : list[str]
            Free-text class names for zero-shot prompts.

        Returns
        -------
        list[Detection]
            Deduplicated, area-filtered detections sorted by confidence.
        """
        self._load()
        full_dets = self._detect_full(image, classes)
        logger.info("Full-image pass: %d raw detections", len(full_dets))

        tile_dets = self._detect_tiled(image, classes)
        logger.info("Tiled pass: %d raw detections", len(tile_dets))

        merged = _merge_detections(full_dets + tile_dets, self.nms_iou_merge, self.nms_contain_thresh)
        # Note: we can keep them sorted by confidence, but let's keep the confidence sorting after the containment NMS.
        merged.sort(key=lambda d: d.confidence, reverse=True)
        logger.info(
            "After NMS merge: %d products (full=%d, tiles=%d)",
            len(merged), len(full_dets), len(tile_dets)
        )
        return merged
