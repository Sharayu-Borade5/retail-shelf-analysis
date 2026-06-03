"""
Product detection using YOLO-World (open-vocabulary).

YOLO-World lets us specify arbitrary text classes at inference time, so we
don't need a retail-specific fine-tuned model.  We use broad packaging nouns
("bottle", "can", "bag", …) to maximise recall on heterogeneous shelf items.

Tiled inference (SAHI-style)
-----------------------------
For large shelf images (~2700 × 2000 px) YOLO-World struggles to detect small
flat packs (biscuit/cookie packets) at full resolution because they occupy only
a tiny fraction of the image.  We solve this by:

  1. Splitting the image into overlapping 640×640 tiles.
  2. Running YOLO on each tile independently.
  3. Re-mapping detected box coordinates back to the original image space.
  4. Merging the full-image and tiled detections with global NMS.

This typically 4-6× recall on flat biscuit/cookie packets that sit on the
lower shelves and are easily missed otherwise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import TILE_SIZE, TILE_OVERLAP, NMS_IOU_MERGE

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2  (pixel coords)
    confidence: float
    class_name: str


# ── NMS helper (pure NumPy, no torchvision required) ─────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Non-maximum suppression.

    Parameters
    ----------
    boxes         : (N, 4) array of [x1, y1, x2, y2]
    scores        : (N,)   confidence scores
    iou_threshold : float

    Returns
    -------
    List of kept indices (sorted by score descending).
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]

    return kept


def _merge_detections(
    all_detections: List[Detection],
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """Merge a pool of detections from different tiles/passes using NMS."""
    if not all_detections:
        return []

    boxes  = np.array([d.bbox for d in all_detections], dtype=float)
    scores = np.array([d.confidence for d in all_detections], dtype=float)
    kept   = _nms(boxes, scores, iou_threshold)
    return [all_detections[i] for i in kept]


# ── Main detector class ───────────────────────────────────────────────────────

class ProductDetector:
    """
    Wraps ultralytics YOLO-World for open-vocabulary product detection.

    Falls back to standard YOLOv8 (COCO) and filters to relevant classes
    if YOLO-World weights are unavailable.
    """

    # COCO class-ids that correspond to packaged goods
    _COCO_FALLBACK_IDS = {
        39: "bottle",
        41: "cup",
        45: "bowl",
        73: "book",   # some cartons are labelled as books
        76: "scissors",
    }
    # Keep only bottle / cup from COCO fallback
    _COCO_KEEP = {39, 41}

    def __init__(
        self,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        min_area: int = 800,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.min_area       = min_area
        self.device         = device
        self._model         = None
        self._use_world     = False

    # ── lazy model loading ────────────────────────────────────────────────

    def _load(self, detection_classes: list[str]) -> None:
        if self._model is not None:
            return

        from ultralytics import YOLO  # deferred import – avoids slow startup

        try:
            self._model = YOLO("yolov8s-worldv2.pt")
            self._model.set_classes(detection_classes)
            self._use_world = True
            logger.info("Loaded YOLO-World (yolov8s-worldv2.pt)")
        except Exception as exc:
            logger.warning(
                "YOLO-World unavailable (%s) – falling back to YOLOv8m (COCO)", exc
            )
            self._model = YOLO("yolov8m.pt")
            self._use_world = False

    # ── internal helpers ──────────────────────────────────────────────────

    def _predict(self, image: np.ndarray, detection_classes: list[str]) -> List[Detection]:
        """Run YOLO on a single image/tile and return raw detections."""
        results = self._model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_area:
                    continue

                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])

                # For COCO fallback, keep only relevant class ids
                if not self._use_world and cls_id not in self._COCO_KEEP:
                    continue

                if self._use_world:
                    cls_name = (
                        detection_classes[cls_id]
                        if cls_id < len(detection_classes)
                        else "product"
                    )
                else:
                    cls_name = self._COCO_FALLBACK_IDS.get(cls_id, "product")

                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_name=cls_name)
                )

        return detections

    def _build_tiles(
        self,
        image: np.ndarray,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Slice *image* into overlapping TILE_SIZE × TILE_SIZE patches.

        Returns a list of (tile_img, x_offset, y_offset) tuples where
        (x_offset, y_offset) is the top-left corner of the tile in the
        original image coordinate system.
        """
        H, W = image.shape[:2]
        step = int(TILE_SIZE * (1 - TILE_OVERLAP))  # stride between tile starts

        # At least one tile even if image is smaller than TILE_SIZE
        xs = list(range(0, max(W - TILE_SIZE, 0) + 1, step)) or [0]
        ys = list(range(0, max(H - TILE_SIZE, 0) + 1, step)) or [0]
        # Ensure we always include the far-right / far-bottom edge
        if xs[-1] + TILE_SIZE < W:
            xs.append(max(W - TILE_SIZE, 0))
        if ys[-1] + TILE_SIZE < H:
            ys.append(max(H - TILE_SIZE, 0))

        tiles = []
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + TILE_SIZE, H)
                x1 = min(x0 + TILE_SIZE, W)
                tile = image[y0:y1, x0:x1]
                tiles.append((tile, x0, y0))

        return tiles

    def _detect_tiled(
        self,
        image: np.ndarray,
        detection_classes: list[str],
    ) -> List[Detection]:
        """
        Run detection on overlapping tiles of the full image.

        Boxes from each tile are remapped to original-image coordinates
        before being returned.
        """
        tiles = self._build_tiles(image)
        all_tile_dets: List[Detection] = []

        for tile_img, x_off, y_off in tiles:
            tile_dets = self._predict(tile_img, detection_classes)
            for d in tile_dets:
                # Remap box to original image coordinates
                tx1, ty1, tx2, ty2 = d.bbox
                orig_bbox = (
                    tx1 + x_off,
                    ty1 + y_off,
                    tx2 + x_off,
                    ty2 + y_off,
                )
                # Re-check area in original space (tiles scale may differ)
                orig_area = (orig_bbox[2] - orig_bbox[0]) * (orig_bbox[3] - orig_bbox[1])
                if orig_area < self.min_area:
                    continue
                all_tile_dets.append(
                    Detection(
                        bbox=orig_bbox,
                        confidence=d.confidence,
                        class_name=d.class_name,
                    )
                )

        return all_tile_dets

    # ── public API ────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        detection_classes: list[str],
    ) -> List[Detection]:
        """
        Run detection on a BGR numpy image (as returned by cv2.imread).

        Combines a full-image pass with a tiled pass, then merges with NMS.
        Returns a list of Detection objects sorted by confidence (desc).
        """
        self._load(detection_classes)

        # Pass 1 — full image (catches large/prominent items)
        full_dets = self._predict(image, detection_classes)
        logger.info("Full-image pass: %d raw detections", len(full_dets))

        # Pass 2 — tiled (catches small flat packs on lower shelves)
        tile_dets = self._detect_tiled(image, detection_classes)
        logger.info("Tiled pass: %d raw detections across all tiles", len(tile_dets))

        # Merge both passes and apply global NMS (more aggressive threshold)
        combined = full_dets + tile_dets
        detections = _merge_detections(combined, iou_threshold=NMS_IOU_MERGE)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info(
            "After NMS merge: %d products (full=%d, tiles=%d)",
            len(detections), len(full_dets), len(tile_dets),
        )
        return detections
