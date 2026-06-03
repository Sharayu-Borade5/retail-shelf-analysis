"""
Product detection using YOLO-World (open-vocabulary).

YOLO-World lets us specify arbitrary text classes at inference time, so we
don't need a retail-specific fine-tuned model.  We use broad packaging nouns
("bottle", "can", "bag", …) to maximise recall on heterogeneous shelf items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2  (pixel coords)
    confidence: float
    class_name: str


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

    # ── public API ────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        detection_classes: list[str],
    ) -> List[Detection]:
        """
        Run detection on a BGR numpy image (as returned by cv2.imread).

        Returns a list of Detection objects sorted by confidence (desc).
        """
        self._load(detection_classes)

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

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_area:
                    continue

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # For COCO fallback, keep only relevant class ids
                if not self._use_world and cls_id not in self._COCO_KEEP:
                    continue

                if self._use_world:
                    cls_name = detection_classes[cls_id] if cls_id < len(detection_classes) else "product"
                else:
                    cls_name = self._COCO_FALLBACK_IDS.get(cls_id, "product")

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name=cls_name,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info("Detected %d products", len(detections))
        return detections
