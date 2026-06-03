"""
src/visualization.py
====================
Annotated shelf image generator.

Draws on the original image:
  • Coloured bounding boxes  (unique colour per brand)
  • Brand label + confidence above each box
  • Price tag row lines (shelf separators)
  • Share-of-Shelf legend panel on the right
  • OCR text strip along the bottom

Output is saved to outputs/visualizations/.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import Detection
from src.segmentation import SegmentationResult

logger = logging.getLogger(__name__)

# Colour palette for up to 50 distinct brands (BGR)
_PALETTE = [
    (0, 200, 255), (255, 80, 80), (80, 255, 80), (255, 200, 0),
    (200, 0, 255), (0, 180, 100), (255, 140, 0), (0, 80, 255),
    (180, 0, 180), (0, 220, 220), (255, 255, 0), (200, 200, 200),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (255, 128, 0),
    (0, 255, 128), (128, 0, 255), (255, 0, 0), (0, 0, 255),
    (100, 200, 100), (200, 100, 200), (100, 100, 200), (200, 200, 100),
    (50, 150, 250), (250, 50, 150), (150, 250, 50), (50, 250, 150),
    (150, 50, 250), (250, 150, 50),
]


class ShelfVisualizer:
    """
    Creates annotated shelf images.

    Parameters
    ----------
    font_scale : float
        OpenCV font scale for labels.
    box_thickness : int
        Bounding box line thickness.
    legend_width : int
        Pixel width of the legend panel appended to the right.
    """

    def __init__(
        self,
        font_scale: float = 0.42,
        box_thickness: int = 2,
        legend_width: int = 230,
    ):
        self.font_scale    = font_scale
        self.box_thickness = box_thickness
        self.legend_width  = legend_width
        self._brand_colors: Dict[str, Tuple[int, int, int]] = {}
        self._color_idx    = 0

    def _brand_color(self, brand: str) -> Tuple[int, int, int]:
        if brand not in self._brand_colors:
            self._brand_colors[brand] = _PALETTE[self._color_idx % len(_PALETTE)]
            self._color_idx += 1
        return self._brand_colors[brand]

    def _draw_label(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ):
        font      = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, self.font_scale, thickness)
        # Background rectangle
        cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 2, y + baseline), color, -1)
        # Text
        lum = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        fg  = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(img, text, (x + 1, y), font, self.font_scale, fg, thickness, cv2.LINE_AA)

    def _draw_legend(
        self,
        canvas: np.ndarray,
        sos: Dict[str, float],
        img_h: int,
    ) -> np.ndarray:
        """Append a SOS legend panel on the right side."""
        panel = np.zeros((img_h, self.legend_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fs     = 0.38
        th     = 1
        header = "Brand  SOS %"
        cv2.putText(panel, header, (8, 18), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(panel, (4, 22), (self.legend_width - 4, 22), (80, 80, 80), 1)

        y = 40
        for brand, pct in sos.items():
            if y > img_h - 10:
                break
            color = self._brand_color(brand)
            cv2.rectangle(panel, (6, y - 8), (18, y + 2), color, -1)
            label = f"{brand}: {pct}%"
            cv2.putText(panel, label, (22, y), font, fs, (200, 200, 200), th, cv2.LINE_AA)
            y += 16

        return np.hstack([canvas, panel])

    def _draw_ocr_strip(
        self,
        canvas: np.ndarray,
        ocr_labels: List[str],
        strip_height: int = 28,
    ) -> np.ndarray:
        """Append an OCR text strip at the bottom."""
        img_w   = canvas.shape[1]
        strip   = np.zeros((strip_height, img_w, 3), dtype=np.uint8)
        strip[:] = (20, 20, 20)
        text = "OCR: " + " | ".join(ocr_labels[:30])  # cap at 30 items
        cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 220, 180), 1, cv2.LINE_AA)
        return np.vstack([canvas, strip])

    def draw(
        self,
        image: np.ndarray,
        detections: List[Detection],
        brands: List[str],
        confidences: List[float],
        seg_result: SegmentationResult,
        sos: Dict[str, float],
        ocr_labels: List[str],
    ) -> np.ndarray:
        """
        Produce an annotated copy of the shelf image.

        Parameters
        ----------
        image : np.ndarray
            Original BGR image (not modified).
        detections, brands, confidences
            Parallel lists from detection + classification.
        seg_result : SegmentationResult
            Shelf row bands.
        sos : dict[str, float]
            Share-of-shelf percentages (for legend).
        ocr_labels : list[str]
            OCR text items (for bottom strip).

        Returns
        -------
        np.ndarray
            Annotated BGR image with legend and OCR strip.
        """
        canvas = image.copy()

        # ── Shelf row lines ────────────────────────────────────────────────
        img_h, img_w = canvas.shape[:2]
        for y_top, _ in seg_result.shelf_rows[1:]:   # skip image top border
            cv2.line(canvas, (0, y_top), (img_w, y_top), (200, 200, 200), 1)

        # ── Bounding boxes + labels ────────────────────────────────────────
        for det, brand, conf in zip(detections, brands, confidences):
            color = self._brand_color(brand)
            cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, self.box_thickness)
            label = f"{brand} {int(conf * 100)}%"
            self._draw_label(canvas, label, det.x1, max(det.y1 - 2, 12), color)

        # ── Legend ─────────────────────────────────────────────────────────
        canvas = self._draw_legend(canvas, sos, img_h)

        # ── OCR strip ──────────────────────────────────────────────────────
        if ocr_labels:
            canvas = self._draw_ocr_strip(canvas, ocr_labels)

        return canvas

    def save(self, canvas: np.ndarray, output_dir: str, stem: str) -> str:
        """Save annotated image and return the file path."""
        path = os.path.join(output_dir, f"{stem}_annotated.jpg")
        cv2.imwrite(path, canvas)
        return path
