"""
Annotated output image generation.

Draws:
  - Coloured bounding boxes per brand
  - Brand label + confidence above each box
  - Shelf row dividers
  - OCR text overlay in the price-tag strip area
  - Legend (brand → colour mapping)
"""

from __future__ import annotations

import colorsys
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .segmentation import SegmentationResult


def _brand_palette(brand_names: List[str]) -> Dict[str, tuple[int, int, int]]:
    """Generate a visually distinct BGR colour per brand."""
    n = max(len(brand_names), 1)
    palette: Dict[str, tuple[int, int, int]] = {}
    for i, name in enumerate(brand_names):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        palette[name] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    # Always make "Other" grey
    palette["Other"] = (160, 160, 160)
    return palette


class ShelfVisualizer:

    def __init__(self, font_scale: float = 0.55, thickness: int = 2) -> None:
        self.font_scale = font_scale
        self.thickness  = thickness
        self._font      = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(
        self,
        image_bgr: np.ndarray,
        detections_with_brands: List[Tuple],   # [(Detection, brand), ...]
        seg_result: SegmentationResult,
        ocr_texts: List[str],
        image_name: str,
        output_dir: str,
    ) -> str:
        """
        Draw annotations on image and save to output_dir.

        Returns the path of the saved file.
        """
        canvas = image_bgr.copy()
        H, W   = canvas.shape[:2]

        # Build colour palette for all unique brands
        brands = list({brand for _, brand in detections_with_brands})
        palette = _brand_palette(brands)

        # ── shelf row dividers ────────────────────────────────────────────
        for row in seg_result.rows:
            cv2.line(canvas, (0, row.y_top),    (W, row.y_top),    (200, 200, 200), 1)
            cv2.line(canvas, (0, row.y_bottom), (W, row.y_bottom), (200, 200, 200), 1)

        # ── bounding boxes + labels ───────────────────────────────────────
        for det, brand in detections_with_brands:
            x1, y1, x2, y2 = det.bbox
            colour = palette.get(brand, (180, 180, 180))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, self.thickness)

            label = f"{brand} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, self._font, self.font_scale, 1
            )
            bg_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                canvas, (x1, bg_y1), (x1 + tw + 4, y1), colour, cv2.FILLED
            )
            cv2.putText(
                canvas, label,
                (x1 + 2, y1 - baseline - 2),
                self._font, self.font_scale, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # ── OCR overlay (bottom strip of image) ──────────────────────────
        if ocr_texts:
            overlay_h = 28
            overlay   = canvas[H - overlay_h: H, 0: W].copy()
            cv2.rectangle(
                canvas, (0, H - overlay_h), (W, H), (0, 0, 0), cv2.FILLED
            )
            short_text = " | ".join(ocr_texts[:12])   # first 12 items
            cv2.putText(
                canvas, f"OCR: {short_text}",
                (6, H - 8),
                self._font, 0.42, (0, 255, 255), 1, cv2.LINE_AA,
            )

        # ── legend (top-right) ────────────────────────────────────────────
        canvas = self._draw_legend(canvas, palette, seg_result.brand_sos_pct)

        # ── save ──────────────────────────────────────────────────────────
        stem     = os.path.splitext(image_name)[0]
        out_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
        cv2.imwrite(out_path, canvas)
        return out_path

    # ── helpers ───────────────────────────────────────────────────────────

    def _draw_legend(
        self,
        canvas: np.ndarray,
        palette: Dict[str, tuple],
        sos_pct: Dict[str, float],
    ) -> np.ndarray:
        H, W = canvas.shape[:2]
        brands_sorted = sorted(sos_pct, key=sos_pct.get, reverse=True)

        row_h  = 20
        pad    = 6
        legend_w = 240
        legend_h = row_h * len(brands_sorted) + 2 * pad + 20

        # Semi-transparent background
        x0 = W - legend_w - 10
        y0 = 10
        sub = canvas[y0: y0 + legend_h, x0: x0 + legend_w]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.35, black, 0.65, 0, sub)
        canvas[y0: y0 + legend_h, x0: x0 + legend_w] = sub

        cv2.putText(
            canvas, "Brand SOS %",
            (x0 + pad, y0 + 16),
            self._font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        for i, brand in enumerate(brands_sorted):
            colour = palette.get(brand, (180, 180, 180))
            y_row  = y0 + 24 + i * row_h
            cv2.rectangle(canvas, (x0 + pad, y_row), (x0 + pad + 12, y_row + 12), colour, cv2.FILLED)
            pct_str = f"{sos_pct.get(brand, 0):.1f}%"
            label   = f"{brand}: {pct_str}"
            cv2.putText(
                canvas, label,
                (x0 + pad + 16, y_row + 11),
                self._font, 0.42, (220, 220, 220), 1, cv2.LINE_AA,
            )

        return canvas
