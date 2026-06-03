"""
src/segmentation.py
===================
Shelf row detection using two complementary approaches:

1. Hough Lines (primary — as specified in the assignment)
   ----------------------------------------------------------
   • Convert to grayscale → Canny edge detection
   • HoughLinesP to detect long horizontal line segments
   • These correspond to physical shelf separators / price tag bars
   • Cluster Y-positions of detected lines into distinct shelf rows

2. Geometric clustering (fallback)
   ----------------------------------
   • Uses Y-centre of detected bounding boxes to infer row boundaries
   • Applied when Hough Lines finds fewer than 2 separators (e.g. very
     dark images or images with no visible shelf bars)

Output
------
For each image the segmenter returns:
  • shelf_rows   : list of (y_top, y_bottom) band tuples
  • row_labels   : per-detection row index assignment
  • sos_pct      : share-of-shelf percentage per brand (area-based)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Output of the shelf segmentation step."""
    shelf_rows: List[Tuple[int, int]]           # (y_top, y_bottom) per row
    row_assignments: List[int]                   # row index for each detection
    brand_areas: Dict[str, float]                # total bbox area per brand
    total_area: float
    method: str                                  # "hough" or "geometric"


# ── Hough Lines helper ─────────────────────────────────────────────────────────

def _detect_shelf_lines(
    image: np.ndarray,
    canny_low: int   = 50,
    canny_high: int  = 150,
    threshold: int   = 80,
    min_length: int  = 150,
    max_gap: int     = 30,
    row_gap_min: int = 40,
) -> List[int]:
    """
    Detect Y-coordinates of horizontal shelf separator bars using Hough Lines.

    Steps
    -----
    1. Convert BGR → grayscale
    2. Apply Canny edge detector
    3. Run HoughLinesP to find line segments
    4. Keep only near-horizontal segments (|Δy| < 8 px)
    5. Cluster Y-positions (within row_gap_min pixels → same shelf bar)
    6. Return cluster median Y values sorted top → bottom

    Returns
    -------
    list[int]
        Sorted Y-coordinates of detected shelf separators.
    """
    img_h, img_w = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min(min_length, img_w // 3),
        maxLineGap=max_gap,
    )

    if lines is None:
        return []

    # Keep near-horizontal segments (|dy| < 8 px)
    ys: List[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 8:
            ys.append((y1 + y2) // 2)

    if not ys:
        return []

    # Cluster close Y values
    ys.sort()
    clusters: List[List[int]] = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] < row_gap_min:
            clusters[-1].append(y)
        else:
            clusters.append([y])

    return [int(np.median(c)) for c in clusters]


# ── Geometric fallback ─────────────────────────────────────────────────────────

def _geometric_rows(
    detections: List[Detection],
    img_h: int,
    row_gap_min: int = 40,
) -> List[int]:
    """
    Infer shelf separator Y-coordinates from detection Y-centres.
    Returns a list of Y split points between adjacent rows.
    """
    if not detections:
        return []
    centres = sorted(set(int(d.cy) for d in detections))
    if len(centres) < 2:
        return []
    # Find large vertical gaps between consecutive centres
    splits: List[int] = []
    prev = centres[0]
    for cy in centres[1:]:
        if cy - prev > row_gap_min:
            splits.append((cy + prev) // 2)
        prev = cy
    return splits


# ── Row band builder ───────────────────────────────────────────────────────────

def _splits_to_bands(splits: List[int], img_h: int) -> List[Tuple[int, int]]:
    """Convert list of Y split points to (y_top, y_bottom) bands."""
    boundaries = [0] + sorted(splits) + [img_h]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _assign_row(detection: Detection, bands: List[Tuple[int, int]]) -> int:
    """Return the row index whose band contains the detection's Y-centre."""
    cy = detection.cy
    for idx, (y_top, y_bot) in enumerate(bands):
        if y_top <= cy < y_bot:
            return idx
    return len(bands) - 1


# ── Public API ─────────────────────────────────────────────────────────────────

class ShelfSegmenter:
    """
    Detect shelf rows and assign detected products to rows.

    Parameters
    ----------
    canny_low, canny_high : int
        Canny edge detector thresholds.
    hough_threshold : int
        Minimum Hough accumulator votes for a line.
    min_line_length : int
        Minimum line length in pixels.
    max_line_gap : int
        Maximum gap within a line segment.
    row_gap_min : int
        Minimum pixel gap between two distinct shelf rows.
    """

    def __init__(
        self,
        canny_low: int       = 50,
        canny_high: int      = 150,
        hough_threshold: int = 80,
        min_line_length: int = 150,
        max_line_gap: int    = 30,
        row_gap_min: int     = 40,
    ):
        self.canny_low       = canny_low
        self.canny_high      = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap    = max_line_gap
        self.row_gap_min     = row_gap_min

    def segment(
        self,
        image: np.ndarray,
        detections: List[Detection],
        brands: List[str],
    ) -> SegmentationResult:
        """
        Segment the shelf image into rows and compute brand areas.

        Parameters
        ----------
        image : np.ndarray
            Full BGR shelf image.
        detections : list[Detection]
            Product bounding boxes.
        brands : list[str]
            Brand assigned to each detection (parallel to ``detections``).

        Returns
        -------
        SegmentationResult
        """
        img_h, img_w = image.shape[:2]

        # Try Hough Lines first
        splits = _detect_shelf_lines(
            image,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            threshold=self.hough_threshold,
            min_length=self.min_line_length,
            max_gap=self.max_line_gap,
            row_gap_min=self.row_gap_min,
        )
        method = "hough"

        if len(splits) < 2:
            logger.debug("Hough Lines found %d splits; using geometric fallback", len(splits))
            splits = _geometric_rows(detections, img_h, self.row_gap_min)
            method = "geometric"

        logger.info(
            "Shelf segmentation (%s): %d rows from %d splits",
            method, len(splits) + 1, len(splits)
        )

        bands          = _splits_to_bands(splits, img_h)
        row_assignments = [_assign_row(d, bands) for d in detections]

        # Compute brand areas
        brand_areas: Dict[str, float] = {}
        for det, brand in zip(detections, brands):
            brand_areas[brand] = brand_areas.get(brand, 0.0) + det.area
        total_area = sum(brand_areas.values()) or 1.0

        return SegmentationResult(
            shelf_rows=bands,
            row_assignments=row_assignments,
            brand_areas=brand_areas,
            total_area=total_area,
            method=method,
        )
