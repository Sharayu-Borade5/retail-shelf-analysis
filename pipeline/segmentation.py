"""
Shelf segmentation and share-of-shelf (SOS) estimation.

Approach
--------
We cluster detected bounding boxes into horizontal shelf rows using a
1-D gap-based algorithm on the vertical centre of each box.  Within each
row we sum the widths occupied by each brand to compute SOS.

This geometric approach requires no additional model – it reuses the
detection output – and mirrors how humans reason about shelf allocation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from config import SHELF_ROW_GAP


@dataclass
class ShelfRow:
    row_id: int
    y_top: int
    y_bottom: int
    detections: list = field(default_factory=list)  # list of (Detection, brand)


@dataclass
class SegmentationResult:
    rows: List[ShelfRow]
    brand_pixel_width: Dict[str, int]     # total horizontal px per brand
    brand_sos_pct: Dict[str, float]       # share of shelf (%)
    total_shelf_width: int


def cluster_into_rows(
    detections_with_brands: List[Tuple],   # [(Detection, brand_name), ...]
    image_height: int,
    row_gap: int = SHELF_ROW_GAP,
) -> List[ShelfRow]:
    """
    Group detections into shelf rows by vertical centre coordinate.

    Two detections belong to the same row if the gap between their
    vertical centres is ≤ row_gap pixels.  Rows are returned top-to-bottom.
    """
    if not detections_with_brands:
        return []

    # Sort by y-centre
    sorted_dets = sorted(
        detections_with_brands,
        key=lambda x: (x[0].bbox[1] + x[0].bbox[3]) / 2,
    )

    rows: List[ShelfRow] = []
    current_row_dets = [sorted_dets[0]]
    current_y_centre = (sorted_dets[0][0].bbox[1] + sorted_dets[0][0].bbox[3]) / 2

    for det, brand in sorted_dets[1:]:
        y_centre = (det.bbox[1] + det.bbox[3]) / 2
        if abs(y_centre - current_y_centre) <= row_gap:
            current_row_dets.append((det, brand))
            # Update running mean y-centre
            current_y_centre = np.mean(
                [(d.bbox[1] + d.bbox[3]) / 2 for d, _ in current_row_dets]
            )
        else:
            rows.append(_make_row(len(rows), current_row_dets))
            current_row_dets = [(det, brand)]
            current_y_centre  = y_centre

    rows.append(_make_row(len(rows), current_row_dets))
    return rows


def _make_row(row_id: int, dets: List[Tuple]) -> ShelfRow:
    all_y1 = [d.bbox[1] for d, _ in dets]
    all_y2 = [d.bbox[3] for d, _ in dets]
    return ShelfRow(
        row_id=row_id,
        y_top=min(all_y1),
        y_bottom=max(all_y2),
        detections=dets,
    )


def compute_sos(
    rows: List[ShelfRow],
    image_width: int,
) -> SegmentationResult:
    """
    Compute brand-level share of shelf.

    For each detected product we use its bounding box width as a proxy for
    the horizontal shelf space it occupies.  Overlapping boxes are NOT
    de-duplicated here – the detector's NMS already handles that.
    """
    brand_px: Dict[str, int] = defaultdict(int)
    total_px = 0

    for row in rows:
        for det, brand in row.detections:
            w = det.bbox[2] - det.bbox[0]
            brand_px[brand] += w
            total_px += w

    if total_px == 0:
        return SegmentationResult(
            rows=rows,
            brand_pixel_width=dict(brand_px),
            brand_sos_pct={},
            total_shelf_width=image_width,
        )

    sos_pct = {b: round(100.0 * px / total_px, 2) for b, px in brand_px.items()}
    sos_pct = dict(sorted(sos_pct.items(), key=lambda x: x[1], reverse=True))

    return SegmentationResult(
        rows=rows,
        brand_pixel_width=dict(brand_px),
        brand_sos_pct=sos_pct,
        total_shelf_width=image_width,
    )


def get_shelf_bands(rows: List[ShelfRow]) -> List[tuple[int, int]]:
    """Return (y_top, y_bottom) tuples for all shelf rows (for OCR)."""
    return [(row.y_top, row.y_bottom) for row in rows]


class ShelfSegmenter:
    """High-level wrapper used by main pipeline."""

    def segment(
        self,
        detections_with_brands: List[Tuple],
        image_width: int,
        image_height: int,
        row_gap: int = SHELF_ROW_GAP,
    ) -> SegmentationResult:
        rows = cluster_into_rows(detections_with_brands, image_height, row_gap)
        return compute_sos(rows, image_width)
