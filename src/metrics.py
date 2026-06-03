"""
src/metrics.py
==============
Business metrics computation for the retail shelf analysis pipeline.

Metrics computed
----------------
share_of_shelf (SOS)
    For each brand: sum of all detected bounding-box areas divided by the
    total detected product area, expressed as a percentage.

    SOS_brand = Σ bbox_area_brand / Σ bbox_area_all_products × 100

on_shelf_availability (OSA)
    Prototype OSA = product count per brand.  A future improvement would
    compare counts against a planogram to detect out-of-stock gaps.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def compute_sos(brand_areas: Dict[str, float], total_area: float) -> Dict[str, float]:
    """
    Compute Share-of-Shelf percentages.

    Parameters
    ----------
    brand_areas : dict[str, float]
        Mapping of brand → total bounding-box area (px²).
    total_area : float
        Sum of all product areas (denominator).

    Returns
    -------
    dict[str, float]
        Brand → SOS percentage, sorted descending, rounded to 2 dp.
    """
    if total_area <= 0:
        return {}
    sos = {brand: round(area / total_area * 100, 2) for brand, area in brand_areas.items()}
    return dict(sorted(sos.items(), key=lambda x: x[1], reverse=True))


def compute_osa(brands: List[str]) -> Dict[str, int]:
    """
    Compute On-Shelf Availability (product count per brand).

    Parameters
    ----------
    brands : list[str]
        Brand assigned to each detected product.

    Returns
    -------
    dict[str, int]
        Brand → product count, sorted descending.
    """
    counts: Dict[str, int] = {}
    for brand in brands:
        counts[brand] = counts.get(brand, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def generate_report(
    image_name: str,
    detections_count: int,
    brands: List[str],
    brand_areas: Dict[str, float],
    total_area: float,
    ocr_labels: List[str],
    shelf_rows_count: int,
    annotated_path: str,
    segmentation_method: str,
) -> dict:
    """
    Generate the full JSON result dictionary for one image.

    Returns
    -------
    dict
        Matches the schema specified in the assignment.
    """
    osa = compute_osa(brands)
    sos = compute_sos(brand_areas, total_area)
    return {
        "image_name":           image_name,
        "total_products":       detections_count,
        "brands":               osa,
        "share_of_shelf":       sos,
        "ocr_labels":           ocr_labels,
        "shelf_rows_detected":  shelf_rows_count,
        "segmentation_method":  segmentation_method,
        "annotated_image":      annotated_path,
    }
