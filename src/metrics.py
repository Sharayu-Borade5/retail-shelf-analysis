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
    ocr_price_tags: List[str],
    product_text: List[str],
    price_text: List[str],
    shelf_rows_count: int,
    annotated_path: str,
    segmentation_method: str,
    products_details: List[dict],
) -> dict:
    """
    Generate the full JSON result dictionary for one image.

    Returns
    -------
    dict
        Matches the schema specified in the assignment and the requested fixes.
    """
    from configs.config import SKU_TO_DISPLAY_BRAND
    
    display_brands = []
    for p in products_details:
        sku = p["sku"]
        disp = SKU_TO_DISPLAY_BRAND.get(sku, sku)
        display_brands.append(disp)
        
    osa = compute_osa(display_brands)
    sos = compute_sos(brand_areas, total_area)
    return {
        "image_name":           image_name,
        "total_products":       detections_count,
        "brands":               osa,
        "brand_counts":         osa,
        "share_of_shelf":       sos,
        "ocr_price_tags":       ocr_price_tags,
        "products":             products_details,
        "product_text":         product_text,
        "price_text":           price_text,
        "shelf_rows_detected":  shelf_rows_count,
        "segmentation_method":  segmentation_method,
        "annotated_image":      annotated_path,
    }


def generate_validation_report(all_reports: List[dict], output_path: str):
    """
    Generate validation report with sanity warnings for submission checking (Fix 8).
    """
    import json
    warnings = []
    for r in all_reports:
        img_name = r["image_name"]
        brands = r["brand_counts"]  # Dict of brand -> count

        if "beverages" in img_name.lower():
            # Beverage shelf checks
            mm_count = brands.get("Minute Maid", 0)
            if mm_count < 4:
                warnings.append(f"[{img_name}] Minute Maid count is {mm_count}, expected at least 4 visible cartons.")
            df_count = brands.get("Dark Fantasy", 0)
            if df_count > 0:
                warnings.append(f"[{img_name}] Impossible brand detected: Dark Fantasy count is {df_count}, expected 0.")
            oreo_count = brands.get("Oreo", 0)
            if oreo_count > 0:
                warnings.append(f"[{img_name}] Impossible brand detected: Oreo count is {oreo_count}, expected 0.")

        elif "snacks" in img_name.lower():
            # Snacks shelf checks
            oreo_count = brands.get("Oreo", 0)
            if oreo_count > 2:
                warnings.append(f"[{img_name}] Oreo count is inflated: got {oreo_count}, expected at most 2 visible packets.")
            trop_count = brands.get("Tropicana", 0)
            if trop_count > 0:
                warnings.append(f"[{img_name}] Impossible brand detected on snack shelf: Tropicana count is {trop_count}, expected 0.")

        elif "dairy" in img_name.lower():
            # Dairy shelf checks
            fanta_count = brands.get("Fanta", 0)
            if fanta_count > 0:
                warnings.append(f"[{img_name}] Impossible brand detected on dairy shelf: Fanta count is {fanta_count}, expected 0.")
            trop_count = brands.get("Tropicana", 0)
            if trop_count > 0:
                warnings.append(f"[{img_name}] Impossible brand detected on dairy shelf: Tropicana count is {trop_count}, expected 0.")
            lays_count = brands.get("Lay's", 0)
            if lays_count > 0:
                warnings.append(f"[{img_name}] Impossible brand detected on dairy shelf: Lay's count is {lays_count}, expected 0.")

    validation_status = "PASSED" if not warnings else "WARNING"
    report = {
        "status": validation_status,
        "total_warnings": len(warnings),
        "warnings": warnings
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

