"""
Retail Shelf Analysis Pipeline  –  entry point.

Usage
-----
    # Single image
    python main.py --images images/shelf_01.jpg

    # Multiple images
    python main.py --images images/shelf_01.jpg images/shelf_02.jpg images/shelf_03.jpg

    # All images in a folder
    python main.py --images_dir images/

    # GPU inference
    python main.py --images images/shelf_01.jpg --device cuda

    # Adjust detection sensitivity
    python main.py --images images/shelf_01.jpg --conf 0.20 --row_gap 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2

# Local modules
import config as cfg
from pipeline import (
    BrandClassifier,
    OCREngine,
    ProductDetector,
    ShelfSegmenter,
    ShelfVisualizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shelf_pipeline")


# ── Pipeline class ─────────────────────────────────────────────────────────

class ShelfAnalysisPipeline:
    """
    Orchestrates detection → classification → OCR → segmentation → output.

    All sub-components use lazy initialisation so the first call is slower
    (model loading) but subsequent calls are fast.
    """

    def __init__(
        self,
        conf_threshold: float = cfg.DETECTION_CONF,
        iou_threshold: float  = cfg.DETECTION_IOU,
        min_area: int         = cfg.MIN_BBOX_AREA,
        row_gap: int          = cfg.SHELF_ROW_GAP,
        device: str           = "cpu",
        output_dir: str       = cfg.OUTPUT_DIR,
    ) -> None:
        self.output_dir = output_dir
        self.row_gap    = row_gap
        os.makedirs(output_dir, exist_ok=True)

        self.detector   = ProductDetector(conf_threshold, iou_threshold, min_area, device)
        self.classifier = BrandClassifier(cfg.BRAND_PROMPTS, cfg.CLIP_MODEL, cfg.CLIP_PRETRAINED, device)
        self.ocr        = OCREngine(cfg.OCR_LANGUAGES, cfg.OCR_CONF, gpu=(device != "cpu"))
        self.segmenter  = ShelfSegmenter()
        self.visualizer = ShelfVisualizer()

    # ── public API ─────────────────────────────────────────────────────────

    def run(self, image_path: str) -> Dict:
        """
        Analyse one shelf image.

        Returns a dictionary matching the required output schema:
        {
            "image_name": str,
            "total_products": int,
            "brands": { brand: count, ... },
            "ocr_labels": [ str, ... ],
            "share_of_shelf_pct": { brand: float, ... },
            "annotated_image": str,   # path to saved annotated image
        }
        """
        image_name = os.path.basename(image_path)
        logger.info("─── Processing: %s ───", image_name)

        # 1. Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        H, W = image.shape[:2]
        logger.info("Image size: %dx%d", W, H)

        # 2. Detect products
        detections = self.detector.detect(image, cfg.DETECTION_CLASSES)
        logger.info("  → %d product regions detected", len(detections))

        if not detections:
            logger.warning("No products detected – returning empty result")
            return self._empty_result(image_name)

        # 3. Crop detected regions and classify brands (batched for speed)
        crops = [image[d.bbox[1]:d.bbox[3], d.bbox[0]:d.bbox[2]] for d in detections]
        brand_results = self.classifier.classify_batch(crops)

        detections_with_brands = [
            (det, brand)
            for det, (brand, _conf) in zip(detections, brand_results)
        ]

        # 4. Shelf segmentation (group into rows, compute SOS)
        seg_result = self.segmenter.segment(
            detections_with_brands, W, H, self.row_gap
        )

        # 5. OCR on price-tag strips
        from pipeline.segmentation import get_shelf_bands
        bands     = get_shelf_bands(seg_result.rows)
        ocr_texts = self.ocr.extract(image, bands if bands else None)

        # 6. Aggregate brand counts
        brand_counts: Dict[str, int] = {}
        for _, brand in detections_with_brands:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        brand_counts = dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True))

        # 7. Visualise & save annotated image
        annotated_path = self.visualizer.annotate(
            image,
            detections_with_brands,
            seg_result,
            ocr_texts,
            image_name,
            self.output_dir,
        )

        result = {
            "image_name": image_name,
            "total_products": len(detections),
            "brands": brand_counts,
            "ocr_labels": ocr_texts,
            "share_of_shelf_pct": seg_result.brand_sos_pct,
            "shelf_rows_detected": len(seg_result.rows),
            "annotated_image": annotated_path,
        }

        # Save JSON alongside annotated image
        stem     = os.path.splitext(image_name)[0]
        json_path = os.path.join(self.output_dir, f"{stem}_result.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

        logger.info(
            "  → brands=%s | ocr_items=%d | annotated=%s",
            list(brand_counts.keys())[:5],
            len(ocr_texts),
            annotated_path,
        )
        return result

    def run_batch(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for path in image_paths:
            try:
                results.append(self.run(path))
            except Exception as exc:
                logger.error("Failed on %s: %s", path, exc)
                results.append({"image_name": os.path.basename(path), "error": str(exc)})
        return results

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(image_name: str) -> Dict:
        return {
            "image_name": image_name,
            "total_products": 0,
            "brands": {},
            "ocr_labels": [],
            "share_of_shelf_pct": {},
            "shelf_rows_detected": 0,
            "annotated_image": None,
        }


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Retail shelf analysis: detection + brand classification + OCR + SOS"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--images", nargs="+", metavar="FILE",
        help="One or more shelf image paths",
    )
    group.add_argument(
        "--images_dir", metavar="DIR",
        help="Directory containing shelf images (.jpg / .png / .jpeg)",
    )
    p.add_argument("--output_dir", default=cfg.OUTPUT_DIR, help="Output directory")
    p.add_argument("--device",    default="cpu",  choices=["cpu", "cuda"])
    p.add_argument("--conf",      type=float, default=cfg.DETECTION_CONF)
    p.add_argument("--iou",       type=float, default=cfg.DETECTION_IOU)
    p.add_argument("--row_gap",   type=int,   default=cfg.SHELF_ROW_GAP)
    return p.parse_args()


def main():
    args = _parse_args()

    if args.images:
        image_paths = args.images
    else:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        d     = Path(args.images_dir)
        image_paths = [str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in exts]
        if not image_paths:
            logger.error("No images found in %s", args.images_dir)
            sys.exit(1)

    pipeline = ShelfAnalysisPipeline(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        row_gap=args.row_gap,
        device=args.device,
        output_dir=args.output_dir,
    )

    results = pipeline.run_batch(image_paths)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'IMAGE':<30} {'PRODUCTS':>9} {'BRANDS':>8} {'OCR ITEMS':>10}")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['image_name']:<30}  ERROR: {r['error']}")
        else:
            print(
                f"{r['image_name']:<30} "
                f"{r['total_products']:>9} "
                f"{len(r['brands']):>8} "
                f"{len(r['ocr_labels']):>10}"
            )
    print("=" * 70)
    print(f"\nOutputs saved to: {args.output_dir}\n")

    # Full JSON dump to stdout (can be piped / redirected)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
