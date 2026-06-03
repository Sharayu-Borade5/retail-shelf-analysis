"""
main.py
=======
Retail Shelf Analysis Pipeline — Entry Point

Usage
-----
# Analyse all images in data/images/
python main.py

# Analyse specific images
python main.py --images data/images/shelf_snacks.jpg data/images/shelf_beverages.jpg

# Use a custom image folder
python main.py --images_dir path/to/folder

# GPU mode
python main.py --images_dir data/images/ --device cuda

Pipeline stages
---------------
1. Shelf Row Segmentation   (Hough Lines → geometric fallback)
2. Product Detection        (YOLO-World zero-shot + tiled inference)
3. Brand Classification     (OpenCLIP zero-shot)
4. OCR Extraction           (PaddleOCR → EasyOCR fallback)
5. Business Metrics         (SOS, OSA)
6. Visualization            (annotated image)
7. JSON Output              (structured result)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shelf_pipeline")

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from configs.config import (
    DATA_DIR, OUTPUT_VIS_DIR, OUTPUT_RES_DIR,
    DETECTION_CLASSES, DETECTION_CONF, DETECTION_IOU, MIN_BBOX_AREA,
    TILE_SIZE, TILE_OVERLAP, NMS_IOU_MERGE, NMS_CONTAIN_THRESH,
    CLIP_MODEL, CLIP_PRETRAINED, CLIP_MIN_CONF, BRAND_PROMPTS,
    OCR_LANGUAGES, OCR_CONF, PRICE_PATTERN,
    ALLOWED_BEVERAGES, ALLOWED_DAIRY, ALLOWED_SNACKS, SKU_TO_PARENT_BRAND,
    CANNY_LOW, CANNY_HIGH, HOUGH_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, ROW_GAP_MIN,
    VIS_FONT_SCALE, VIS_BOX_THICK, VIS_LEGEND_W,
)

# ── Pipeline modules ────────────────────────────────────────────────────────────
from src.detector      import ProductDetector
from src.classifier    import BrandClassifier
from src.ocr           import OCREngine
from src.segmentation  import ShelfSegmenter
from src.metrics       import compute_sos, compute_osa, generate_report, generate_validation_report
from src.visualization import ShelfVisualizer

# Suppress third-party noise
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("easyocr").setLevel(logging.WARNING)
logging.getLogger("ppocr").setLevel(logging.WARNING)


class RetailShelfPipeline:
    """
    End-to-end retail shelf analysis pipeline.

    Instantiate once and call ``process()`` for each image.
    All heavy models are loaded lazily on first use and reused across images.
    """

    def __init__(self, device: str = "cpu"):
        self.ocr        = OCREngine(languages=OCR_LANGUAGES, min_conf=OCR_CONF)
        self.detector   = ProductDetector(
            conf=DETECTION_CONF,
            iou=DETECTION_IOU,
            min_area=MIN_BBOX_AREA,
            tile_size=TILE_SIZE,
            tile_overlap=TILE_OVERLAP,
            nms_iou_merge=NMS_IOU_MERGE,
            nms_contain_thresh=NMS_CONTAIN_THRESH,
        )
        self.classifier = BrandClassifier(
            brand_prompts=BRAND_PROMPTS,
            model_name=CLIP_MODEL,
            pretrained=CLIP_PRETRAINED,
            min_conf=CLIP_MIN_CONF,
            device=device,
            ocr_engine=self.ocr,
        )
        self.segmenter  = ShelfSegmenter(
            canny_low=CANNY_LOW,
            canny_high=CANNY_HIGH,
            hough_threshold=HOUGH_THRESHOLD,
            min_line_length=MIN_LINE_LENGTH,
            max_line_gap=MAX_LINE_GAP,
            row_gap_min=ROW_GAP_MIN,
        )
        self.visualizer = ShelfVisualizer(
            font_scale=VIS_FONT_SCALE,
            box_thickness=VIS_BOX_THICK,
            legend_width=VIS_LEGEND_W,
        )

    def process(self, image_path: str) -> dict:
        """
        Process a single shelf image through the full pipeline.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to the input image.

        Returns
        -------
        dict
            JSON-serialisable result matching the assignment schema.
        """
        stem = Path(image_path).stem
        name = Path(image_path).name
        logger.info("─── Processing: %s ───", name)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        logger.info("Image size: %dx%d", image.shape[1], image.shape[0])

        # ── Step 1: Product Detection ──────────────────────────────────────
        detections = self.detector.detect(image, DETECTION_CLASSES)
        logger.info("→ %d product regions detected", len(detections))

        if not detections:
            logger.warning("No products detected in %s", name)
            return generate_report(name, 0, [], {}, 0.0, [], [], [], 0, "", "none")

        # Determine allowed brands based on shelf type (Fix 3)
        name_lower = name.lower()
        if "dairy" in name_lower:
            allowed = ALLOWED_DAIRY
        elif "beverage" in name_lower or "beverages" in name_lower:
            allowed = ALLOWED_BEVERAGES
        elif "snack" in name_lower or "snacks" in name_lower:
            allowed = ALLOWED_SNACKS
        else:
            allowed = None

        # ── Step 2: Brand Classification ───────────────────────────────────
        crops = [
            image[d.y1:d.y2, d.x1:d.x2]
            for d in detections
            if d.y2 > d.y1 and d.x2 > d.x1
        ]
        results, crop_texts = self.classifier.classify_batch(crops, allowed_brands=allowed)

        # Resolve brand/SKU hierarchy and apply context validation (Fix 2 & 3)
        brands = []
        skus = []
        confidences = []
        for res in results:
            sku = res[0]
            conf = res[1]
            parent_brand = SKU_TO_PARENT_BRAND.get(sku, sku)

            # Context filtering
            if allowed is not None and parent_brand not in allowed:
                logger.info("Context violation: %s (SKU: %s) not allowed on shelf %s. Overriding to Other.", parent_brand, sku, name)
                parent_brand = "Other"
                sku = "Other"
                conf = 0.0

            brands.append(parent_brand)
            skus.append(sku)
            confidences.append(conf)

        products_details = [
            {"brand": b, "sku": s, "confidence": c}
            for b, s, c in zip(brands, skus, confidences)
        ]

        # Process product_text (from package crops)
        product_text_set = set()
        for texts in crop_texts:
            for t in texts:
                if t.strip():
                    product_text_set.add(t.strip())
        product_text = sorted(list(product_text_set))

        # ── Step 3: Shelf Row Segmentation ─────────────────────────────────
        seg = self.segmenter.segment(image, detections, brands)

        # ── Step 4: Shelf Label OCR Extraction ─────────────────────────────
        price_text = self.ocr.extract(image, seg.shelf_rows)

        # Extract price tags using PRICE_PATTERN
        import re
        price_re = re.compile(PRICE_PATTERN)
        bare_num_re = re.compile(r"\b\d{2,3}\b")
        ocr_price_tags_set = set()
        for text in price_text:
            currency_matches = price_re.findall(text)
            if currency_matches:
                for m in currency_matches:
                    ocr_price_tags_set.add(m.strip())
            else:
                # Remove percentages and weights to avoid extracting them as prices
                cleaned_text = re.sub(r"\d+\s*%|\d+\s*(?:g|ml|kg|l|oz|gm)\b", "", text, flags=re.IGNORECASE)
                nums = bare_num_re.findall(cleaned_text)
                for num in nums:
                    val = int(num)
                    if 10 <= val <= 500:
                        ocr_price_tags_set.add(f"₹{val}")
        ocr_price_tags = sorted(list(ocr_price_tags_set), key=lambda x: int(re.sub(r"\D", "", x)))

        # ── Step 5: Business Metrics ───────────────────────────────────────
        sos = compute_sos(seg.brand_areas, seg.total_area)

        # ── Step 6: Visualization ──────────────────────────────────────────
        canvas = self.visualizer.draw(
            image, detections, brands, confidences, seg, sos, ocr_price_tags, skus
        )
        vis_path = self.visualizer.save(canvas, OUTPUT_VIS_DIR, stem)
        logger.info("→ annotated: %s", vis_path)

        # ── Step 7: JSON Report ────────────────────────────────────────────
        report = generate_report(
            image_name=name,
            detections_count=len(detections),
            brands=brands,
            brand_areas=seg.brand_areas,
            total_area=seg.total_area,
            ocr_price_tags=ocr_price_tags,
            product_text=product_text,
            price_text=price_text,
            shelf_rows_count=len(seg.shelf_rows),
            annotated_path=vis_path,
            segmentation_method=seg.method,
            products_details=products_details,
        )

        top_brands = list(report["brands"].keys())[:5]
        logger.info(
            "→ brands=%s | ocr_price_tags=%d | annotated=%s",
            top_brands, len(ocr_price_tags), vis_path,
        )
        return report


def _find_images(images_dir: Optional[str], images: Optional[List[str]]) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if images:
        return [p for p in images if Path(p).suffix.lower() in exts]
    if images_dir:
        return sorted(
            str(p) for p in Path(images_dir).iterdir()
            if p.suffix.lower() in exts
        )
    # Default: scan data/images/
    return sorted(
        str(p) for p in Path(DATA_DIR).iterdir()
        if p.suffix.lower() in exts
    )


def main():
    parser = argparse.ArgumentParser(description="Retail Shelf Analysis Pipeline")
    parser.add_argument("--images",     nargs="+", help="Specific image paths")
    parser.add_argument("--images_dir", help="Folder of images to process")
    parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    image_paths = _find_images(args.images_dir, args.images)
    if not image_paths:
        logger.error("No images found. Put shelf images in data/images/ or use --images.")
        sys.exit(1)

    logger.info("Found %d image(s) to process", len(image_paths))
    pipeline = RetailShelfPipeline(device=args.device)
    all_results = []

    for path in image_paths:
        try:
            result = pipeline.process(path)
            # Save individual JSON
            json_path = os.path.join(OUTPUT_RES_DIR, Path(path).stem + "_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            all_results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", path, exc)
            raise

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'IMAGE':<35} {'PRODUCTS':>8}  {'BRANDS':>6}  {'OCR':>5}  {'ROWS':>4}")
    print("=" * 70)
    for r in all_results:
        print(
            f"{r['image_name']:<35} {r['total_products']:>8}  "
            f"{len(r['brands']):>6}  {len(r['ocr_price_tags']):>5}  "
            f"{r['shelf_rows_detected']:>4}"
        )
    print("=" * 70)

    # Save combined JSON
    combined_path = os.path.join(OUTPUT_RES_DIR, "all_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    # Generate and save validation report (Fix 8)
    val_report_path = os.path.join(OUTPUT_RES_DIR, "validation_report.json")
    generate_validation_report(all_results, val_report_path)
    
    print(f"\nOutputs saved to: {OUTPUT_VIS_DIR}  (images)")
    print(f"                  {OUTPUT_RES_DIR}  (JSON)")
    print(f"                  {val_report_path}  (validation report)")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
