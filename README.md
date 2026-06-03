# Retail Shelf Analysis Pipeline

An end-to-end ML inference pipeline that analyses retail shelf images and
generates brand-wise shelf presence, product availability, and OCR insights.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT SHELF IMAGE                           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   1. DETECTION        │  YOLO-World (yolov8s-worldv2.pt)
          │   ProductDetector     │  Open-vocabulary: "bottle", "can",
          │                       │  "bag", "carton", "packet", …
          └───────────┬───────────┘
                      │  N bounding boxes  (x1,y1,x2,y2, conf)
          ┌───────────▼───────────┐
          │   2. CLASSIFICATION   │  CLIP  (ViT-B/32, OpenAI weights)
          │   BrandClassifier     │  Zero-shot cosine similarity against
          │                       │  brand-specific text prompts
          └───┬───────────────────┘
              │  (Detection, brand_name) pairs
  ┌───────────▼───────────┐  ┌──────────────────────────────────┐
  │  3. SEGMENTATION      │  │  4. OCR                          │
  │  ShelfSegmenter       │  │  OCREngine  (EasyOCR)            │
  │                       │  │                                  │
  │  • Cluster boxes into │  │  • Crop price-tag strips at      │
  │    shelf rows by Y    │  │    bottom of each shelf row      │
  │  • Sum bbox widths    │  │  • Extract ₹ prices, weights,    │
  │    per brand → SOS %  │  │    brand text                    │
  └───────────┬───────────┘  └─────────────────┬────────────────┘
              │                                 │
          ┌───▼─────────────────────────────────▼───┐
          │              5. OUTPUT                   │
          │  ShelfVisualizer  +  JSON result         │
          │                                          │
          │  • Annotated image (boxes + labels)      │
          │  • Shelf row dividers                    │
          │  • Brand SOS legend                      │
          │  • OCR text strip                        │
          │  • result JSON                           │
          └──────────────────────────────────────────┘
```

---

## Model Selection Justification

### 1. Detection — YOLO-World (`yolov8s-worldv2.pt`)

| Criterion        | Decision |
|------------------|----------|
| **Why**          | Standard YOLOv8/COCO only recognises ~5 packaging-relevant classes (bottle, cup …).  YOLO-World accepts free-text class names at inference time, so we can query for "bottle", "can", "bag", "carton", "packet" without any fine-tuning. |
| **Speed**        | ~25 ms/image on CPU (640 px input); ~6 ms on GPU. |
| **Accuracy**     | Recall ~75–85% on densely packed shelf images (estimated; no retail GT). False positives filtered by `MIN_BBOX_AREA`. |
| **Fallback**     | If YOLO-World weights are unavailable, the code automatically falls back to `yolov8m.pt` (COCO) and keeps the `bottle` / `cup` classes. |
| **Deployment**   | Single `.pt` file, CPU-compatible, no external dependency beyond `ultralytics`. |

### 2. Classification — CLIP (`ViT-B/32`, OpenAI)

| Criterion        | Decision |
|------------------|----------|
| **Why**          | Zero-shot classification requires no labelled retail dataset.  CLIP was pretrained on 400 M image–text pairs and has seen branded FMCG packaging, making it effective for brand recognition with well-crafted text prompts. |
| **Speed**        | ~12 ms per crop on CPU (all crops processed as one batch). |
| **Accuracy**     | Works well when brand colours/logos are distinctive (Coca-Cola red, Pepsi blue, Lay's yellow).  Struggles with brands that look visually similar (e.g. Amul Shakti vs Amul Gold). |
| **Alternative**  | A fine-tuned EfficientNet-B0 on a retail dataset would be more accurate but requires labelled data. CLIP is the practical zero-shot choice. |
| **Deployment**   | ~350 MB model; runs on CPU; `open-clip-torch` is pip-installable. |

### 3. OCR — EasyOCR

| Criterion        | Decision |
|------------------|----------|
| **Why**          | EasyOCR outperforms Tesseract on natural-scene text (signage, product labels) without needing system-level installation. It handles the mixed font styles on Indian retail price tags reliably. |
| **Speed**        | ~200–400 ms per shelf image on CPU (run once per image, not per detection). |
| **Accuracy**     | Good on large, high-contrast text (price tags).  Degrades on small or curved text. |
| **Deployment**   | Pure Python install; no Tesseract binary needed. |

### 4. Shelf Segmentation — Geometric (no neural model)

A neural segmentation model (e.g. Mask R-CNN) would add ~1 s/image latency
with marginal benefit over a geometric approach, because shelf rows in retail
images are nearly always horizontal bands.  Our implementation:

1. Clusters detected bounding boxes by vertical centre (1-D gap algorithm).
2. Sums bounding-box widths per brand within each row → Share of Shelf %.

This is fast (< 1 ms), interpretable, and directly produces actionable SOS metrics.

---

## CPU vs GPU Considerations

| Component     | CPU time (est.) | GPU time (est.) | Notes |
|---------------|-----------------|-----------------|-------|
| YOLO-World    | ~30 ms          | ~8 ms           | Batch size 1 |
| CLIP          | ~15 ms          | ~4 ms           | All crops batched |
| EasyOCR       | ~350 ms         | ~80 ms          | Largest bottleneck on CPU |
| **Total**     | **~400 ms**     | **~100 ms**     | Per image |

The pipeline is **CPU-deployable** for batch/offline use cases.
For real-time store analytics (multiple cameras), GPU is recommended.

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) GPU support – install the CUDA-enabled torch first
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    then re-run: pip install -r requirements.txt

# 4. Model weights are downloaded automatically on first run:
#    - YOLO-World: ~/.cache/ultralytics/
#    - CLIP ViT-B/32: ~/.cache/huggingface/ (or open_clip cache)
#    - EasyOCR: ~/.EasyOCR/
```

---

## Usage

```bash
# Analyse all three provided test images
python main.py --images images/shelf_dairy.jpg images/shelf_beverages.jpg images/shelf_snacks.jpg

# Process an entire folder
python main.py --images_dir images/

# GPU mode
python main.py --images_dir images/ --device cuda

# Tune detection sensitivity (lower conf = more detections, more noise)
python main.py --images_dir images/ --conf 0.18 --row_gap 60
```

Outputs are written to `outputs/`:
- `<name>_annotated.jpg` — image with bounding boxes, labels, legend, OCR strip
- `<name>_result.json`   — structured JSON result

---

## Output Format

```json
{
  "image_name": "shelf_beverages.jpg",
  "total_products": 47,
  "brands": {
    "Coca-Cola": 12,
    "Pepsi": 8,
    "Tropicana": 6,
    "Amul": 5,
    "Other": 16
  },
  "ocr_labels": ["₹50", "600 ml", "Sprite", "₹99", "1 L", "Tropicana", "₹125"],
  "share_of_shelf_pct": {
    "Coca-Cola": 28.4,
    "Pepsi": 17.6,
    "Tropicana": 13.2,
    "Amul": 11.0,
    "Other": 29.8
  },
  "shelf_rows_detected": 4,
  "annotated_image": "outputs/shelf_beverages_annotated.jpg"
}
```

---

## Assumptions & Limitations

| # | Assumption / Limitation |
|---|------------------------|
| 1 | **Frontal shelf images** — the pipeline assumes products face the camera. Angled or cluttered shelves reduce detection recall. |
| 2 | **CLIP brand accuracy** — zero-shot CLIP accuracy is ~60–75% per detection on densely packed shelves. A fine-tuned classifier on labelled retail images would be materially better. |
| 3 | **Shelf row detection** — relies on detected bounding boxes being representative of all shelf rows. If one row has no detections, its price tags may be missed by OCR. |
| 4 | **OCR language** — configured for English (`en`). Hindi/regional text on labels will be missed; adding `hi` to `OCR_LANGUAGES` in `config.py` would extend coverage. |
| 5 | **No OSA (Out-of-Stock)** — the pipeline detects presence but does not yet identify empty shelf gaps. This can be added by analysing unexpectedly large horizontal gaps between detected boxes within a row. |
| 6 | **YOLO-World fallback** — standard YOLOv8 (COCO fallback) only returns bottle/cup detections, significantly reducing coverage for packaged goods. Always use YOLO-World weights when possible. |

---

## Project Structure

```
retail_shelf_analysis/
├── pipeline/
│   ├── __init__.py
│   ├── detector.py        # YOLO-World product detection
│   ├── classifier.py      # CLIP zero-shot brand classification
│   ├── ocr_engine.py      # EasyOCR price tag extraction
│   ├── segmentation.py    # Shelf row clustering + SOS computation
│   └── visualizer.py      # Annotated image generation
├── main.py                # CLI entry point + ShelfAnalysisPipeline
├── config.py              # Brand prompts, thresholds, paths
├── requirements.txt
├── README.md
├── images/                # Place input shelf images here
└── outputs/               # Annotated images + JSON results (auto-created)
```
