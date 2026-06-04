# Retail Shelf Analysis Pipeline

An end-to-end ML inference pipeline that analyses retail shelf images and
generates brand-level shelf presence, share-of-shelf, and OCR insights.

---

## Architecture

```mermaid
flowchart TD
    A[Input Image] --> B[Shelf Segmentation\nCanny + Hough Lines]
    A --> F[PaddleOCR\nPrice Tag Extraction]
    B --> C[YOLO-World Detection\nZero-shot + Tiled Inference]
    C --> D[Crop Products]
    D --> E[OpenCLIP Classification\nBrand Similarity]
    E --> G[Business Metrics\nSOS + OSA]
    F --> G
    G --> H[Visualization\noutputs/visualizations/]
    G --> I[JSON Output\noutputs/results/]
```

See [`docs/architecture.md`](docs/architecture.md) for detailed data-flow diagrams.

---

## Project Structure

```
retail_shelf_analysis/
├── main.py                      # Pipeline entry point + CLI
├── requirements.txt
├── README.md
│
├── configs/
│   └── config.py                # All parameters + brand prompts
│
├── src/
│   ├── detector.py              # YOLO-World zero-shot detection
│   ├── classifier.py            # OpenCLIP brand classification
│   ├── ocr.py                   # PaddleOCR / EasyOCR extraction
│   ├── segmentation.py          # Hough Lines shelf row detection
│   ├── metrics.py               # SOS, OSA, report generation
│   └── visualization.py         # Annotated image generator
│
├── data/
│   └── images/                  # Input shelf images
│
├── outputs/
│   ├── visualizations/          # Annotated JPEG outputs
│   └── results/                 # JSON result files
│
└── docs/
    └── architecture.md          # Mermaid architecture diagram
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle-gpu    # GPU PaddlePaddle

# 4. Model weights are downloaded automatically on first run:
#    YOLO-World  : ~/.cache/ultralytics/
#    OpenCLIP    : ~/.cache/huggingface/ or open_clip cache
#    PaddleOCR   : ~/.paddleocr/
#    EasyOCR     : ~/.EasyOCR/
```

---

## Usage

```bash
# Process all images in data/images/ (default)
python main.py

# Process specific images
python main.py --images data/images/shelf_snacks.jpg data/images/shelf_beverages.jpg

# Process a folder
python main.py --images_dir data/images/

# GPU mode
python main.py --images_dir data/images/ --device cuda
```

Outputs:
- `outputs/visualizations/<name>_annotated.jpg` — annotated image
- `outputs/results/<name>_result.json`           — structured JSON
- `outputs/results/all_results.json`             — combined JSON

---

## Output Format

```json
{
  "image_name": "shelf_beverages.jpg",
  "total_products": 52,
  "brands": {
    "Coca-Cola": 10,
    "Pepsi": 8,
    "Real": 6,
    "Other": 28
  },
  "share_of_shelf": {
    "Coca-Cola": 22.4,
    "Pepsi": 16.8,
    "Real": 12.1,
    "Other": 48.7
  },
  "ocr_labels": ["₹50", "₹99", "₹125", "Tropicana", "NESTEA"],
  "shelf_rows_detected": 4,
  "segmentation_method": "hough",
  "annotated_image": "outputs/visualizations/shelf_beverages_annotated.jpg"
}
```

---

## Model Selection Rationale

### 1. Detection — YOLO-World (`yolov8s-worldv2.pt`)

| Criterion | Decision |
|-----------|----------|
| **Specified** | Grounding DINO |
| **Used** | YOLO-World |
| **Why** | Grounding DINO requires CUDA compilation on Windows which fails on CPU-only machines. YOLO-World provides identical open-vocabulary zero-shot detection via free-text class prompts (`"bottle"`, `"biscuit"`, `"snack packet"`) with a simple `pip install ultralytics`. Both models share the same paradigm: large-scale vision-language pretraining enabling detection of arbitrary object categories without fine-tuning. |
| **Tiled Inference** | Large shelf images (1400×1100 px) cause YOLO-World to miss small flat products occupying <0.5% of image area. SAHI-style 640×640 tiling with 30% overlap improves recall from ~23 → ~55 detected snack packs. |
| **Speed** | 3–5 s/image CPU (full-image + ~12 tiles). ~500 ms on GPU. |

### 2. Classification — OpenCLIP (`ViT-B/32`, OpenAI)

| Criterion | Decision |
|-----------|----------|
| **Why** | Zero-shot classification requires no labelled retail dataset. CLIP was pretrained on 400M image–text pairs and recognises branded FMCG packaging via descriptive text prompts. |
| **Confidence Gate** | `CLIP_MIN_CONF = 0.22`: if the best cosine similarity is below threshold, the crop is labelled "Other". Eliminates cross-category errors (e.g. a Sprite bottle matching "ITC Dark Fantasy biscuit"). |
| **Speed** | ~15 ms/image on CPU (all crops batched). |
| **Alternative** | A fine-tuned EfficientNet-B0 on a retail dataset would be more accurate but requires labelled data — not available in a zero-shot setting. |

### 3. OCR — PaddleOCR (primary) / EasyOCR (fallback)

| Criterion | Decision |
|-----------|----------|
| **Why PaddleOCR** | PaddleOCR outperforms EasyOCR on small, densely-packed price tag text and handles mixed-font Indian retail labels reliably. Specified explicitly in the assignment. |
| **Fallback** | EasyOCR is used automatically if `paddleocr` is not installed, ensuring the pipeline runs on all environments. |
| **Noise Filter** | A 6-stage filter removes garbage tokens: length check, bare number rejection, noise-char ratio, Levenshtein brand allowlist, random-capitalisation heuristic, real-word / price pattern check. |
| **Speed** | ~300–500 ms/image on CPU. |

### 4. Shelf Segmentation — Hough Lines + Geometric Fallback

| Criterion | Decision |
|-----------|----------|
| **Primary** | Canny edge detection → `HoughLinesP` to find horizontal shelf separator bars. Physical shelf bars are long, straight, high-contrast horizontal lines — Hough transform detects them reliably. |
| **Fallback** | When Hough finds fewer than 2 lines (very dark images or no visible bars), Y-centres of detected bounding boxes are clustered geometrically. |
| **Why not SAM2** | SAM2 adds ~1 s/image latency and is not needed when shelf rows are horizontal bands — a well-known image prior we can exploit cheaply. |

---

## CPU vs GPU

| Component | CPU | GPU | Bottleneck |
|-----------|-----|-----|------------|
| YOLO-World (full image) | ~30 ms | ~8 ms | — |
| YOLO-World (tiled, 12 tiles) | ~2–4 s | ~200 ms | Main CPU bottleneck |
| OpenCLIP | ~15 ms | ~4 ms | — |
| PaddleOCR | ~350 ms | ~80 ms | — |
| **Total per image** | **~5–8 s** | **~500 ms** | |

The pipeline is fully **CPU-deployable** for batch/offline store audits.
For real-time multi-camera analytics, GPU is recommended.

---

## Accuracy vs Speed Tradeoffs

| Setting | Detection Recall | Latency | Notes |
|---------|-----------------|---------|-------|
| Full-image only | Low (misses small packs) | Fast | Not recommended for dense shelves |
| Tiled inference (default) | High | Moderate | Best balance |
| Lower `DETECTION_CONF` | Higher recall, more FP | Same | Tune per shelf type |
| Larger CLIP model (ViT-L) | Better brand accuracy | +3× slower | Use if GPU available |

---

## Assumptions & Limitations

| # | Assumption / Limitation |
|---|------------------------|
| 1 | **Frontal shelf images** — products must face the camera. Angled or heavily occluded products reduce detection recall. |
| 2 | **Zero-shot CLIP accuracy ~60–75%** — brands with visually similar packaging (Amul Shakti vs Amul Gold) may be confused. A fine-tuned classifier would improve accuracy significantly. |
| 3 | **Tiled inference overhead** — SAHI-style tiling adds 2–4 s/image on CPU. For real-time use, GPU inference or a faster detector (RT-DETR) is preferred. |
| 4 | **OCR language** — configured for English only. Adding `"hi"` to `OCR_LANGUAGES` in `configs/config.py` extends to Hindi. |
| 5 | **OCR noise** — EasyOCR/PaddleOCR occasionally produces garbled tokens on small or curved labels. The 6-stage filter removes most noise but some short random-letter tokens may remain. |
| 6 | **Prototype OSA** — On-Shelf Availability is approximated as product count per brand. True OSA requires a planogram to identify expected vs actual product positions. |
| 7 | **YOLO-World fallback model** — if YOLO-World weights fail to download, `yolov8m.pt` (COCO) is used as fallback; this only detects `bottle` and `cup` classes, severely reducing coverage. |

---

## 🚀 Production Pipeline Redesign (Architectural Critique)

While the zero-shot prototype provides a fast, training-free baseline, deploying a retail-shelf intelligence system at scale requires addressing several fundamental limitations. Below is an architectural critique and a blueprint for a production-grade pipeline.

### Proposed Production Architecture

```mermaid
flowchart TD
    A[Input Image] --> B[Fine-Tuned YOLOv11-seg]
    B --> C[Product Masks & Crops]
    B --> D[Shelf Rows & Space Bands]
    C --> E[DINOv2 Feature Extractor]
    E --> F[FAISS Reference Gallery\nTop-K SKU Candidates]
    C --> G[Crop OCR\nProduct Packaging Text]
    A --> H[Shelf Label & Price Tag OCR]
    G --> I[Bayesian Evidence Fusion\nScore = w_vis * Vis + w_ocr * OCR + w_ctx * Context]
    F --> I
    H --> I
    D --> I
    I --> J[Final SKU Predictions]
    J --> K[Shelf Analytics\nSOS + OSA + Planogram Alignment]
```

---

### Key Redesign Components

#### 1. Zero-Shot Detection vs. Fine-tuned Instance Segmentation
* **Prototype Issue:** YOLO-World/Grounding DINO relies on zero-shot class descriptions (`"bottle"`, `"snack packet"`). Because the detector has never seen the actual products, it struggles with retail-shelf boundaries, causing adjacent bags to merge into a single detection, or logos and price tags to trigger false detections.
* **Production Redesign:** Fine-tune a custom instance segmentation model (e.g., **YOLOv11-seg** or **RT-DETR**) on the provided COCO brand dataset (`Brand Detection.coco.zip`).
* **Impact:** Fine-tuning teaches the model exact packing contours and aspect ratios. Pixel-precise instance masks replace axis-aligned bounding boxes, yielding highly accurate Share of Shelf (SOS) calculations and preventing double-counting on overlapping items.

#### 2. Deterministic "OCR-First" Mappings vs. Bayesian Evidence Fusion
* **Prototype Issue:** Mappings are dictated by an OCR-first dictionary look-up. If the word `"masala"` is read, the system immediately maps the crop to `"Lay's Magic Masala"`. If the product is actually a competitor pack (e.g., *Kurkure Masala* or *Bingo Masala*), the visual classifier is bypassed, leaking incorrect brand metrics.
* **Production Redesign:** Replace the hard-override logic with a **Bayesian Evidence Fusion** scoring function:
  $$\text{Score}(Brand) = w_{\text{visual}} \cdot P(Brand \mid \text{Visual}) + w_{\text{ocr}} \cdot P(Brand \mid \text{OCR}) + w_{\text{context}} \cdot P(Brand \mid \text{Context})$$
* **Impact:** OCR results contribute as probabilistic evidence instead of absolute truth. If the visual similarity for *Kurkure* is $0.90$ and the OCR extracts the token `"masala"`, the system correctly labels the product *Kurkure Masala* instead of overriding it to *Lay's*.

#### 3. General CLIP Classifier vs. Custom SKU metric learning
* **Prototype Issue:** General-purpose vision-language models like OpenCLIP ViT-B/32 are trained on generic internet imagery. They easily recognize broad concepts but cannot distinguish between visually similar SKUs under the same parent brand (e.g., *Amul Gold* vs. *Amul Taaza* vs. *Amul Shakti* milk packets).
* **Production Redesign:** Replace CLIP with a metric learning backbone (e.g., ConvNeXt or Vision Transformer) fine-tuned on the brand dataset using ArcFace or Triplet Loss to map product crops into a tight, highly-discriminative SKU embedding space.

#### 4. IoU-Based NMS vs. Embedding-Based Duplicate Suppression
* **Prototype Issue:** Setting fixed IoU thresholds is a trade-off: a lower threshold (e.g., `0.38` for beverages) merges valid adjacent bottles, while a higher threshold (e.g., `0.60` for snacks) fails to suppress overlapping double-detections of the same pack.
* **Production Redesign:** Extract feature embeddings for overlapping bounding boxes using a foundation model like **DINOv2**. Overlapping regions with high cosine similarity ($\ge 0.90$) are automatically collapsed into a single facing. This prevents double-counting without merging adjacent distinct products.

#### 5. Strip-Only OCR vs. Multi-Zone Text Fusion
* **Prototype Issue:** OCR is restricted to localized shelf separator strips. Text printed on the actual product packaging is completely ignored.
* **Production Redesign:** Run text extraction across three distinct zones:
  1. **Product Face Crop:** To identify variant text (e.g., `"Classic"`, `"Spicy Treat"`, `"Sugar Free"`).
  2. **Direct Shelf Label:** Located immediately beneath the product to extract barcodes and POS descriptions.
  3. **Adjacent Price Tags:** To read promotions and pricing details.
* **Impact:** All extracted text tokens are compiled and passed to the fusion engine, improving resolution of ambiguous packaging variants.

#### 6. Missing Visual Retrieval (DINOv2 + FAISS)
* **Prototype Issue:** Whenever a brand updates its packaging graphic or introduces a new SKU, standard supervised classifiers must be retrained.
* **Production Redesign:** Build a visual search engine:
  * Extract features from detected product crops using a frozen **DINOv2-ViT-B** or **DINOv2-ViT-L** backbone.
  * Index a reference gallery of standard SKU packaging photos in a **FAISS** vector database.
  * Use k-Nearest Neighbors (k-NN) search to retrieve the closest matching SKU in the reference gallery.
* **Impact:** Onboarding new products or seasonal packaging designs requires simply dropping a reference photo into the gallery directory—requiring **zero model retraining**.

