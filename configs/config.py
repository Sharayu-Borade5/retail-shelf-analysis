"""
configs/config.py
=================
Central configuration for the Retail Shelf Analysis pipeline.
All tunable parameters, model settings, and brand definitions live here.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data", "images")
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
OUTPUT_RES_DIR = os.path.join(BASE_DIR, "outputs", "results")

for _d in [OUTPUT_VIS_DIR, OUTPUT_RES_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Detection ──────────────────────────────────────────────────────────────────
# Model: YOLO-World yolov8s-worldv2.pt (zero-shot open-vocabulary)
# Specified model: Grounding DINO — see README for why YOLO-World is used instead
DETECTION_CLASSES = [
    # Assignment-specified prompts
    "product package", "bottle", "can", "snack packet",
    "milk bottle", "juice carton",
    # Additional single-word nouns (YOLO-World scores these higher than phrases)
    "biscuit", "cracker", "cookie", "wafer",
]
DETECTION_CONF  = 0.12    # kept low; tile NMS handles duplicate suppression
DETECTION_IOU   = 0.45
MIN_BBOX_AREA   = 500     # px²; filters shelf fixtures / price-tag detections

# Tiled inference (SAHI-style) — dramatically improves recall on small products
TILE_SIZE       = 640
TILE_OVERLAP    = 0.30    # 30% overlap so edge products are seen by two tiles
NMS_IOU_MERGE   = 0.25    # more aggressive than per-tile IOU; removes tile duplicates

# ── Classification (OpenCLIP) ──────────────────────────────────────────────────
CLIP_MODEL      = "ViT-B/32"
CLIP_PRETRAINED = "openai"
CLIP_MIN_CONF   = 0.22    # below this → "Other"; prevents cross-category errors

# Visually-descriptive text prompts per brand.
# More specific = better CLIP discrimination.
BRAND_PROMPTS: dict[str, list[str]] = {
    # ── Beverages ──────────────────────────────────────────────────────────
    "Coca-Cola": [
        "Coca-Cola red plastic bottle with white script",
        "Coke dark cola beverage red bottle",
        "Diet Coke black slim can",
    ],
    "Pepsi": [
        "Pepsi blue cola bottle with red white globe",
        "Pepsi beverage dark blue can",
    ],
    "Sprite": [
        "Sprite green transparent bottle lemon lime soda",
        "Sprite clear green beverage",
    ],
    "7UP": [
        "7UP green bottle with 7up logo lemon lime",
        "7up clear refreshing lemon soda",
    ],
    "Mountain Dew": [
        "Mountain Dew neon green bottle citrus soda",
        "Mtn Dew green fluorescent beverage",
    ],
    "Fanta": [
        "Fanta orange fruit soda bottle",
        "Fanta bright orange refreshing drink",
    ],
    "Thums Up": [
        "Thums Up dark cola India bottle",
        "Thums Up strong cola brown drink",
    ],
    "Limca": [
        "Limca lemon lime soda pale green bottle India",
    ],
    "Mirinda": [
        "Mirinda orange soda bottle orange coloured",
    ],
    "Nestea": [
        "Nestea iced tea ready to drink bottle",
    ],
    "Lipton": [
        "Lipton ice tea bottle yellow label tea",
        "Lipton lemon peach iced tea drink",
    ],
    "Tropicana": [
        "Tropicana orange juice carton tetra pak",
        "Tropicana fruit juice pack with orange image",
    ],
    "Real": [
        "Real fruit juice carton Dabur India",
        "Real juice tetrapak with fruit image",
    ],
    "Minute Maid": [
        "Minute Maid juice drink bottle Coca-Cola",
        "Minute Maid orange pulpy juice",
    ],
    "B Natural": [
        "B Natural ITC fruit juice bottle",
    ],
    "Paper Boat": [
        "Paper Boat traditional Indian drink pouch",
        "Paper Boat aam panna or kokum drink",
    ],
    "Red Bull": [
        "Red Bull energy drink silver blue slim can",
        "Red Bull blue can energy drink",
    ],
    "Gatorade": [
        "Gatorade sports electrolyte drink bottle",
        "Gatorade colourful bottle",
    ],
    "Nescafe": [
        "Nescafe coffee instant coffee bottle brown",
        "Nescafe Classic red coffee jar",
    ],
    "Amul Kool": [
        "Amul Kool flavoured milk bottle",
        "Amul Kool chocolate or badam milk",
    ],
    # ── Dairy ──────────────────────────────────────────────────────────────
    "Amul": [
        "Amul milk bottle India red white",
        "Amul Taaza milk blue bottle",
        "Amul Shakti milk bottle yellow",
        "Amul Gold full cream milk bottle",
        "Amul butter yellow box",
        "Amul cheese product India",
        "Amul Pro nutritional drink",
        "Amul dairy product India",
    ],
    "Mother Dairy": [
        "Mother Dairy milk bottle blue white India",
        "Mother Dairy toned milk",
    ],
    "Danone": [
        "Danone Activia yoghurt cup with fruit",
        "Danone Actimel probiotic small bottle",
    ],
    "Nestle": [
        "Nestle A+ milk bottle blue India",
        "Nestle dairy milk product",
    ],
    "Yakult": [
        "Yakult probiotic fermented milk small bottle",
        "Yakult small white or pink drink",
    ],
    "Epigamia": [
        "Epigamia Greek yoghurt cup",
        "Epigamia milk shake bottle",
    ],
    "Hersheys": [
        "Hersheys chocolate milk shake brown bottle",
        "Hersheys syrup dark brown",
    ],
    "Milky Mist": [
        "Milky Mist butter or paneer India",
        "Milky Mist dairy product yellow",
    ],
    "Britannia Dairy": [
        "Britannia cheese slices packet yellow",
        "Britannia dairy product cheese",
    ],
    # ── Snacks ─────────────────────────────────────────────────────────────
    "Lays": [
        "Lays potato chips bag yellow",
        "Lay's flavoured chips packet yellow red",
    ],
    "Doritos": [
        "Doritos tortilla chips bag triangular",
        "Doritos nacho chips colourful bag",
    ],
    "Cheetos": [
        "Cheetos cheese puffs orange bag",
        "Cheetos crunchy snack orange packet",
    ],
    "Pringles": [
        "Pringles potato crisps cylindrical tube can",
        "Pringles chips red green yellow can",
    ],
    "Kurkure": [
        "Kurkure spicy puffed snack orange packet India",
        "Kurkure Indian namkeen snack",
    ],
    "Bingo": [
        "Bingo Mad Angles snack triangular chips",
        "Bingo chips Indian snack orange",
    ],
    "Uncle Chipps": [
        "Uncle Chipps potato chips India packet",
    ],
    "Malkist": [
        "Malkist cracker biscuit Monde Nissin",
        "Malkist chocolate cream biscuit",
    ],
    "Oreo": [
        "Oreo chocolate sandwich cookie blue packet",
        "Oreo cream biscuit with filling",
    ],
    "Parle": [
        "Parle-G glucose biscuit yellow packet India",
        "Monaco salted cracker Parle",
        "Parle Hide and Seek chocolate biscuit",
    ],
    "Britannia": [
        "Britannia Good Day cashew butter cookies",
        "Britannia Marie Gold thin biscuit",
        "Britannia Tiger glucose biscuit",
        "Britannia Milano dark chocolate cookies",
        "Britannia Hide and Seek Fab biscuit",
    ],
    "ITC Dark Fantasy": [
        "Dark Fantasy choco fills biscuit dark purple blue ITC",
        "ITC Dark Fantasy choco fills cookie dark packaging",
    ],
    "Other": [
        "generic unbranded retail product on supermarket shelf",
    ],
}

# ── OCR ────────────────────────────────────────────────────────────────────────
# Primary: PaddleOCR; auto-fallback to EasyOCR if PaddleOCR not installed
OCR_LANGUAGES = ["en"]
OCR_CONF      = 0.45

# ── Shelf Segmentation ─────────────────────────────────────────────────────────
# Hough Lines approach (as specified in the assignment)
CANNY_LOW        = 50
CANNY_HIGH       = 150
HOUGH_THRESHOLD  = 80     # minimum votes for a line to be accepted
MIN_LINE_LENGTH  = 150    # minimum shelf bar length in pixels
MAX_LINE_GAP     = 30     # maximum gap allowed within a shelf line
ROW_GAP_MIN      = 40     # minimum pixel gap between two shelf rows

# ── Visualization ──────────────────────────────────────────────────────────────
VIS_FONT_SCALE = 0.42
VIS_BOX_THICK  = 2
VIS_LEGEND_W   = 230    # width of legend panel appended to right of image
