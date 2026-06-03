import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Detection ────────────────────────────────────────────────────────────────
# YOLO-World open-vocabulary classes.
# IMPORTANT: Keep classes as short, concrete nouns — multi-word phrases score
# lower with YOLO-World than single-word category names.
DETECTION_CLASSES = [
    # Generic containers
    "bottle", "can", "carton", "bag", "packet", "box", "cup", "container", "jar",
    # Snack-specific (single-word; YOLO-World handles these well)
    "biscuit", "cracker", "cookie", "wafer", "chips", "snack",
    # Broader fallbacks
    "food package", "product",
]
DETECTION_CONF = 0.12   # low; tile-level NMS removes duplicates
DETECTION_IOU  = 0.45
MIN_BBOX_AREA  = 300    # tiles are smaller so raw pixel areas are smaller

# ── Tiled inference (SAHI-style) ─────────────────────────────────────────────
# Tiles the image into overlapping 640×640 patches so small flat packs that
# YOLO-World misses at full resolution become visible at tile scale.
TILE_SIZE    = 640
TILE_OVERLAP = 0.30   # 30% overlap between adjacent tiles

# ── Classification (CLIP) ────────────────────────────────────────────────────
CLIP_MODEL      = "ViT-B/32"
CLIP_PRETRAINED = "openai"

# Each brand maps to a list of descriptive text prompts.
# CLIP selects the brand whose average prompt similarity is highest.
BRAND_PROMPTS: dict[str, list[str]] = {
    # ── Beverages ─────────────────────────────────────────────────────────
    "Coca-Cola": [
        "a bottle of Coca-Cola with red label",
        "Coke soft drink bottle",
        "Thums Up cola bottle with red label",
        "Diet Coke bottle",
    ],
    "Pepsi": [
        "a bottle of Pepsi with blue label",
        "Pepsi cola blue can",
        "Pepsi soft drink bottle",
    ],
    "Sprite": [
        "a green Sprite soda bottle",
        "Sprite lemon lime drink bottle",
    ],
    "Fanta": [
        "orange Fanta bottle",
        "Fanta orange soda",
    ],
    "Limca": [
        "Limca lemon soda green bottle",
        "Limca drink bottle",
    ],
    "Mountain Dew": [
        "Mountain Dew green bottle",
        "green Mountain Dew soda",
    ],
    "7UP": [
        "7UP clear bottle with green label",
        "Seven Up soda bottle",
    ],
    "Mirinda": [
        "Mirinda orange drink bottle",
        "Mirinda soda",
    ],
    "Tropicana": [
        "Tropicana orange juice carton",
        "Tropicana fruit juice pack",
        "Tropicana juice tetra pack",
    ],
    "Real": [
        "Real fruit juice carton",
        "Dabur Real juice pack",
        "Real richly juice tetra pack",
    ],
    "Minute Maid": [
        "Minute Maid pulpy orange juice",
        "Minute Maid juice carton",
        "Minute Maid apple juice",
    ],
    "Nestea": [
        "Nestea iced tea bottle",
        "Nestle Nestea green bottle",
    ],
    "Lipton": [
        "Lipton iced tea bottle",
        "Lipton lemon tea",
    ],
    "Gatorade": [
        "Gatorade sports drink bottle",
        "Gatorade electrolyte drink",
    ],
    "Red Bull": [
        "Red Bull energy drink can",
        "Red Bull slim can",
    ],
    "Nescafe": [
        "Nescafe coffee drink bottle",
        "Nescafe Classic coffee",
    ],
    "B Natural": [
        "B Natural fruit drink bottle",
        "ITC B Natural juice bottle",
    ],
    "Paper Boat": [
        "Paper Boat drink pouch",
        "Paper Boat aam panna drink",
        "Paper Boat traditional drink",
    ],
    # ── Dairy ─────────────────────────────────────────────────────────────
    "Amul": [
        "Amul milk bottle with blue label",
        "Amul Gold milk",
        "Amul Taaza toned milk",
        "Amul butter yellow wrapper",
        "Amul cheese block",
        "Amul Kool flavoured milk",
        "Amul Shakti milk",
        "Amul Pro chocolate drink",
        "Amul cheese spread",
        "Amul Royale flavoured milk",
    ],
    "Mother Dairy": [
        "Mother Dairy toned milk bottle",
        "Mother Dairy milk blue bottle",
    ],
    "Nestle": [
        "Nestle a+ yogurt cup",
        "Nestle dahi curd",
        "Nestle a+ dahi",
    ],
    "Yakult": [
        "Yakult probiotic drink small bottle",
        "Yakult fermented milk drink",
    ],
    "Danone": [
        "Activia yogurt cup",
        "Actimel probiotic drink bottle",
        "Danone yogurt",
    ],
    "Epigamia": [
        "Epigamia Greek yogurt",
        "Epigamia milk shake bottle",
    ],
    "Hersheys": [
        "Hersheys chocolate milk shake bottle",
        "Hersheys cocoa drink",
    ],
    "Milky Mist": [
        "Milky Mist butter packet",
        "Milky Mist cheese",
        "Milky Mist mozzarella",
    ],
    "Britannia Dairy": [
        "Britannia cheese slices packet",
        "Britannia cheese",
    ],
    # ── Snacks ────────────────────────────────────────────────────────────
    "Lays": [
        "Lay's potato chips yellow bag",
        "Lays Classic chips bag",
        "Lays India Magic Masala chips",
        "Lays Chile Limon chips",
    ],
    "Doritos": [
        "Doritos nacho chips triangular bag",
        "Doritos Cool Ranch chips",
        "Doritos tortilla chips",
    ],
    "Cheetos": [
        "Cheetos puffed corn snack bag",
        "Cheetos orange crunchy snack",
    ],
    "Kurkure": [
        "Kurkure Indian snack bag",
        "Kurkure puffed snack",
    ],
    "Uncle Chipps": [
        "Uncle Chipps potato chips bag",
        "Uncle Chipps plain salted",
    ],
    "Bingo": [
        "Bingo chips packet",
        "Bingo Mad Angles snack",
        "Bingo Original Style chips",
    ],
    "Pringles": [
        "Pringles chips cylindrical can",
        "Pringles potato crisps tube",
    ],
    "Parle": [
        "Parle-G glucose biscuits yellow packet",
        "Monaco salted crackers",
        "Parle Hide and Seek biscuits",
        "Parle biscuit pack",
    ],
    "Britannia": [
        "Britannia Good Day cashew cookies",
        "Britannia Marie Gold biscuits",
        "Britannia Tiger biscuits",
        "Britannia Hide and Seek Fab",
        "Britannia Milano cookies",
    ],
    "Oreo": [
        "Oreo chocolate sandwich cookies",
        "Oreo cream biscuit blue packet",
    ],
    "ITC Snacks": [
        "Dark Fantasy choco fills biscuits",
        "Bingo Mad Angles achaari masti",
    ],
    "Malkist": [
        "Malkist cracker biscuits",
        "Malkist layered crackers",
    ],
    "Other": [
        "generic product on a retail shelf",
        "unknown packaged food product",
    ],
}

# ── OCR ─────────────────────────────────────────────────────────────────────
OCR_LANGUAGES   = ["en"]
OCR_CONF        = 0.45   # raised from 0.35 to suppress low-confidence junk
# Price-tag rows tend to sit in the bottom 18% of each shelf band
PRICE_ROW_FRAC  = 0.18

# ── Shelf segmentation ───────────────────────────────────────────────────────
# Minimum vertical gap (px) to split detections into separate shelf rows
SHELF_ROW_GAP   = 40
