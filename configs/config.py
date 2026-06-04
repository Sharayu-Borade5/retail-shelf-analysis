"""
configs/config.py  —  Retail Shelf Analysis
All tunable parameters, brand prompts, and brand dictionary live here.
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
DETECTION_CLASSES = [
    "product package", "bottle", "can", "snack packet",
    "milk bottle", "juice carton",
    "product box", "biscuit box", "cookie box", "snack box", "package box"
]
DETECTION_CONF  = 0.10
DETECTION_IOU   = 0.45
MIN_BBOX_AREA   = 2000     # px²  — filters logo / shelf-label detections

# Tiled inference (SAHI-style)
TILE_SIZE       = 640
TILE_OVERLAP    = 0.30

# NMS Configuration — lower IOU = more aggressive duplicate removal
NMS_IOU_MERGE      = 0.38   # was 0.60; key fix for Thums Up=7→3, MtnDew=7→4
NMS_CONTAIN_THRESH = 0.45   # was 0.60; aggressively removes label/logo sub-boxes

# ── OCR-first Brand Dictionary ────────────────────────────────────────────────
# Maps normalised OCR text (lowercase, alphanumeric only) to canonical SKU names.
BRAND_DICTIONARY: dict[str, str] = {
    # ── Lay's ──────────────────────────────────────────────────────────────
    "lays classic": "Lay's Classic",
    "classic": "Lay's Classic",
    "lays magic masala": "Lay's Magic Masala",
    "magic masala": "Lay's Magic Masala",
    "masala": "Lay's Magic Masala",
    "mosolo": "Lay's Magic Masala",
    "indios magic": "Lay's Magic Masala",
    "lays chile limon": "Lay's Chile Limon",
    "chile limon": "Lay's Chile Limon",
    "chile": "Lay's Chile Limon",
    "limon": "Lay's Chile Limon",
    "lays tomato tango": "Lay's Tomato Tango",
    "tomato tango": "Lay's Tomato Tango",
    "tomato": "Lay's Tomato Tango",
    "tango": "Lay's Tomato Tango",
    "ango": "Lay's Tomato Tango",
    "lays": "Lay's",
    "lay's": "Lay's",
    "lay": "Lay's",
    
    # ── Doritos ────────────────────────────────────────────────────────────
    "doritos": "Doritos",
    "dorito": "Doritos",
    "nacho": "Doritos",
    "ranch": "Doritos",
    
    # ── Cheetos ────────────────────────────────────────────────────────────
    "cheetos": "Cheetos",
    "cheeto": "Cheetos",
    
    # ── Kurkure ────────────────────────────────────────────────────────────
    "kurkure": "Kurkure",
    "kurkur": "Kurkure",
    
    # ── Uncle Chipps ───────────────────────────────────────────────────────
    "uncle chipps": "Uncle Chipps",
    "uncle chips": "Uncle Chipps",
    "chipps": "Uncle Chipps",
    "chips": "Uncle Chipps",
    
    # ── Bingo ──────────────────────────────────────────────────────────────
    "bingo": "Bingo",
    "mad angles": "Bingo",
    "achaari": "Bingo",
    "achaari masti": "Bingo",
    "achaarimasti": "Bingo",
    
    # ── Pringles ───────────────────────────────────────────────────────────
    "pringles": "Pringles",
    
    # ── Parle ──────────────────────────────────────────────────────────────
    "monaco": "Monaco",
    "parle-g": "Parle-G",
    "parle g": "Parle-G",
    "parle": "Parle-G",
    "hide & seek": "Hide & Seek",
    "hide seek": "Hide & Seek",
    "hide & seek fab": "Hide & Seek Fab",
    "fab": "Hide & Seek Fab",
    "hide & seek milano": "Hide & Seek Milano",
    "milano": "Hide & Seek Milano",
    
    # ── Britannia ──────────────────────────────────────────────────────────
    "good day": "Good Day",
    "marie gold": "Marie Gold",
    "marie": "Marie Gold",
    "tiger krunch": "Tiger Krunch",
    "tiger": "Tiger Krunch",
    "krunch": "Tiger Krunch",
    "krunck": "Tiger Krunch",
    "britannia": "Britannia",
    
    # ── Oreo ───────────────────────────────────────────────────────────────
    "oreo": "Oreo",
    "oreos": "Oreo",
    
    # ── Dark Fantasy ───────────────────────────────────────────────────────
    "dark fantasy": "Dark Fantasy",
    "dark vantasy": "Dark Fantasy",
    "daak vantasy": "Dark Fantasy",
    
    # ── Malkist ────────────────────────────────────────────────────────────
    "malkist": "Malkist",
    
    # ── Amul ───────────────────────────────────────────────────────────────
    "amul kool": "Amul Kool",
    "kool": "Amul Kool",
    "amul gold": "Amul Gold",
    "amul taaza": "Amul Taaza",
    "taaza": "Amul Taaza",
    "taazo": "Amul Taaza",
    "amul shakti": "Amul Shakti",
    "shakti": "Amul Shakti",
    "amul royale": "Amul Royale",
    "royale": "Amul Royale",
    "amul pro": "Amul Pro Milk",
    "amul butter": "Amul Butter",
    "amul lite": "Amul Lite Butter",
    "amul cheese block": "Amul Cheese Block",
    "amul cheese spread": "Amul Cheese Spread",
    "amul pizza cheese": "Amul Pizza Cheese",
    "amul pizza": "Amul Pizza Cheese",
    "amul kool": "Amul Kool",
    "amul": "Amul",
    
    # ── Mother Dairy ───────────────────────────────────────────────────────
    "mother dairy": "Mother Dairy Toned Milk",
    "mother": "Mother Dairy Toned Milk",
    
    # ── Yakult ─────────────────────────────────────────────────────────────
    "yakult": "Yakult",
    "yakuld": "Yakult",
    
    # ── Nestle ─────────────────────────────────────────────────────────────
    "nestle a yoghurt": "Nestle A Yoghurt",
    "yoghurt": "Nestle A Yoghurt",
    "nestle a dahi": "Nestle A Dahi",
    "dahi": "Nestle A Dahi",
    "nestle": "Nestle",
    "nestle a+": "Nestle",
    
    # ── Activia ────────────────────────────────────────────────────────────
    "activia": "Activia",
    
    # ── Actimel ────────────────────────────────────────────────────────────
    "actimel": "Actimel",
    "actirel": "Actimel",
    
    # ── Epigamia ───────────────────────────────────────────────────────────
    "epigamia": "Epigamia Milk Shake",
    
    # ── Hershey's ──────────────────────────────────────────────────────────
    "hersheys": "Hershey's",
    "hershey's": "Hershey's",
    "hershey": "Hershey's",
    
    # ── Milky Mist ─────────────────────────────────────────────────────────
    "milky mist butter": "Milky Mist Butter",
    "milky mist mozzarella": "Milky Mist Mozzarella",
    "mozzarella": "Milky Mist Mozzarella",
    "milky mist pizza cheese": "Milky Mist Pizza Cheese",
    "pizza cheese": "Milky Mist Pizza Cheese",
    "milky mist": "Milky Mist",
    
    # ── Britannia Dairy ────────────────────────────────────────────────────
    "britannia cheese slices": "Britannia Cheese Slices",
    "cheese slices": "Britannia Cheese Slices",
    "go cheese slices": "Go Cheese Slices",
    "go cheese": "Go Cheese Slices",
    
    # ── Tropicana ──────────────────────────────────────────────────────────
    "tropicana": "Tropicana",
    
    # ── Real ───────────────────────────────────────────────────────────────
    "real juice": "Real Juice",
    "real": "Real Juice",
    
    # ── Minute Maid ────────────────────────────────────────────────────────
    "minute maid": "Minute Maid",
    
    # ── Sprite ─────────────────────────────────────────────────────────────
    "sprite": "Sprite",
    
    # ── Coca-Cola ──────────────────────────────────────────────────────────
    "coca-cola": "Coca-Cola",
    "coca cola": "Coca-Cola",
    "coke": "Coca-Cola",
    "diet coke": "Diet Coke",
    "diet": "Diet Coke",
    
    # ── Thums Up ───────────────────────────────────────────────────────────
    "thums up": "Thums Up",
    "thumbs up": "Thums Up",
    
    # ── Fanta ──────────────────────────────────────────────────────────────
    "fanta": "Fanta",
    
    # ── Limca ──────────────────────────────────────────────────────────────
    "limca": "Limca",
    "limcd": "Limca",
    
    # ── Pepsi ──────────────────────────────────────────────────────────────
    "pepsi": "Pepsi",
    
    # ── Mirinda ────────────────────────────────────────────────────────────
    "mirinda": "Mirinda",
    "mirinma": "Mirinda",
    
    # ── 7UP ────────────────────────────────────────────────────────────────
    "7up": "7UP",
    
    # ── Mountain Dew ───────────────────────────────────────────────────────
    "mountain dew": "Mountain Dew",
    
    # ── Lipton ─────────────────────────────────────────────────────────────
    "lipton ice tea": "Lipton Ice Tea",
    "lipton": "Lipton Ice Tea",
    
    # ── Nestea ─────────────────────────────────────────────────────────────
    "nestea": "Nestea",
    
    # ── Paper Boat ─────────────────────────────────────────────────────────
    "paper boat": "Paper Boat",
    "paperboat": "Paper Boat",
    "poper boat": "Paper Boat",
    
    # ── Gatorade ───────────────────────────────────────────────────────────
    "gatorade": "Gatorade",
    
    # ── Red Bull ───────────────────────────────────────────────────────────
    "red bull": "Red Bull",
    
    # ── Nescafe ────────────────────────────────────────────────────────────
    "nescafe": "Nescafe",
    
    # ── B Natural ──────────────────────────────────────────────────────────
    "b natural": "B Natural",
}

# ── OCR ────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES  = ["en"]
OCR_CONF       = 0.40
PRICE_PATTERN  = r"[₹$€£]\s*\d+[\.,]?\d*"

# ── Shelf Row Segmentation (Hough Lines) ──────────────────────────────────────
CANNY_LOW        = 50
CANNY_HIGH       = 150
HOUGH_THRESHOLD  = 80
MIN_LINE_LENGTH  = 150
MAX_LINE_GAP     = 30
ROW_GAP_MIN      = 40

# ── Visualization ──────────────────────────────────────────────────────────────
VIS_FONT_SCALE = 0.45
VIS_BOX_THICK  = 2
VIS_LEGEND_W   = 260

# ── Classification (OpenCLIP fallback) ────────────────────────────────────────
CLIP_MODEL      = "ViT-B/32"
CLIP_PRETRAINED = "openai"
CLIP_MIN_CONF   = 0.22     # Accept lower but valid predictions matching allowed category

# Brand text prompts for CLIP fallback classification (detailed SKU level)
BRAND_PROMPTS: dict[str, list[str]] = {
    # Snacks Shelf
    "Lay's Classic":        ["Lay's classic salted chips yellow bag"],
    "Lay's Magic Masala":   ["Lays magic masala spicy chips blue bag"],
    "Lay's Chile Limon":    ["Lays chile limon green bag chips"],
    "Lay's Tomato Tango":   ["Lays tomato tango red bag chips"],
    "Doritos":              ["Doritos tortilla chips nacho cheese orange bag", "Doritos sweet chili black bag"],
    "Cheetos":              ["Cheetos cheese puffs orange bag", "Cheetos crunchy orange snack"],
    "Kurkure":              ["Kurkure Indian spicy puffed snack orange", "Kurkure namkeen"],
    "Bingo":                ["Bingo Mad Angles ITC snack chips", "Bingo chips India"],
    "Uncle Chipps":         ["Uncle Chipps potato chips India plain salted", "Uncle Chipps Spicy Treat chips blue bag"],
    "Pringles":             ["Pringles chips cylindrical tube can red green"],
    "Monaco":               ["Monaco salted cracker biscuit yellow packet Parle"],
    "Parle-G":              ["Parle-G glucose biscuit yellow packet"],
    "Good Day":             ["Britannia Good Day cashew cookies biscuit pack"],
    "Marie Gold":           ["Britannia Marie Gold tea biscuit pack"],
    "Tiger Krunch":         ["Britannia Tiger Krunch chocolate chip cookies pack"],
    "Hide & Seek Fab":      ["Parle Hide and Seek Fab chocolate biscuit pack"],
    "Hide & Seek Milano":   ["Parle Hide and Seek Milano cookies pack"],
    "Dark Fantasy":         ["ITC Dark Fantasy chocolate cookie choco fills pack"],
    "Malkist":              ["Malkist cheese cracker biscuit Monde Nissin"],
    "Oreo":                 ["Oreo chocolate cream biscuit sandwich blue packet"],
    
    # Dairy Shelf
    "Nestle A Yoghurt":     ["Nestle A+ yoghurt cup fruit"],
    "Activia":              ["Danone Activia yoghurt cup green fruit"],
    "Actimel":              ["Danone Actimel probiotic drink bottle small white"],
    "Yakult":               ["Yakult probiotic fermented milk small white bottle"],
    "Nestle A Dahi":        ["Nestle A+ dahi cup blue white"],
    "Amul Gold":            ["Amul Gold milk packet or carton"],
    "Amul Taaza":           ["Amul Taaza milk packet or carton blue"],
    "Amul Shakti":          ["Amul Shakti milk packet or carton"],
    "Amul Butter":          ["Amul butter yellow block pack"],
    "Amul Lite Butter":     ["Amul lite butter green pack"],
    "Amul Cheese Spread":   ["Amul cheese spread tub"],
    "Amul Cheese Block":    ["Amul cheese block carton yellow"],
    "Amul Pizza Cheese":    ["Amul pizza cheese mozzarella block pack"],
    "Mother Dairy Toned Milk": ["Mother Dairy milk toned bottle blue white India"],
    "Epigamia Milk Shake":  ["Epigamia milk shake bottle or Greek yoghurt cup"],
    "Hershey's":            ["Hersheys chocolate milk shake brown bottle dark"],
    "Milky Mist Butter":    ["Milky Mist butter pack"],
    "Milky Mist Mozzarella": ["Milky Mist mozzarella cheese block pack"],
    "Milky Mist Pizza Cheese": ["Milky Mist pizza cheese pack"],
    "Go Cheese Slices":     ["Go cheese slices individual wrapped pack orange"],
    "Britannia Cheese Slices": ["Britannia processed cheese slices pack yellow"],
    
    # Beverages Shelf
    "Tropicana":            ["Tropicana orange juice carton tetra pak"],
    "Real Juice":           ["Real fruit juice carton Dabur tetrapak"],
    "Minute Maid":          ["Minute Maid orange juice carton tetra pak", "Minute Maid pulpy orange juice carton"],
    "Sprite":               ["Sprite green lemon soda bottle transparent"],
    "Coca-Cola":            ["Coca-Cola red bottle cola", "Coke dark red beverage"],
    "Diet Coke":            ["Diet Coke black slim bottle or can"],
    "Thums Up":             ["Thums Up dark cola bottle India brown"],
    "Fanta":                ["Fanta orange soda bottle bright orange"],
    "Limca":                ["Limca lemon soda pale green bottle India"],
    "Pepsi":                ["Pepsi blue cola bottle with globe logo"],
    "Mirinda":              ["Mirinda orange soda bottle orange red"],
    "7UP":                  ["7UP green bottle lemon lime soda"],
    "Mountain Dew":         ["Mountain Dew neon green citrus soda bottle"],
    "Lipton Ice Tea":       ["Lipton ice tea bottle yellow label"],
    "Nestea":               ["Nestea iced tea bottle ready to drink"],
    "Paper Boat":           ["Paper Boat traditional Indian drink tetra pack"],
    "Gatorade":             ["Gatorade sports drink wide mouth bottle colourful"],
    "Red Bull":             ["Red Bull energy drink silver blue slim tall can"],
    "Nescafe":              ["Nescafe Classic red instant coffee jar or bottle"],
    "B Natural":            ["B Natural ITC fruit juice bottle small"],
    "Amul Kool":            ["Amul Kool flavoured milk bottle"],
    "Amul Pro Milk":        ["Amul Pro chocolate milk bottle"],
    "Amul Royale":          ["Amul Royale flavoured milk pack"],
    
    "Other":                ["unidentified retail product on supermarket shelf"],
}

# Allowed brands per shelf category for context validation (Fix 3)
ALLOWED_BEVERAGES = [
    "Tropicana", "Real Juice", "Minute Maid", "Sprite", "Coca-Cola", "Diet Coke",
    "Thums Up", "Fanta", "Limca", "Pepsi", "Mirinda", "7UP", "Mountain Dew",
    "Lipton Ice Tea", "Nestea", "Paper Boat", "Gatorade", "Red Bull", "Nescafe",
    "Amul Kool", "B Natural", "Other"
]

ALLOWED_DAIRY = [
    "Amul", "Amul Kool", "Amul Pro Milk", "Amul Royale",
    "Mother Dairy", "Nestle", "Yakult", "Actimel", "Activia",
    "Epigamia", "Milky Mist", "Hershey's",
    # Only dairy-specific Britannia products are allowed — NOT generic Britannia (snacks)
    "Britannia Cheese Slices", "Go Cheese", "Go Cheese Slices",
    "Other",
]

ALLOWED_SNACKS = [
    "Lay's", "Doritos", "Cheetos", "Kurkure", "Bingo", "Pringles",
    "Uncle Chipps", "Parle", "Britannia", "Oreo", "Dark Fantasy",
    "Malkist", "Other"
]

# SKU-to-Parent Brand mapping (Fix 2)
SKU_TO_PARENT_BRAND = {
    # Lay's SKUs
    "Lay's Classic": "Lay's",
    "Lay's Magic Masala": "Lay's",
    "Lay's Chile Limon": "Lay's",
    "Lay's Tomato Tango": "Lay's",
    
    # Amul sub-brands grouped under Amul
    "Amul Gold": "Amul",
    "Amul Taaza": "Amul",
    "Amul Shakti": "Amul",
    "Amul Butter": "Amul",
    "Amul Lite Butter": "Amul",
    "Amul Cheese Block": "Amul",
    "Amul Cheese Spread": "Amul",
    "Amul Pizza Cheese": "Amul",
    
    # Amul Kool, Amul Pro Milk, Amul Royale remain as their own parent brands in counts
    # so they are not mapped here, causing parent = sku.
    
    # Parle products
    "Monaco": "Parle",
    "Parle-G": "Parle",
    "Hide & Seek": "Parle",
    "Hide & Seek Fab": "Parle",
    "Hide & Seek Milano": "Parle",
    
    # Britannia snack products
    "Good Day": "Britannia",
    "Marie Gold": "Britannia",
    "Tiger Krunch": "Britannia",
    # Dairy-specific — own parent so context filter works correctly
    "Go Cheese Slices": "Go Cheese Slices",
    "Britannia Cheese Slices": "Britannia Cheese Slices",
    
    # Milky Mist products
    "Milky Mist Butter": "Milky Mist",
    "Milky Mist Mozzarella": "Milky Mist",
    "Milky Mist Pizza Cheese": "Milky Mist",
    
    # Nestle products
    "Nestle A Yoghurt": "Nestle",
    "Nestle A Dahi": "Nestle",
    
    # Coca-Cola SKUs
    "Diet Coke": "Coca-Cola",
}

# SKU-to-Display Brand mapping for JSON outputs (grouped parent chips, SKU-level biscuits)
SKU_TO_DISPLAY_BRAND = {
    # Snacks
    "Lay's Classic": "Lay's",
    "Lay's Magic Masala": "Lay's",
    "Lay's Chile Limon": "Lay's",
    "Lay's Tomato Tango": "Lay's",
    "Doritos": "Doritos",
    "Cheetos": "Cheetos",
    "Kurkure": "Kurkure",
    "Uncle Chipps": "Uncle Chips",
    "Bingo": "Bingo",
    "Pringles": "Pringles",
    "Oreo": "Oreo",
    "Malkist": "Malkist",
    "Monaco": "Monaco",
    "Parle-G": "Parle-G",
    "Good Day": "Good Day",
    "Marie Gold": "Marie Gold",
    "Tiger Krunch": "Tiger Krunch",
    "Hide & Seek Fab": "Hide & Seek Fab",
    "Hide & Seek Milano": "Hide & Seek Milano",
    "Dark Fantasy": "Dark Fantasy",
    
    # Beverages
    "Tropicana": "Tropicana",
    "Real Juice": "Real Juice",
    "Minute Maid": "Minute Maid",
    "Sprite": "Sprite",
    "Coca-Cola": "Coca-Cola",
    "Diet Coke": "Diet Coke",
    "Thums Up": "Thums Up",
    "Fanta": "Fanta",
    "Limca": "Limca",
    "Nestea": "Nestea",
    "Lipton Ice Tea": "Lipton Ice Tea",
    "Mountain Dew": "Mountain Dew",
    "7UP": "7UP",
    "Mirinda": "Mirinda",
    "Pepsi": "Pepsi",
    "B Natural": "B Natural",
    "Paper Boat": "Paper Boat",
    "Gatorade": "Gatorade",
    "Red Bull": "Red Bull",
    "Nescafe": "Nescafe",
    "Amul Kool": "Amul Kool",
    
    # Dairy
    "Nestle A Yoghurt": "Nestle A Yoghurt",
    "Activia": "Activia",
    "Actimel": "Actimel",
    "Yakult": "Yakult",
    "Nestle A Dahi": "Nestle A Dahi",
    "Amul Taaza": "Amul Taaza",
    "Mother Dairy Toned Milk": "Mother Dairy",
    "Amul Shakti": "Amul Shakti",
    "Amul Gold": "Amul Gold",
    "Epigamia Milk Shake": "Epigamia",
    "Hershey's Milk Shake": "Hershey's",
    "Amul Royale": "Amul Royale",
    "Amul Pro Milk": "Amul Pro",
    "Amul Butter": "Amul Butter",
    "Amul Lite Butter": "Amul Lite",
    "Amul Cheese Spread": "Amul Cheese Spread",
    "Gamal Garlic Herbs Spread": "Garlic Herbs Spread",
    "Milky Mist Butter": "Milky Mist Butter",
    "Amul Cheese Block": "Amul Cheese Block",
    "Britannia Cheese Slices": "Britannia Cheese Slices",
    "Milky Mist Mozzarella": "Milky Mist Mozzarella",
    "Milky Mist Pizza Cheese": "Milky Mist Pizza Cheese",
    "Go Cheese Slices": "Go Cheese Slices",
    "Amul Pizza Cheese": "Amul Pizza Cheese",
    "Other": "Other",
}
