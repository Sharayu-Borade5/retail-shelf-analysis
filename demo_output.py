"""
demo_output.py – Generates realistic sample outputs for all three test
shelf images WITHOUT running any ML models.

Use this to see the expected JSON schema and annotated image layout
while model weights are being downloaded / environment is being set up.

Run:
    python demo_output.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Expected outputs derived from visual inspection of the three images ─────

SAMPLE_RESULTS = [
    {
        "image_name": "shelf_dairy.jpg",
        "total_products": 54,
        "brands": {
            "Amul":           22,
            "Danone":          8,
            "Nestle":          6,
            "Yakult":          3,
            "Mother Dairy":    4,
            "Hersheys":        4,
            "Epigamia":        3,
            "Milky Mist":      4,
        },
        "ocr_labels": [
            "₹25 NESTLE A+ YOGHURT 90g",
            "₹30 ACTIVIA YOGHURT 100g",
            "₹25 ACTIMEL DRINK 100ml",
            "₹28 YAKULT DRINK 65ml",
            "₹20 NESTLE A+ DAHI 100g",
            "₹62 AMUL TAAZA TONED MILK 1L",
            "₹54 MOTHER DAIRY TONED MILK 1L",
            "₹60 AMUL SHAKTI MILK 1L",
            "₹62 AMUL GOLD MILK 1L",
            "₹45 EPIGAMIA MILK SHAKE 180ml",
            "₹25 AMUL KOOL 200ml",
            "₹40 HERSHEYS MILK SHAKE 200ml",
            "₹30 AMUL ROYALE 200ml",
            "₹35 AMUL PRO MILK 200ml",
            "₹58 AMUL BUTTER 100g",
            "₹52 AMUL LITE BUTTER 100g",
            "₹60 AMUL CHEESE SPREAD 200g",
            "₹60 GARLIC & HERBS SPREAD 200g",
            "₹55 MILKY MIST BUTTER 100g",
            "₹70 AMUL CHEESE BLOCK 200g",
            "₹60 BRITANNIA CHEESE SLICES 200g",
            "₹140 MILKY MIST MOZZARELLA 200g",
            "₹120 MILKY MIST PIZZA CHEESE 200g",
            "₹85 GO CHEESE SLICES 200g",
            "₹130 AMUL PIZZA CHEESE",
        ],
        "share_of_shelf_pct": {
            "Amul":        43.2,
            "Danone":      14.8,
            "Nestle":      11.1,
            "Mother Dairy": 7.4,
            "Hersheys":     7.4,
            "Milky Mist":   7.4,
            "Epigamia":     4.6,
            "Yakult":       4.1,
        },
        "shelf_rows_detected": 5,
    },
    {
        "image_name": "shelf_beverages.jpg",
        "total_products": 62,
        "brands": {
            "Tropicana":    12,
            "Coca-Cola":    10,
            "Pepsi":         9,
            "Amul":          3,
            "Real":          5,
            "Minute Maid":   4,
            "Mountain Dew":  4,
            "7UP":           3,
            "Sprite":        3,
            "Fanta":         4,
            "Limca":         3,
            "Nestea":        2,
            "Lipton":        2,
            "Red Bull":      3,
            "Nescafe":       3,
            "Gatorade":      3,
            "B Natural":     4,
            "Paper Boat":    3,
        },
        "ocr_labels": [
            "₹125 TROPICANA 1L",
            "₹99 REAL JUICE 1L",
            "₹105 MINUTE MAID 1L",
            "₹50 SPRITE 600ml",
            "₹50 COCA COLA 600ml",
            "₹55 DIET COKE 600ml",
            "₹50 THUMS UP 600ml",
            "₹50 FANTA 600ml",
            "₹50 LIMCA 600ml",
            "₹60 NESTEA 500ml",
            "₹60 LIPTON ICE TEA 500ml",
            "₹50 MOUNTAIN DEW 600ml",
            "₹50 7UP 600ml",
            "₹50 MIRINDA 600ml",
            "₹50 PEPSI 600ml",
            "₹40 B NATURAL 200ml",
            "₹35 PAPER BOAT 200ml",
            "₹75 GATORADE 500ml",
            "₹110 RED BULL 250ml",
            "₹40 NESCAFE 180ml",
            "₹30 AMUL KOOL 200ml",
        ],
        "share_of_shelf_pct": {
            "Tropicana":    19.4,
            "Coca-Cola":    16.1,
            "Pepsi":        14.5,
            "Real":          8.1,
            "Minute Maid":   6.5,
            "Mountain Dew":  6.5,
            "Fanta":         6.5,
            "B Natural":     6.5,
            "7UP":           4.8,
            "Sprite":        4.8,
            "Limca":         4.8,
            "Red Bull":      4.8,
            "Nescafe":       4.8,
            "Gatorade":      4.8,
            "Nestea":        3.2,
            "Lipton":        3.2,
            "Amul":          4.8,
            "Paper Boat":    4.8,
        },
        "shelf_rows_detected": 4,
    },
    {
        "image_name": "shelf_snacks.jpg",
        "total_products": 45,
        "brands": {
            "Lays":           5,
            "Doritos":        3,
            "Cheetos":        2,
            "Kurkure":        3,
            "Uncle Chipps":   2,
            "Bingo":          4,
            "Pringles":       3,
            "Parle":          5,
            "Britannia":      8,
            "Oreo":           2,
            "ITC Snacks":     4,
            "Malkist":        4,
        },
        "ocr_labels": [
            "₹20 LAY'S 52.5g",
            "₹28 DORITOS 44g",
            "₹20 CHEETOS 35g",
            "₹20 KURKURE 60g",
            "₹20 UNCLE CHIPPS 60g",
            "₹20 BINGO! 52g",
            "₹99 PRINGLES 107g",
            "₹10 MONACO 63g",
            "₹10 PARLE-G 80g",
            "₹20 GOOD DAY 81g",
            "₹20 MARIE GOLD 120g",
            "₹20 OREO 120g",
            "₹10 TIGER KRUNCH 60g",
            "₹20 HIDE & SEEK FAB 112g",
            "₹30 HIDE & SEEK MILANO 75g",
            "₹30 DARK FANTASY 75g",
            "₹10 MALKIST 65g",
        ],
        "share_of_shelf_pct": {
            "Britannia":   22.2,
            "Parle":       13.9,
            "Lays":        13.9,
            "Bingo":       11.1,
            "ITC Snacks":  11.1,
            "Malkist":     11.1,
            "Kurkure":      8.3,
            "Pringles":     8.3,
            "Doritos":      8.3,
            "Uncle Chipps": 5.6,
            "Cheetos":      5.6,
            "Oreo":         5.6,
        },
        "shelf_rows_detected": 4,
    },
]


def draw_sample_annotation(image_path: str, result: dict, out_dir: str) -> str:
    """Draw a minimal annotation on the image using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Cannot load {image_path} – skipping visualisation")
        return ""

    H, W = img.shape[:2]

    # Draw shelf row dividers
    n_rows = result["shelf_rows_detected"]
    row_h  = H // (n_rows + 1)
    for i in range(1, n_rows + 1):
        y = i * row_h
        cv2.line(img, (0, y), (W, y), (180, 180, 180), 2)

    # Overlay brand SOS legend (top-right)
    sos = result["share_of_shelf_pct"]
    brands_sorted = sorted(sos, key=sos.get, reverse=True)
    legend_x = W - 280
    cv2.rectangle(img, (legend_x - 5, 5), (W - 5, 30 + 22 * len(brands_sorted)),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Brand SOS %", (legend_x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, b in enumerate(brands_sorted):
        text = f"{b}: {sos[b]:.1f}%"
        cv2.putText(img, text, (legend_x, 44 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    # OCR strip at bottom
    ocr_labels = result["ocr_labels"][:8]
    cv2.rectangle(img, (0, H - 30), (W, H), (0, 0, 0), cv2.FILLED)
    short = " | ".join(ocr_labels)
    cv2.putText(img, f"OCR: {short}", (6, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    stem    = os.path.splitext(result["image_name"])[0]
    outpath = os.path.join(out_dir, f"{stem}_annotated.jpg")
    cv2.imwrite(outpath, img)
    return outpath


def main():
    print("\n=== Retail Shelf Analysis – Demo Output ===\n")

    for result in SAMPLE_RESULTS:
        image_path = os.path.join("images", result["image_name"])
        ann_path   = draw_sample_annotation(image_path, result, OUTPUT_DIR)
        result["annotated_image"] = ann_path

        json_path = os.path.join(OUTPUT_DIR, result["image_name"].replace(".jpg", "_result.json"))
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

        print(f"[{result['image_name']}]")
        print(f"  Total products : {result['total_products']}")
        print(f"  Brands         : {list(result['brands'].keys())}")
        ocr_preview = [s.replace("₹", "Rs.") for s in result['ocr_labels'][:5]]
        print(f"  OCR labels     : {ocr_preview} ...")
        print(f"  SOS (top 3)    : { {k: result['share_of_shelf_pct'][k] for k in list(result['share_of_shelf_pct'])[:3]} }")
        print(f"  Saved JSON     : {json_path}")
        if ann_path:
            print(f"  Annotated img  : {ann_path}")
        print()

    print("Full JSON output:\n")
    safe = json.dumps(SAMPLE_RESULTS, indent=2, ensure_ascii=True)
    print(safe)


if __name__ == "__main__":
    main()
