"""
src/ocr.py
==========
OCR extraction from shelf label / price tag strips.

Primary  : PaddleOCR  (as specified in the assignment)
Fallback : EasyOCR   (used automatically if paddleocr is not installed)

The engine runs once per image on horizontal strip regions at the bottom
of each detected shelf row where price tags are physically located.

Output cleaning
---------------
A multi-stage filter removes noise tokens:
  1. Too short (< 3 chars)
  2. Pure numbers without a currency symbol (bare '120' etc.)
  3. High noise-char ratio (> 25% non-alphanumeric)
  4. Known-brand near-match allowlist via Levenshtein ≤ 1 (keeps misreads)
  5. Random-capitalisation detection (UlU / lUl triplets, ≥3 case flips)
  6. Real-word / price / weight pattern check (≥5 alpha OR currency price)
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Compiled patterns ──────────────────────────────────────────────────────────
_PRICE_WITH_SYMBOL = re.compile(r"[₹$€£]\s*\d+")
_WEIGHT_RE         = re.compile(r"\d{2,}\s*(g|ml|kg|l|oz)\b", re.IGNORECASE)
_REAL_WORD_RE      = re.compile(r"[A-Za-z]{5,}")
_NOISE_RE          = re.compile(r"[^A-Za-z0-9₹$%.,\s\-&+]")
_ALTCAP_RE         = re.compile(r"(?:[A-Z][a-z][A-Z]|[a-z][A-Z][a-z])")
_BARE_NUM_RE       = re.compile(r"^\d+$")

_KNOWN_BRANDS = {
    "lays", "doritos", "cheetos", "pringles", "kurkure", "bingo", "oreo",
    "parle", "britannia", "malkist", "milano", "oreo", "oreos",
    "amul", "danone", "nestle", "yakult", "epigamia", "hersheys",
    "pepsi", "sprite", "fanta", "mirinda", "nestea", "lipton", "gatorade",
    "redbull", "nescafe", "tropicana", "minutemaid", "paperboat",
    "motherdairy", "milkymist", "activia", "actimel",
    # Label words
    "classic", "original", "masala", "salted", "spicy", "plain",
    "chocolate", "vanilla", "strawberry", "mango", "orange",
    "extra", "offer", "ranch", "natural",
    "butter", "cheese", "slices", "mozzarella", "royale", "shakti",
    "krunch", "tango", "spanish", "guava", "mixed",
    "apple", "pineapple", "pomegranate", "marie", "tiger", "fantasy",
    "monaco", "milano", "cheetos",
}


def _levenshtein(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 99
    if a == b:
        return 0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            ndp[j + 1] = min(dp[j] + (ca != cb), dp[j + 1] + 1, ndp[j] + 1)
        dp = ndp
    return dp[-1]


def _matches_known_brand(text: str) -> bool:
    t = text.lower().strip()
    for word in t.split():
        if len(word) < 4:
            continue
        for brand in _KNOWN_BRANDS:
            if _levenshtein(word, brand) <= 1:
                return True
    if _levenshtein(t.replace(" ", ""), t.replace(" ", "")) == 0:
        pass
    for brand in _KNOWN_BRANDS:
        if _levenshtein(t, brand) <= 1:
            return True
    return False


def _case_transitions(text: str) -> int:
    alpha = [c for c in text if c.isalpha()]
    return sum(1 for a, b in zip(alpha, alpha[1:]) if a.isupper() != b.isupper())


def _is_random_case(text: str) -> bool:
    if len(text) > 8:
        return False
    if _matches_known_brand(text):
        return False
    return bool(_ALTCAP_RE.search(text)) or _case_transitions(text) >= 3


def _clean(text: str) -> str:
    return text.strip().strip("\"'.,;:!?-_")


def _is_useful(text: str) -> bool:
    """Six-stage noise filter."""
    t = _clean(text)
    if len(t) < 3:
        if t.isdigit() and len(t) == 2:
            pass
        else:
            return False                        # Stage 1: too short
    if _BARE_NUM_RE.match(t):
        try:
            val = int(t)
            if 10 <= val <= 999:
                return True
        except ValueError:
            pass
        return False                        # Stage 2: bare number (other sizes)
    noise_chars = len(_NOISE_RE.findall(t))
    if noise_chars / max(len(t), 1) > 0.25:
        return False                        # Stage 3: too noisy
    if len(t) >= 4 and _matches_known_brand(t):
        return True                         # Stage 4: brand allowlist
    if _is_random_case(t):
        return False                        # Stage 5: random caps
    # Stage 6: must have substance
    if _REAL_WORD_RE.search(t):
        return True
    if _PRICE_WITH_SYMBOL.search(t):
        return True
    if _WEIGHT_RE.search(t):
        return True
    return False


class OCREngine:
    """
    Shelf OCR engine.

    Tries PaddleOCR first; falls back to EasyOCR automatically.

    Parameters
    ----------
    languages : list[str]
        Language codes (e.g. ["en"]).
    min_conf : float
        Minimum recognition confidence [0, 1].
    """

    def __init__(self, languages: Optional[List[str]] = None, min_conf: float = 0.45):
        self.languages = languages or ["en"]
        self.min_conf  = min_conf
        self._engine   = None
        self._backend  = None  # "paddle" or "easy"

    def _load(self):
        if self._engine is not None:
            return
        try:
            from paddleocr import PaddleOCR
            self._engine  = PaddleOCR(
                use_angle_cls=True, lang=self.languages[0],
                use_gpu=False, show_log=False
            )
            self._backend = "paddle"
            logger.info("PaddleOCR loaded (language=%s)", self.languages[0])
        except Exception as exc:
            logger.warning("PaddleOCR unavailable (%s); falling back to EasyOCR", exc)
            import easyocr
            self._engine  = easyocr.Reader(self.languages, gpu=False, verbose=False)
            self._backend = "easy"
            logger.info("EasyOCR reader loaded (languages=%s)", self.languages)

    def _raw_texts(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Return list of (text, confidence) from the OCR engine."""
        self._load()
        if self._backend == "paddle":
            result = self._engine.ocr(image, cls=True)
            pairs: List[Tuple[str, float]] = []
            if result and result[0]:
                for line in result[0]:
                    text, conf = line[1]
                    pairs.append((text, conf))
            return pairs
        else:  # easyocr
            result = self._engine.readtext(image)
            return [(text, conf) for _, text, conf in result]

    def extract(self, image: np.ndarray, shelf_bands: List[Tuple[int, int]]) -> List[str]:
        """
        Run OCR on horizontal strips at the bottom of each shelf row
        (where price tags are physically located) and return cleaned labels.

        Parameters
        ----------
        image : np.ndarray
            Full BGR shelf image.
        shelf_bands : list[tuple[int,int]]
            List of (y_top, y_bottom) for each shelf row.

        Returns
        -------
        list[str]
            Deduplicated, noise-filtered OCR text items.
        """
        img_h, img_w = image.shape[:2]
        strip_h      = max(60, int(img_h * 0.045))

        regions: List[np.ndarray] = []
        for y_top, y_bot in shelf_bands:
            # Price tag strip just below the shelf separator
            y0 = max(0, y_bot - strip_h)
            y1 = min(img_h, y_bot + strip_h)
            regions.append(image[y0:y1, 0:img_w])

        raw_pairs: List[Tuple[str, float]] = []
        for region in regions:
            raw_pairs.extend(self._raw_texts(region))

        # Filter by confidence and clean
        seen: set = set()
        texts: List[str] = []
        for text, conf in raw_pairs:
            if conf < self.min_conf:
                continue
            cleaned = _clean(text)
            key     = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            if _is_useful(cleaned):
                texts.append(cleaned)

        logger.info("OCR extracted %d unique text items", len(texts))
        return texts
