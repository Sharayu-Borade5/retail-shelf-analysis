"""
OCR for shelf price tags using EasyOCR.

Strategy
--------
Price tags in Indian retail shelves consistently appear as yellow label
strips along the bottom edge of each shelf row.  We:

  1. Detect horizontal shelf bands from the y-coordinates of product
     bounding boxes (or fall back to evenly-spaced horizontal strips).
  2. For each band, crop the bottom PRICE_ROW_FRAC of its height (where
     price labels sit).
  3. Run EasyOCR on each crop and keep results above OCR_CONF threshold.
  4. Optionally run on the full image as a supplement to catch any missed text.

Noise-reduction pipeline
------------------------
Raw OCR on shelf images produces many noisy tokens ("CcW", "checs?", "EITAA").
We apply a multi-stage filter:

  a. Strip edge punctuation, collapse whitespace.
  b. Reject tokens < 3 chars.
  c. Reject tokens where > 25% of characters are non-alphanumeric.
  d. Reject tokens that look like random-case noise (alternating caps pattern).
  e. Accept tokens that match a known brand/product name (fuzzy, edit dist ≤ 1)
     even if they would otherwise fail.
  f. For remaining tokens, require a real word (≥ 4 alpha chars in the longest
     word), a price pattern, or a weight pattern.
  g. Case-insensitive deduplication (keep the first-seen, best-cased version).
"""

from __future__ import annotations

import logging
import re
from typing import List, Set

import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

from config import OCR_CONF, OCR_LANGUAGES, PRICE_ROW_FRAC

logger = logging.getLogger(__name__)

# ── Compiled regexes ────────────────────────────────────────────────────────────────────────
_PRICE_RE      = re.compile(r"[\u20b9$€£]?\s*\d+[\.,]?\d*")
_WEIGHT_RE     = re.compile(r"\d{2,}\s*(g|ml|kg|l|oz)\b", re.IGNORECASE)
_REAL_WORD_RE  = re.compile(r"[A-Za-z]{5,}")       # ≥ 5 consecutive letters (raised from 4)
_NOISE_RE      = re.compile(r"[^A-Za-z0-9\u20b9$%\.,\s\-&\+]")  # non-clean chars
# Alternating-caps: catch even a single upper–lower–upper or lower–upper–lower triplet
_ALTCAP_RE     = re.compile(r"(?:[A-Z][a-z][A-Z]|[a-z][A-Z][a-z])")
# Bare number (no currency/unit): e.g. "106", "120" — marginal value, filtered out
_BARE_NUM_RE   = re.compile(r"^\d+$")

# ── Known brand / product name allowlist ───────────────────────────────────────────────────────
# These are checked with edit-distance ≤ 1 so OCR near-misses are still kept.
# Only tokens ≥ 4 chars are checked against this list to prevent spurious matches.
_KNOWN_BRANDS: List[str] = [
    # Snacks
    "lays", "doritos", "cheetos", "kurkure", "bingo", "pringles",
    "parle", "parle-g", "parleg", "britannia", "oreo", "malkist",
    "hide seek", "hideseek", "good day", "marie gold", "mariegold",
    "tiger", "dark fantasy", "darkfantasy", "monaco", "milano",
    # Beverages
    "coca-cola", "cocacola", "pepsi", "sprite", "fanta", "limca",
    "mountain dew", "7up", "mirinda", "tropicana", "real", "minute maid",
    "nestea", "lipton", "gatorade", "red bull", "nescafe", "paper boat",
    # Dairy
    "amul", "mother dairy", "nestle", "yakult", "danone", "epigamia",
    "hersheys", "milky mist", "activia", "actimel", "dahi", "yoghurt",
    # Common label words (≥ 5 chars to avoid false positives)
    "classic", "original", "masala", "salted", "spicy", "plain",
    "chocolate", "vanilla", "strawberry", "mango", "orange",
    "extra", "offer", "price", "ranch", "natural",
    "butter", "cheese", "slices", "mozzarella", "royale", "shakti",
    "krunch", "tango", "spanish", "diet coke", "guava", "mixed",
    "apple", "pineapple", "pomegranate",
]


def _levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance (for short strings, O(len(a)*len(b)))."""
    if abs(len(a) - len(b)) > 2:      # fast-reject
        return 99
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _matches_known_brand(text: str) -> bool:
    """True if *text* fuzzy-matches any known brand/product with edit dist ≤ 1.
    Only applied to tokens of at least 4 characters to prevent spurious matches
    on very short noise strings like 'ays', 'Rea', 'Riol'.
    """
    t = text.lower().strip()
    if len(t) < 4:          # too short for reliable fuzzy matching
        return False
    for brand in _KNOWN_BRANDS:
        if t == brand or _levenshtein(t, brand) <= 1:
            return True
    # Also check individual words of multi-word brands
    words = t.split()
    for word in words:
        if len(word) < 4:
            continue
        for brand in _KNOWN_BRANDS:
            if word == brand or _levenshtein(word, brand) <= 1:
                return True
    return False


def _clean(text: str) -> str:
    """Strip leading/trailing noise and normalise whitespace."""
    t = text.strip(" [](){}|\\/'\";:,!")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _case_transitions(text: str) -> int:
    """Count how many times casing flips between adjacent alpha characters."""
    alpha = [c for c in text if c.isalpha()]
    return sum(1 for a, b in zip(alpha, alpha[1:]) if a.isupper() != b.isupper())


def _is_random_case(text: str) -> bool:
    """
    Heuristic: returns True if the token looks like random capitalisation noise.
    Two triggers (either is sufficient):
      1. Contains a UlU or lUl triplet (original check).
      2. Has ≥ 3 case flips in a word of ≤ 8 chars (catches ChiPPs-style noise).
    Both are only applied when the token does NOT match a known brand.
    """
    if len(text) > 8:
        return False   # long strings may legitimately mix case
    if _matches_known_brand(text):
        return False   # known brand near-match — keep it
    return bool(_ALTCAP_RE.search(text)) or _case_transitions(text) >= 3


def _is_useful(text: str) -> bool:
    """
    Multi-stage usefulness filter.

    Returns True only for tokens that carry meaningful label information.
    """
    t = _clean(text)

    # Stage 1: minimum length
    if len(t) < 3:
        return False

    # Stage 2: bare numbers with no unit/currency (low signal, high noise)
    if _BARE_NUM_RE.match(t):
        return False

    # Stage 3: noise character ratio (> 25% → reject)
    noise_hits = len(_NOISE_RE.findall(t))
    if noise_hits / max(len(t), 1) > 0.25:
        return False

    # Stage 4: brand allowlist (tokens ≥ 4 chars only)
    if len(t) >= 4 and _matches_known_brand(t):
        return True

    # Stage 5: reject random-case noise
    if _is_random_case(t):
        return False

    # Stage 6: must contain a price-with-symbol, a weight, OR a real word (≥ 5 letters)
    price_with_symbol = re.search(r"[\u20b9$€£]\s*\d+", t)
    return bool(
        price_with_symbol
        or _WEIGHT_RE.search(t)
        or _REAL_WORD_RE.search(t)
    )


class OCREngine:
    """EasyOCR-based text extractor for shelf price tags."""

    def __init__(
        self,
        languages: list[str] | None = None,
        conf_threshold: float = OCR_CONF,
        gpu: bool = False,
    ) -> None:
        self.languages      = languages or OCR_LANGUAGES
        self.conf_threshold = conf_threshold
        self.gpu            = gpu
        self._reader        = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        logger.info("EasyOCR reader loaded (languages=%s)", self.languages)

    # ── internal helpers ──────────────────────────────────────────────────

    def _run_on_region(self, region: np.ndarray) -> List[str]:
        if region.size == 0:
            return []
        results = self._reader.readtext(region, detail=1)
        texts = []
        for (_bbox, text, conf) in results:
            cleaned = _clean(text)
            if conf >= self.conf_threshold and _is_useful(cleaned):
                texts.append(cleaned)
        return texts

    # ── public API ────────────────────────────────────────────────────────

    def extract(
        self,
        image_bgr: np.ndarray,
        shelf_row_bands: List[tuple[int, int]] | None = None,
    ) -> List[str]:
        """
        Extract text from price-tag regions of a shelf image.

        Parameters
        ----------
        image_bgr        : full-resolution BGR image
        shelf_row_bands  : list of (y_top, y_bottom) pixel ranges for each
                           shelf row.  If None, the image is divided into
                           4 equal horizontal bands.

        Returns
        -------
        Deduplicated (case-insensitive) list of OCR strings.
        """
        self._load()

        H, W = image_bgr.shape[:2]

        if shelf_row_bands is None:
            n_rows = 4
            step   = H // n_rows
            shelf_row_bands = [
                (i * step, min((i + 1) * step, H)) for i in range(n_rows)
            ]

        all_texts: List[str] = []

        for y_top, y_bot in shelf_row_bands:
            row_h    = y_bot - y_top
            tag_top  = y_bot - int(row_h * PRICE_ROW_FRAC)
            tag_top  = max(tag_top, y_top)
            tag_crop = image_bgr[tag_top:y_bot, 0:W]
            all_texts.extend(self._run_on_region(tag_crop))

        # Supplement: full-image pass (catches text outside price strips)
        full_texts = self._run_on_region(image_bgr)
        all_texts.extend(full_texts)

        # Case-insensitive deduplication (keep first-seen, best-cased version)
        seen_lower: Set[str] = set()
        unique: List[str] = []
        for t in all_texts:
            key = t.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                unique.append(t)

        logger.info("OCR extracted %d unique text items", len(unique))
        return unique
