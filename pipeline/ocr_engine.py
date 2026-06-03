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
"""

from __future__ import annotations

import logging
import re
from typing import List

import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

from config import OCR_CONF, OCR_LANGUAGES, PRICE_ROW_FRAC

logger = logging.getLogger(__name__)

# Patterns that look like price / product labels on Indian retail shelves
_PRICE_RE   = re.compile(r"[₹$€£]?\s*\d+[\.,]?\d*")
_WEIGHT_RE  = re.compile(r"\d+\s*(g|ml|kg|l|oz)\b", re.IGNORECASE)
_BRAND_TEXT = re.compile(r"[A-Za-z]{3,}")   # any word >= 3 chars

def _is_useful(text: str) -> bool:
    t = text.strip()
    return bool(
        t and (
            _PRICE_RE.search(t)
            or _WEIGHT_RE.search(t)
            or _BRAND_TEXT.search(t)
        )
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
            if conf >= self.conf_threshold and _is_useful(text):
                texts.append(text.strip())
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
        Deduplicated list of OCR strings.
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

        # Deduplicate while preserving insertion order
        seen: set[str] = set()
        unique: List[str] = []
        for t in all_texts:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        logger.info("OCR extracted %d unique text items", len(unique))
        return unique
