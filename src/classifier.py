"""
src/classifier.py
=================
Brand classification using OpenCLIP (ViT-B/32 OpenAI weights).

Each detected crop is compared against per-brand text prompts using
cosine similarity.  The brand with the highest similarity is assigned.

Confidence gate
---------------
CLIP_MIN_CONF (default 0.22) guards against cross-category errors:
if the best cosine similarity is below the threshold the crop is
labelled "Other" instead of a potentially wrong brand.

Example: a bottle of Sprite should not be labelled "ITC Dark Fantasy"
just because that prompt scored 0.20 on the dark background.
"""
from __future__ import annotations

import logging
import ssl
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.config import BRAND_DICTIONARY
from rapidfuzz import fuzz

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)


def clean_str(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()


def match_ocr_to_brand(
    ocr_pairs: List[Tuple[str, float]], 
    brand_dict: Dict[str, str],
    allowed_brands: Optional[List[str]] = None
) -> Optional[str]:
    """Helper to match OCR text detected on crops to a SKU using rapidfuzz fuzz.ratio on cleaned strings.
    Only considers OCR candidates with confidence > 0.65 (Fix 4).
    Evaluates all OCR pairs and returns the overall highest scoring match.
    """
    from configs.config import SKU_TO_PARENT_BRAND
    
    # Filter dictionary choices if allowed_brands is specified
    active_brand_dict = brand_dict
    if allowed_brands is not None:
        active_brand_dict = {
            k: v for k, v in brand_dict.items()
            if SKU_TO_PARENT_BRAND.get(v, v) in allowed_brands
        }
        
    if not active_brand_dict:
        return None
        
    # Map cleaned choices to original keys
    cleaned_choices = {clean_str(k): k for k in active_brand_dict.keys()}
    
    best_overall_score = 0.0
    best_overall_ocr_conf = 0.0
    best_overall_key = None
    
    for text, conf in ocr_pairs:
        if conf <= 0.65:
            continue
        cleaned_query = clean_str(text)
        if not cleaned_query:
            continue
            
        # Match using fuzz.ratio scorer
        for cleaned_choice in cleaned_choices.keys():
            score = fuzz.ratio(cleaned_query, cleaned_choice)
            # Find the highest fuzzy score. Tie-break using OCR confidence.
            if score > best_overall_score or (abs(score - best_overall_score) < 1e-5 and conf > best_overall_ocr_conf):
                best_overall_score = score
                best_overall_ocr_conf = conf
                best_overall_key = cleaned_choices[cleaned_choice]
                
    if best_overall_score > 75.0 and best_overall_key is not None:
        return active_brand_dict[best_overall_key]
        
    return None


class BrandClassifier:
    """
    OpenCLIP zero-shot brand classifier.

    Parameters
    ----------
    brand_prompts : dict[str, list[str]]
        Maps brand name → list of descriptive text prompts.
    model_name : str
        OpenCLIP model architecture (default "ViT-B/32").
    pretrained : str
        Pre-trained weights tag (default "openai").
    min_conf : float
        Minimum cosine similarity to accept a brand prediction.
        Below this threshold the crop is returned as "Other".
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        brand_prompts: Dict[str, List[str]],
        model_name: str = "ViT-B/32",
        pretrained: str = "openai",
        min_conf: float = 0.22,
        device: str = "cpu",
        ocr_engine: Optional[object] = None,
    ):
        self.brand_prompts = brand_prompts
        self.model_name    = model_name
        self.pretrained    = pretrained
        self.min_conf      = min_conf
        self.device        = device
        self.ocr_engine    = ocr_engine
        self._model        = None
        self._preprocess   = None
        self._text_embs    = None
        self._brand_names: List[str] = []

    def _load(self):
        if self._model is not None:
            return
        try:
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
        except Exception:
            import clip
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)

        self._model.eval().to(self.device)

        # Pre-compute averaged text embeddings per brand
        try:
            import open_clip
            tokenize_fn = open_clip.get_tokenizer(self.model_name)
        except Exception:
            import clip
            tokenize_fn = clip.tokenize

        brand_embs: List[torch.Tensor] = []
        for brand, prompts in self.brand_prompts.items():
            self._brand_names.append(brand)
            tokens = tokenize_fn(prompts).to(self.device)
            with torch.no_grad():
                emb = self._model.encode_text(tokens).float()
                emb = F.normalize(emb, dim=-1).mean(dim=0, keepdim=True)
                emb = F.normalize(emb, dim=-1)
            brand_embs.append(emb)

        self._text_embs = torch.cat(brand_embs, dim=0)  # (num_brands, D)
        logger.info("Loaded CLIP %s → %d brands", self.model_name, len(self._brand_names))

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Classify a single BGR crop. Returns (brand_name, confidence)."""
        if self.ocr_engine is not None and crop_bgr.size > 0:
            try:
                ocr_pairs = self.ocr_engine._raw_texts(crop_bgr)
                matched_brand = match_ocr_to_brand(ocr_pairs, BRAND_DICTIONARY)
                if matched_brand:
                    return matched_brand, 1.0
            except Exception as e:
                logger.warning("OCR classification fallback to CLIP due to error: %s", e)

        self._load()
        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self._model.encode_image(tensor).float()
            img_emb = F.normalize(img_emb, dim=-1)
        sims      = (img_emb @ self._text_embs.T).squeeze(0)
        best_idx  = int(sims.argmax().item())
        best_conf = float(sims[best_idx].item())
        if best_conf < self.min_conf:
            return "Other", best_conf
        return self._brand_names[best_idx], best_conf

    def classify_batch(
        self,
        crops_bgr: List[np.ndarray],
        allowed_brands: Optional[List[str]] = None
    ) -> Tuple[List[Tuple[str, float]], List[List[str]]]:
        """Classify a batch of BGR crops with optional category restrictions.
        
        Returns:
            - List of (brand_name, confidence)
            - List of lists of OCR texts detected on each crop
        """
        self._load()
        if not crops_bgr:
            return [], []

        results: List[Optional[Tuple[str, float]]] = [None] * len(crops_bgr)
        all_crop_texts: List[List[str]] = [[] for _ in crops_bgr]
        clip_indices: List[int] = []

        # 1. OCR-first pass
        for idx, crop in enumerate(crops_bgr):
            if self.ocr_engine is not None and crop.size > 0:
                try:
                    ocr_pairs = self.ocr_engine._raw_texts(crop)
                    texts = [p[0] for p in ocr_pairs]
                    all_crop_texts[idx] = texts
                    
                    matched_brand = match_ocr_to_brand(ocr_pairs, BRAND_DICTIONARY, allowed_brands)
                    if matched_brand:
                        results[idx] = (matched_brand, 1.0)
                        logger.info("OCR match: Crop %d -> %s", idx, matched_brand)
                except Exception as e:
                    logger.warning("OCR on crop %d failed: %s", idx, e)
            
            if results[idx] is None:
                clip_indices.append(idx)

        # 2. CLIP Fallback for unmatched crops
        if clip_indices:
            fallback_crops = [crops_bgr[i] for i in clip_indices]
            import cv2
            tensors = []
            for crop in fallback_crops:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensors.append(self._preprocess(Image.fromarray(rgb)))

            batch = torch.stack(tensors, dim=0).to(self.device)
            with torch.no_grad():
                img_embs = self._model.encode_image(batch).float()
                img_embs = F.normalize(img_embs, dim=-1)

            # Filter categories for CLIP classification according to allowed_brands
            from configs.config import SKU_TO_PARENT_BRAND
            brand_names = self._brand_names
            text_embs = self._text_embs
            
            if allowed_brands is not None:
                indices = [
                    i for i, b in enumerate(self._brand_names)
                    if SKU_TO_PARENT_BRAND.get(b, b) in allowed_brands
                ]
                if indices:
                    brand_names = [self._brand_names[i] for i in indices]
                    text_embs = self._text_embs[indices]
                else:
                    logger.warning("No allowed brands found in BRAND_PROMPTS. Defaulting to all.")

            sims       = img_embs @ text_embs.T
            best_idxs  = sims.argmax(dim=1).tolist()
            best_confs = sims.max(dim=1).values.tolist()

            for i, idx in enumerate(clip_indices):
                conf = float(best_confs[i])
                brand = brand_names[best_idxs[i]]
                if conf < self.min_conf:
                    results[idx] = ("Other", conf)
                else:
                    results[idx] = (brand, conf)
                logger.info("CLIP fallback: Crop %d -> %s (conf: %.3f)", idx, results[idx][0], conf)

        final_results = [r if r is not None else ("Other", 0.0) for r in results]
        return final_results, all_crop_texts
