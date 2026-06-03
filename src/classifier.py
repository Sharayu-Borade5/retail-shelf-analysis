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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)


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
    ):
        self.brand_prompts = brand_prompts
        self.model_name    = model_name
        self.pretrained    = pretrained
        self.min_conf      = min_conf
        self.device        = device
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

    def classify_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Classify a batch of BGR crops. Returns list of (brand_name, confidence)."""
        self._load()
        if not crops_bgr:
            return []
        import cv2
        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._preprocess(Image.fromarray(rgb)))

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            img_embs = self._model.encode_image(batch).float()
            img_embs = F.normalize(img_embs, dim=-1)

        sims       = img_embs @ self._text_embs.T
        best_idxs  = sims.argmax(dim=1).tolist()
        best_confs = sims.max(dim=1).values.tolist()

        results: List[Tuple[str, float]] = []
        for idx, conf in zip(best_idxs, best_confs):
            if float(conf) < self.min_conf:
                results.append(("Other", float(conf)))
            else:
                results.append((self._brand_names[idx], float(conf)))
        return results
