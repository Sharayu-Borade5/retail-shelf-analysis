"""
Brand classification using OpenAI CLIP (zero-shot).

For each detected product crop, we compute cosine similarity between the
image embedding and pre-computed text embeddings for every brand prompt.
The brand with the highest average similarity across its prompts wins.
"""

from __future__ import annotations

import logging
import ssl
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow CLIP weight download on systems with SSL cert issues
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)


class BrandClassifier:
    """
    CLIP-based zero-shot brand classifier using openai/CLIP.

    Parameters
    ----------
    brand_prompts : dict mapping brand name -> list of text descriptions
    model_name    : CLIP architecture string (e.g. 'ViT-B/32')
    device        : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        brand_prompts: Dict[str, List[str]],
        model_name: str = "ViT-B/32",
        pretrained: str = "openai",   # kept for API compatibility, unused
        device: str = "cpu",
    ) -> None:
        self.brand_prompts = brand_prompts
        self.device        = device
        self._model        = None
        self._preprocess   = None
        self._text_embs    = None
        self._brand_names: List[str] = []
        self._model_name   = model_name

    # ── lazy initialisation ───────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        import clip  # openai-clip

        self._model, self._preprocess = clip.load(self._model_name, device=self.device)
        self._model.eval()
        self._clip_module = clip
        self._precompute_text_embeddings()
        logger.info("Loaded CLIP %s – %d brands", self._model_name, len(self._brand_names))

    def _precompute_text_embeddings(self) -> None:
        brand_vectors: List[torch.Tensor] = []
        self._brand_names = list(self.brand_prompts.keys())

        with torch.no_grad():
            for brand, prompts in self.brand_prompts.items():
                if not prompts:
                    prompts = ["a packaged product on a retail shelf"]
                tokens = self._clip_module.tokenize(prompts).to(self.device)
                text_emb = self._model.encode_text(tokens).float()
                text_emb = F.normalize(text_emb, dim=-1)
                brand_vectors.append(text_emb.mean(dim=0))

        self._text_embs = torch.stack(brand_vectors, dim=0)
        self._text_embs = F.normalize(self._text_embs, dim=-1)

    # ── public API ────────────────────────────────────────────────────────

    def classify(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        self._load()
        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_emb = self._model.encode_image(tensor).float()
            img_emb = F.normalize(img_emb, dim=-1)

        sims     = (img_emb @ self._text_embs.T).squeeze(0)
        best_idx = int(sims.argmax().item())
        return self._brand_names[best_idx], float(sims[best_idx].item())

    def classify_batch(self, crops_bgr: List[np.ndarray]) -> List[tuple[str, float]]:
        self._load()
        if not crops_bgr:
            return []

        import cv2
        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self._preprocess(pil))

        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            img_embs = self._model.encode_image(batch).float()
            img_embs = F.normalize(img_embs, dim=-1)

        sims       = img_embs @ self._text_embs.T
        best_idxs  = sims.argmax(dim=1).tolist()
        best_confs = sims.max(dim=1).values.tolist()

        return [(self._brand_names[i], float(c)) for i, c in zip(best_idxs, best_confs)]
