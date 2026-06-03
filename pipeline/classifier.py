"""
Brand classification using CLIP (zero-shot).

For each detected product crop, we compute cosine similarity between the
image embedding and pre-computed text embeddings for every brand prompt.
The brand with the highest average similarity across its prompts wins.

Zero-shot CLIP works surprisingly well for branded FMCG packaging because
the model has seen product images paired with brand names during pretraining.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class BrandClassifier:
    """
    CLIP-based zero-shot brand classifier.

    Parameters
    ----------
    brand_prompts : dict mapping brand name -> list of text descriptions
    model_name    : CLIP model architecture (default ViT-B/32)
    pretrained    : weight source (default "openai")
    device        : "cpu" or "cuda"
    """

    def __init__(
        self,
        brand_prompts: Dict[str, List[str]],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        self.brand_prompts = brand_prompts
        self.device        = device
        self._model        = None
        self._preprocess   = None
        self._tokenizer    = None
        self._text_embs    = None   # shape: (num_brands, D)
        self._brand_names: List[str] = []
        self._model_name   = model_name
        self._pretrained   = pretrained

    # ── lazy initialisation ───────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        import open_clip  # pip install open-clip-torch

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        self._model.eval().to(self.device)
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

        self._precompute_text_embeddings()
        logger.info(
            "Loaded CLIP %s – %d brands registered",
            self._model_name,
            len(self._brand_names),
        )

    def _precompute_text_embeddings(self) -> None:
        """Encode all brand prompts once; store mean-pooled embeddings."""
        brand_vectors: List[torch.Tensor] = []
        self._brand_names = list(self.brand_prompts.keys())

        with torch.no_grad():
            for brand, prompts in self.brand_prompts.items():
                if not prompts:
                    # "Other" – use a generic description
                    prompts = ["a packaged product on a retail shelf"]
                tokens = self._tokenizer(prompts).to(self.device)
                text_emb = self._model.encode_text(tokens)           # (P, D)
                text_emb = F.normalize(text_emb, dim=-1)
                brand_vectors.append(text_emb.mean(dim=0))           # (D,)

        self._text_embs = torch.stack(brand_vectors, dim=0)          # (B, D)
        self._text_embs = F.normalize(self._text_embs, dim=-1)

    # ── public API ────────────────────────────────────────────────────────

    def classify(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Classify a single BGR crop (numpy array).

        Returns (brand_name, confidence_score).
        """
        self._load()

        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            img_emb = self._model.encode_image(tensor)               # (1, D)
            img_emb = F.normalize(img_emb, dim=-1)

        sims = (img_emb @ self._text_embs.T).squeeze(0)              # (B,)
        best_idx = int(sims.argmax().item())
        best_conf = float(sims[best_idx].item())

        return self._brand_names[best_idx], best_conf

    def classify_batch(
        self, crops_bgr: List[np.ndarray]
    ) -> List[tuple[str, float]]:
        """Classify multiple crops in one forward pass for speed."""
        self._load()

        if not crops_bgr:
            return []

        import cv2
        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self._preprocess(pil))

        batch = torch.stack(tensors, dim=0).to(self.device)          # (N, C, H, W)

        with torch.no_grad():
            img_embs = self._model.encode_image(batch)               # (N, D)
            img_embs = F.normalize(img_embs, dim=-1)

        sims = img_embs @ self._text_embs.T                          # (N, B)
        best_idxs  = sims.argmax(dim=1).tolist()
        best_confs = sims.max(dim=1).values.tolist()

        return [
            (self._brand_names[idx], float(conf))
            for idx, conf in zip(best_idxs, best_confs)
        ]
