"""Text and visual embedders for retrieval.

* ``ClinicalTextEncoder`` wraps a clinical SBERT sentence encoder for report vectors.
* ``VisualEncoder`` reuses the *existing* ``vlm_model.vision_encoder.VisionEncoder`` so
  visual-retrieval embeddings live in the same CLIP space the VLM itself consumes. The
  256 patch tokens are mean-pooled to one vector per image.

Both return L2-normalized ``float32`` arrays so a FAISS inner-product index equals cosine.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class ClinicalTextEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Loaded text encoder %s", model_name)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


class VisualEncoder:
    """Pooled CLIP patch embeddings, using the same vision tower as the VLM."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from vlm_model.vision_encoder import VisionEncoder as _VisionEncoder

        self.torch = torch
        self.device = device
        self.encoder = _VisionEncoder(model_name=model_name).to(device)
        self.encoder.eval()
        self.image_processor = self.encoder.image_processor
        logger.info("Loaded visual encoder %s", model_name)

    @property
    def dim(self) -> int:
        return self.encoder.hidden_size

    def encode_paths(self, image_paths: List[str], normalize: bool = True) -> np.ndarray:
        from data.image_processing import load_and_process_image

        torch = self.torch
        vecs = []
        with torch.no_grad():
            for path in image_paths:
                pixel_values = load_and_process_image(path, self.image_processor)
                pixel_values = pixel_values.unsqueeze(0).to(self.device)
                patches = self.encoder(pixel_values)        # (1, 256, D)
                pooled = patches.mean(dim=1)                # (1, D)
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                vecs.append(pooled.float().cpu().numpy())
        return np.concatenate(vecs, axis=0).astype(np.float32)
