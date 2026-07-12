"""Load the frozen VLM from a ``rag_eval.yaml`` ``model`` block.

This is a thin wrapper around the existing model so the RAG layer never has to edit
inference.py. It differs from ``inference.load_vlm`` in two ways needed for evaluation:

* the connector checkpoint is *optional* — the retrieval and prompt-formatting path can
  be smoke-tested before any connector has been trained;
* ``torch_dtype`` / ``device`` can be overridden from the RAG config (e.g. float32 on CPU).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import yaml

from vlm_model.vlm import VLMForCausalLM
from training.checkpoint import load_connector_checkpoint

logger = logging.getLogger(__name__)


def load_vlm_from_cfg(cfg: dict):
    """Build and return a frozen ``VLMForCausalLM`` described by ``cfg['model']``."""
    model_cfg = cfg.get("model", {})
    inner_config_path = model_cfg["config"]
    device = model_cfg.get("device", "cpu")

    with open(inner_config_path, "r") as f:
        inner_config = yaml.safe_load(f)

    # Allow the RAG config to override the LM dtype (float32 is friendlier on CPU).
    dtype = model_cfg.get("torch_dtype")
    if dtype:
        inner_config.setdefault("language_model", {})["torch_dtype"] = dtype

    model = VLMForCausalLM(inner_config)

    checkpoint = model_cfg.get("checkpoint")
    if checkpoint and os.path.isdir(checkpoint):
        load_connector_checkpoint(model.connector, checkpoint)
        logger.info("Loaded connector checkpoint from %s", checkpoint)
    else:
        logger.warning(
            "No connector checkpoint loaded (checkpoint=%r). Running with an "
            "untrained connector — fine for exercising retrieval / prompt formatting, "
            "but generated answers will not be meaningful.",
            checkpoint,
        )

    model = model.to(device)
    model.eval()
    return model
