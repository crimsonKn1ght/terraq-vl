"""Turn retrieved image-report pairs into a structured context block and splice it
into the existing VLM prompt.

The only change versus ``inference.run_inference`` (inference.py:43) is that a text
block of retrieved clinical references is inserted *after* the ``<image>`` token and
*before* the question. Keeping ``<image>`` first is essential: ``prepare_inputs_embeds``
(vlm.py:88) substitutes the 256 visual embeddings at the *first* image-token position,
so the model, tokenizer and connector are all untouched — this is pure inference-time
grounding with no retraining.
"""

from __future__ import annotations

from typing import List, Optional

from vlm_model.utils import IMAGE_TOKEN
from retrieval.base import RetrievedPair
from ragcore import phase3_stubs


# Defaults are medical (the project's primary domain). All three are overridable via a
# `prompt:` block in the config, so another domain (e.g. astronomy) reuses this file as-is.
SYSTEM_PROMPT = (
    "You are a careful medical assistant. Use the retrieved clinical references only "
    "if they are relevant to the image and question; do not fabricate findings."
)
DEFAULT_REFERENCES_HEADER = "[Retrieved clinical references]"
DEFAULT_REFERENCES_FOOTER = "[End of references]"


def format_context(
    pairs: List[RetrievedPair],
    cfg: dict,
    question: Optional[str] = None,
) -> str:
    """Render retrieved pairs as a compact, source-attributed reference block.

    Returns an empty string when there are no pairs (the no-retrieval baseline), so the
    resulting prompt is byte-for-byte the plain prompt in that case.
    """
    if not pairs:
        return ""

    eval_cfg = cfg.get("eval", {})
    prompt_cfg = cfg.get("prompt", {})
    max_chars = int(eval_cfg.get("max_context_chars", 700))
    header = prompt_cfg.get("references_header", DEFAULT_REFERENCES_HEADER)
    footer = prompt_cfg.get("references_footer", DEFAULT_REFERENCES_FOOTER)

    if cfg.get("phase3", {}).get("adaptive_context_window", False):
        pairs = phase3_stubs.adaptive_window(pairs, question or "", cfg)

    lines = [header]
    used = len(header)
    for idx, pair in enumerate(pairs, start=1):
        modality = pair.meta.get("modality", "image")
        report = " ".join(pair.report.split())  # collapse whitespace
        entry = f"[{idx}] ({modality}) {report}"
        # Enforce the global character budget so references never push the question /
        # answer out of the model's trained context window (max_length: 512).
        if used + len(entry) + 1 > max_chars:
            remaining = max_chars - used - 1
            if remaining <= 0:
                break
            entry = entry[:remaining].rstrip()
            lines.append(entry)
            break
        lines.append(entry)
        used += len(entry) + 1

    lines.append(footer)
    return "\n".join(lines)


def build_rag_conversation(
    question: str,
    context_block: str,
    image_token: str = IMAGE_TOKEN,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Compose the Qwen-style chat string with the image, references, and question.

    Mirrors the f-string at inference.py:43; the references slot in between the image
    token and the question. When ``context_block`` is empty this reduces to the original
    no-retrieval prompt (with the medical system prompt).
    """
    user_body = f"{image_token}\n"
    if context_block:
        user_body += f"{context_block}\n"
    user_body += question

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
