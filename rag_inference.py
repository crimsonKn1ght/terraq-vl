"""Retrieval-augmented inference — the RAG twin of ``inference.run_inference``.

Flow: retrieve top-k image-report pairs -> format them into a structured context block
-> prepend after the <image> token -> call the existing ``model.generate``. No model,
tokenizer or connector changes; grounding happens entirely through the prompt.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import yaml

from vlm_model.utils import IMAGE_TOKEN
from data.image_processing import load_and_process_image
from retrieval.base import BaseRetriever
from ragcore.context_format import format_context, build_rag_conversation, SYSTEM_PROMPT
from ragcore.model_loader import load_vlm_from_cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _decode_answer(tokenizer, output_ids) -> str:
    """Strip the chat scaffolding from generated ids (mirrors inference.py:67-74)."""
    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    return response.strip()


def rag_answer(
    model,
    image_path: str,
    question: str,
    retriever: Optional[BaseRetriever],
    cfg: dict,
    device: Optional[str] = None,
    retrieval_query: Optional[str] = None,
) -> str:
    """Answer one VQA query, optionally grounded with retrieved references.

    ``retriever=None`` reproduces the plain (Phase-1 baseline) prompt exactly.
    ``retrieval_query`` overrides the text used to search the corpus (defaults to the
    question); the Phase-3 query-decomposition hook feeds it.
    """
    device = device or cfg.get("model", {}).get("device", "cpu")
    eval_cfg = cfg.get("eval", {})
    top_k = int(cfg.get("retrieval", {}).get("top_k", 3))

    search_text = retrieval_query or question
    pairs = (
        retriever.retrieve(image_path, search_text, top_k)
        if retriever is not None
        else []
    )
    context_block = format_context(pairs, cfg, question=question)
    system_prompt = cfg.get("prompt", {}).get("system", SYSTEM_PROMPT)
    conversation = build_rag_conversation(question, context_block, IMAGE_TOKEN, system_prompt)

    pixel_values = load_and_process_image(image_path, model.image_processor)
    pixel_values = pixel_values.unsqueeze(0).to(device)

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    encoded = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generate_kwargs = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 64)),
        "do_sample": False,  # greedy: reproducible exact-match scoring
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    output_ids = model.generate(
        input_ids=input_ids,
        images=pixel_values,
        attention_mask=attention_mask,
        **generate_kwargs,
    )
    return _decode_answer(tokenizer, output_ids)


def main():
    parser = argparse.ArgumentParser(description="Retrieval-augmented VLM inference")
    parser.add_argument("--config", type=str, default="configs/rag_eval.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override retrieval mode: no_retrieval | dense_visual | sparse_bm25 | hybrid",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mode = args.mode or cfg.get("retrieval", {}).get("retrieval_mode", "hybrid")
    model = load_vlm_from_cfg(cfg)

    from retrieval.factory import build_retriever

    retriever = build_retriever(mode, cfg)
    answer = rag_answer(model, args.image, args.question, retriever, cfg)

    print(f"\nMode:     {mode}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")


if __name__ == "__main__":
    main()
