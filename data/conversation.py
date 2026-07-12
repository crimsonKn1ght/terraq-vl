from typing import List, Dict, Tuple

import torch
from transformers import PreTrainedTokenizer

from vlm_model.utils import IGNORE_INDEX, IMAGE_TOKEN


def tokenize_conversation(
    conversations: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    image_token_id: int,
    max_length: int = 2048,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    human_msg = ""
    assistant_msg = ""
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            human_msg = value
        elif role == "gpt":
            assistant_msg = value

    system_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    user_text = f"<|im_start|>user\n{human_msg}<|im_end|>\n"
    assistant_prefix = "<|im_start|>assistant\n"
    assistant_text = f"{assistant_msg}<|im_end|>\n"

    prompt_str = system_text + user_text + assistant_prefix
    full_str = prompt_str + assistant_text

    prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_str, add_special_tokens=False)

    image_token_str_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    final_ids = []
    for tid in full_tokens:
        final_ids.append(tid)
    final_ids = full_tokens

    image_placeholder_positions = []
    for idx, tid in enumerate(final_ids):
        if tid == image_token_str_id:
            image_placeholder_positions.append(idx)

    if not image_placeholder_positions:
        image_text_tokens = tokenizer.encode(IMAGE_TOKEN, add_special_tokens=False)
        if len(image_text_tokens) > 1:
            for start_idx in range(len(final_ids) - len(image_text_tokens) + 1):
                if (
                    final_ids[start_idx : start_idx + len(image_text_tokens)]
                    == image_text_tokens
                ):
                    final_ids = (
                        final_ids[:start_idx]
                        + [image_token_id]
                        + final_ids[start_idx + len(image_text_tokens) :]
                    )
                    prompt_len = len(prompt_tokens) - len(image_text_tokens) + 1
                    break
            else:
                prompt_len = len(prompt_tokens)
        else:
            prompt_len = len(prompt_tokens)
    else:
        prompt_len = len(prompt_tokens)

    if len(final_ids) > max_length:
        final_ids = final_ids[:max_length]

    input_ids = torch.tensor(final_ids, dtype=torch.long)

    labels = input_ids.clone()
    labels[:prompt_len] = IGNORE_INDEX

    return input_ids, labels
