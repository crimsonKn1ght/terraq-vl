import argparse
import logging

import yaml
import torch

from vlm_model.vlm import VLMForCausalLM
from vlm_model.utils import IMAGE_TOKEN
from data.image_processing import load_and_process_image
from training.checkpoint import load_connector_checkpoint, load_lora_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_vlm(config_path: str, connector_checkpoint: str, device: str = "cuda") -> VLMForCausalLM:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = VLMForCausalLM(config)
    load_connector_checkpoint(model.connector, connector_checkpoint)

    # Stage-2 checkpoints also carry a LoRA adapter (lora/ subdir). The config's `lora` block already
    # built the adapter structure on the LLM; here we load the trained weights into it. Stage-1
    # checkpoints have no lora/ subdir, so this is a no-op for them.
    if getattr(model.language_model, "is_lora", False):
        if load_lora_adapter(model.language_model.model, connector_checkpoint):
            logger.info(f"Loaded LoRA adapter from {connector_checkpoint}/lora")
        else:
            logger.warning(
                f"Config has a `lora` block but no lora/ adapter found in {connector_checkpoint}; "
                "running with randomly-initialized adapters."
            )

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from {connector_checkpoint}")
    return model


def run_inference(
    model: VLMForCausalLM,
    image_path: str,
    prompt: str = "Describe this image in detail.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    pixel_values = load_and_process_image(image_path, model.image_processor)
    pixel_values = pixel_values.unsqueeze(0).to(device)

    conversation = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{IMAGE_TOKEN}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"

    encoded = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": 0.9,
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    # The LLM weights are bf16 (per config) but the connector/image embeds are fp32. Training
    # reconciled this via Accelerate's autocast; mirror that here so the dtypes match.
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids=input_ids,
            images=pixel_values,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="VLM Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to connector checkpoint dir")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_vlm(args.config, args.checkpoint, args.device)

    response = run_inference(
        model=model,
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
