from PIL import Image
from transformers import CLIPImageProcessor


def load_and_process_image(
    image_path: str, image_processor: CLIPImageProcessor
) -> "torch.Tensor":
    import torch

    image = Image.open(image_path).convert("RGB")
    processed = image_processor(images=image, return_tensors="pt")
    return processed["pixel_values"].squeeze(0)  # (3, 224, 224)
