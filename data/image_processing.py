"""Image loading + CLIP preprocessing (the domain-specific seam for non-RGB inputs)."""

from PIL import Image
from transformers import CLIPImageProcessor


def load_and_process_image(
    image_path: str, image_processor: CLIPImageProcessor
) -> "torch.Tensor":
    """Load an image as RGB and run the CLIP processor, returning a ``(3, 224, 224)`` tensor.

    This is the domain-specific seam: replace it to handle non-RGB inputs (FITS, DICOM, multi-band)
    when porting to another domain.
    """
    import torch

    image = Image.open(image_path).convert("RGB")
    processed = image_processor(images=image, return_tensors="pt")
    return processed["pixel_values"].squeeze(0)  # (3, 224, 224)
