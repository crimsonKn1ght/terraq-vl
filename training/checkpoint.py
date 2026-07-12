import json
import os
from typing import Optional, Any

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


def save_connector_checkpoint(
    connector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    output_dir: str,
    peft_model: Optional[nn.Module] = None,
) -> str:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_file(connector.state_dict(), os.path.join(checkpoint_dir, "connector.safetensors"))

    # Stage-2: also persist the LoRA adapter next to the connector. Stage-1 passes peft_model=None,
    # so its checkpoint dirs stay byte-compatible (no lora/ subdir).
    if peft_model is not None:
        peft_model.save_pretrained(os.path.join(checkpoint_dir, "lora"))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )

    meta = {"step": step, "loss": loss}
    with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return checkpoint_dir


def load_lora_adapter(peft_model: nn.Module, checkpoint_path: str) -> bool:
    """Load a saved LoRA adapter into an already-LoRA-wrapped model. No-op (returns False) if the
    checkpoint has no ``lora/`` subdir, so Stage-1 checkpoints load unchanged."""
    lora_dir = os.path.join(checkpoint_path, "lora")
    if not os.path.isdir(lora_dir):
        return False

    from peft import set_peft_model_state_dict

    adapter_file = os.path.join(lora_dir, "adapter_model.safetensors")
    state_dict = load_file(adapter_file)
    set_peft_model_state_dict(peft_model, state_dict)
    return True


def load_connector_checkpoint(
    connector: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> int:
    connector_path = os.path.join(checkpoint_path, "connector.safetensors")
    state_dict = load_file(connector_path)
    connector.load_state_dict(state_dict)

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if (optimizer is not None or scheduler is not None) and os.path.exists(
        training_state_path
    ):
        training_state = torch.load(training_state_path, weights_only=True)
        if optimizer is not None:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])

    meta_path = os.path.join(checkpoint_path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("step", 0)

    return 0
