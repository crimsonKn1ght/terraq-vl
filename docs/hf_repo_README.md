---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model:
- Qwen/Qwen2.5-3B-Instruct
- openai/clip-vit-large-patch14
tags:
- remote-sensing
- vision-language-model
- vrsbench
- llava
- lora
---

# TerraQ-VL — Remote-Sensing Vision-Language Model (VRSBench)

A LLaVA-style VLM for remote sensing: a frozen **CLIP ViT-L/14** vision encoder and a frozen
**Qwen2.5-3B-Instruct** LLM bridged by a trainable MLP connector, trained on **VRSBench**
(aerial/satellite imagery → detailed caption + visual QA).

- **Stage 1** trains only the connector (vision tower and LLM frozen).
- **Stage 2** warm-starts that connector and adds **LoRA** adapters (r=16) on the LLM.

Code, training, and reproduction: **https://github.com/crimsonKn1ght/TerraQ-VL**

## Repository layout

```
stage-1/                          stage-2/
  checkpoints/                      checkpoints/
    checkpoint-100 … checkpoint-3270  checkpoint-200 … checkpoint-2180
    (connector.safetensors +          (+ lora/adapter_model.safetensors)
     training_state.pt + meta.json)
  config/    pretrain_vrsbench.yaml  config/    finetune_vrsbench_stage2.yaml
  curves/    held-out loss (png/csv/json)
  predictions/  greedy preds on the held-out test split
  data/      val.json + test.json (the disjoint held-out splits)
  MODEL_CARD.md   base models, hyperparameters, per-checkpoint train + val loss
  manifest.json   every file with size + sha256
```

Checkpoints are stored as **raw, directly-loadable dirs** (no unzip needed). See each
`stage-*/MODEL_CARD.md` for the exact per-checkpoint train and validation loss.

## Training in brief

| | Stage 1 (connector) | Stage 2 (connector + LoRA) |
|---|---|---|
| Trainable | MLP connector | connector (warm-started from stage-1 `checkpoint-3270`) + LoRA on Qwen |
| Epochs / steps | 3 / 3270 | 1 / 2180 |
| Effective batch | 128 | 64 |
| Warm-start | — | stage-1 `checkpoint-3270` |

**Data split** is three-way and disjoint **by image**: train / `val.json` (validation, used during
training) / `test.json` (held out for final eval only, never trained or selected on). Held-out test =
1,367 caption+VQA records over 197 images. Stage-1 token-weighted validation loss falls **1.79 → 1.21**
across the 33 checkpoints (see `stage-1/curves/`).

## Load & run

Get the code, then point inference at a checkpoint dir from this repo:

```bash
python inference.py --config configs/finetune_vrsbench_stage2.yaml \
    --checkpoint stage-2/checkpoints/checkpoint-2180 \
    --image your_image.jpg \
    --prompt "Describe this remote sensing image." --temperature 0
```

Pass the matching stage config so the (Stage-2) LoRA structure is built before the checkpoint loads;
`inference.py` restores the connector and (Stage 2) the LoRA adapter automatically.

Rebuild the exact training images from VRSBench with
`scripts/build_vrsbench_trainset.py --seed 42` — `val.json` / `test.json` here pin the held-out sets.

## Citation

```bibtex
@misc{gourab_roy_2026,
	author       = { Gourab Roy },
	title        = { terraq-vl (Revision f7ddb21) },
	year         = 2026,
	url          = { https://huggingface.co/grKnight/terraq-vl },
	doi          = { 10.57967/hf/9584 },
	publisher    = { Hugging Face }
}
```

DOI: [10.57967/hf/9584](https://doi.org/10.57967/hf/9584)

## License — research / non-commercial

The **code** (on GitHub) is MIT, but these **trained weights are not**: they are LoRA-adapted from
**Qwen2.5-3B-Instruct** and fine-tuned on **VRSBench**, so the checkpoints inherit the most
restrictive terms of both:

- **Qwen2.5-3B-Instruct** → **Qwen Research License** (`qwen-research`): **non-commercial**; commercial
  use needs a separate license from Alibaba Cloud. It also forbids using outputs to improve any
  non-Qwen LLM.
- **VRSBench** → **CC-BY-NC-4.0**: **non-commercial + attribution** (source imagery from DOTA-v2 /
  DIOR under their own academic terms).

**Net: research / non-commercial, attribution required.** Full breakdown in the GitHub
[Model & Data Licensing](https://github.com/crimsonKn1ght/TerraQ-VL#model--data-licensing) section.
