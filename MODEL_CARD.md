---
license: cc-by-sa-4.0
base_model:
  - Qwen/Qwen2.5-1.5B-Instruct
  - openai/clip-vit-large-patch14
datasets:
  - UniverseTBD/AstroLLaVA_convos
language:
  - en
pipeline_tag: image-text-to-text
tags:
  - astraq-vl
  - vision-language-model
  - llava
  - astronomy
  - multimodal
  - image-captioning
  - connector
---

# AstraQ-VL Stage-1 (connector alignment)

A LLaVA-style vision–language connector that lets **Qwen2.5-1.5B-Instruct** describe astronomy
images encoded by **CLIP ViT-L/14**. Only the connector (~3.9M params) is trained; both backbones
stay frozen. This is the **Stage-1 feature-alignment** stage, trained for **3 epochs** on
[`UniverseTBD/AstroLLaVA_convos`](https://huggingface.co/datasets/UniverseTBD/AstroLLaVA_convos)
with a **disjoint held-out test split** so it can be evaluated on unseen images.

> ⚠️ This repo ships the **connector checkpoint only** (`connector.safetensors`, ~16 MB). It is
> **not** a standalone `transformers` model — it needs the custom VLM code from the
> [astraq-vl](https://github.com/crimsonKn1ght/astraq-vl) repo plus the two base models
> (auto-downloaded from the Hub) to run.

## Downloads (per-epoch bundles)

Each bundle holds that epoch's checkpoint, its **held-out** predictions (`predictions_test_ep*.jsonl`),
the training config, the `test.json` split, and a `REPRODUCE.md`:

| Bundle | Checkpoint | |
|--------|-----------|--|
| [`astraq-vl-stage1-ep3.zip`](https://huggingface.co/grKnight/astraq-vl-stage1/blob/main/astraq-vl-stage1-ep3.zip) | `checkpoint-3789` (epoch 3, final) | **recommended** |
| [`astraq-vl-stage1-ep2.zip`](https://huggingface.co/grKnight/astraq-vl-stage1/blob/main/astraq-vl-stage1-ep2.zip) | `checkpoint-2500` (≈ epoch 2) | |
| [`astraq-vl-stage1-ep1.zip`](https://huggingface.co/grKnight/astraq-vl-stage1/blob/main/astraq-vl-stage1-ep1.zip) | `checkpoint-1300` (≈ epoch 1) | |

> **Superseded files.** An earlier release (`*-legacy-1epoch-no-heldout-*`) was trained to ~1 epoch
> only and evaluated on training images (no held-out split, so possible leakage). Kept for record;
> use the `ep1`/`ep2`/`ep3` bundles above.

## Architecture

```
image ─► CLIP ViT-L/14 (frozen) ─► MLP connector (TRAINED) ─► Qwen2.5-1.5B-Instruct (frozen) ─► text
                                    1024 → 1536 → 1536
```

- **Vision:** `openai/clip-vit-large-patch14`, penultimate layer patch features (frozen)
- **Connector:** 2-layer MLP with GELU, 1024→1536→1536 (the only trained weights)
- **LLM:** `Qwen/Qwen2.5-1.5B-Instruct` (frozen)
- **Trainable / total:** 3,935,232 / 1,850,414,592 (0.21%)

## Training

| | |
|---|---|
| Data | `UniverseTBD/AstroLLaVA_convos`, per-image held-out split: train 161,653 recs / 29,151 imgs, test 3,271 recs / 591 imgs (41 corrupt skipped) |
| Image prep | long side ≤ 384 px, JPEG |
| Objective | next-token cross-entropy on answer tokens only (connector-only) |
| Epochs / steps | 3 epochs, 3,789 update steps |
| Effective batch | 128 (per-device 8 × grad-accum 16) |
| LR / schedule | 1e-3, cosine with 3% warmup |
| Precision | bf16 (autocast) |
| Max length | 512 (+256 image tokens) |
| Hardware | 1× RTX 6000 Ada (48 GB), ~26 samples/s, ~38 GB VRAM |
| Loss | ~2.08 → ~1.50 |

`checkpoint-3789` (epoch 3) is the recommended checkpoint; `checkpoint-1300`/`-2500` are the
epoch-1/epoch-2 points for comparison.

## Usage

```bash
# 1. get the code
git clone https://github.com/crimsonKn1ght/astraq-vl && cd astraq-vl
pip install -r requirements.txt

# 2. download + unzip the recommended bundle
hf download grKnight/astraq-vl-stage1 astraq-vl-stage1-ep3.zip --local-dir .
unzip astraq-vl-stage1-ep3.zip -d ckpt

# 3. caption an image (CLIP + Qwen auto-download on first run)
python inference.py \
  --config ckpt/pretrain_astraq_vl.yaml \
  --checkpoint ckpt/checkpoint-3789 \
  --image your_astro_image.jpg \
  --prompt "Describe this astronomical image." \
  --temperature 0
```

The bundled `predictions_test_ep*.jsonl` hold the held-out outputs with their reference captions.

## Capabilities & limitations

**What it does well** — it grounds on *coarse visual structure* (object class / morphology), and
this **generalizes to held-out images**. On unseen test images, quality improved monotonically with
training: epoch 1 misidentified objects, epoch 2 fixed the object *category*, and epoch 3 recovered
*specific* objects — e.g. correctly naming **SN 1987A and its ring** and the **Dumbbell Nebula**, on
images it never trained on. Because these are held-out, that's genuine generalization, not
memorization.

**What it doesn't** — it **hallucinates fine details** (exact catalog numbers, telescopes, dates,
distances), filling specifics from the frozen LLM's prior rather than the pixels. This is the
expected Stage-1 ceiling: the connector supplies a coarse visual category and the frozen LLM
improvises the rest. For factual specificity, a **Stage-2 fine-tune** (unfreezing the LLM via LoRA
on the QA pairs) is the fix — more Stage-1 epochs do not help. That model is now released at
[`grKnight/astraq-vl-stage2`](https://huggingface.co/grKnight/astraq-vl-stage2).

The held-out comparison above is a **qualitative spot check** on a few samples, not a full
quantitative benchmark.

## Reproduction

Each bundle includes a `REPRODUCE.md` pinning the exact code commit, base models, and package
versions (`torch 2.8.0+cu128`, `transformers 5.12.1`). The split is seeded, so the build reproduces
the exact train/test partition.

```
build:  python scripts/build_astrollava_trainset.py --include-qa --max-image-size 384 --test-fraction 0.02 --seed 42
train:  python train.py --config configs/pretrain_astraq_vl.yaml
eval:   python scripts/batch_inference.py --records-json datasets/astrollava_llava/test.json --num-samples 0 ...
```

## License & attribution

- **Weights:** `cc-by-sa-4.0`, inherited from the training data.
- **Training data:** [`UniverseTBD/AstroLLaVA_convos`](https://huggingface.co/datasets/UniverseTBD/AstroLLaVA_convos)
  (CC-BY-SA-4.0); imagery from NASA APOD, ESO, and NASA/ESA Hubble.
- **Base models:** Qwen2.5-1.5B-Instruct (Apache-2.0), CLIP ViT-L/14 (OpenAI, MIT).
- Built on the AstroLLaVA work ([arXiv:2504.08583](https://arxiv.org/abs/2504.08583)).
