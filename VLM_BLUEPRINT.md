# VLM Recreation Blueprint

A domain-agnostic recipe for recreating this codebase's Vision-Language Model (VLM) on **any new
domain** (satellite imagery, histopathology, industrial inspection, remote sensing, documents, …).

This repo (AstraQ-VL) is a **minimal LLaVA-style VLM**: a frozen vision encoder + a small trainable
MLP connector + a frozen instruction-tuned LLM, trained in two stages, with an optional
inference-time Retrieval-Augmented Generation (RAG) grounding layer. Everything here generalizes;
astronomy and medicine appear only as *worked examples*.

**How to use this document:** read §1–§2 for the mental model, copy the code in §3–§7 as your
skeleton, then work the **Porting Checklist (§11)** to decide what to keep vs. swap for your domain.

---

## Table of Contents

1. [The idea in one picture](#1-the-idea-in-one-picture)
2. [Core dimensions & constants (swap table)](#2-core-dimensions--constants-swap-table)
3. [The critical mechanism: splicing image embeddings](#3-the-critical-mechanism-splicing-image-embeddings)
4. [Data format & tokenization](#4-data-format--tokenization)
5. [Two-stage training](#5-two-stage-training)
6. [Config reference (annotated YAML)](#6-config-reference-annotated-yaml)
7. [Inference](#7-inference)
8. [Optional: RAG grounding layer](#8-optional-rag-grounding-layer)
9. [Evaluation methodology](#9-evaluation-methodology)
10. [Environment & hardware](#10-environment--hardware)
11. [Porting checklist: keep vs. swap](#11-porting-checklist-keep-vs-swap)
12. [Gotchas & hard-won lessons](#12-gotchas--hard-won-lessons)
13. [Directory layout to recreate](#13-directory-layout-to-recreate)

---

## 1. The idea in one picture

```
Image (3, 224, 224)
    ↓
[Vision Encoder: FROZEN]  e.g. CLIP ViT-L/14  → N patch tokens (B, N, D_vision)
    ↓
[MLP Connector: TRAINABLE] Linear→GELU→Linear → (B, N, D_llm)
    ↓ (spliced in at the <image> token position, concatenated with)
Text token embeddings (B, T, D_llm) from the LLM's own embedding table
    ↓
[LLM: FROZEN (Stage 1) / FROZEN + LoRA (Stage 2)]  → next-token loss
```

**Training objective:** plain next-token prediction on image–text pairs. The visual token positions
are masked out of the loss (`label = -100`); only the assistant's text tokens produce gradients.

**Why this design:**
- **Simplicity**: no Q-Former, no cross-attention. Just a linear projection with a GELU.
- **Efficiency**: vision encoder + LLM are frozen; you train ~4M connector params (Stage 1) or
  ~22M connector+LoRA params (Stage 2), not billions.
- **Proven**: this is the LLaVA alignment recipe.

**Two stages:**
- **Stage 1 (alignment):** train **only** the connector to map frozen vision features into the
  LLM's embedding space. Grounds *coarse* structure (object class/morphology) but hallucinates fine
  specifics (names, numbers, dates) filled from the LLM's prior.
- **Stage 2 (instruction tuning):** warm-start the Stage-1 connector, keep training it, **and**
  fine-tune the LLM with **LoRA** adapters on the same caption+QA data. Vision encoder stays frozen.
  Targets the Stage-1 ceiling by letting the LLM learn from pixels, not just the connector.

---

## 2. Core dimensions & constants (swap table)

The whole model is defined by a handful of numbers. To retarget a new base model pair, change these
consistently everywhere (config + connector dims are the only hard couplings):

| Thing | This repo's value | Where it comes from | Change if… |
|---|---|---|---|
| Vision encoder | `openai/clip-vit-large-patch14` | config `vision_encoder.model_name` | new visual domain |
| Vision hidden size `D_vision` | **1024** | CLIP ViT-L config | encoder swap → **connector input changes** |
| Patch/visual tokens `N` | **256** = (224/14)² | `(image_size // patch_size)²`, CLS dropped | encoder swap |
| Image size | **224×224** | CLIP image processor | encoder swap |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` | config `language_model.model_name` | need bigger/smaller LLM |
| LLM hidden size `D_llm` | **1536** | Qwen2.5-1.5B config | LLM swap → **connector output changes** |
| Connector shape | `Linear(1024→1536)→GELU→Linear(1536→1536)` | `vlm_model/connector.py` | tracks `D_vision`,`D_llm` |
| Image placeholder token | `"<image>"` | `vlm_model/utils.py` | keep (added as special token) |
| Loss-ignore index | `-100` | `vlm_model/utils.py` | keep (PyTorch convention) |
| Trainable params | ~3.9M (Stage 1) / ~22.4M (Stage 2) | connector (+LoRA) | informational |

**Constants file** (`vlm_model/utils.py`), copy verbatim:

```python
import torch
import torch.nn as nn

IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
```

---

## 3. The critical mechanism: splicing image embeddings

This is the heart of the whole system: the one function to understand deeply. It builds
`inputs_embeds` by replacing the single `<image>` token with the `N` projected visual embeddings,
and masks those positions out of the loss.

### 3a. The three sub-modules

**Vision encoder** (`vlm_model/vision_encoder.py`), frozen; returns patch features, drops CLS:

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from .utils import freeze_module

class VisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14",
                 select_layer=-2, select_feature="patch"):
        super().__init__()
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        freeze_module(self.model)

    @property
    def hidden_size(self):  # D_vision
        return self.model.config.hidden_size

    @property
    def num_patches(self):  # N
        return (self.model.config.image_size // self.model.config.patch_size) ** 2

    @torch.no_grad()
    def forward(self, images):
        outputs = self.model(pixel_values=images, output_hidden_states=True)
        features = outputs.hidden_states[self.select_layer]   # penultimate layer (-2)
        if self.select_feature == "patch":
            features = features[:, 1:, :]  # drop CLS token → (B, 256, 1024)
        return features
```

> **Note:** `select_layer=-2` (penultimate hidden state) is the LLaVA convention: the last layer is
> too specialized for CLIP's contrastive objective. Keep this even when swapping encoders that expose
> `output_hidden_states`.

**Connector** (`vlm_model/connector.py`), the *only* thing trained in Stage 1:

```python
import torch.nn as nn

class VisionLanguageConnector(nn.Module):
    def __init__(self, vision_hidden_size=1024, llm_hidden_size=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, vision_features):
        return self.mlp(vision_features)
```

**Language model** (`vlm_model/language_model.py`), frozen base; adds the `<image>` token; optional
LoRA for Stage 2:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import freeze_module, IMAGE_TOKEN

# Qwen2.5 attention + MLP projections: standard LoRA target set. Adjust names for other LLMs.
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]

class LanguageModel(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 torch_dtype=torch.bfloat16, lora=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Register <image> as a real special token and grow the embedding table by one row.
        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        freeze_module(self.model)  # always start fully frozen
        self.is_lora = lora is not None
        if self.is_lora:
            self._apply_lora(lora)

    def _apply_lora(self, lora):
        from peft import LoraConfig, get_peft_model   # lazy import: Stage 1 needs no peft
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("lora_alpha", 32)),
            lora_dropout=float(lora.get("lora_dropout", 0.05)),
            target_modules=list(lora.get("target_modules", DEFAULT_LORA_TARGET_MODULES)),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

    @property
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
```

### 3b. The splice (`vlm_model/vlm.py`)

The composite model wires the three together. `encode_images` runs the frozen vision tower under
`no_grad` then the trainable connector (so gradients flow *only* into the connector). The full
`prepare_inputs_embeds` is worth reading in the source; the essential loop per sample is:

```python
def prepare_inputs_embeds(self, input_ids, attention_mask, labels, images=None):
    embed_tokens = self.language_model.get_input_embeddings()
    if images is None:                                   # text-only path
        return embed_tokens(input_ids), attention_mask, labels

    image_embeds = self.encode_images(images)            # (B, N, D_llm)

    # for each sample: find the <image> position, then rebuild the sequence as
    #   [ text_embeds_before ] + [ N visual embeds ] + [ text_embeds_after ]
    img_pos = torch.where(ids == self.image_token_id)[0][0].item()
    before = embed_tokens(ids[:img_pos])
    after  = embed_tokens(ids[img_pos + 1:])
    combined_embeds = torch.cat([before, image_embeds[i], after], dim=0)

    # labels: the N visual positions become IGNORE_INDEX so they never contribute to the loss
    image_labels = torch.full((self.num_patches,), IGNORE_INDEX, ...)
    combined_labels = torch.cat([labels_before, image_labels, labels_after], dim=0)

    # attention_mask: the N visual positions are all 1s (attended)
    image_mask = torch.ones(self.num_patches, ...)
    combined_mask = torch.cat([mask_before, image_mask, mask_after], dim=0)

    # then right-pad all samples in the batch to the same max length
    return padded_embeds, padded_mask, padded_labels
```

`forward()` and `generate()` both call `prepare_inputs_embeds` first, then pass `inputs_embeds`
(never `input_ids`) into the LLM. **This is why no model/tokenizer surgery is needed**: the image
enters purely as embeddings at one token slot.

`enable_gradient_checkpointing()` (Stage 2 only) calls `llm.gradient_checkpointing_enable()` and
sets `config.use_cache = False` (mutually exclusive with checkpointing). It handles the PeftModel
config-proxy case.

---

## 4. Data format & tokenization

### 4a. LLaVA JSON schema (the universal interchange format)

```json
[
  {
    "id": "sample_0",
    "image": "relative/path/inside/image_dir.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt",   "value": "A detailed answer/caption here."}
    ]
  }
]
```

- `image` is **relative** to `data.image_dir`.
- The `<image>` placeholder goes in the **human** turn.
- Only the assistant (`gpt`) turn is supervised.

### 4b. Dataset (`data/dataset.py`)

`LLaVAPretrainDataset` loads the JSON, and for each item preprocesses the image (§4d) and tokenizes
the conversation (§4c). It has a **robustness feature worth keeping**: on a failed sample it tries
the next up-to-10 indices, and finally falls back to a dummy sample, so one corrupt image never
kills a training run.

### 4c. Conversation tokenization + label masking (`data/conversation.py`)

Uses the Qwen chat template and masks everything except the assistant answer:

```python
def tokenize_conversation(conversations, tokenizer, image_token_id, max_length=2048):
    # keep only the last human/gpt turn (this repo flattens multi-turn to single-turn)
    for turn in conversations:
        if turn["from"] == "human":    human_msg = turn["value"]
        elif turn["from"] == "gpt":    assistant_msg = turn["value"]

    system_text     = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    user_text       = f"<|im_start|>user\n{human_msg}<|im_end|>\n"
    assistant_prefix = "<|im_start|>assistant\n"
    assistant_text  = f"{assistant_msg}<|im_end|>\n"

    prompt_str = system_text + user_text + assistant_prefix   # everything up to the answer
    full_str   = prompt_str + assistant_text

    prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_tokens   = tokenizer.encode(full_str,   add_special_tokens=False)

    input_ids = torch.tensor(full_tokens[:max_length], dtype=torch.long)
    labels = input_ids.clone()
    labels[:prompt_len] = IGNORE_INDEX        # mask the prompt; supervise only the answer
    return input_ids, labels
```

> **Swap the chat template** (`<|im_start|>…<|im_end|>`) to match your LLM (Llama-3, Mistral, etc.).
> The masking logic is identical regardless of template.

### 4d. Image preprocessing (`data/image_processing.py`)

```python
def load_and_process_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    processed = image_processor(images=image, return_tensors="pt")
    return processed["pixel_values"].squeeze(0)   # (3, 224, 224)
```

> **This is the #1 file to rewrite for a non-photographic domain** (FITS astronomy frames, DICOM
> medical, multi-band remote sensing). You must handle dynamic range / bit depth / channels and
> produce whatever tensor your vision tower expects.

### 4e. Collator (`data/collator.py`)

Right-pads `input_ids` (with `pad_token_id`) and `labels` (with `-100`), builds the `attention_mask`,
and stacks images into `(B, 3, 224, 224)`. Caps at `max_length`. Straightforward: copy as-is.

### 4f. Dataset builder (`scripts/build_astrollava_trainset.py`, pattern to imitate)

Streams a HuggingFace dataset → emits `train.json` + `images/`. Reusable ideas:
- **Per-image, seeded train/test split** (`--test-fraction`, `--seed`): route an image *and all its
  records* to one side so captions/QA never leak across the split.
- **Two record types:** caption pairs (rotate through several prompt phrasings) and flattened
  single-turn QA (`--include-qa`).
- **Robust image handling:** `PILImage.MAX_IMAGE_PIXELS = None` (disable decompression-bomb guard for
  trusted huge frames), `img.thumbnail((max, max))` to cap disk, per-row try/except to skip corrupt
  rows instead of aborting, `decode=False` so decode errors surface per-row.
- **`os._exit(0)` at the end** to dodge a `datasets` streaming finalizer crash on interpreter
  shutdown (would otherwise return a false non-zero exit code to shell scripts).

---

## 5. Two-stage training

### 5a. Entry point (`train.py`)

```python
config = yaml.safe_load(open(args.config))
accelerator = Accelerator(
    mixed_precision="bf16" if config["training"].get("bf16", True) else "no",
    gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 32),
)
model = VLMForCausalLM(config)

# Stage 2: warm-start the connector from a Stage-1 checkpoint
if config.get("stage1_checkpoint"):
    load_connector_checkpoint(model.connector, config["stage1_checkpoint"])

# Stage 2: gradient checkpointing to fit the LLM backward pass
if config["training"].get("gradient_checkpointing", False):
    model.enable_gradient_checkpointing()

# log every trainable tensor by name+shape: a fast sanity check on what's frozen
for name, param in model.named_parameters():
    if param.requires_grad:
        logger.info(f"  [TRAINABLE] {name}: {param.shape}")

dataset = LLaVAPretrainDataset(data_cfg["train_data_path"], data_cfg["image_dir"],
                               model.tokenizer, model.image_processor,
                               model.image_token_id, data_cfg.get("max_length", 2048))
VLMTrainer(model, dataset, config, accelerator).train()
```

### 5b. Trainer loop (`training/trainer.py`): key decisions

- **Optimizer:** `AdamW(betas=(0.9, 0.999), weight_decay=cfg)`. Optional **split LR**: give the
  warm-started connector its own (lower) LR than fresh LoRA via `connector_lr`.
- **Scheduler:** cosine decay with linear warmup (`training/lr_scheduler.py`):

```python
def lr_lambda(step):
    if step < num_warmup_steps:
        return step / max(1, num_warmup_steps)
    progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
```

- **Eval modes:** vision encoder always `.eval()`. LLM `.eval()` in Stage 1, but `.train()` in
  Stage 2 (so LoRA dropout is active).
- **Gradient-flow assertion** (do this once, *before* `zero_grad`): catches a silently detached
  image path, the most common wiring bug:

```python
assert any(p.grad is not None for p in unwrapped.connector.parameters()), \
    "Connector received no gradient: the image-embedding merge path is detached from the loss."
if is_lora:
    assert any(p.grad is not None for p in lora_params), \
        "No LoRA parameter received gradient: the LLM adapters are detached from the loss."
```

- **Grad clipping:** `clip_grad_norm_(trainable, max_grad_norm=1.0)`.
- **Accumulation:** `with accelerator.accumulate(model):`, effective batch =
  `per_device_batch_size × gradient_accumulation_steps × num_processes`.
- **Logging:** loss / LR / samples-per-sec every `logging_steps`.

### 5c. Checkpoint format (`training/checkpoint.py`)

Deliberately **not** a full `transformers` model, only the trained deltas:

```
checkpoint-<step>/
├── connector.safetensors        # always (the trained MLP)
├── lora/                         # Stage 2 only (peft_model.save_pretrained)
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── training_state.pt            # optimizer + scheduler state (for resume)
└── meta.json                    # {"step": ..., "loss": ...}
```

- `save_connector_checkpoint(..., peft_model=None)` → Stage-1 dirs have no `lora/` and stay
  byte-compatible.
- `load_connector_checkpoint()` restores the connector (and optionally optimizer/scheduler).
- `load_lora_adapter()` is a **no-op if there's no `lora/` subdir**, so Stage-1 checkpoints load
  unchanged through the same code path.

### 5d. Hyperparameters that mattered

| Parameter | Stage 1 | Stage 2 | Rationale |
|---|---|---|---|
| Learning rate | 1e-3 – 2e-3 | 2e-4 | Connector-only tolerates high LR; LoRA+LLM needs lower |
| Schedule | cosine, 3% warmup | cosine, 3% warmup | Components are pretrained → short warmup |
| Weight decay | 0.0 | 0.0 | Tiny trainable set needs no regularization |
| Effective batch | 128–256 | 64 | Stage 2's full-LLM backward is the memory driver |
| Epochs | 1–3 | 1 | Alignment / instruction-tuning conventions |
| Precision | bf16 | bf16 + grad-checkpointing | Ampere+ GPU required for bf16 |
| Max length | 512 (+256 image tokens) | 512 | Longer OOMs the large-vocab fp32 loss |

---

## 6. Config reference (annotated YAML)

Everything is config-driven; the model reads `config["vision_encoder"|"language_model"|"connector"
|"data"|"training"]`. Copy and edit these two files.

**Stage 1 (`configs/pretrain_stage1.yaml`):**

```yaml
vision_encoder:
  model_name: openai/clip-vit-large-patch14
  select_layer: -2          # penultimate hidden state (LLaVA convention)
  select_feature: patch     # drop CLS, keep the 256 patch tokens
language_model:
  model_name: Qwen/Qwen2.5-1.5B-Instruct
  torch_dtype: bfloat16
connector:
  vision_hidden_size: 1024  # = D_vision (must match the encoder)
  llm_hidden_size: 1536     # = D_llm    (must match the LLM)
data:
  train_data_path: datasets/<yourset>/train.json
  image_dir: datasets/<yourset>/images
  max_length: 512
training:
  output_dir: ./checkpoints/stage1
  num_epochs: 1
  per_device_batch_size: 8
  gradient_accumulation_steps: 16   # effective batch = 128
  learning_rate: 0.001
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  bf16: true                        # needs Ampere-or-newer GPU
  dataloader_num_workers: 8
  logging_steps: 10
  save_steps: 100
  seed: 42
```

**Stage 2 (`configs/finetune_..._stage2.yaml`)** (adds the `lora` block, `stage1_checkpoint`, and
`training.stage: 2`):

```yaml
language_model:
  model_name: Qwen/Qwen2.5-1.5B-Instruct
  torch_dtype: bfloat16
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
connector:
  vision_hidden_size: 1024
  llm_hidden_size: 1536
stage1_checkpoint: ./checkpoints/stage1/checkpoint-XXXX   # warm-start the connector
data:
  train_data_path: datasets/<yourset>/train.json
  image_dir: datasets/<yourset>/images
  max_length: 512
training:
  stage: 2                          # connector + LoRA (Stage 1 = connector only)
  output_dir: ./checkpoints/stage2
  num_epochs: 1
  per_device_batch_size: 4          # full-LLM backward is the memory driver; drop to 2 if OOM
  gradient_accumulation_steps: 16   # effective batch = 64
  gradient_checkpointing: true      # required to fit the LLM backward pass
  learning_rate: 0.0002
  # connector_lr: 0.00005           # (optional) lower LR for the pretrained connector
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  bf16: true
  dataloader_num_workers: 8
  logging_steps: 10
  save_steps: 200
  seed: 42
```

Run:
```bash
python train.py --config configs/pretrain_stage1.yaml          # or accelerate launch train.py ...
python train.py --config configs/finetune_..._stage2.yaml
```

---

## 7. Inference (`inference.py`)

```python
def run_inference(model, image_path, prompt, max_new_tokens=256, temperature=0.7, device="cuda"):
    pixel_values = load_and_process_image(image_path, model.image_processor).unsqueeze(0).to(device)

    conversation = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{IMAGE_TOKEN}\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n")

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"                     # left-pad for generation
    encoded = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": 0.9,
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    # CRITICAL: LLM weights are bf16 but the connector/image embeds are fp32. Training reconciled
    # this via Accelerate autocast; mirror it here or you get a dtype-mismatch error.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids = model.generate(input_ids=encoded["input_ids"].to(device),
                                    images=pixel_values,
                                    attention_mask=encoded["attention_mask"].to(device),
                                    **generate_kwargs)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
```

Loading restores the connector and (if the config has a `lora` block) the LoRA adapter from the
checkpoint dir: Stage-1 checkpoints simply skip the LoRA load. Use `--temperature 0` for
deterministic/reproducible outputs.

---

## 8. Optional: RAG grounding layer

An **inference-time** retrieval layer that reduces hallucination *without any retraining*. It
retrieves reference image–report pairs, formats them into a text block, and **prepends that block
after the `<image>` token and before the question**. Because `<image>` stays first,
`prepare_inputs_embeds` is untouched: the model, tokenizer, and connector are all unchanged.

```
image ─► <image> token ─► [retrieved reference block] ─► question ─► model.generate
                          └────── inserted here, pure prompt-level grounding ──────┘
```

### 8a. The prepend (`ragcore/context_format.py`, `rag_inference.py`)

```python
def build_rag_conversation(question, context_block, image_token="<image>", system_prompt=...):
    user_body = f"{image_token}\n"
    if context_block:
        user_body += f"{context_block}\n"     # references go BETWEEN image and question
    user_body += question
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_body}<|im_end|>\n"
            f"<|im_start|>assistant\n")
```

The reference block is character-budgeted (`max_context_chars`) so references never push the
question out of the trained context window. When there are no pairs, the prompt is byte-for-byte the
plain no-retrieval prompt (clean baseline).

### 8b. Pluggable seams (the reusable abstractions)

- **`BaseRetriever`** (`retrieval/base.py`): every retriever returns the same `RetrievedPair`
  (`pair_id, image_path, report, score, meta`). Modes: `no_retrieval | dense_visual | sparse_bm25 |
  hybrid`, selected by `retrieval/factory.py` (lazy imports so faiss/sentence-transformers aren't
  needed unless used).
- **`CorpusLoader`** (`corpus/base.py`): every dataset adapter yields `CorpusRecord`
  (`pair_id, image_path, report, meta`); `corpus/registry.py` maps `corpus.name` → loader. New
  domain = one new adapter (e.g. `galaxy_zoo.py` alongside `iu_xray.py`, `roco.py`).
- **Encoders** (`retrieval/encoders.py`): SBERT for report text; **the same vision tower as the VLM**
  for visual retrieval (256 patches mean-pooled to one L2-normalized vector). Both return
  `float32`, so a FAISS inner-product index == cosine.
- **Hybrid** (`retrieval/hybrid.py`): dense-text + dense-visual + BM25, fused by **weighted
  reciprocal-rank fusion** (`RRF_K=60`), then **cross-encoder rerank** to `top_k`.
- **Index:** FAISS `IndexFlatIP` (exact, correct at sample scale; swap to IVF/HNSW at large scale) +
  a BM25 index.

### 8c. Domain swap = a YAML edit

`configs/rag_eval.yaml` (medical) vs `configs/rag_eval_astro.yaml` (astronomy) differ only in:
`corpus.name`, `eval.benchmark`, `retrieval.text_encoder_id`, and an optional `prompt:` block
(`system`, `references_header`, `references_footer`). The retrieval machinery, ablation driver, eval
runner, and prepend are reused **unchanged**.

Run: `python run_rag.py --config configs/rag_eval_<domain>.yaml` (add `--synthetic` for a
no-download CPU smoke test). Outputs `results_<mode>.json` + `comparison.md` per mode.

> **Caveat:** RAG only yields meaningful numbers with a **trained connector** (`model.checkpoint`
> set). With a null checkpoint the pipeline runs end-to-end but generations are random, useful only
> to validate wiring.

---

## 9. Evaluation methodology

Score a **held-out split** (carved out *before* training, per-image, seeded) with a shared pipeline.
The metric set (`scripts/score_predictions.py`, `eval/metrics_nli.py`):

| Metric | What it measures | Direction |
|---|---|---|
| ROUGE-L F1 | longest-common-subsequence overlap w/ reference | ↑ |
| Token-F1 | bag-of-tokens overlap | ↑ |
| Exact match | normalized string equality (good for yes/no VQA) | ↑ |
| NLI consistency | `P(entail) − P(contradict)` via an MNLI classifier, in [-1,1] | ↑ |
| Contradiction rate | fraction of predictions that outright contradict the reference | ↓ |
| SBERT cosine | semantic similarity of prediction vs reference | ↑ |
| **Specificity hallucination** | **domain-specific: see below** | ↓ |

### The specificity-hallucination metric (a transferable idea)

Reusable recipe for "does the model invent precise facts it can't support?":
1. **Regex-extract "specifics"** from both prediction and reference, in astronomy: catalog numbers
   (`NGC 1234`, `M31`), instrument names (Hubble/JWST/ALMA…), measurements (light-years, parsecs,
   Kelvin, magnitude), redshift (`z = 0.5`), years.
2. **Unsupported specifics** = `specifics(prediction) − specifics(reference)`.
3. **Hallucination rate** = fraction of records with ≥1 unsupported specific;
   **precision proxy** = supported / total predicted specifics.

To port: replace the astronomy regex vocab (`INSTRUMENTS`, `CATALOG_PATTERNS`, `MEASUREMENT_PATTERN`,
etc.) with your domain's precise-fact patterns (gene names, part numbers, drug dosages, coordinates…).
Everything else in `specificity_row()` / `aggregate_specificity()` is domain-agnostic.

Also swap the NLI model (`roberta-large-mnli` default) for a domain-tuned checkpoint (MedNLI/SciNLI)
and read NLI numbers as *relative-across-models*, not absolute; general MNLI mishandles domain
negation/hedging.

Reporting rigor worth copying: **paired bootstrap confidence intervals** across common records
(`scripts/bootstrap_full_heldout_ci.py`) and per-split summaries (`overall` / `caption` / `qa`).

---

## 10. Environment & hardware

**`requirements.txt`:**

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0
accelerate>=0.28.0
datasets>=2.19.0
Pillow>=10.0.0
pyyaml>=6.0
tqdm>=4.66.0
safetensors>=0.4.0
peft>=0.11.0            # Stage-2 LoRA
# RAG layer (optional):
faiss-cpu>=1.7.4
sentence-transformers>=2.6.0
rank-bm25>=0.2.2
```

- **Python ≥ 3.10; CUDA 11.8+.** bf16 requires an **Ampere-or-newer** GPU (RTX 30xx/40xx, A-series);
  **avoid T4/V100**.
- **GPU memory scales with caption length**, not just batch. Reference measurements on an RTX 6000
  Ada (48 GB): Stage 1 (long captions, `max_length 512`, batch 8) ≈ **38 GB, ~26 samples/s**;
  Stage 2 (adds full-LLM backward, batch 4 + grad-checkpointing) ≈ **~15 samples/s**. Short-caption
  sets fit in 6–8 GB. On 24 GB: drop batch, raise grad-accum.
- **Cloud/RunPod workflow** (`RUNPOD.md`, `scripts/runpod_setup.sh`, `runpod_train.sh`): clone into
  a **persistent network volume** (`/workspace`), point `HF_HOME` there, smoke-test on 50 samples,
  then build the full set + train. CLIP + Qwen are public (no HF token).
- **`.gitignore`** the generated bulk: `checkpoints/`, `datasets/`, `hf_cache/`, RAG indexes/results
  (`rag_index*/`, `rag_results*/`), `eval_runs/`, `*.jsonl`, `*.zip`.

---

## 11. Porting checklist: keep vs. swap

### Keep **unchanged** (the reusable core)

- [ ] `vlm_model/connector.py`: MLP connector (dims come from config)
- [ ] `vlm_model/vlm.py`: `prepare_inputs_embeds` splice + masking (the crown jewel)
- [ ] `vlm_model/utils.py`: constants + freeze helpers
- [ ] `data/dataset.py`, `data/collator.py`: dataset + collation
- [ ] `data/conversation.py`: label masking (only swap the chat-template strings)
- [ ] `training/trainer.py`, `training/lr_scheduler.py`, `training/checkpoint.py`
- [ ] `train.py`, `inference.py`
- [ ] The entire RAG core (`ragcore/`, `retrieval/`, `eval/runner.py`, `eval/ablation.py`)
- [ ] The eval harness + specificity-metric *structure*

### Swap **per domain**

- [ ] **Vision tower** (config `vision_encoder.model_name`). ⚠️ **Biggest caveat:** a non-CLIP tower
      (AstroCLIP, BiomedCLIP, a custom encoder) is *not* a drop-in `CLIPVisionModel`, changes
      `D_vision` (→ connector input dim), and **requires retraining the connector from scratch**.
      Standard CLIP on RGB is fine only for a prototype.
- [ ] **`data/image_processing.py`**: domain image loading (FITS, DICOM, multi-band, dynamic range).
- [ ] **LLM** (config `language_model.model_name`) + LoRA `target_modules` (names differ per LLM) +
      the chat-template strings in `data/conversation.py` and `inference.py`.
- [ ] **Dataset builder**: imitate `scripts/build_astrollava_trainset.py` for your source; keep the
      seeded per-image split.
- [ ] **RAG:** a new `CorpusLoader` adapter + a benchmark builder + `retrieval.text_encoder_id` +
      the `prompt:` wording block.
- [ ] **Eval:** the specificity regex vocab + a domain NLI checkpoint.

### Minimal path to a working prototype in a new domain

1. Assemble data in **LLaVA JSON** (§4a) with a held-out split.
2. Copy the whole `vlm_model/`, `data/`, `training/` + `train.py`/`inference.py` unchanged.
3. Adjust **only** `configs/pretrain_stage1.yaml` (paths + the two connector dims if you changed the
   base pair) and `data/image_processing.py` if your images aren't standard RGB.
4. Train Stage 1 → spot-check → train Stage 2 (LoRA) → held-out eval.
5. (Optional) add the RAG layer once a connector is trained.

---

## 12. Gotchas & hard-won lessons

- **Dtype mismatch at inference.** LLM weights are bf16 but the connector/image embeds are fp32.
  Training hid this via Accelerate autocast; at inference you **must** wrap `generate` in
  `torch.autocast(dtype=torch.bfloat16)` or it errors.
- **Gradient checkpointing ⊥ KV cache.** `enable_gradient_checkpointing()` must also set
  `config.use_cache = False`. For a `PeftModel`, read the config via the base-model proxy.
- **The connector must receive gradient.** Assert it once per run (§5b): a detached image path
  trains silently to nothing. `encode_images` runs the vision tower under `no_grad` but the connector
  *outside* it, so `inputs_embeds` requires grad and checkpointing works without
  `enable_input_require_grads`.
- **Mask visual tokens from the loss** (`label = -100` for the N patch positions): otherwise the
  model tries to "predict" image embeddings and loss goes NaN.
- **Per-image (not per-record) train/test split.** Keep an image's caption and all its QA on the same
  side, or you leak and overstate held-out gains.
- **`max_length` drives memory more than batch.** The large-vocab fp32 loss OOMs at long sequences;
  512 (+256 image tokens ≈ 768 effective) is a safe default. Prefer shrinking `max_length` or batch
  over disabling bf16.
- **Decompression-bomb guard** hard-errors mid-stream on huge (100+ MP) frames: set
  `PILImage.MAX_IMAGE_PIXELS = None` for trusted data and cap with `img.thumbnail(...)`.
- **`datasets` streaming finalizer** can throw on interpreter shutdown and return a false failure to
  shell scripts; `os._exit(0)` after flushing sidesteps it.
- **Checkpoints are deltas, not models.** They need this repo's code + the two base models (+`peft`
  for Stage 2) to run; they are *not* standalone `transformers` checkpoints. Ship a `REPRODUCE.md`
  pinning the code commit, base-model ids, data-build command (with seed), and package versions.
- **LoRA / no-LoRA share one load path.** `load_lora_adapter()` is a no-op when there's no `lora/`
  subdir, so Stage-1 and Stage-2 checkpoints load through identical code.

---

## 13. Directory layout to recreate

```
<project>/
├── train.py                     # training entry point
├── inference.py                 # single-image inference
├── requirements.txt
├── configs/
│   ├── pretrain_stage1.yaml     # Stage 1 (connector only)
│   └── finetune_..._stage2.yaml # Stage 2 (connector + LoRA)
├── vlm_model/
│   ├── utils.py                 # constants, freeze/param helpers
│   ├── vision_encoder.py        # frozen vision tower wrapper
│   ├── connector.py             # trainable MLP  ← the only Stage-1 trainable
│   ├── language_model.py        # frozen LLM (+ optional LoRA), <image> token
│   └── vlm.py                   # composite model + prepare_inputs_embeds  ← crown jewel
├── data/
│   ├── image_processing.py      # image → tensor  ← domain-specific
│   ├── conversation.py          # tokenize + label-mask
│   ├── dataset.py               # LLaVA JSON dataset
│   └── collator.py              # batch padding + attention mask
├── training/
│   ├── lr_scheduler.py          # cosine + warmup
│   ├── checkpoint.py            # save/load connector (+ LoRA)
│   └── trainer.py               # the training loop
├── scripts/
│   └── build_<domain>_trainset.py   # HF dataset → LLaVA JSON (+ seeded split)
│   └── score_predictions.py         # held-out metrics (+ specificity)
└── (optional RAG layer)
    ├── rag_inference.py         # retrieve → prepend → generate
    ├── run_rag.py               # build index + run ablation
    ├── ragcore/                 # context_format, model_loader, phase3_stubs
    ├── retrieval/               # base, factory, encoders, faiss/bm25, hybrid, reranker
    ├── corpus/                  # base, registry, per-dataset adapters, build_index
    ├── eval/                    # runner, ablation, benchmarks, metrics_em, metrics_nli
    └── configs/rag_eval_<domain>.yaml
```

---

*Generated as a recreation guide for building a new-domain VLM from this codebase's patterns.
No source files were modified in producing this document.*
