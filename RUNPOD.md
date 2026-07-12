# Training the remote-sensing VLM (TerraQ-VL) on RunPod

This trains the **MLP connector** (Stage 1) — and optionally the LLM's **LoRA adapters** (Stage 2) —
to align remote-sensing images with text, using **VRSBench** (~29.6k aerial/satellite images with
human-verified captions + VQA pairs, from DOTA-v2 / DIOR via GoogleEarth). CLIP stays frozen; the LLM
is **Qwen2.5-3B-Instruct** (frozen in Stage 1, LoRA-tuned in Stage 2).

> The original astronomy path (AstroLLaVA + Qwen2.5-1.5B) still ships as the backbone — build with
> `scripts/build_astrollava_trainset.py` and train a `configs/*astraq*.yaml` config instead.

## 0. Get the code on GitHub

The repo already lives at `https://github.com/crimsonKn1ght/TerraQ-VL`. On the pod you just clone it.

## 1. Create the pod

- **Template:** "RunPod PyTorch 2.x" (CUDA + PyTorch preinstalled).
- **GPU:** Qwen2.5-3B is heavier than the 1.5B backbone. Recommend **≥ 40 GB** — RTX 6000 Ada / A6000
  (48 GB), A100 (40/80 GB). A **24 GB** card (RTX 3090 / 4090) can still do **Stage 1** if you drop
  `per_device_batch_size` to 2 in `configs/pretrain_vrsbench.yaml` (and raise
  `gradient_accumulation_steps` to keep the effective batch at 128). Training defaults to **bf16**, so
  use an **Ampere-or-newer** GPU — **avoid T4 and V100**.
- **Network Volume:** create one (e.g. **50 GB**) and attach it; it mounts at `/workspace`. This is
  what makes the model downloads, the ~8.4 GB VRSBench image archive, and your checkpoints survive a
  pod stop/terminate. The container disk is wiped on terminate.

## 2. Clone into the volume and set up

Open the pod's **web terminal** (or SSH/JupyterLab) and clone **into `/workspace`** so everything
lands on the persistent volume:

```bash
cd /workspace
git clone https://github.com/crimsonKn1ght/TerraQ-VL.git
cd TerraQ-VL

# Smoke-test the whole pipeline on 50 source images first (fast, catches setup issues):
bash scripts/runpod_setup.sh 50
bash scripts/runpod_train.sh        # runs a few steps and saves a checkpoint

# Happy? Build the full set and train for real:
bash scripts/runpod_setup.sh        # full VRSBench set (downloads ~8.4 GB of images once)
bash scripts/runpod_train.sh
```

`runpod_setup.sh` installs `requirements.txt`, points `HF_HOME` at `/workspace/hf_cache`, and builds
`datasets/vrsbench_llava/{train.json,test.json,images/}` (caption **and** VQA turns, with a seeded 2%
held-out test split). `runpod_train.sh` runs `train.py --config configs/pretrain_vrsbench.yaml`.

> **Download note.** The first `runpod_setup.sh` (smoke or full) downloads `VRSBench_train.json`
> (~65 MB) and `Images_train.zip` (~8.4 GB) into `HF_HOME` on the volume; only referenced images are
> extracted, so `50` builds fast once the zip is present. CLIP and Qwen2.5-3B are public weights — no
> Hugging Face token needed.

## 3. Watch it train

Loss and learning rate print every 10 steps; checkpoints save every 100 steps to
`checkpoints/vrsbench-stage1/`. Each Stage-1 checkpoint is just the connector (`connector.safetensors`
+ optimizer state + `meta.json`). With the default config (3 epochs, effective batch 128) the run is
short.

If you hit **out-of-memory**, lower `per_device_batch_size` in `configs/pretrain_vrsbench.yaml` (try 2)
and raise `gradient_accumulation_steps` to keep the effective batch at 128. Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already exported by `runpod_train.sh`) also helps.

## 4. Stage 2 (optional): LoRA instruction tuning

After Stage 1, point `stage1_checkpoint` in `configs/finetune_vrsbench_stage2.yaml` at your final
Stage-1 checkpoint (e.g. `./checkpoints/vrsbench-stage1/checkpoint-XXXX`), then:

```bash
bash scripts/runpod_train.sh configs/finetune_vrsbench_stage2.yaml
```

Stage 2 adds the full-3B backward pass, so it uses `per_device_batch_size: 2` + gradient
checkpointing. Each Stage-2 checkpoint holds the continued-trained `connector.safetensors` **and** a
`lora/adapter_model.safetensors`.

## 5. Get the checkpoint off the pod

It's already on the `/workspace` volume, so it persists. To also pull it to your laptop:

```bash
# on the pod:
runpodctl send checkpoints/vrsbench-stage1
# it prints a one-time code; on your laptop:
runpodctl receive <code>
```

(Or download via the JupyterLab file browser.)

## 6. Stop or terminate

**Stop** keeps the volume (small storage fee) so you can resume later. **Terminate** frees the GPU;
your data/checkpoints survive only because they're on the network volume. Don't terminate before
step 5 if you didn't use a volume.

## 7. Use the trained connector

```bash
python inference.py --config configs/pretrain_vrsbench.yaml \
  --checkpoint checkpoints/vrsbench-stage1/checkpoint-<step> \
  --image datasets/vrsbench_llava/images/<some_image>.jpg \
  --prompt "Describe this remote-sensing image."
```

For Stage 2, pass `configs/finetune_vrsbench_stage2.yaml` and the Stage-2 checkpoint (the loader
restores both the connector and the LoRA adapter).

## Notes / limits

- **Prototype-grade vision:** standard CLIP at 224×224 on RGB cutouts. A production remote-sensing VLM
  would swap in an RS-specialized vision tower (which changes the feature dim and **requires retraining
  the connector**) and handle higher resolution / multi-band inputs — a separate effort.
- **Licensing (non-commercial):** VRSBench is **CC-BY-NC-4.0** (images from DOTA-v2 / DIOR) and
  **Qwen2.5-3B-Instruct** is under the **Qwen Research License** (non-commercial). Any weights you train
  or data you redistribute inherit those terms — keep attribution and cite VRSBench. See the
  **Model & Data Licensing** section in `README.md`.
- To train on your own data instead, emit the same LLaVA JSON shape (see
  `scripts/build_vrsbench_trainset.py`) and point `data.train_data_path` / `data.image_dir` at it.
