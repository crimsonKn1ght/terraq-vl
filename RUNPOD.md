# Training the astronomy VLM on RunPod

This trains the **MLP connector only** (CLIP and Qwen stay frozen) to align astronomy
images with text, using **AstroLLaVA_convos** (~29.8k real image–caption + Q&A pairs from
NASA APOD / ESO / Hubble). It's a small job (~8–12 GB VRAM), so a single 24 GB
Ampere-or-newer GPU is plenty.

(A Galaxy10 DECaLS path also ships — `scripts/build_galaxy_trainset.py` +
`configs/pretrain_astro.yaml` — but that's a classification set with synthesized captions,
useful only as a morphology probe. AstroLLaVA is the recommended training data.)

## 0. Get the code on GitHub (one time, from your PC)

The local folder isn't a git repo yet. Push it so the pod can clone it:

```bash
git init
git add .
git commit -m "astronomy VLM"
git branch -M main
git remote add origin https://github.com/<you>/astraq-vl.git
git push -u origin main
```

## 1. Create the pod

- **Template:** "RunPod PyTorch 2.x" (CUDA + PyTorch preinstalled).
- **GPU:** any Ampere or newer with ≥16 GB — RTX 3090 / 4090 / A40 / A5000. Community
  Cloud is cheapest (~$0.2–0.4/hr). **Avoid T4 and V100** — training defaults to bf16,
  which needs Ampere+.
- **Network Volume:** create one (e.g. 50 GB) and attach it; it mounts at `/workspace`.
  This is what makes model downloads, the dataset, and your checkpoints survive a pod
  stop/terminate. The container disk is wiped on terminate.

## 2. Clone into the volume and set up

Open the pod's **web terminal** (or SSH/JupyterLab) and clone **into `/workspace`** so
everything lands on the persistent volume:

```bash
cd /workspace
git clone https://github.com/<you>/astraq-vl.git
cd astraq-vl

# Smoke-test the whole pipeline on 50 samples first (fast, catches setup issues):
bash scripts/runpod_setup.sh 50
bash scripts/runpod_train.sh        # should run a few steps and save a checkpoint

# Happy? Build the full set and train for real:
bash scripts/runpod_setup.sh        # full AstroLLaVA set (~30k images, downloads a few GB)
bash scripts/runpod_train.sh
```

`runpod_setup.sh` installs `requirements.txt`, points `HF_HOME` at `/workspace/hf_cache`,
and builds `datasets/astrollava_llava/{train.json,images/}` (caption pairs **and** the
GPT-4 Q&A turns, via `--include-qa`). `runpod_train.sh` runs
`train.py --config configs/pretrain_astraq_vl.yaml`.

CLIP and Qwen are public weights — no Hugging Face token needed. (Only set
`huggingface-cli login` if you later switch to a gated dataset.)

## 3. Watch it train

Loss and learning rate print every 10 steps; checkpoints save every 100 steps to
`checkpoints/astraq-vl-stage1/`. Each checkpoint is just the connector (~16 MB:
`connector.safetensors` + optimizer state + `meta.json`). With the default config
(3 epochs, effective batch 128) the full run is short — roughly an hour on a 4090.

If you hit out-of-memory, lower `per_device_batch_size` to 4 in
`configs/pretrain_astraq_vl.yaml` and raise `gradient_accumulation_steps` to keep the
effective batch at 128. (On a 24 GB card you can instead raise it to 16.)

## 4. Get the checkpoint off the pod

It's already on the `/workspace` volume, so it persists. To also pull it to your laptop:

```bash
# on the pod:
runpodctl send checkpoints/astro-stage1
# it prints a one-time code; on your laptop:
runpodctl receive <code>
```

(Or download via the JupyterLab file browser.)

## 5. Stop or terminate

**Stop** keeps the volume (you pay a small storage fee) so you can resume later.
**Terminate** frees the GPU; your data/checkpoints survive only because they're on the
network volume. Don't terminate before step 4 if you didn't use a volume.

## 6. Use the trained connector

Point inference at the checkpoint dir:

```bash
python inference.py --config configs/pretrain_astraq_vl.yaml \
  --checkpoint checkpoints/astraq-vl-stage1/checkpoint-<step> \
  --image datasets/astrollava_llava/images/astrollava_train_0.png \
  --prompt "Describe this astronomical image."
```

## Notes / limits

- This is a **prototype-grade** port: standard CLIP at 224×224 on RGB cutouts. A
  production astronomy VLM would swap in an astronomy vision tower (e.g. AstroCLIP) and
  handle FITS / dynamic range — that changes the feature dim and **requires retraining
  the connector**, so it's a separate effort.
- AstroLLaVA captions are human-written (APOD/ESO/Hubble) and the Q&A turns are
  GPT-4-generated, under CC-BY-SA-4.0 — keep the attribution if you redistribute.
- To train on your own data instead, emit the same LLaVA JSON shape (see
  `scripts/build_astrollava_trainset.py`) and point `data.train_data_path` /
  `data.image_dir` at it.
