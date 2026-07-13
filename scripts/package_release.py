"""Assemble a release bundle (folder + zip) for a trained TerraQ-VL stage.

Collects everything worth keeping for a stage into one tidy, self-describing folder — checkpoints
(zipped), the training config, training/validation curves, held-out predictions, the raw log, the
data splits — plus a generated MODEL_CARD.md and a manifest.json (every file with size + sha256).
The result is what scripts/upload_to_hf.py pushes to the Hub.

This is pure assembly: run inference (scripts/batch_inference.py) and the curve
(scripts/plot_training_curve.py) FIRST, then point this script at their outputs. Nothing here needs
a GPU. stdlib + pyyaml only.

Usage (from repo root) — Stage 2, bundling the final and best-val checkpoints:
    python scripts/package_release.py --stage 2 \
        --title terraq-vl-stage2 \
        --config configs/finetune_vrsbench_stage2.yaml \
        --checkpoint checkpoints/vrsbench-stage2/checkpoint-2180 \
                     checkpoints/vrsbench-stage2/checkpoint-1800 \
        --curve-stem curve_stage2 \
        --log train_stage2.log \
        --predictions predictions_test_stage2.jsonl \
        --data datasets/vrsbench_llava/test.json datasets/vrsbench_llava/val.json

Usage — Stage 1, one per-epoch checkpoint + its held-out predictions each:
    python scripts/package_release.py --stage 1 \
        --title terraq-vl-stage1 \
        --config configs/pretrain_vrsbench.yaml \
        --checkpoint checkpoints/vrsbench-stage1/checkpoint-1100 \
                     checkpoints/vrsbench-stage1/checkpoint-2200 \
                     checkpoints/vrsbench-stage1/checkpoint-3270 \
        --curve-stem curve_stage1 \
        --predictions predictions_test_stage1_ep1.jsonl \
                      predictions_test_stage1_ep2.jsonl \
                      predictions_test_stage1_ep3.jsonl \
        --data datasets/vrsbench_llava/test.json
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_dir(src: Path, dest_zip: Path) -> None:
    """Zip a directory, storing paths relative to the dir's parent (so it unzips to <name>/...)."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_meta(ckpt: Path) -> dict:
    meta_path = ckpt / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package a TerraQ-VL stage into a release folder + zip.")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--title", required=True, help="Release/folder name, e.g. terraq-vl-stage2.")
    p.add_argument("--config", required=True, help="Training config YAML for this stage.")
    p.add_argument("--checkpoint", nargs="+", required=True, help="Checkpoint dir(s) to bundle (zipped).")
    p.add_argument("--predictions", nargs="*", default=[], help="Held-out prediction JSONL(s).")
    p.add_argument("--curve-stem", default=None,
                   help="Stem of the curve outputs (.png/.csv/.json) from plot_training_curve.py.")
    p.add_argument("--log", default=None, help="Raw training stdout log to include.")
    p.add_argument("--data", nargs="*", default=[], help="Data split files to include (e.g. test.json).")
    p.add_argument("--out", default="release", help="Parent output dir (default: release/).")
    p.add_argument("--repo-url", default="https://github.com/crimsonKn1ght/TerraQ-VL",
                   help="Source repo URL for provenance.")
    p.add_argument(
        "--no-checkpoint-zip",
        action="store_true",
        help="Do not copy/zip the checkpoints into the bundle (and skip the whole-bundle zip). Use "
        "when uploading ALL checkpoints as raw dirs separately (e.g. via upload_to_hf.py) to avoid "
        "duplicating gigabytes on disk and on the Hub. The model card still reads each meta.json.",
    )
    return p.parse_args()


def build_model_card(args, ckpts, cfg, bundle: Path) -> str:
    ve = cfg.get("vision_encoder", {})
    lm = cfg.get("language_model", {})
    tr = cfg.get("training", {})
    lora = lm.get("lora")
    eff_batch = tr.get("per_device_batch_size", "?")
    accum = tr.get("gradient_accumulation_steps", 1)
    try:
        eff_batch = f"{tr['per_device_batch_size'] * accum} (per-device {tr['per_device_batch_size']} x accum {accum})"
    except Exception:
        pass

    lines = [
        f"# {args.title}",
        "",
        f"TerraQ-VL Stage-{args.stage} release. Source: {args.repo_url} @ `{git_commit()}`.",
        "",
        "## Model",
        f"- Vision encoder (frozen): `{ve.get('model_name', '?')}` (select_layer {ve.get('select_layer', '?')})",
        f"- LLM: `{lm.get('model_name', '?')}` "
        + ("(frozen; connector-only)" if args.stage == 1 else "(frozen base + LoRA adapters)"),
    ]
    if lora:
        lines.append(
            f"- LoRA: r={lora.get('r')}, alpha={lora.get('lora_alpha')}, dropout={lora.get('lora_dropout')}, "
            f"targets={lora.get('target_modules')}"
        )
    if args.stage == 2 and cfg.get("stage1_checkpoint"):
        lines.append(f"- Connector warm-started from Stage-1 checkpoint: `{cfg['stage1_checkpoint']}`")

    lines += [
        "",
        "## Training",
        f"- Effective batch: {eff_batch}, epochs {tr.get('num_epochs', '?')}, "
        f"LR {tr.get('learning_rate', '?')}, warmup_ratio {tr.get('warmup_ratio', '?')}, "
        f"bf16 {tr.get('bf16', '?')}",
        f"- Validation: every {tr.get('eval_steps', tr.get('save_steps', '?'))} steps on the "
        f"disjoint `val.json` split (token-weighted loss).",
        "",
        "## Checkpoints ("
        + ("raw dirs under `checkpoints/`" if args.no_checkpoint_zip else "zipped under `checkpoints/`")
        + ")",
        "",
        "| checkpoint | train loss | val loss |",
        "|---|---|---|",
    ]
    for ckpt in ckpts:
        meta = read_meta(Path(ckpt))
        loss = meta.get("loss")
        vloss = meta.get("val_loss")
        lines.append(
            f"| {Path(ckpt).name} | {loss if loss is not None else 'n/a'} "
            f"| {vloss if vloss is not None else 'n/a'} |"
        )

    lines += [
        "",
        "## Contents",
        "- `checkpoints/` — "
        + ("raw checkpoint dir(s)" if args.no_checkpoint_zip else "zipped checkpoint dir(s)")
        + ": `connector.safetensors`"
        + (" + `lora/` adapter" if args.stage == 2 else "")
        + " + `training_state.pt` + `meta.json`",
        "- `config/` — the exact training/inference config YAML",
        "- `curves/` — training + held-out validation loss curve (png/csv/json)",
        "- `predictions/` — greedy captions on the held-out `test.json` (response + reference)",
        "- `logs/` — raw training stdout",
        "- `data/` — the held-out split(s) used (regenerate images with the builder)",
        "- `manifest.json` — every file with size + sha256",
        "",
        "## Inference",
        "```bash",
        f"python inference.py --config {Path(args.config).name} \\",
        f"    --checkpoint <unzipped checkpoint dir> \\",
        "    --image your_image.jpg \\",
        '    --prompt "Describe this remote sensing image." --temperature 0',
        "```",
        "Both the connector and (Stage 2) the LoRA adapter load automatically from the checkpoint dir; "
        "pass this stage's config so the adapter structure is built first.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_parent = Path(args.out)
    bundle = out_parent / args.title
    if bundle.exists():
        shutil.rmtree(bundle)
    subdirs = ["config", "curves", "predictions", "logs", "data"]
    if not args.no_checkpoint_zip:
        subdirs.insert(0, "checkpoints")
    for sub in subdirs:
        (bundle / sub).mkdir(parents=True, exist_ok=True)

    # Config
    shutil.copy2(args.config, bundle / "config" / Path(args.config).name)

    # Checkpoints -> one zip each (so best-val and final stay separately downloadable). With
    # --no-checkpoint-zip we only validate them (the card still reports each meta.json) and leave the
    # raw dirs to be uploaded separately — no gigabytes duplicated on disk or on the Hub.
    for ckpt in args.checkpoint:
        ckpt_path = Path(ckpt)
        if not (ckpt_path / "connector.safetensors").exists():
            raise SystemExit(f"MISSING connector.safetensors in {ckpt_path} — is training done / path right?")
        if not args.no_checkpoint_zip:
            zip_dir(ckpt_path, bundle / "checkpoints" / f"{ckpt_path.name}.zip")

    # Curves (produced beforehand by plot_training_curve.py --out <stem> --plot)
    if args.curve_stem:
        for ext in ("png", "csv", "json"):
            src = Path(f"{args.curve_stem}.{ext}")
            if src.exists():
                shutil.copy2(src, bundle / "curves" / src.name)

    # Predictions, log, data splits
    for pred in args.predictions:
        src = Path(pred)
        if src.exists():
            shutil.copy2(src, bundle / "predictions" / src.name)
        else:
            print(f"  warning: predictions not found, skipping: {pred}")
    if args.log and Path(args.log).exists():
        shutil.copy2(args.log, bundle / "logs" / Path(args.log).name)
    for d in args.data:
        src = Path(d)
        if src.exists():
            shutil.copy2(src, bundle / "data" / src.name)

    # Model card
    (bundle / "MODEL_CARD.md").write_text(build_model_card(args, args.checkpoint, cfg, bundle))

    # Manifest: every file with size + sha256
    manifest = []
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest.append(
                {"path": str(p.relative_to(bundle)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            )
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Whole-bundle zip alongside the folder (skipped when checkpoints live outside the bundle —
    # re-zipping only the light artifacts adds little, and with big checkpoints it would double GBs).
    total_mb = sum(m["bytes"] for m in manifest) / 1e6
    print(f"\nPackaged {len(manifest)} files ({total_mb:.1f} MB) into folder: {bundle}")
    if not args.no_checkpoint_zip:
        top_zip = out_parent / f"{args.title}.zip"
        if top_zip.exists():
            top_zip.unlink()
        zip_dir(bundle, top_zip)
        print(f"  whole-bundle zip: {top_zip}")
    print(f"\nUpload with:\n  python scripts/upload_to_hf.py --repo-id <user>/{args.title} --path {bundle}")


if __name__ == "__main__":
    main()
