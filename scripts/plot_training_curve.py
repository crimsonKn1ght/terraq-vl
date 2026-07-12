"""Extract training metrics from the trainer's stdout log and emit CSV / JSON (+ optional PNG).

The training loop (training/trainer.py) logs one line every `logging_steps` to stdout, e.g.:

    2026-06-27 14:02:19,364 - training.trainer - INFO - Step 10/2525 | Loss: 1.4726 | LR: 2.67e-05 | Samples/s: 15.3

That stdout capture is the canonical metric source (no TensorBoard/W&B/CSV is written during
training). This script parses those lines into tidy rows of (step, loss, lr, samples_per_sec),
writes CSV + JSON, prints summary stats, and optionally renders a loss/LR curve PNG.

Usage (from repo root):
    # parse a captured log -> training_curve.csv / .json (+ .png if matplotlib is installed)
    python scripts/plot_training_curve.py train_stage2.log --out training_curve --plot

    # pipe it in instead of a file
    python scripts/plot_training_curve.py - < train_stage2.log

    # sparse fallback: read {step,loss} from checkpoint meta.json files (no log needed)
    python scripts/plot_training_curve.py --meta-dir checkpoints/astraq-vl-stage2 --out curve_from_meta
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Matches the trainer's log line; tolerant of the logging prefix and surrounding whitespace.
STEP_RE = re.compile(
    r"Step\s+(\d+)\s*/\s*(\d+)\s*\|\s*Loss:\s*([0-9.]+)\s*\|\s*"
    r"LR:\s*([0-9.eE+\-]+)\s*\|\s*Samples/s:\s*([0-9.]+)"
)


def parse_log(lines) -> list:
    """Return [{step,total_steps,loss,lr,samples_per_sec}, ...] from trainer stdout lines."""
    rows = []
    for line in lines:
        m = STEP_RE.search(line)
        if not m:
            continue
        step, total, loss, lr, sps = m.groups()
        rows.append(
            {
                "step": int(step),
                "total_steps": int(total),
                "loss": float(loss),
                "lr": float(lr),
                "samples_per_sec": float(sps),
            }
        )
    return rows


def parse_meta_dir(meta_dir: str) -> list:
    """Sparse fallback: read {step, loss} from each checkpoint-*/meta.json."""
    rows = []
    for meta in sorted(Path(meta_dir).glob("checkpoint-*/meta.json")):
        try:
            d = json.loads(meta.read_text())
            rows.append({"step": int(d["step"]), "loss": float(d["loss"])})
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"  skipping {meta}: {exc}", file=sys.stderr)
    rows.sort(key=lambda r: r["step"])
    return rows


def write_outputs(rows: list, out_stem: str) -> None:
    if not rows:
        raise SystemExit("No metric rows parsed — is this the trainer's stdout log?")

    fields = list(rows[0].keys())
    csv_path = Path(f"{out_stem}.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")

    json_path = Path(f"{out_stem}.json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows -> {csv_path} and {json_path}")


def summarize(rows: list) -> None:
    losses = [r["loss"] for r in rows]
    first, last = rows[0], rows[-1]
    min_row = min(rows, key=lambda r: r["loss"])
    print("\nSummary")
    print(f"  steps logged : {len(rows)}  (step {first['step']} -> {last['step']})")
    print(f"  loss         : start {first['loss']:.4f} | final {last['loss']:.4f} | "
          f"min {min_row['loss']:.4f} @ step {min_row['step']}")
    if "samples_per_sec" in last:
        avg_sps = sum(r["samples_per_sec"] for r in rows) / len(rows)
        print(f"  throughput   : ~{avg_sps:.1f} samples/s (avg of logged points)")


def plot(rows: list, out_stem: str, title: str = "Training loss") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless (works on a pod with no display)
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping PNG (CSV/JSON still written). "
              "`pip install matplotlib` to enable.", file=sys.stderr)
        return

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, losses, color="tab:blue", label="loss")
    ax1.set_xlabel("update step")
    ax1.set_ylabel("training loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    if "lr" in rows[0]:
        ax2 = ax1.twinx()
        ax2.plot(steps, [r["lr"] for r in rows], color="tab:orange", alpha=0.6, label="lr")
        ax2.set_ylabel("learning rate", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(title)
    fig.tight_layout()
    png_path = Path(f"{out_stem}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Wrote plot -> {png_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse trainer stdout log into metric CSV/JSON (+PNG).")
    p.add_argument("logfile", nargs="?", default=None,
                   help="Trainer stdout log file, or '-' for stdin. Omit if using --meta-dir.")
    p.add_argument("--meta-dir", default=None,
                   help="Instead of a log, read sparse {step,loss} from checkpoint-*/meta.json here.")
    p.add_argument("--out", default="training_curve", help="Output stem (.csv/.json/.png).")
    p.add_argument("--plot", action="store_true", help="Also render a PNG (needs matplotlib).")
    p.add_argument("--title", default="Training loss", help="Plot title.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.meta_dir:
        rows = parse_meta_dir(args.meta_dir)
    elif args.logfile in (None, "-"):
        if args.logfile is None and sys.stdin.isatty():
            raise SystemExit("Provide a log file, '-' for stdin, or --meta-dir. See --help.")
        rows = parse_log(sys.stdin)
    else:
        rows = parse_log(Path(args.logfile).read_text(encoding="utf-8", errors="ignore").splitlines())

    write_outputs(rows, args.out)
    summarize(rows)
    if args.plot:
        plot(rows, args.out, args.title)


if __name__ == "__main__":
    main()
