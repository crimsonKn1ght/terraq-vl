"""Extract training metrics from the trainer's stdout log and emit CSV / JSON (+ optional PNG).

The training loop (training/trainer.py) logs one line every `logging_steps` to stdout, e.g.:

    2026-06-27 14:02:19,364 - training.trainer - INFO - Step 10/2525 | Loss: 1.4726 | LR: 2.67e-05 | Samples/s: 15.3

and, when a validation set is configured, an extra line every `eval_steps`:

    2026-06-27 14:05:41,002 - training.trainer - INFO - Step 100/2525 | Val loss: 1.5210

That stdout capture is the canonical metric source (no TensorBoard/W&B/CSV is written during
training). This script parses those lines into tidy rows of (step, loss, lr, samples_per_sec) with an
optional merged `val_loss`, writes CSV + JSON, prints summary stats, and optionally renders a curve
PNG (train + held-out validation loss overlaid, plus LR).

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

# Matches the trainer's train-loss log line; tolerant of the logging prefix and surrounding whitespace.
STEP_RE = re.compile(
    r"Step\s+(\d+)\s*/\s*(\d+)\s*\|\s*Loss:\s*([0-9.]+)\s*\|\s*"
    r"LR:\s*([0-9.eE+\-]+)\s*\|\s*Samples/s:\s*([0-9.]+)"
)

# Matches the trainer's held-out validation line, e.g. "Step 100/2525 | Val loss: 1.5210" (also
# "Final val loss: ..."). Logged every eval_steps; steps line up with the train-loss steps.
VAL_RE = re.compile(r"Step\s+(\d+)\s*/\s*\d+\s*\|\s*(?:Final\s+val\s+loss|Val\s+loss):\s*([0-9.]+)")


def parse_log(lines) -> list:
    """Return [{step,total_steps,loss,lr,samples_per_sec[,val_loss]}, ...] from trainer stdout lines.

    Train-loss and validation-loss lines are logged separately at the same step; the val loss is
    merged onto the matching train row (and a ``val_loss`` column is added to every row only when at
    least one validation line was seen, so logs without validation keep their original shape).
    """
    rows = []
    val_by_step = {}
    for line in lines:
        m = STEP_RE.search(line)
        if m:
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
            continue
        mv = VAL_RE.search(line)
        if mv:
            val_by_step[int(mv.group(1))] = float(mv.group(2))

    if val_by_step:
        for r in rows:
            r["val_loss"] = val_by_step.get(r["step"])
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
    val_rows = [r for r in rows if r.get("val_loss") is not None]
    if val_rows:
        best_val = min(val_rows, key=lambda r: r["val_loss"])
        print(f"  val loss     : final {val_rows[-1]['val_loss']:.4f} | "
              f"best {best_val['val_loss']:.4f} @ step {best_val['step']}  "
              f"(best-val checkpoint to keep)")
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
    ax1.plot(steps, losses, color="tab:blue", label="train loss")
    ax1.set_xlabel("update step")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    # Overlay the held-out validation loss (if the log carried it): train falling while val flattens
    # or rises is the overfitting signal; both still falling means more training helps.
    val_pts = [(r["step"], r["val_loss"]) for r in rows if r.get("val_loss") is not None]
    if val_pts:
        vsteps, vlosses = zip(*val_pts)
        ax1.plot(vsteps, vlosses, color="tab:green", marker="o", markersize=3, label="val loss")
        ax1.legend(loc="upper right")

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
