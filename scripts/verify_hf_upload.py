"""Verify a local archive was fully uploaded to a Hugging Face repo.

For each LOCALDIR=REPOPREFIX pair, checks that every local file exists on the Hub at
REPOPREFIX/<relpath> with a matching byte size, and prints a single OK / NOT‑OK verdict. Use it right
before terminating a pod, so you know nothing silently failed to upload.

    export HF_TOKEN=hf_xxx           # only needed for a private repo
    # default layout (matches the archival flow) — just pass the repo:
    python scripts/verify_hf_upload.py --repo-id grKnight/terraq-vl
    # or spell out pairs explicitly:
    python scripts/verify_hf_upload.py --repo-id grKnight/terraq-vl \
        --pair checkpoints/vrsbench-stage1=stage-1/checkpoints \
        --pair release/terraq-vl-stage2=stage-2

Exit code 0 = everything present with matching sizes; 1 = something is missing / truncated.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# (local dir, repo path prefix) — the standard preserve-everything layout.
DEFAULT_PAIRS = [
    ("checkpoints/vrsbench-stage1", "stage-1/checkpoints"),
    ("checkpoints/vrsbench-stage2", "stage-2/checkpoints"),
    ("release/terraq-vl-stage1", "stage-1"),
    ("release/terraq-vl-stage2", "stage-2"),
]


def compare(
    pairs: List[Tuple[str, str]], remote_sizes: Dict[str, int], check_size: bool
) -> Tuple[int, List[str], List[Tuple[str, int, int]], List[str]]:
    """Pure comparison: returns (files_checked, missing, size_mismatch, skipped_pairs)."""
    checked = 0
    missing: List[str] = []
    mismatch: List[Tuple[str, int, int]] = []
    skipped: List[str] = []
    for local_dir, prefix in pairs:
        base = Path(local_dir)
        if not base.is_dir():
            skipped.append(local_dir)
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(base).as_posix()
            repo_path = f"{prefix}/{rel}" if prefix else rel
            checked += 1
            if repo_path not in remote_sizes:
                missing.append(repo_path)
            elif check_size and remote_sizes[repo_path] != f.stat().st_size:
                mismatch.append((repo_path, f.stat().st_size, remote_sizes[repo_path]))
    return checked, missing, mismatch, skipped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a local archive is fully present on a HF repo.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--pair", action="append", default=[], metavar="LOCALDIR=REPOPREFIX",
                   help="Local dir mapped to its repo path prefix. Repeatable. "
                   "Omit to use the default stage-1/stage-2 layout.")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    p.add_argument("--no-size", action="store_true", help="Check presence only, skip size match.")
    p.add_argument("--token", default=None, help="HF token (else $HF_TOKEN; only needed if private).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pair:
        pairs = []
        for spec in args.pair:
            if "=" not in spec:
                sys.exit(f"--pair must be LOCALDIR=REPOPREFIX, got: {spec}")
            local, prefix = spec.split("=", 1)
            pairs.append((local, prefix))
    else:
        pairs = DEFAULT_PAIRS

    from huggingface_hub import HfApi

    api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))
    # path -> size for every file on the Hub (RepoFile has .size; RepoFolder does not).
    remote_sizes: Dict[str, int] = {}
    for item in api.list_repo_tree(args.repo_id, recursive=True, repo_type=args.repo_type):
        size = getattr(item, "size", None)
        if size is not None:
            remote_sizes[item.path] = size

    checked, missing, mismatch, skipped = compare(pairs, remote_sizes, not args.no_size)

    for s in skipped:
        print(f"  (skipped — local dir not found: {s})")
    print(f"Checked {checked} local files against {args.repo_id} ({len(remote_sizes)} files on the Hub).")

    if not missing and not mismatch:
        print(f"\n✅ OK — all {checked} files are on the Hub" + ("" if args.no_size else " with matching sizes") + ".")
        sys.exit(0)

    if missing:
        print(f"\n❌ MISSING {len(missing)} file(s) — re-upload these:")
        for m in missing[:50]:
            print(f"    {m}")
        if len(missing) > 50:
            print(f"    … and {len(missing) - 50} more")
    if mismatch:
        print(f"\n⚠️  SIZE MISMATCH {len(mismatch)} file(s) (truncated/failed upload — re-upload):")
        for path, local_sz, hub_sz in mismatch[:50]:
            print(f"    {path}  local {local_sz} vs hub {hub_sz}")
    print("\nNOT OK — re-run the matching upload_to_hf.py command(s), then re-check.")
    sys.exit(1)


if __name__ == "__main__":
    main()
