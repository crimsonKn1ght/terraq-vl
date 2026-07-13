"""Upload a release folder (or a single file) to a Hugging Face Hub repo.

Creates the repo if it does not exist, then uploads. Auth via the HF_TOKEN env var (a write token
from https://huggingface.co/settings/tokens) or --token.

    export HF_TOKEN=hf_xxx
    # a whole release folder (default: a model repo)
    python scripts/upload_to_hf.py --repo-id grKnight/terraq-vl-stage2 --path release/terraq-vl-stage2
    # just the zip, into a subdir
    python scripts/upload_to_hf.py --repo-id grKnight/terraq-vl-stage2 \
        --path release/terraq-vl-stage2.zip --path-in-repo bundles/

Notes:
- --repo-type model (default) / dataset / space.
- --private creates the repo private (ignored if it already exists).
- Large checkpoints upload fine over the Hub's LFS; keep HF_TOKEN a *write* token.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a folder/file to a Hugging Face Hub repo.")
    p.add_argument("--repo-id", required=True, help="Target repo, e.g. grKnight/terraq-vl-stage2.")
    p.add_argument("--path", required=True, help="Local folder or file to upload.")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    p.add_argument("--path-in-repo", default=None,
                   help="Destination path inside the repo (default: repo root for a folder, "
                   "the filename for a single file).")
    p.add_argument("--private", action="store_true", help="Create the repo private (if new).")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--token", default=None, help="HF write token (else uses $HF_TOKEN).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("No token: set HF_TOKEN (write token from huggingface.co/settings/tokens) or pass --token.")

    path = Path(args.path)
    if not path.exists():
        sys.exit(f"Path does not exist: {path}")

    # Imported here so the rest of the repo does not hard-depend on huggingface_hub's upload API.
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    msg = args.commit_message or f"Upload {path.name}"
    if path.is_dir():
        api.upload_folder(
            folder_path=str(path),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo or ".",
            commit_message=msg,
        )
    else:
        dest = args.path_in_repo or path.name
        if dest.endswith("/"):
            dest = dest + path.name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=msg,
        )

    url = f"https://huggingface.co/{args.repo_id}"
    if args.repo_type != "model":
        url = f"https://huggingface.co/{args.repo_type}s/{args.repo_id}"
    print(f"\nUploaded {path} -> {url}")


if __name__ == "__main__":
    main()
