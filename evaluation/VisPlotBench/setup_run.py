"""Organize flat runs-{id}-{lang}/ folders into runs/runs-{id}/ and fix repo_root.

Usage:
    python setup_run.py --run-id 1            # move runs-1-python/ etc. into runs/runs-1/
    python setup_run.py --run-id 1 --dry-run  # preview without moving
"""

import argparse
import json
import shutil
from pathlib import Path


def fix_repo_root(manifest_path: Path, repo_root: str) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["repo_root"] = repo_root
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def find_flat_run_dirs(base: Path, run_id: str) -> dict[str, Path]:
    """Return {lang: path} for all runs-{id}-{lang}/ at top level."""
    prefix = f"runs-{run_id}-"
    result: dict[str, Path] = {}
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and entry.name.startswith(prefix):
            lang = entry.name[len(prefix):]
            if lang:
                result[lang] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize flat run folders into runs/runs-{id}/ structure")
    parser.add_argument("--run-id", required=True, help="Run identifier (e.g. 1, 2)")
    parser.add_argument("--base-dir", default=".", help="Project root (default: current dir)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without moving")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    run_id = args.run_id
    runs_parent = base / "runs"
    run_dir = runs_parent / f"runs-{run_id}"
    repo_root = str(base) + "/"

    flat_dirs = find_flat_run_dirs(base, run_id)

    # Also look for exec-logs-{id}
    exec_logs_src = base / f"exec-logs-{run_id}"

    if not flat_dirs and run_dir.is_dir():
        # Already migrated -- just fix repo_root
        prefix = f"runs-{run_id}-"
        lang_dirs = sorted(
            d for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix) and (d / "manifest.json").exists()
        )
        if lang_dirs:
            print(f"runs/runs-{run_id}/ already exists with {len(lang_dirs)} language folders.")
            print("Fixing repo_root in manifests...")
            for d in lang_dirs:
                mf = d / "manifest.json"
                if not args.dry_run:
                    fix_repo_root(mf, repo_root)
                lang = d.name[len(prefix):]
                print(f"  {d.name}/manifest.json -> repo_root = {repo_root}")
            print("Done.")
            return
        print(f"runs/runs-{run_id}/ exists but has no language subfolders.")
        return

    if not flat_dirs:
        print(f"No runs-{run_id}-* folders found in {base}")
        return

    print(f"Found {len(flat_dirs)} language folder(s) for run {run_id}:")
    for lang, src in flat_dirs.items():
        print(f"  {src.name}/  ->  runs/runs-{run_id}/{src.name}/")

    if exec_logs_src.is_dir():
        print(f"  exec-logs-{run_id}/  ->  runs/runs-{run_id}/exec-logs/")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    runs_parent.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    moved = 0
    for lang, src in flat_dirs.items():
        dst = run_dir / src.name
        if dst.exists():
            print(f"  [SKIP] {dst.relative_to(base)} already exists")
            continue
        shutil.move(str(src), str(dst))
        mf = dst / "manifest.json"
        if mf.exists():
            fix_repo_root(mf, repo_root)
        moved += 1

    if exec_logs_src.is_dir():
        dst_logs = run_dir / "exec-logs"
        if not dst_logs.exists():
            shutil.move(str(exec_logs_src), str(dst_logs))
            print(f"  Moved exec-logs-{run_id}/ -> runs/runs-{run_id}/exec-logs/")

    print(f"\nMigrated {moved} folder(s) into runs/runs-{run_id}/")
    print(f"repo_root set to: {repo_root}")


if __name__ == "__main__":
    main()
