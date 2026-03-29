#!/usr/bin/env python3
"""
Re-run failed R entries through the upgraded checker (with interactive code
transforms), then merge results back into passed.jsonl / failed.jsonl.

Usage:
    python rerun_failed.py [--result-dir ./results] [--output-dir ./png_output] [--workers 8] [--timeout 30]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from checker import _process_one
from utils import find_rscript


def should_retry(obj: dict) -> bool:
    """Decide if a failed entry is worth retrying."""
    stage = obj.get("failure_stage", "")
    msg = obj.get("failure_message", "")
    if "no package called" in msg:
        return True
    if stage == "render" and ("CRAN without setting a mirror" in msg or "empty" in msg.lower()):
        return True
    if stage == "extract":
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Re-run failed R entries with upgraded checker")
    parser.add_argument("--result-dir", default="./results", help="Directory with passed/failed JSONL")
    parser.add_argument("--output-dir", default="./png_output", help="PNG output directory")
    parser.add_argument("--workers", "-j", type=int, default=8)
    parser.add_argument("--timeout", "-t", type=int, default=30)
    args = parser.parse_args()

    find_rscript()

    result_dir = Path(args.result_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    passed_path = result_dir / "passed.jsonl"
    failed_path = result_dir / "failed.jsonl"

    if not failed_path.exists():
        print("No failed.jsonl found.")
        sys.exit(0)

    existing_passed = []
    if passed_path.exists():
        with open(passed_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_passed.append(line)

    retry_tasks = []
    skip_entries = []
    with open(failed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if should_retry(obj):
                clean_obj = {k: v for k, v in obj.items()
                             if k not in ("failure_stage", "failure_message", "failure_category")}
                uuid = clean_obj.get("uuid", "unknown")
                src = clean_obj.get("_source", "data")
                png_name = f"{Path(src).stem}_{uuid}.png"
                retry_tasks.append((clean_obj, str(output_dir / png_name), args.timeout))
            else:
                skip_entries.append(line)

    total_retry = len(retry_tasks)
    print(f"Existing passed : {len(existing_passed)}")
    print(f"Failed entries  : {total_retry + len(skip_entries)}")
    print(f"  Retrying      : {total_retry}")
    print(f"  Skipping      : {len(skip_entries)}")
    print(f"Workers={args.workers}  Timeout={args.timeout}s")
    print()

    if total_retry == 0:
        print("Nothing to retry.")
        sys.exit(0)

    t0 = time.time()
    new_passed = 0
    new_failed = 0
    new_failed_lines = list(skip_entries)
    new_passed_lines = list(existing_passed)
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, task): task for task in retry_tasks}
        for future in as_completed(futures):
            result = future.result()
            line = json.dumps(result["obj"], ensure_ascii=False)
            if result["status"] == "passed":
                new_passed_lines.append(line)
                new_passed += 1
            else:
                new_failed_lines.append(line)
                new_failed += 1
            done += 1
            if done % 50 == 0 or done == total_retry:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Progress: {done}/{total_retry}  "
                    f"(+{new_passed} passed, +{new_failed} still failed)  "
                    f"[{rate:.1f} items/s]",
                    end="", flush=True,
                )

    with open(passed_path, "w", encoding="utf-8") as f:
        for line in new_passed_lines:
            f.write(line + "\n")
    with open(failed_path, "w", encoding="utf-8") as f:
        for line in new_failed_lines:
            f.write(line + "\n")

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 55}")
    print(f"Retried         : {total_retry}")
    print(f"Newly passed    : {new_passed}")
    print(f"Still failed    : {new_failed}")
    print(f"Time            : {elapsed:.1f}s")
    print(f"{'=' * 55}")
    print(f"FINAL passed.jsonl : {len(new_passed_lines)}")
    print(f"FINAL failed.jsonl : {len(new_failed_lines)}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
