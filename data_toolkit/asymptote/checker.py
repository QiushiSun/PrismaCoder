#!/usr/bin/env python3
"""
JSONL Asymptote Checker — extract Asymptote code from JSONL output fields,
render to PNG via asy, and split into passed / failed JSONL files.

Usage:
    python checker.py <jsonl_path_or_dir> [options]

Options:
    --output  -o   PNG output directory            (default: ./png_output)
    --result  -r   Directory for result JSONLs     (default: ./results)
    --workers -j   Parallel render workers         (default: 8)
    --timeout -t   Per-script render timeout (s)   (default: 60)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from utils import extract_asy_code, find_asy, render_asy_to_png


def _process_one(args: tuple) -> dict:
    """Process a single entry in a worker process."""
    obj, png_path_str, timeout = args

    output_text = obj.get("output", "")
    if not output_text:
        obj["failure_stage"] = "extract"
        obj["failure_message"] = "No 'output' field or empty"
        return {"status": "failed", "obj": obj}

    asy_code, method = extract_asy_code(output_text)
    if asy_code is None:
        obj["failure_stage"] = "extract"
        obj["failure_message"] = "No Asymptote code found in output"
        return {"status": "failed", "obj": obj}

    ok, msg = render_asy_to_png(asy_code, png_path_str, timeout=timeout)
    if not ok:
        obj["failure_stage"] = "render"
        obj["failure_message"] = msg
        return {"status": "failed", "obj": obj}

    obj["png_path"] = Path(png_path_str).name
    obj["asy_extract_method"] = method
    return {"status": "passed", "obj": obj}


def load_entries(jsonl_path: Path) -> list[tuple[dict, int]]:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as e:
                obj = {
                    "_source": jsonl_path.name, "_line": line_no,
                    "failure_stage": "json_parse", "failure_message": str(e),
                }
            entries.append((obj, line_no))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Extract Asymptote from JSONL, render to PNG, split passed/failed"
    )
    parser.add_argument(
        "input", help="Path to a .jsonl file or directory containing .jsonl files",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory for PNGs (default: ./png_output)",
    )
    parser.add_argument(
        "--result", "-r", default=None,
        help="Directory for passed.jsonl / failed.jsonl (default: ./results)",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=8, help="Parallel render workers",
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=60, help="Per-script render timeout (s)",
    )
    args = parser.parse_args()

    find_asy()

    input_path = Path(args.input).resolve()
    if input_path.is_file():
        jsonl_files = [input_path]
    elif input_path.is_dir():
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: {input_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output).resolve() if args.output
        else Path("./png_output").resolve()
    )
    result_dir = (
        Path(args.result).resolve() if args.result
        else Path("./results").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    passed_path = result_dir / "passed.jsonl"
    failed_path = result_dir / "failed.jsonl"

    print(f"Input         : {input_path}")
    print(f"JSONL files   : {len(jsonl_files)}")
    print(f"PNG output    : {output_dir}")
    print(f"Result dir    : {result_dir}")
    print(f"Workers={args.workers}  Timeout={args.timeout}s")
    print()

    all_tasks = []
    json_parse_failures = []
    for jf in jsonl_files:
        entries = load_entries(jf)
        for obj, line_no in entries:
            if "failure_stage" in obj and obj["failure_stage"] == "json_parse":
                json_parse_failures.append(obj)
                continue
            uuid = obj.get("uuid", f"line{line_no}")
            png_name = f"{jf.stem}_{uuid}.png"
            png_path_str = str(output_dir / png_name)
            all_tasks.append((obj, png_path_str, args.timeout))

    total = len(all_tasks) + len(json_parse_failures)
    print(f"Total entries : {total}  (render tasks: {len(all_tasks)}, "
          f"json errors: {len(json_parse_failures)})")

    t0 = time.time()
    passed_count = 0
    failed_count = len(json_parse_failures)

    with (
        open(passed_path, "w", encoding="utf-8") as passed_fh,
        open(failed_path, "w", encoding="utf-8") as failed_fh,
    ):
        for fail_obj in json_parse_failures:
            failed_fh.write(json.dumps(fail_obj, ensure_ascii=False) + "\n")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, task): task for task in all_tasks}

            pbar = tqdm(
                as_completed(futures),
                total=len(all_tasks),
                desc="Rendering",
                unit="entry",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] P:{postfix}",
            )
            pbar.set_postfix_str(f"{passed_count}passed {failed_count}failed")

            for future in pbar:
                result = future.result()
                if result["status"] == "passed":
                    passed_fh.write(
                        json.dumps(result["obj"], ensure_ascii=False) + "\n"
                    )
                    passed_count += 1
                else:
                    failed_fh.write(
                        json.dumps(result["obj"], ensure_ascii=False) + "\n"
                    )
                    failed_count += 1

                pass_rate = 100 * passed_count / (passed_count + failed_count)
                pbar.set_postfix_str(
                    f"{passed_count}passed {failed_count}failed ({pass_rate:.1f}%)"
                )

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"Total   : {total}")
    print(f"Passed  : {passed_count}")
    print(f"Failed  : {failed_count}")
    print(f"Rate    : {100 * passed_count / total:.1f}%")
    print(f"Time    : {elapsed:.1f}s  ({total / elapsed:.1f} items/s)")
    print(f"{'=' * 50}")

    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
