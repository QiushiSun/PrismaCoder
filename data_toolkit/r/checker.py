#!/usr/bin/env python3
"""
JSONL R Checker — extract R code from JSONL output fields, render to PNG
via Rscript, and split into passed / failed JSONL files.

Handles interactive R code (Shiny, plotly, rgl) by transforming it into
static-renderable code before execution.

Usage:
    python checker.py <jsonl_path_or_dir> [options]

Options:
    --output  -o   PNG output directory            (default: ./png_output)
    --result  -r   Directory for result JSONLs     (default: ./results)
    --workers -j   Parallel render workers         (default: 8)
    --timeout -t   Per-script render timeout (s)   (default: 30)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils import (
    classify_interactive,
    extract_r_code,
    find_rscript,
    render_r_to_png,
    transform_plotly_ggplotly,
    transform_plotly_native,
    transform_rgl,
    transform_shiny_code,
)


def _process_one(args: tuple) -> dict:
    """Process a single entry in a worker process."""
    obj, png_path_str, timeout = args

    output_text = obj.get("output", "")
    if not output_text:
        obj["failure_stage"] = "extract"
        obj["failure_message"] = "No 'output' field or empty"
        return {"status": "failed", "obj": obj}

    r_code, method = extract_r_code(output_text)
    if r_code is None:
        obj["failure_stage"] = "extract"
        obj["failure_message"] = "No R code found in output"
        return {"status": "failed", "obj": obj}

    interactive_type = classify_interactive(r_code)
    render_mode = "static"

    if interactive_type == "shiny_no_plot":
        obj["failure_stage"] = "interactive_no_plot"
        obj["failure_message"] = "Shiny app with no renderable plot output"
        obj["failure_category"] = "interactive_no_plot"
        return {"status": "failed", "obj": obj}
    elif interactive_type == "shiny_plot":
        r_code = transform_shiny_code(r_code)
        method = f"{method}+shiny_extract"
    elif interactive_type == "plotly_ggplotly":
        r_code = transform_plotly_ggplotly(r_code)
        method = f"{method}+plotly_ggplotly"
    elif interactive_type == "plotly_native":
        r_code = transform_plotly_native(r_code, png_path_str)
        method = f"{method}+plotly_native"
        render_mode = "plotly_native"
    elif interactive_type == "rgl":
        r_code = transform_rgl(r_code, png_path_str)
        method = f"{method}+rgl"
        render_mode = "rgl"

    ok, msg = render_r_to_png(r_code, png_path_str, timeout=timeout, mode=render_mode)
    if not ok:
        obj["failure_stage"] = "render"
        obj["failure_message"] = msg
        if interactive_type:
            obj["failure_category"] = f"interactive_{interactive_type}"
        else:
            if "no package called" in msg:
                obj["failure_category"] = "missing_package"
            elif "timed out" in msg.lower():
                obj["failure_category"] = "timeout"
            else:
                obj["failure_category"] = "code_error"
        return {"status": "failed", "obj": obj}

    obj["png_path"] = Path(png_path_str).name
    obj["r_extract_method"] = method
    return {"status": "passed", "obj": obj}


def load_entries(jsonl_path):
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
        description="Extract R code from JSONL, render to PNG, split passed/failed"
    )
    parser.add_argument("input", help="Path to a .jsonl file or directory")
    parser.add_argument("--output", "-o", default=None, help="PNG output directory (default: ./png_output)")
    parser.add_argument("--result", "-r", default=None, help="Result JSONL directory (default: ./results)")
    parser.add_argument("--workers", "-j", type=int, default=8, help="Parallel render workers")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="Per-script render timeout (s)")
    args = parser.parse_args()

    find_rscript()

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

    output_dir = Path(args.output).resolve() if args.output else Path("./png_output").resolve()
    result_dir = Path(args.result).resolve() if args.result else Path("./results").resolve()
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
        for obj, line_no in load_entries(jf):
            if "failure_stage" in obj and obj["failure_stage"] == "json_parse":
                json_parse_failures.append(obj)
                continue
            uuid = obj.get("uuid", f"line{line_no}")
            png_name = f"{jf.stem}_{uuid}.png"
            all_tasks.append((obj, str(output_dir / png_name), args.timeout))

    total = len(all_tasks) + len(json_parse_failures)
    print(f"Total entries : {total}  (render: {len(all_tasks)}, json errors: {len(json_parse_failures)})")

    t0 = time.time()
    passed_count = 0
    failed_count = len(json_parse_failures)
    done = 0

    with (
        open(passed_path, "w", encoding="utf-8") as passed_fh,
        open(failed_path, "w", encoding="utf-8") as failed_fh,
    ):
        for obj in json_parse_failures:
            failed_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, task): task for task in all_tasks}
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "passed":
                    passed_fh.write(json.dumps(result["obj"], ensure_ascii=False) + "\n")
                    passed_count += 1
                else:
                    failed_fh.write(json.dumps(result["obj"], ensure_ascii=False) + "\n")
                    failed_count += 1

                done += 1
                if done % 100 == 0 or done == len(all_tasks):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  Progress: {done}/{len(all_tasks)}  "
                        f"({passed_count} passed, {failed_count} failed)  "
                        f"[{rate:.1f} items/s]",
                        end="", flush=True,
                    )

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 50}")
    print(f"Total   : {total}")
    print(f"Passed  : {passed_count}")
    print(f"Failed  : {failed_count}")
    print(f"Time    : {elapsed:.1f}s  ({total / elapsed:.1f} items/s)")
    print(f"{'=' * 50}")

    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
