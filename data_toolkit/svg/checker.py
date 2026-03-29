#!/usr/bin/env python3
"""
JSONL SVG Checker — extract SVG from JSONL output fields, validate, render to
PNG, and split results into passed / failed JSONL files.

Usage:
    python checker.py <jsonl_path_or_dir> [options]

Options:
    --output  -o   PNG output directory          (default: ./png_output)
    --result  -r   Directory for result JSONLs   (default: ./results)
    --width   -w   PNG width in px               (default: 1600)
    --dpi     -d   Output DPI                    (default: 96)

Output files (inside --result dir):
    passed.jsonl   — entries that rendered successfully
    failed.jsonl   — entries that failed at any stage
"""

import argparse
import json
import sys
import time
from pathlib import Path

from utils import extract_svg, render_svg_to_png, validate_svg_string


def process_jsonl(jsonl_path, output_dir, passed_fh, failed_fh, width=1600, dpi=96):
    counters = {"total": 0, "passed": 0, "failed": 0}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            counters["total"] += 1

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as e:
                failed_entry = {
                    "_source": jsonl_path.name, "_line": line_no,
                    "failure_stage": "json_parse", "failure_message": str(e),
                }
                failed_fh.write(json.dumps(failed_entry, ensure_ascii=False) + "\n")
                counters["failed"] += 1
                continue

            uuid = obj.get("uuid", f"line{line_no}")
            output_text = obj.get("output", "")
            if not output_text:
                obj["failure_stage"] = "extract"
                obj["failure_message"] = "No 'output' field or empty"
                failed_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counters["failed"] += 1
                continue

            svg_str, method = extract_svg(output_text)
            if svg_str is None:
                obj["failure_stage"] = "extract"
                obj["failure_message"] = "No SVG content found in output"
                failed_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counters["failed"] += 1
                continue

            ok, msg = validate_svg_string(svg_str)
            if not ok:
                obj["failure_stage"] = "xml_validate"
                obj["failure_message"] = msg
                failed_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counters["failed"] += 1
                continue

            png_name = f"{jsonl_path.stem}_{uuid}.png"
            png_path = output_dir / png_name
            ok, msg = render_svg_to_png(svg_str, png_path, width=width, dpi=dpi)
            if not ok:
                obj["failure_stage"] = "render"
                obj["failure_message"] = msg
                failed_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counters["failed"] += 1
                continue

            obj["png_path"] = png_name
            obj["svg_extract_method"] = method
            passed_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            counters["passed"] += 1

    return counters


def main():
    parser = argparse.ArgumentParser(
        description="Extract SVG from JSONL, validate, render to PNG, split passed/failed"
    )
    parser.add_argument("input", help="Path to a .jsonl file or directory containing .jsonl files")
    parser.add_argument("--output", "-o", default=None, help="PNG output directory (default: ./png_output)")
    parser.add_argument("--result", "-r", default=None, help="Result JSONL directory (default: ./results)")
    parser.add_argument("--width", "-w", type=int, default=1600, help="Output PNG width in px")
    parser.add_argument("--dpi", "-d", type=int, default=96, help="Output DPI")
    args = parser.parse_args()

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
    print(f"Width={args.width}px  DPI={args.dpi}")
    print()

    t0 = time.time()
    total_counters = {"total": 0, "passed": 0, "failed": 0}

    with (
        open(passed_path, "w", encoding="utf-8") as passed_fh,
        open(failed_path, "w", encoding="utf-8") as failed_fh,
    ):
        for jf in jsonl_files:
            print(f"Processing: {jf.name} ... ", end="", flush=True)
            c = process_jsonl(jf, output_dir, passed_fh, failed_fh, width=args.width, dpi=args.dpi)
            for k in total_counters:
                total_counters[k] += c[k]
            print(f"done  ({c['passed']} passed, {c['failed']} failed)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"Total   : {total_counters['total']}")
    print(f"Passed  : {total_counters['passed']}")
    print(f"Failed  : {total_counters['failed']}")
    print(f"Time    : {elapsed:.1f}s")
    print(f"{'=' * 50}")

    sys.exit(1 if total_counters["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
