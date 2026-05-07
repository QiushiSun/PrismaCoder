"""Retroactively recover problems from `failed_batches/failed_batch_*.txt`.

Re-applies the latest `parse_problems` (with strict=False fallback + object
recovery) to each dumped raw response, builds proper records (with SVG validation
and recipe metadata reconstructed via the same RNG seed used by `process_batch`),
and appends them to `problems.jsonl`. Successfully recovered files are moved to
`failed_batches/recovered/` so they are not re-processed.

Usage:
  conda run -n vpb python recover_failed_batches.py \
      --rollout-dir rollout-0420-claude46-mmcode-svg
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from generate_problems import (
    load_train_examples,
    load_test_examples,
    load_ingredients,
    build_batch_inputs,
    parse_problems,
    validate_svg,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--problems-per-batch", type=int, default=3)
    parser.add_argument("--model", default="claude-sonnet-4-6-aws")
    parser.add_argument(
        "--icl-source", choices=["train", "test"], default=None,
        help="Override ICL source for recipe reconstruction. "
             "Defaults to value in <rollout-dir>/manifest.json.",
    )
    parser.add_argument(
        "--difficulty-mode", choices=["standard", "hard"], default=None,
        help="Override difficulty mode for recipe reconstruction. "
             "Defaults to value in <rollout-dir>/manifest.json (or 'standard' if absent).",
    )
    args = parser.parse_args()

    rollout = Path(args.rollout_dir)
    failed_dir = rollout / "failed_batches"
    if not failed_dir.exists():
        print(f"No failed_batches/ in {rollout}, nothing to do")
        return

    out_path = rollout / "problems.jsonl"
    recovered_dir = failed_dir / "recovered"
    recovered_dir.mkdir(exist_ok=True)

    icl_source = args.icl_source
    difficulty_mode = args.difficulty_mode
    manifest_path = rollout / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    if icl_source is None:
        icl_source = manifest.get("icl_source", "train")
    if difficulty_mode is None:
        difficulty_mode = manifest.get("difficulty_mode", "standard")
    print(f"ICL source:      {icl_source}")
    print(f"Difficulty mode: {difficulty_mode}")

    print("Loading train examples + ingredients (for recipe reconstruction)...")
    train_examples = load_train_examples()
    test_examples = load_test_examples() if icl_source == "test" else None
    ingredients = load_ingredients()

    failed_files = sorted(failed_dir.glob("failed_batch_*.txt"))
    print(f"Found {len(failed_files)} failed-batch dumps")

    appended = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for ff in failed_files:
            try:
                batch_id = int(ff.stem.replace("failed_batch_", ""))
            except ValueError:
                print(f"  SKIP {ff.name}: cannot parse batch_id")
                continue

            raw = ff.read_text(encoding="utf-8")
            parsed = parse_problems(raw)
            if not parsed:
                print(f"  SKIP {ff.name}: still unparseable ({len(raw)} chars)")
                continue

            recipe, icl = build_batch_inputs(
                batch_id, train_examples, test_examples, ingredients,
                icl_source=icl_source, difficulty_mode=difficulty_mode,
            )
            icl_ids = [ex["problem_id"] for ex in icl]

            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                io = item.get("input_output", {})
                if isinstance(io, str):
                    try:
                        io = json.loads(io)
                    except json.JSONDecodeError:
                        io = {}

                svg_str = item.get("visualization_svg", "") or ""
                svg_ok, svg_reason = validate_svg(svg_str)

                record = {
                    "id": f"mmcode_aug_{batch_id * args.problems_per_batch + i:06d}",
                    "batch_id": batch_id,
                    "recipe": recipe,
                    "icl_source": icl_source,
                    "difficulty_mode": difficulty_mode,
                    "icl_example_ids": icl_ids,
                    "model": args.model,
                    "question": item.get("question", ""),
                    "solution": item.get("solution", ""),
                    "input_output": {
                        "inputs": io.get("inputs", []),
                        "outputs": io.get("outputs", []),
                    },
                    "visualization_svg": svg_str if svg_ok else "",
                    "visualization_caption": item.get("visualization_caption", ""),
                    "svg_valid": svg_ok,
                    "svg_reason": svg_reason,
                    "difficulty": item.get("difficulty", recipe["difficulty"]),
                    "tags": item.get("tags", recipe["algorithm_tags"]),
                    "recovered_from_failed_batch": True,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                appended += 1

            shutil.move(str(ff), recovered_dir / ff.name)
            print(f"  {ff.name} -> recovered {len(parsed)} problems")

    print(f"\nAppended {appended} recovered problems to {out_path}")


if __name__ == "__main__":
    main()
