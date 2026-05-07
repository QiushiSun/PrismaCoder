"""Post-process the MMCode augmented dataset:

1. Detect duplicate IDs (a side-effect of the chunk-1 orphan-process incident
   on 2026-04-20: the same batch_ids were processed by two concurrent writers,
   producing different problems labelled with the same `id`).
2. Re-assign globally unique sequential IDs.
3. Track provenance via `original_id` and `dup_seq` (when applicable).
4. Save to `problems.dedup.jsonl` (does NOT modify the source file).

Usage:
  conda run -n vpb python dedup_problems.py \
      --input rollout-0420-claude46-mmcode-svg/problems.jsonl \
      --output rollout-0420-claude46-mmcode-svg/problems.dedup.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source .jsonl")
    parser.add_argument("--output", required=True, help="Cleaned .jsonl with unique IDs")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = []
    with in_path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARN: skipping malformed line {ln}: {exc}")

    print(f"Read {len(records)} records from {in_path}")

    id_counts = Counter(r.get("id", "") for r in records)
    n_dup_ids = sum(1 for c in id_counts.values() if c > 1)
    n_extra = sum(c - 1 for c in id_counts.values() if c > 1)
    print(f"  Duplicate IDs: {n_dup_ids} (extra copies: {n_extra})")

    seen: dict[str, int] = {}
    out_records = []
    for i, r in enumerate(records):
        old_id = r.get("id", f"unknown_{i}")
        seen[old_id] = seen.get(old_id, 0) + 1
        new_id = f"mmcode_aug_{i:06d}"
        new_record = dict(r)
        new_record["id"] = new_id
        new_record["original_id"] = old_id
        if seen[old_id] > 1:
            new_record["dup_seq"] = seen[old_id]
        out_records.append(new_record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    new_id_counts = Counter(r["id"] for r in out_records)
    n_dup_after = sum(1 for c in new_id_counts.values() if c > 1)
    print(f"Wrote {len(out_records)} records to {out_path}")
    print(f"  Duplicate IDs after dedup: {n_dup_after}")

    valid_svg = sum(1 for r in out_records if r.get("svg_valid", False))
    print(f"  Valid SVGs: {valid_svg} ({valid_svg / len(out_records) * 100:.1f}%)")


if __name__ == "__main__":
    main()
