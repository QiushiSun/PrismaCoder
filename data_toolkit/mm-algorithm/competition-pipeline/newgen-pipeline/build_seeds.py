"""
Convert leetcode_native records into ICL seed records for the newgen pipeline.

For each leetcode_records.jsonl entry, copy image[0] (only the first image)
into seeds/images/<qid>.png, and emit a slim record into seeds/records.jsonl
with the fields the prompt builder needs.

Usage:
  conda run -n vpb python build_seeds_lc.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
LC_RECORDS = BASE / "seeds_sample" / "leetcode_records.jsonl"
LC_IMAGE_ROOT = BASE / "seeds_sample"  # paths in JSONL are relative to here
SEEDS_DIR = BASE / "seeds"
SEED_IMG_DIR = SEEDS_DIR / "images"


def main() -> None:
    if not LC_RECORDS.exists():
        raise SystemExit(f"missing {LC_RECORDS}")
    SEED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    out_recs: list[dict] = []
    n_skipped = 0

    with LC_RECORDS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r["id"]  # e.g. "leetcode_3493"

            imgs = r.get("images") or ([r["image"]] if r.get("image") else [])
            if not imgs:
                n_skipped += 1
                continue

            # Try several reasonable image-path layouts.
            cand = [
                (LC_IMAGE_ROOT / imgs[0]).resolve(),
                (LC_IMAGE_ROOT / "images" / Path(imgs[0]).name).resolve(),
            ]
            src_first = next((p for p in cand if p.exists()), None)
            if src_first is None:
                n_skipped += 1
                continue

            dst_name = f"{qid}{src_first.suffix}"
            dst = SEED_IMG_DIR / dst_name
            if not dst.exists():
                shutil.copy2(src_first, dst)

            out_recs.append({
                "qid":         qid,
                "title":       r.get("title") or "",
                "description": r.get("query") or "",  # already markdown-ised
                "solution":    r.get("code") or "",
                "tags":        r.get("tags") or [],
                "difficulty":  r.get("difficulty") or "Unknown",
                "image_path":  f"images/{dst_name}",
            })

    out_path = SEEDS_DIR / "records.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_recs)} seed records → {out_path.relative_to(BASE)}")
    print(f"Images: {len(list(SEED_IMG_DIR.glob('*')))} files in {SEED_IMG_DIR.relative_to(BASE)}")
    print(f"Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
