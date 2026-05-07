"""
Step 5 — package verified Competition-Data solutions into SFT records.

For each verified/<sid>.json (passed all tests):
  • Copy referenced images into sft/images/<sid>_<i>.<ext>
  • Build a multimodal query: <image> tokens + cleaned problem text +
    "Solve in Python (STDIN/STDOUT)."
  • Strip Python comments from the solution code.
  • Emit one SFT record into sft/<source>_solved.jsonl, plus aggregate
    sft/comp_solved_train.jsonl.

Usage:
  conda run -n vpb python build_sft_competitive.py
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMP_ROOT = BASE / "seeds_sample"  # in the original repo this was Competition-Data root
SEEDS_PATH = BASE / "seeds" / "records.jsonl"
TRANS_DIR = BASE / "translations"
VER_DIR = BASE / "verified"
SFT_DIR = BASE / "sft"
SFT_IMG_DIR = SFT_DIR / "images"


def strip_python_comments(src: str) -> str:
    if not src or not src.strip():
        return src
    src_norm = src if src.endswith("\n") else src + "\n"
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src_norm).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return src
    cmts: dict[int, list[tuple[int, int]]] = {}
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            cmts.setdefault(tok.start[0], []).append((tok.start[1], tok.end[1]))
    out: list[str] = []
    for i, line in enumerate(src_norm.splitlines(keepends=True), start=1):
        if i in cmts:
            for s, e in sorted(cmts[i], reverse=True):
                line = line[:s] + line[e:]
            line = line.rstrip()
            if not line.strip():
                continue
            line += "\n"
        out.append(line)
    cleaned = "".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip() + "\n"


COMP_QUERY_PROMPT = """You are an exceptionally intelligent coding assistant with a deep understanding of Python programming and a keen ability to interpret visual data. Your responses are consistently accurate, reliable, and thoughtful.

**Objective:**
You are given a competitive-programming problem and one or more accompanying diagrams. Solve the problem in Python: read all input from stdin, write all output to stdout.

**Note**
- The diagram contains essential context that the prose alone may omit; pay close attention to its labels, arrows, and structural cues.
- Output the complete Python solution as a single fenced ```python``` block — no commentary outside the block.

**Problem:**
{problem_text}
"""


def main() -> None:
    SFT_DIR.mkdir(exist_ok=True)
    SFT_IMG_DIR.mkdir(exist_ok=True)

    seeds: dict[str, dict] = {}
    with SEEDS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seeds[r["sid"]] = r

    by_source: dict[str, list[dict]] = {}
    n_total = 0
    n_skipped = 0
    diff_counter: Counter = Counter()

    for vp in sorted(VER_DIR.glob("*.json")):
        sid = vp.stem
        ver = json.loads(vp.read_text(encoding="utf-8"))
        seed = seeds.get(sid)
        if not seed:
            n_skipped += 1
            continue

        # Resolve EN problem text
        if seed.get("lang") == "jp":
            tp = TRANS_DIR / f"{sid}.txt"
            if not tp.exists():
                n_skipped += 1
                continue
            problem_text = tp.read_text(encoding="utf-8")
        else:
            problem_text = seed.get("problem_text_en") or ""

        # Copy images, retain original ordering
        dst_imgs: list[str] = []
        ok_imgs = True
        for i, rel in enumerate(seed.get("image_paths") or []):
            src_abs = COMP_ROOT / rel
            if not src_abs.exists():
                ok_imgs = False
                break
            dst_name = f"{sid}_{i}{src_abs.suffix}"
            dst_abs = SFT_IMG_DIR / dst_name
            if not dst_abs.exists():
                shutil.copy2(src_abs, dst_abs)
            dst_imgs.append(f"images/{dst_name}")
        if not ok_imgs or not dst_imgs:
            n_skipped += 1
            continue

        n_image_tokens = len(dst_imgs)
        image_block = "\n".join(["<image>"] * n_image_tokens)
        query = (
            f"{image_block}\n\n" +
            COMP_QUERY_PROMPT.format(problem_text=problem_text.strip())
        )

        code_clean = strip_python_comments(ver["code"]).rstrip()

        rec = {
            "id":               sid,
            "source":           seed["source"],
            "problem_id":       seed["problem_id"],
            "title":            seed.get("title"),
            "url":              seed.get("url"),
            "lang":             seed.get("lang"),
            "image":            dst_imgs[0],
            "images":           dst_imgs,
            "query":            query,
            "code":             code_clean,
            "response":         f"```python\n{code_clean}\n```",
            "n_existing_tests": ver["n_existing_tests"],
            "n_extra_tests":    ver["n_extra_tests"],
            "n_total_tests":    ver["n_total"],
            "test_cases":       ver["all_tests"],
        }
        by_source.setdefault(seed["source"], []).append(rec)
        n_total += 1
        diff_counter[seed["source"]] += 1

    # Write per-source jsonl + aggregate
    for source, recs in by_source.items():
        out = SFT_DIR / f"{source}_solved.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {out.relative_to(BASE)}: {len(recs)} records")

    all_recs = sum(by_source.values(), [])
    out_main = SFT_DIR / "comp_solved_train.jsonl"
    with out_main.open("w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {out_main.relative_to(BASE)}: {len(all_recs)} records")

    summary = {
        "ts":          datetime.now().isoformat(timespec="seconds"),
        "n_total":     n_total,
        "n_skipped":   n_skipped,
        "by_source":   dict(diff_counter),
        "image_count_distribution":
            dict(sorted(Counter(len(r["images"]) for r in all_recs).items())),
        "n_test_distribution":
            dict(sorted(Counter(r["n_total_tests"] for r in all_recs).items())),
    }
    (SFT_DIR / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
