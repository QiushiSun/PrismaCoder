"""
Cross-source seed builder for the Hard-augmentation pipeline.

Pulls Hard / Medium-Hard records from MULTIPLE sources to widen the ICL
inspiration pool:
  • leetcode_native      — clean class-Solution style (ICL pool)
  • leetcode_newgen      — generated, class-Solution style (ICL pool)
  • codeforces / atcoder / aizu solved — STDIN/STDOUT competitive (inspiration)
  • mmcode_full          — STDIN/STDOUT competitive (inspiration)

Important: only LC-native + LC-newgen are used as direct ICL examples
(class-Solution output style consistency).  CF / AC / Aizu / MMCode are
written into a separate `inspiration/` sidecar pool — the prompt can show
these as "domain inspiration" without their solutions, to push the LLM
toward harder algorithmic content.

This open-source variant reads from `seeds_sample/`, which contains a
small subsample of each source for smoke-testing.  When you point the
pipeline at your own corpus, override the paths via env vars:

  CROSS_AUG_LC_NATIVE     — path to leetcode_records.jsonl
  CROSS_AUG_LC_NEWGEN     — path to lc-newgen train.sft.clean.jsonl
  CROSS_AUG_COMP_DIR      — dir holding *_solved.jsonl files
  CROSS_AUG_MMCODE_FULL   — path to mmcode_full_records.jsonl

  All upstream image paths are resolved relative to the parent dir of
  each source jsonl unless you set CROSS_AUG_*_IMG_ROOT to override it.

Outputs:
  seeds/icl_records.jsonl        — class-Solution ICL pool (Hard + Medium)
  seeds/inspiration_records.jsonl — competitive inspiration (Hard only)
  seeds/images/<qid>.<ext>        — first image of each ICL/insp record

Usage:
  conda run -n vpb python build_seeds.py
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SAMPLE_DIR = BASE / "seeds_sample"

# ── source paths (overridable via env) ────────────────────────────────────

def _path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default)).expanduser().resolve()


LC_NATIVE        = _path("CROSS_AUG_LC_NATIVE",
                         SAMPLE_DIR / "leetcode_native_records.jsonl")
LC_NEWGEN        = _path("CROSS_AUG_LC_NEWGEN",
                         SAMPLE_DIR / "leetcode_newgen_records.jsonl")
COMP_DIR         = _path("CROSS_AUG_COMP_DIR",
                         SAMPLE_DIR / "competition_solved")
MMC_FULL         = _path("CROSS_AUG_MMCODE_FULL",
                         SAMPLE_DIR / "mmcode_full_records.jsonl")

LC_NATIVE_IMG    = _path("CROSS_AUG_LC_NATIVE_IMG_ROOT",  LC_NATIVE.parent)
LC_NEWGEN_IMG    = _path("CROSS_AUG_LC_NEWGEN_IMG_ROOT",  LC_NEWGEN.parent)
COMP_IMG         = _path("CROSS_AUG_COMP_IMG_ROOT",       COMP_DIR)
MMC_FULL_IMG     = _path("CROSS_AUG_MMCODE_FULL_IMG_ROOT", MMC_FULL.parent)

SEEDS_DIR = BASE / "seeds"
ICL_OUT   = SEEDS_DIR / "icl_records.jsonl"
INSP_OUT  = SEEDS_DIR / "inspiration_records.jsonl"
IMG_DIR   = SEEDS_DIR / "images"


def _copy_first_image(record_id: str, image_path: str,
                      src_root: Path, dst_dir: Path) -> str:
    """Copy `image_path` (relative to src_root) to dst_dir/<id>.<ext>.
    Tries a few candidate layouts; returns the new relative path or "" on miss."""
    cand = [
        (src_root / image_path),
        (src_root / "images" / Path(image_path).name),
        (src_root / Path(image_path).name),
    ]
    src_abs = next((p for p in cand if p.exists()), None)
    if src_abs is None:
        return ""
    dst_name = f"{record_id}{src_abs.suffix}"
    dst_abs = dst_dir / dst_name
    if not dst_abs.exists():
        try:
            shutil.copy2(src_abs, dst_abs)
        except Exception:
            return ""
    return f"images/{dst_name}"


def harvest_lc_native() -> list[dict]:
    """LC-native records: Hard + Medium (skip Easy)."""
    if not LC_NATIVE.exists():
        return []
    out = []
    with LC_NATIVE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            diff = r.get("difficulty") or "Unknown"
            if diff not in {"Hard", "Medium"}:
                continue
            qid = r["id"]
            imgs = r.get("images") or ([r["image"]] if r.get("image") else [])
            if not imgs:
                continue
            new_img = _copy_first_image(qid, imgs[0], LC_NATIVE_IMG, IMG_DIR)
            if not new_img:
                continue
            out.append({
                "qid": qid,
                "source": "lc_native",
                "title": r.get("title") or "",
                "description": r.get("query") or "",
                "solution": r.get("code") or "",
                "tags": r.get("tags") or [],
                "difficulty": diff,
                "image_path": new_img,
            })
    return out


def harvest_lc_newgen() -> list[dict]:
    """LC-newgen Hard subset only — closest to our target."""
    if not LC_NEWGEN.exists():
        return []
    out = []
    with LC_NEWGEN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            diff = r.get("difficulty") or r.get("recipe", {}).get("difficulty") or "Unknown"
            if diff != "Hard":
                continue
            qid = r["id"]
            imgs = r.get("images") or ([r["image"]] if r.get("image") else [])
            if not imgs:
                continue
            new_img = _copy_first_image(qid, imgs[0], LC_NEWGEN_IMG, IMG_DIR)
            if not new_img:
                continue
            out.append({
                "qid": qid,
                "source": "lc_newgen",
                "title": r.get("title") or "",
                "description": r.get("query") or "",
                "solution": r.get("code") or "",
                "tags": [],
                "difficulty": diff,
                "image_path": new_img,
            })
    return out


def harvest_competition() -> list[dict]:
    """Competition (CF/AC/Aizu) records — used as inspiration only."""
    out: list[dict] = []
    if not COMP_DIR.exists():
        return out
    for src in ["codeforces", "atcoder", "aizu"]:
        p = COMP_DIR / f"{src}_solved.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r["id"]
                first_img = r.get("image") or ((r.get("images") or [None])[0])
                if not first_img:
                    continue
                new_img = _copy_first_image(qid, first_img, COMP_IMG, IMG_DIR)
                if not new_img:
                    continue
                out.append({
                    "qid": qid,
                    "source": src,
                    "title": (r.get("title") or "").strip(),
                    "description": r.get("query") or "",
                    "solution": "",
                    "tags": [],
                    "difficulty": "Hard",
                    "image_path": new_img,
                })
    return out


def harvest_mmcode_full() -> list[dict]:
    """MMCode_full records — Hard / Medium-Hard / Very-Hard subset, inspiration."""
    if not MMC_FULL.exists():
        return []
    out = []
    keep_diffs = {"HARD", "MEDIUM_HARD", "VERY_HARD"}
    with MMC_FULL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            meta = r.get("metadata") or {}
            diff = (meta.get("difficulty") or r.get("difficulty") or "").upper()
            if diff not in keep_diffs:
                continue
            qid = r["id"]
            first_img = r.get("image") or ((r.get("images") or [None])[0])
            if not first_img:
                continue
            new_img = _copy_first_image(qid, first_img, MMC_FULL_IMG, IMG_DIR)
            if not new_img:
                continue
            out.append({
                "qid": qid,
                "source": "mmcode_full",
                "upstream": meta.get("upstream_source") or "?",
                "title": (meta.get("name") or "").strip(),
                "description": r.get("query") or "",
                "solution": "",
                "tags": meta.get("tags") or [],
                "difficulty": diff.title().replace("_", "-"),
                "image_path": new_img,
            })
    return out


def main() -> None:
    SEEDS_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    print("Harvesting LC-native ...")
    icl_native = harvest_lc_native()
    print(f"  → {len(icl_native)} records")

    print("Harvesting LC-newgen (Hard only) ...")
    icl_newgen = harvest_lc_newgen()
    print(f"  → {len(icl_newgen)} records")

    print("Harvesting competition (CF/AC/Aizu) ...")
    insp_comp = harvest_competition()
    print(f"  → {len(insp_comp)} records")

    print("Harvesting MMCode_full (Hard subset) ...")
    insp_mmc = harvest_mmcode_full()
    print(f"  → {len(insp_mmc)} records")

    icl_all = icl_native + icl_newgen
    insp_all = insp_comp + insp_mmc

    with ICL_OUT.open("w", encoding="utf-8") as f:
        for r in icl_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with INSP_OUT.open("w", encoding="utf-8") as f:
        for r in insp_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"ICL pool:         {len(icl_all)}  → {ICL_OUT.name}")
    src_counter = Counter(r["source"] for r in icl_all)
    diff_counter = Counter(r["difficulty"] for r in icl_all)
    print(f"  by source:      {dict(src_counter)}")
    print(f"  by difficulty:  {dict(diff_counter)}")

    print()
    print(f"Inspiration pool: {len(insp_all)}  → {INSP_OUT.name}")
    src_counter = Counter(r["source"] for r in insp_all)
    diff_counter = Counter(r["difficulty"] for r in insp_all)
    print(f"  by source:      {dict(src_counter)}")
    print(f"  by difficulty:  {dict(diff_counter)}")
    print()
    print(f"Images: {len(list(IMG_DIR.glob('*')))}")


if __name__ == "__main__":
    main()
