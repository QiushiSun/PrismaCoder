"""
Step 1 — parse Competition-Data raw JSONs into unified seed records.

Input  (CodeForces / AtCoder / Aizu *_data_with_images.json):
  schema:  {images: [...], messages: [{role, content_en, content_ja}, ...],
            title, url, problem_id|task_id, test_cases: [{input, output}]}

Output:  seeds/records.jsonl  — one record per problem
  schema:  {sid, source, problem_id, title, url, lang, problem_text,
            problem_text_jp, image_paths, parseable_tests, raw_tests,
            n_parseable_tests, needs_augmentation}

A test case is "parseable" if its input contains a newline (real OJ test
cases preserved across ingestion) — otherwise the input was flattened
during scraping and we can't reconstruct delimiters reliably.

Usage:
  conda run -n vpb python parse_seeds.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent

# SEEDS_DIR may point to either:
#   • a folder with `<source>_data_with_images.json` directly inside it, OR
#   • the original repo layout with CodeForces/, AtCoder/, Aizu/ subdirs.
# We try the flat layout first, then fall back to the repo layout.
import os  # noqa: E402
_seeds_root = Path(os.environ.get("SEEDS_DIR", BASE / "seeds_sample"))

def _find_source(name: str, ja_fallback_filename: str | None = None) -> Path:
    flat = _seeds_root / f"{name}_data_with_images.json"
    if flat.exists():
        return flat
    nested = _seeds_root / name.title() / f"{name}_data_with_images.json"
    if nested.exists():
        return nested
    return flat   # so missing-file errors point at the canonical place

SOURCES = [
    ("codeforces", _find_source("codeforces"), "en"),
    ("atcoder",    _find_source("atcoder"),    "jp"),
    ("aizu",       _find_source("aizu"),       "mixed"),
]


def is_parseable(t: dict) -> bool:
    """A test case is 'parseable' iff its INPUT survived ingestion intact.

    Heuristics (input only — output newlines say nothing about input shape):
      • inputs with a newline are always fine
      • a pure-numeric single token (e.g. '42', '-7') is fine
      • a single-line of space-separated tokens is fine ONLY when the
        problem expects a single line of input (we can't tell, so
        accept inputs ≤ 30 chars to bias toward simple cases)
    """
    inp = (t.get("input") or "").strip()
    if not inp:
        return False
    if "\n" in inp:
        return True
    if re.fullmatch(r"-?\d+", inp):
        return True
    if len(inp) <= 30 and re.fullmatch(r"[\d\s\-\.]+", inp):
        return True
    return False


def get_user_message(rec: dict, source: str) -> tuple[str, str, str]:
    """Return (lang, problem_text_en, problem_text_jp).

    For codeforces: content_en is real English.
    For atcoder: content_en field is mislabelled (still JP); we use content_ja.
    For aizu: prefer content_en, fall back to content_ja, then content.
    """
    for m in rec.get("messages") or []:
        if m.get("role") != "user":
            continue
        en = (m.get("content_en") or "").strip()
        ja = (m.get("content_ja") or "").strip()
        plain = (m.get("content") or "").strip()
        if source == "codeforces":
            return "en", en, ""
        if source == "atcoder":
            # AtCoder's content_en is actually JP — store in JP slot only
            return "jp", "", ja or en or plain
        if source == "aizu":
            if en:
                return "en", en, ja
            if ja:
                return "jp", "", ja
            return "en", plain, ""
    return "?", "", ""


def main() -> None:
    out_dir = BASE / "seeds"
    out_dir.mkdir(exist_ok=True)
    rec_out = out_dir / "records.jsonl"

    all_records: list[dict] = []
    seen_sids: set[str] = set()    # dedupe (AtCoder JSON has each task 2x)
    n_dups = 0
    stats: Counter = Counter()

    for source, src_path, lang_default in SOURCES:
        if not src_path.exists():
            print(f"  ⚠ missing {src_path}")
            continue
        data = json.loads(src_path.read_text(encoding="utf-8"))
        n_with_imgs_on_disk = 0
        n_skipped_no_imgs = 0

        for r in data:
            problem_id = r.get("problem_id") or r.get("task_id") or ""
            if not problem_id:
                continue

            # Resolve image paths.  Try a few common layouts and store the
            # path RELATIVE TO SEEDS_DIR so downstream stages can rebuild
            # full paths consistently.
            img_paths_rel: list[str] = []
            img_paths_abs: list[Path] = []
            src_root = src_path.parent
            missing = False
            for img in r.get("images") or []:
                # Probe several places: <src_root>/<img>, <src_root>/images/<basename>,
                # <SEEDS_DIR>/images/<basename>.
                cand = [
                    src_root / img,
                    src_root / "images" / Path(img).name,
                    _seeds_root / "images" / Path(img).name,
                ]
                p_abs = next((p for p in cand if p.exists()), None)
                if p_abs is None:
                    missing = True
                    break
                img_paths_abs.append(p_abs)
                # Use a stable rel path under seeds_root if possible, else basename
                try:
                    img_paths_rel.append(str(p_abs.resolve().relative_to(_seeds_root)))
                except ValueError:
                    img_paths_rel.append(p_abs.name)
            if missing or not img_paths_rel:
                n_skipped_no_imgs += 1
                continue
            n_with_imgs_on_disk += 1

            lang, text_en, text_jp = get_user_message(r, source)

            # Classify test cases
            tests = r.get("test_cases") or []
            parseable = [t for t in tests if is_parseable(t)]

            sid = f"{source}_{problem_id}"
            if sid in seen_sids:
                n_dups += 1
                continue
            seen_sids.add(sid)
            all_records.append({
                "sid":                  sid,
                "source":               source,
                "problem_id":           problem_id,
                "title":                (r.get("title") or "").strip(),
                "url":                  r.get("url") or "",
                "lang":                 lang,
                "problem_text_en":      text_en,
                "problem_text_jp":      text_jp,
                "image_paths":          img_paths_rel,
                "n_images":             len(img_paths_rel),
                "n_raw_tests":          len(tests),
                "n_parseable_tests":    len(parseable),
                "needs_augmentation":   len(parseable) < 2,
                "raw_tests":            tests,
                "parseable_tests":      parseable,
            })
            stats[f"{source}/needs_aug"] += int(len(parseable) < 2)
            stats[f"{source}/total"] += 1
        print(
            f"[{source}] {n_with_imgs_on_disk} kept, {n_skipped_no_imgs} dropped (missing imgs), "
            f"{n_dups} dropped as duplicate sid, lang_default={lang_default}"
        )
        n_dups = 0

    with rec_out.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records)} seeds → {rec_out.relative_to(BASE)}")
    print()
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    n_jp = sum(1 for r in all_records if r["lang"] == "jp")
    n_en = sum(1 for r in all_records if r["lang"] == "en")
    n_other = len(all_records) - n_jp - n_en
    print(f"\n  needs translation: {n_jp}")
    print(f"  english already:   {n_en}")
    print(f"  unknown:           {n_other}")
    n_aug = sum(1 for r in all_records if r["needs_augmentation"])
    print(f"  needs test-aug:    {n_aug} ({100*n_aug/max(1,len(all_records)):.1f}%)")


if __name__ == "__main__":
    main()
