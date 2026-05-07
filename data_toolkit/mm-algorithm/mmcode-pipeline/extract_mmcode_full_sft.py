"""
Extract the *raw* MMCode training set (mmcode_full.jsonl) into unified SFT
records, suitable for unification with the existing ICL rollout pack.

Input
─────
  MMCode-Data/mmcode_full.jsonl   (3548 rows = 3285 train + 263 test)
  Per record:
      data_split        'extra' (train) | 'test'  ← we use only 'extra'
      problem_id        e.g. 'cf_771_A'
      name, source, difficulty, tags, raw_tags, skill_types, image_tags
      url, time_limit, memory_limit
      question          markdown statement with `![image](N.png)` placeholders
      input_spec, output_spec, note
      sample_tests      JSON-encoded {inputs:[...], outputs:[...]}
      input_output      JSON-encoded full hidden test cases
      solutions         JSON-encoded list[str]  (mostly Python, sometimes mixed)
      starter_code      str (rare)
      images            list[str (PNG base64)]   ←— we decode + persist
      picture_num       int

Output
──────
  MMCode-Data/sft_full/mmcode_full_records.jsonl
  MMCode-Data/sft_full/images/<problem_id>_<idx>.png
  MMCode-Data/sft_full/mmcode_full_summary.json

Per-record schema (matches the leetcode SFT pack)
───────────────────────────────────────────────────
  id                     "mm_full_<problem_id>"   (e.g. "mm_full_cf_771_A")
  source                 "mmcode"
  augmentation_method    "mmcode-full-train"
  problem_id             original problem id from upstream
  difficulty             EASY/MEDIUM/MEDIUM_HARD/HARD/VERY_HARD/UNKNOWN
  tags                   topic tags (codeforces / atcoder / ... raw labels)
  upstream_source        source platform (codeforces, aizu, atcoder, ...)
  url                    canonical problem URL
  image                  "images/mmcode_full/<problem_id>_0.png" (first)
  images                 list of all images
  query                  multimodal user prompt with <image> tokens inline
                         (built by replacing `![image](N.png)` in `question`)
  code                   Python solution, comment-stripped, docstrings kept
  response               "```python\n{code}\n```"
  metadata               source-specific verification + I/O fields

Quality gate
────────────
A record is kept iff:
  * data_split == "extra"  (drop the 263 test-set records)
  * solutions list is non-empty
  * at least one solution parses with ast.parse() AND looks like Python
  * picture_num matches len(images)  (else log + drop)
  * every image base64 decodes to a PNG magic-number prefix
  * every <image> token in the produced query has a paired image
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import re
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
SRC_JSONL = BASE / "mmcode_full.jsonl"
OUT_DIR = BASE / "sft_full"
IMG_OUT = OUT_DIR / "images"


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Heuristic patterns that disqualify a string from being Python.
_NON_PY_HINTS = re.compile(
    r"#include\s*<|using\s+namespace\s+std|public\s+(?:static\s+)?\w+\s+main"
    r"|\bcout\s*<<|\bcin\s*>>|\bSystem\.out|\bScanner\b"
    r"|\bvoid\s+main\s*\(",
)
_PY_HINTS = re.compile(
    r"\bdef\s+\w|^import\s|^from\s|\bprint\s*\(|\binput\s*\(\)"
    r"|sys\.stdin|sys\.stdout",
    re.MULTILINE,
)


# ───────────────────────────── comment stripping ─────────────────────────────

def strip_python_comments(src: str) -> str:
    """Remove `#` comments. Docstrings + string literals are preserved."""
    if not src or not src.strip():
        return src
    src_normalised = src if src.endswith("\n") else src + "\n"
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src_normalised).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return src
    comments_by_line: dict[int, list[tuple[int, int]]] = {}
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            comments_by_line.setdefault(tok.start[0], []).append(
                (tok.start[1], tok.end[1])
            )
    out_lines: list[str] = []
    for i, line in enumerate(src_normalised.splitlines(keepends=True), start=1):
        if i in comments_by_line:
            for st, en in sorted(comments_by_line[i], reverse=True):
                line = line[:st] + line[en:]
            line = line.rstrip()
            if not line.strip():
                continue
            line += "\n"
        out_lines.append(line)
    out = "".join(out_lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip() + "\n"


# ───────────────────────────── solution selection ─────────────────────────────

def looks_python(code: str) -> bool:
    if not code:
        return False
    head = code[:2500]
    if _NON_PY_HINTS.search(head):
        return False
    if _PY_HINTS.search(head):
        return True
    return False


def parses_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError):
        return False


def select_python_solution(
    sols_raw: str | None,
) -> tuple[str | None, dict[str, int]]:
    """Pick the cleanest Python solution from a JSON-encoded list.

    Strategy:
      1. parse + filter to Python-looking + ast-parses
      2. prefer ones that USE classes/functions (cleaner)
      3. then prefer the shortest (concise)
    """
    diag = {"raw_count": 0, "parsed_python": 0}
    if not sols_raw:
        return None, diag
    try:
        sols = json.loads(sols_raw)
    except (TypeError, json.JSONDecodeError):
        return None, diag
    if not isinstance(sols, list):
        return None, diag
    diag["raw_count"] = len(sols)

    cands: list[str] = []
    for s in sols:
        if not isinstance(s, str):
            continue
        if len(s) > 50_000:           # absurdly long submissions
            continue
        if not looks_python(s):
            continue
        if not parses_python(s):
            continue
        cands.append(s)
    diag["parsed_python"] = len(cands)
    if not cands:
        return None, diag

    def score(s: str) -> tuple[int, int]:
        # prefer solutions with structure, then shorter ones
        has_struct = ("def " in s) or ("class " in s)
        return (0 if has_struct else 1, len(s))

    cands.sort(key=score)
    return cands[0], diag


# ───────────────────────────── image handling ─────────────────────────────

# Markdown-style refs: ![image](1.png), ![image](2.png), … OR with leading
# path like ![image](images/cf_771_A/1.png). We capture the leading numeric
# index either way.
_IMG_REF_RE = re.compile(
    r"!\[image]\(\s*(?:[^()]*?/)?(\d+)\.(?:png|jpe?g|gif|webp)\s*\)",
    re.IGNORECASE,
)


def decode_and_save_images(
    images_b64: list[str], rec_id: str, out_dir: Path
) -> list[str] | None:
    """Decode each base64 PNG and write to disk. Returns list of relative
    image paths, or None on failure (any image fails to decode / not PNG)."""
    out_paths: list[str] = []
    for idx, b64 in enumerate(images_b64):
        if not isinstance(b64, str) or not b64:
            return None
        try:
            raw = base64.b64decode(b64, validate=False)
        except (ValueError, TypeError):
            return None
        if not raw or not raw.startswith(PNG_MAGIC):
            # We could check JPEG too, but the source data is uniformly PNG.
            return None
        rel = f"images/{rec_id}_{idx}.png"
        dst = out_dir / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
        out_paths.append(rel)
    return out_paths


def substitute_image_tokens(question: str, n_images: int) -> tuple[str, int]:
    """Replace ![image](N.png) refs with <image>. Returns (text, n_replaced).

    `N` is treated as 1-indexed. Multiple refs to the same N each become an
    `<image>` (we don't dedupe — preserves authoring intent).
    Refs whose N is out of [1, n_images] are dropped.
    """

    n_replaced = 0

    def _sub(m: re.Match) -> str:
        nonlocal n_replaced
        try:
            n = int(m.group(1))
        except ValueError:
            return ""
        if 1 <= n <= n_images:
            n_replaced += 1
            return "<image>"
        return ""

    return _IMG_REF_RE.sub(_sub, question), n_replaced


# ───────────────────────────── prompt template ─────────────────────────────

QUERY_PROMPT = """You are an exceptionally intelligent coding assistant with a deep understanding of Python programming and a keen ability to interpret visual data. Your responses are consistently accurate, reliable, and thoughtful.

**Objective:**
You will be presented with a competitive-programming style problem and accompanying diagram(s). Please solve the problem in Python (reading from STDIN and writing to STDOUT).

**Note**
- The diagram contains essential context that the prose alone may omit; pay close attention to its labels, arrows, and structural cues.
- Output the complete Python solution as a single fenced ```python``` block — no commentary outside the block.

**Problem:**
{question}
"""


# ───────────────────────────── extraction ─────────────────────────────

SkipReason = str  # one of:
                  # 'wrong_split' | 'no_solutions' | 'no_python_solution'
                  # | 'image_decode_failed' | 'no_image_tokens'
                  # | 'image_count_mismatch'

def build_record(
    raw: dict, *, copy_images: bool = True,
) -> tuple[dict | None, SkipReason | None, dict[str, int]]:
    if raw.get("data_split") != "extra":
        return None, "wrong_split", {}

    problem_id = raw.get("problem_id")
    if not problem_id:
        return None, "no_problem_id", {}
    rec_id = f"mm_full_{problem_id}"

    sols_raw = raw.get("solutions")
    code, sol_diag = select_python_solution(sols_raw)
    if code is None:
        if not sols_raw or sol_diag["raw_count"] == 0:
            return None, "no_solutions", sol_diag
        return None, "no_python_solution", sol_diag

    images_b64 = raw.get("images") or []
    if not images_b64:
        return None, "no_images_in_json", sol_diag
    n_pic = raw.get("picture_num")
    if isinstance(n_pic, int) and n_pic != len(images_b64):
        return None, "image_count_mismatch", sol_diag

    if copy_images:
        rel_paths = decode_and_save_images(images_b64, rec_id, OUT_DIR)
    else:
        rel_paths = [f"images/{rec_id}_{i}.png" for i in range(len(images_b64))]
    if rel_paths is None:
        return None, "image_decode_failed", sol_diag

    question = raw.get("question") or ""
    body, n_tokens = substitute_image_tokens(question, len(rel_paths))
    if n_tokens == 0:
        # `question` lacked ![image](N.png) refs even though the record has
        # images. Prepend `<image>` tokens (one per image) before the prose
        # so the SFT record is still multi-image.
        body = ("<image>\n" * len(rel_paths)) + question
        n_tokens = len(rel_paths)
        synth_token = True
    else:
        synth_token = False

    code_clean = strip_python_comments(code).rstrip()

    query = QUERY_PROMPT.format(question=body)

    metadata = {
        "problem_id":          problem_id,
        "upstream_source":     raw.get("source"),
        "name":                raw.get("name"),
        "difficulty":          raw.get("difficulty"),
        "url":                 raw.get("url"),
        "tags":                _parse_listish(raw.get("tags")),
        "raw_tags":            _parse_listish(raw.get("raw_tags")),
        "skill_types":         _parse_listish(raw.get("skill_types")),
        "image_tags":          raw.get("image_tags") or [],
        "picture_num":         raw.get("picture_num"),
        "time_limit":          raw.get("time_limit"),
        "memory_limit":        raw.get("memory_limit"),
        "input_spec":          raw.get("input_spec"),
        "output_spec":         raw.get("output_spec"),
        "note":                raw.get("note"),
        "starter_code":        raw.get("starter_code"),
        "sample_tests":        _safe_json_loads(raw.get("sample_tests")),
        "input_output":        _safe_json_loads(raw.get("input_output")),
        "n_solutions_raw":     sol_diag.get("raw_count"),
        "n_solutions_python":  sol_diag.get("parsed_python"),
        "image_token_synth":   synth_token,
    }

    rec = {
        "id":                  rec_id,
        "source":              "mmcode",
        "augmentation_method": "mmcode-full-train",
        "image":               rel_paths[0],
        "images":              rel_paths,
        "query":               query,
        "code":                code_clean,
        "response":            f"```python\n{code_clean}\n```",
        "metadata":            metadata,
    }
    return rec, None, sol_diag


def _parse_listish(v: Any) -> list:
    """Some upstream fields are JSON-stringified lists. Return list or []."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            out = json.loads(v.replace("'", '"'))
            if isinstance(out, list):
                return out
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to a tolerant eval of bracketed string-list-of-strings
        try:
            import ast as _ast
            out = _ast.literal_eval(v)
            if isinstance(out, list):
                return out
        except (ValueError, SyntaxError):
            return []
    return []


def _safe_json_loads(v: Any) -> Any:
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return None
    return v


# ───────────────────────────── main ─────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N input records (0 = all)")
    ap.add_argument("--no-copy-images", action="store_true",
                    help="Skip decoding/writing images (useful for dry-runs)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"Reading {SRC_JSONL.relative_to(BASE)} …")
    started = datetime.now()

    records: list[dict] = []
    skip_reasons: Counter = Counter()
    per_split = Counter()
    per_diff = Counter()
    img_count_dist = Counter()
    upstream_dist = Counter()
    n_images_total = 0
    n_token_synth = 0

    with SRC_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["json_parse_failed"] += 1
                continue
            per_split[raw.get("data_split", "?")] += 1
            rec, why, _diag = build_record(
                raw, copy_images=not args.no_copy_images
            )
            if rec is None:
                skip_reasons[why or "unknown"] += 1
                continue
            records.append(rec)
            per_diff[rec["metadata"].get("difficulty") or "?"] += 1
            img_count_dist[len(rec["images"])] += 1
            upstream_dist[rec["metadata"].get("upstream_source") or "?"] += 1
            n_images_total += len(rec["images"])
            if rec["metadata"].get("image_token_synth"):
                n_token_synth += 1

    out_path = out_dir / "mmcode_full_records.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "ts":          started.isoformat(timespec="seconds"),
        "elapsed_sec": round((datetime.now() - started).total_seconds(), 1),
        "input":       str(SRC_JSONL.relative_to(BASE)),
        "n_input_total": sum(per_split.values()),
        "input_split_counts": dict(per_split),
        "n_output":    len(records),
        "n_skipped":   sum(skip_reasons.values()),
        "skip_reasons":            dict(skip_reasons),
        "by_difficulty":           dict(per_diff),
        "by_upstream_source":      dict(upstream_dist),
        "image_count_distribution": dict(sorted(img_count_dist.items())),
        "total_image_files":       n_images_total,
        "image_token_synthesised": n_token_synth,
        "out_jsonl":   str(out_path.relative_to(BASE)),
        "out_images":  str((out_dir / "images").relative_to(BASE)),
    }
    (out_dir / "mmcode_full_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("MMCode FULL train → SFT extraction COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
