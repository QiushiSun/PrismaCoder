"""
Leakage audit for newgen LeetCode SFT records.

Strategy
────────
For each generated record we compute:

  • title_jaccard   — token Jaccard between gen.title and every native title
  • desc_jaccard    — token Jaccard on the first paragraph of the description
                      (after stripping the leading `# Title` line)
  • def_collision   — generated METHOD_NAME == any native method name

We use Jaccard rather than `difflib.SequenceMatcher.quick_ratio` because the
latter is a character-set UPPER BOUND that gives spurious 0.7-0.9 scores for
natural-English text (since alphabets overlap).  Jaccard on word-tokens is
fast (O(|a|+|b|) per pair) and meaningful.

Records whose max title_jaccard OR desc_jaccard exceeds `--sim-cut` are
rejected; rest written to train.sft.clean.jsonl.

Usage:
  conda run -n vpb python audit_leakage_lc.py --sim-cut 0.5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Cross-augmentation audits against the local seeds_sample by default;
# point CROSS_AUG_AUDIT_SOURCES (colon-separated paths) at your own
# upstream-training pool when running on real data.
import os
SAMPLE_DIR = BASE / "seeds_sample"
_DEFAULT_SOURCES = [
    SAMPLE_DIR / "leetcode_native_records.jsonl",
    SAMPLE_DIR / "leetcode_newgen_records.jsonl",
    SAMPLE_DIR / "mmcode_full_records.jsonl",
    SAMPLE_DIR / "competition_solved" / "codeforces_solved.jsonl",
    SAMPLE_DIR / "competition_solved" / "atcoder_solved.jsonl",
    SAMPLE_DIR / "competition_solved" / "aizu_solved.jsonl",
]
_env_sources = os.environ.get("CROSS_AUG_AUDIT_SOURCES", "")
if _env_sources.strip():
    NATIVE_SOURCES = [Path(p).expanduser().resolve()
                      for p in _env_sources.split(":") if p.strip()]
else:
    NATIVE_SOURCES = [p for p in _DEFAULT_SOURCES if p.exists()]
SFT_MAIN = BASE / "sft" / "train.sft.jsonl"
SFT_CLEAN = BASE / "sft" / "train.sft.clean.jsonl"
AUDIT_OUT = BASE / "sft" / "leakage_audit.json"

DEF_RE = re.compile(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE)


def normalise_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_paragraph(s: str, max_chars: int = 600) -> str:
    """Take the first paragraph (up to first blank line) of a description."""
    parts = s.strip().split("\n\n", 1)
    return parts[0][:max_chars] if parts else ""


_TITLE_LINE_RE = re.compile(r"^#\s*(?:\d+\.\s*)?(.+)$", re.MULTILINE)


def extract_title(rec: dict) -> str:
    """Pull the title from explicit field, metadata, or first '# Title' line."""
    for k in ("title", "title_en"):
        if rec.get(k):
            return str(rec[k])
    md = rec.get("metadata") or {}
    for k in ("title", "title_en"):
        if md.get(k):
            return str(md[k])
    q = rec.get("query") or ""
    m = _TITLE_LINE_RE.search(q)
    if m:
        return m.group(1).strip().rstrip(":")
    return ""


def query_body(rec: dict) -> str:
    """Drop the leading '# Title' line so similarity isn't dominated by titles
    we already compare separately."""
    q = rec.get("query") or ""
    lines = q.lstrip().split("\n")
    if lines and lines[0].strip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).lstrip()


_WORD_RE = re.compile(r"[a-z]+")
# Stopwords that pollute Jaccard signal — common English filler that appears
# in nearly every problem description.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "is", "are",
    "be", "was", "were", "with", "as", "at", "by", "this", "that", "these",
    "those", "from", "into", "but", "if", "then", "else", "any", "you", "your",
    "we", "us", "they", "their", "it", "its", "have", "has", "had", "will",
    "can", "may", "do", "does", "not", "all", "each", "such", "more", "most",
    "less", "also", "given", "return", "where", "which", "must", "may",
})


def tokens(s: str, *, min_len: int = 2) -> set[str]:
    return {
        w for w in _WORD_RE.findall(s.lower())
        if len(w) >= min_len and w not in _STOPWORDS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-cut", type=float, default=0.5)
    ap.add_argument("--def-collision-mode", choices=("warn", "drop"), default="warn",
                    help="when generated method_name == native method_name, "
                         "either warn or drop the variant")
    args = ap.parse_args()

    if not SFT_MAIN.exists():
        raise SystemExit(f"missing {SFT_MAIN}")

    # Load all native pools — pre-tokenise so the inner loop is just set ops.
    native_title_toks: list[set[str]] = []
    native_desc_toks: list[set[str]] = []
    native_def_names: set[str] = set()
    n_native_records = 0
    n_native_by_src: dict[str, int] = {}
    for src_path in NATIVE_SOURCES:
        if not src_path.exists():
            print(f"  ⚠ skip missing native source: {src_path}")
            continue
        n_in_src = 0
        with src_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n_native_records += 1
                n_in_src += 1
                t = tokens(extract_title(r))
                d = tokens(first_paragraph(query_body(r), max_chars=1500))
                if t:
                    native_title_toks.append(t)
                if d:
                    native_desc_toks.append(d)
                for m in DEF_RE.finditer(r.get("code") or ""):
                    native_def_names.add(m.group(1))
        n_native_by_src[src_path.parent.name + "/" + src_path.name] = n_in_src
    print(f"  per-source: {n_native_by_src}")

    print(
        f"Loaded {n_native_records} native records "
        f"({len(native_title_toks)} titles, {len(native_desc_toks)} descriptions, "
        f"{len(native_def_names)} unique def names)"
    )

    in_recs: list[dict] = []
    with SFT_MAIN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                in_recs.append(json.loads(line))

    # Tokenise generated records once each; many variants share the same
    # parent_qid → cache by parent_qid for a 5× speedup.
    parent_cache: dict[str, tuple[set[str], set[str]]] = {}

    def gen_tokens(rec: dict) -> tuple[set[str], set[str]]:
        key = rec.get("parent_qid") or rec.get("id")
        if key in parent_cache:
            return parent_cache[key]
        t = tokens(extract_title(rec))
        d = tokens(first_paragraph(query_body(rec), max_chars=1500))
        parent_cache[key] = (t, d)
        return t, d

    survivors: list[dict] = []
    rejected: list[dict] = []
    title_sim_dist: Counter = Counter()
    desc_sim_dist: Counter = Counter()
    def_collisions: list[dict] = []

    for rec in in_recs:
        gen_title_toks, gen_desc_toks = gen_tokens(rec)
        gen_method = rec.get("method_name") or ""

        best_title_sim = 0.0
        for nt in native_title_toks:
            s = jaccard(gen_title_toks, nt)
            if s > best_title_sim:
                best_title_sim = s
                if best_title_sim >= 1.0:
                    break

        best_desc_sim = 0.0
        for nd in native_desc_toks:
            s = jaccard(gen_desc_toks, nd)
            if s > best_desc_sim:
                best_desc_sim = s
                if best_desc_sim >= 1.0:
                    break

        bucket_t = round(best_title_sim, 1)
        bucket_d = round(best_desc_sim, 1)
        title_sim_dist[bucket_t] += 1
        desc_sim_dist[bucket_d] += 1

        def_collision = gen_method in native_def_names
        if def_collision:
            def_collisions.append({
                "id": rec["id"], "method_name": gen_method,
                "title_sim": round(best_title_sim, 3),
                "desc_sim": round(best_desc_sim, 3),
            })

        rec_aug = dict(rec)
        rec_aug["leakage_audit"] = {
            "title_jaccard_max":  round(best_title_sim, 3),
            "desc_jaccard_max":   round(best_desc_sim, 3),
            "def_name_collision": def_collision,
        }

        rejected_reason = None
        if best_title_sim >= args.sim_cut or best_desc_sim >= args.sim_cut:
            rejected_reason = (
                f"sim_cut breach (title={best_title_sim:.2f} desc={best_desc_sim:.2f} "
                f"≥ {args.sim_cut})"
            )
        elif def_collision and args.def_collision_mode == "drop":
            rejected_reason = f"def_name collides with native: {gen_method}"

        if rejected_reason:
            rec_aug["leakage_audit"]["rejected_reason"] = rejected_reason
            rejected.append(rec_aug)
        else:
            survivors.append(rec_aug)

    with SFT_CLEAN.open("w", encoding="utf-8") as f:
        for r in survivors:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "ts":             datetime.now().isoformat(timespec="seconds"),
        "method":         "token-jaccard",
        "n_in":           len(in_recs),
        "n_clean":        len(survivors),
        "n_rejected":     len(rejected),
        "sim_cut":        args.sim_cut,
        "title_jaccard_dist": dict(sorted(title_sim_dist.items())),
        "desc_jaccard_dist":  dict(sorted(desc_sim_dist.items())),
        "n_def_name_collisions": len(def_collisions),
        "def_collision_examples": def_collisions[:10],
    }
    AUDIT_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write rejected jsonl for inspection
    (BASE / "sft" / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
