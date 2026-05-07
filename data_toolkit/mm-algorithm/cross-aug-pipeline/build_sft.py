"""
Package verified variants from `variants/records.jsonl` into SFT records.

For each variant with status == "ok", build a record with:
  • id:      "<qid>_v<seed>"
  • images:  list of paths (one per example)
  • query:   <title> + filled description (with <image> tokens) + signature
             + "Provide a complete Python solution."
  • code:    bare class Solution source extracted from the template
  • response: ```python\n<code>\n```

Output:
  sft/train.sft.jsonl                 (all variants)
  sft/train.sft.hard.jsonl            (n_test_cases >= 5)
  sft/images/<qid>_v<seed>_<k>.png    (copied from variants/images/)
  sft/build_summary.json

Usage:
  conda run -n vpb python build_sft_set_newgen_lc.py
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
VARIANTS_DIR = BASE / "variants"
TEMPLATES_DIR = BASE / "templates"
SFT_DIR = BASE / "sft"
SFT_IMG_DIR = SFT_DIR / "images"

# Re-use solution extractor
sys.path.insert(0, str(BASE))
from run_templates import extract_solution_source  # noqa: E402


def strip_python_comments(src: str) -> str:
    """Remove `#` comments. Docstrings + string literals preserved."""
    if not src or not src.strip():
        return src
    src_norm = src if src.endswith("\n") else src + "\n"
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src_norm).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return src
    comments_by_line: dict[int, list[tuple[int, int]]] = {}
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            comments_by_line.setdefault(tok.start[0], []).append(
                (tok.start[1], tok.end[1])
            )
    out: list[str] = []
    for i, line in enumerate(src_norm.splitlines(keepends=True), start=1):
        if i in comments_by_line:
            for s, e in sorted(comments_by_line[i], reverse=True):
                line = line[:s] + line[e:]
            line = line.rstrip()
            if not line.strip():
                continue
            line += "\n"
        out.append(line)
    import re
    cleaned = "".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip() + "\n"


def build_query(title: str, description: str, signature: str) -> str:
    """Compose the user-side prompt.

    The LLM-rendered DESCRIPTION already starts with `# Title` ~half the time,
    so we only prepend our own header when the description does NOT start with
    `#` (avoids the duplicate-title repro you'd otherwise see).
    """
    desc = description.strip()
    head = "" if desc.startswith("#") else f"# {title.strip()}\n\n"
    return (
        f"{head}{desc}\n\n"
        f"**Function signature:**\n\n"
        f"```python\n"
        f"{signature.strip()}\n"
        f"```\n\n"
        f"Provide a complete Python solution."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-min-tests", type=int, default=5,
                    help="threshold for the .hard subset (n_test_cases >= this)")
    args = ap.parse_args()

    if not (VARIANTS_DIR / "records.jsonl").exists():
        raise SystemExit("variants/records.jsonl missing — run run_templates_lc.py first")

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    SFT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    audit_by_qid: dict[str, dict] = {}
    for p in (BASE / "audit").glob("*.json"):
        try:
            audit_by_qid[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Template-level recipe (records.jsonl) carries inspiration_qid/source —
    # variant rendering drops these, so we re-attach them here by qid.
    insp_by_qid: dict[str, dict] = {}
    rec_path = BASE / "records.jsonl"
    if rec_path.exists():
        with rec_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tr = json.loads(line)
                qid = tr.get("qid")
                if qid:
                    insp_by_qid[qid] = {
                        "inspiration_qid":    tr.get("inspiration_qid"),
                        "inspiration_source": tr.get("inspiration_source"),
                    }

    sft_records: list[dict] = []
    n_skipped = 0
    n_total = 0
    method_name_counts: Counter = Counter()
    diff_counts: Counter = Counter()
    n_test_dist: Counter = Counter()
    n_images_dist: Counter = Counter()

    with (VARIANTS_DIR / "records.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            n_total += 1
            if v.get("status") != "ok":
                n_skipped += 1
                continue

            qid = v["qid"]
            audit = audit_by_qid.get(qid, {})
            tpl_path = TEMPLATES_DIR / f"{qid}.py"
            if not tpl_path.exists():
                n_skipped += 1
                continue

            try:
                solution_src_raw = extract_solution_source(tpl_path.read_text(encoding="utf-8"))
            except Exception:
                n_skipped += 1
                continue
            solution_src = strip_python_comments(solution_src_raw).rstrip()

            # Copy images
            dst_imgs: list[str] = []
            for src_rel in v.get("image_paths") or []:
                src_abs = VARIANTS_DIR / src_rel
                if not src_abs.exists():
                    n_skipped += 1
                    break
                dst_name = src_abs.name
                dst_abs = SFT_IMG_DIR / dst_name
                if not dst_abs.exists():
                    shutil.copy2(src_abs, dst_abs)
                dst_imgs.append(f"images/{dst_name}")
            else:
                # all images copied
                pass
            if not dst_imgs or len(dst_imgs) != len(v.get("image_paths") or []):
                n_skipped += 1
                continue

            query = build_query(v["title"], v["description_filled"], v["signature"])

            rec = {
                "id":                  v["variant_id"],
                "parent_qid":          qid,
                "seed":                v["seed"],
                "image":               dst_imgs[0],
                "images":              dst_imgs,
                "query":               query,
                "code":                solution_src,
                "response":            f"```python\n{solution_src}\n```",
                "title":               v["title"],
                "method_name":         v["method_name"],
                "function_signature":  v["signature"],
                "difficulty":          audit.get("difficulty"),
                "algo_tags":           audit.get("algo_tags") or [],
                "struct_tags":         audit.get("struct_tags") or [],
                "visual_aspect":       audit.get("visual_aspect"),
                "domain":              audit.get("domain"),
                "n_examples":          v["n_examples"],
                "n_test_cases":        v["n_test_cases"],
                "test_cases":          v["test_cases"],
                "in_proc_verification": True,
                "sandbox_check":       v.get("sandbox_check"),
                "icl_qids":            audit.get("icl_qids") or [],
                "inspiration_qid":     insp_by_qid.get(qid, {}).get("inspiration_qid"),
                "inspiration_source":  insp_by_qid.get(qid, {}).get("inspiration_source"),
            }
            sft_records.append(rec)
            method_name_counts[v["method_name"]] += 1
            diff_counts[audit.get("difficulty") or "?"] += 1
            n_test_dist[v["n_test_cases"]] += 1
            n_images_dist[len(dst_imgs)] += 1

    out_main = SFT_DIR / "train.sft.jsonl"
    with out_main.open("w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hard = [r for r in sft_records if r["n_test_cases"] >= args.hard_min_tests]
    out_hard = SFT_DIR / "train.sft.hard.jsonl"
    with out_hard.open("w", encoding="utf-8") as f:
        for r in hard:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    insp_src_counter = Counter(
        r.get("inspiration_source") or "<none>" for r in sft_records
    )

    summary = {
        "ts":              datetime.now().isoformat(timespec="seconds"),
        "n_total_variants": n_total,
        "n_sft":           len(sft_records),
        "n_skipped":       n_skipped,
        "n_hard":          len(hard),
        "by_difficulty":   dict(diff_counts),
        "by_n_test_cases": dict(sorted(n_test_dist.items())),
        "by_n_images":     dict(sorted(n_images_dist.items())),
        "unique_method_names": len(method_name_counts),
        "by_inspiration_source": dict(insp_src_counter),
        "out_main":        str(out_main.relative_to(BASE)),
        "out_hard":        str(out_hard.relative_to(BASE)),
    }
    (SFT_DIR / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
