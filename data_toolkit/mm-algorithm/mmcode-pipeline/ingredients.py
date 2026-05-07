"""Extract and build structured ingredient pools for MMCode compositional generation.

Reads mmcode_full.jsonl (train split only), extracts algorithm tags, visual context
types, difficulty levels, and source styles, then writes ingredients.json.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUTPUT_PATH = BASE / "ingredients.json"

# ── Load train set and extract distributions ─────────────────────────────────

def _resolve_seed_path(filename: str) -> Path:
    """Look for `filename` first at pipeline root, fall back to seeds_sample/."""
    cand_root = BASE / filename
    if cand_root.exists():
        return cand_root
    cand_sample = BASE / "seeds_sample" / filename
    if cand_sample.exists():
        return cand_sample
    return cand_root


def _load_train_ids_to_skip() -> set[str]:
    """Load test problem IDs so we only use train data for ingredients."""
    test_ids: set[str] = set()
    test_path = _resolve_seed_path("mmcode_test.jsonl")
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                test_ids.add(obj.get("problem_id", ""))
    return test_ids


def _parse_tags_field(raw) -> list[str]:
    """Parse the 'tags' field which may be a string repr of a list or an actual list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        if raw.strip():
            return [raw.strip()]
    return []


def extract_from_train() -> dict:
    """Extract ingredient pools from the MMCode train set."""
    test_ids = _load_train_ids_to_skip()

    algo_tags: Counter[str] = Counter()
    img_tags: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    sources: Counter[str] = Counter()

    full_path = _resolve_seed_path("mmcode_full.jsonl")
    with open(full_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("problem_id", "") in test_ids:
                continue

            for t in _parse_tags_field(obj.get("tags")):
                algo_tags[t] += 1

            for t in (obj.get("image_tags") or []):
                img_tags[t] += 1

            diff = obj.get("difficulty")
            difficulties[str(diff) if diff else "Unknown"] += 1
            sources[obj.get("source") or "unknown"] += 1

    return {
        "algo_tags": dict(algo_tags.most_common()),
        "img_tags": dict(img_tags.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "sources": dict(sources.most_common()),
    }


# ── Curated source style descriptions ────────────────────────────────────────

SOURCE_STYLES = {
    "codeforces": (
        "Competitive programming style: formal mathematical statement, tight constraints, "
        "often involves graph theory or DP, multiple test cases in one run"
    ),
    "aizu": (
        "ACM/ICPC style: precise input/output spec, algorithmic focus, "
        "moderate difficulty, clean problem structure"
    ),
    "leetcode": (
        "Interview-style: concise problem, examples with explanation, "
        "focus on data structures and algorithms, single function signature"
    ),
    "atcoder": (
        "Japanese competitive programming style: clean mathematical formulation, "
        "tight time limits, often combinatorial or number-theoretic"
    ),
    "hacker rank": (
        "Coding challenge style: progressive difficulty, story-based framing, "
        "detailed input format specification"
    ),
    "codechef": (
        "Long challenge style: detailed problem story, multiple subtasks with "
        "varying constraints, emphasis on optimization"
    ),
    "open kattis": (
        "University contest style: classic algorithmic problems, "
        "clear I/O specification, focus on correctness"
    ),
    "project euler": (
        "Mathematical puzzle style: elegant number theory or combinatorics, "
        "single numeric answer, requires mathematical insight"
    ),
    "geekforgeeks": (
        "Practice problem style: expected time/space complexity given, "
        "common DS&A patterns, educational focus"
    ),
    "codewars": (
        "Kata-style: creative problem framing, emphasis on clean code, "
        "varied difficulty (kyu system)"
    ),
}

# ── Difficulty levels ─────────────────────────────────────────────────────────

DIFFICULTY_LEVELS = {
    "EASY": "Simple problem solvable with basic loops, conditions, or a single well-known algorithm",
    "MEDIUM": "Requires combining 1-2 algorithms or data structures, moderate implementation effort",
    "MEDIUM_HARD": "Needs non-trivial algorithmic insight, possibly graph/DP with careful state design",
    "HARD": "Complex multi-step algorithm, advanced data structures, or mathematical reasoning required",
    "VERY_HARD": "Expert-level: novel algorithmic idea, multiple advanced techniques combined, tight constraints",
}

# ── Build final ingredients ──────────────────────────────────────────────────

def build_ingredients() -> dict:
    raw = extract_from_train()

    algo_tag_list = [t for t, c in sorted(raw["algo_tags"].items(), key=lambda x: -x[1]) if c >= 3]
    img_tag_list = [t for t in sorted(raw["img_tags"].keys(), key=lambda x: -raw["img_tags"][x])]
    source_list = list(raw["sources"].keys())

    return {
        "algorithm_tags": algo_tag_list,
        "visual_context_types": img_tag_list,
        "difficulty_levels": DIFFICULTY_LEVELS,
        "source_styles": SOURCE_STYLES,
        "source_list": source_list,
        "_raw_counts": {
            "algo_tags": raw["algo_tags"],
            "img_tags": raw["img_tags"],
            "difficulties": raw["difficulties"],
            "sources": raw["sources"],
        },
    }


def main() -> None:
    ingredients = build_ingredients()
    OUTPUT_PATH.write_text(
        json.dumps(ingredients, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Algorithm tags: {len(ingredients['algorithm_tags'])}")
    for t in ingredients["algorithm_tags"][:15]:
        ct = ingredients["_raw_counts"]["algo_tags"].get(t, 0)
        print(f"  {t}: {ct}")
    print(f"  ... ({len(ingredients['algorithm_tags'])} total)")

    print(f"\nVisual context types: {len(ingredients['visual_context_types'])}")
    for t in ingredients["visual_context_types"]:
        ct = ingredients["_raw_counts"]["img_tags"].get(t, 0)
        print(f"  {t}: {ct}")

    print(f"\nDifficulty levels: {len(ingredients['difficulty_levels'])}")
    print(f"Source styles: {len(ingredients['source_styles'])}")
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
