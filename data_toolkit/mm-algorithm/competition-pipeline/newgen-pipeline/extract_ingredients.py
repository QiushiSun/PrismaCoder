"""
Build the ingredient pools used by recipe sampling.

Reads:
  ../sft/leetcode_records.jsonl   (811 multimodal LeetCode SFT records)

Outputs (to rollout/ingredients_lc.json):
  difficulties        {Easy, Medium, Hard} → count
  algo_tags           17 canonical algorithmic flavors → count
  struct_tags         12 canonical data-structure tags → count
  visual_aspects      5 canonical visual contexts → count
  domains             curated LeetCode-style scenario hints (handpicked)

Usage:
  conda run -n vpb python extract_ingredients_lc.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
LC_SRC = BASE / "seeds_sample" / "leetcode_records.jsonl"
OUT = BASE / "ingredients.json"

# ── canonical pools (we coerce upstream tags into these buckets) ──────────────
ALGO_TAGS_CANON: list[str] = [
    "Dynamic Programming",
    "Greedy",
    "Two Pointers",
    "Binary Search",
    "Backtracking",
    "Sliding Window",
    "Bit Manipulation",
    "Bitmask",
    "Topological Sort",
    "Union Find",
    "Shortest Path",
    "Prefix Sum",
    "Stack-Monotonic",
    "Heap (Priority Queue)",
    "Trie",
    "Recursion",
    "Math",
]

STRUCT_TAGS_CANON: list[str] = [
    "Array",
    "String",
    "Tree",
    "Binary Tree",
    "Graph",
    "Linked List",
    "Matrix",
    "Hash Table",
    "Stack",
    "Queue",
    "Heap (Priority Queue)",
    "Binary Search Tree",
]

VISUAL_ASPECTS: list[str] = [
    "tree-diagram (rooted, labeled nodes drawn vertically)",
    "graph-diagram (small directed/undirected graph with labeled nodes)",
    "grid-matrix (2-D grid with values or color-coded cells)",
    "linked-list (chained boxes connected by arrows)",
    "geometric-2d (points, segments, polygons on a coordinate plane)",
]

# Curated domain-flavor hints.  These are *style* / *scenario* nudges; they push
# the LLM away from blindly cloning canonical leetcode topics.
DOMAINS_LC: list[str] = [
    "warehouse robot path on a grid",
    "social-network friendship graph traversal",
    "office floor-plan adjacency reasoning",
    "subway / metro line rail-routing",
    "cellular tower frequency assignment",
    "puzzle island flood-fill",
    "DNA base-pair string transformation",
    "chess-piece reachability count",
    "stock-trade interval scheduling",
    "hotel room availability calendar",
    "river-crossing token transfer",
    "circuit board wire-routing on grid",
    "library book Dewey-decimal tree",
    "delivery drone Manhattan path",
    "amusement park ride queue priorities",
    "language tree dialect lineage",
    "music chord progression states",
    "weather radar grid wind-flow",
    "sports tournament bracket elimination",
    "garden plant adjacency diagrams",
    "supply-chain warehouse network",
    "hospital triage priority queue",
    "school class course-prerequisite DAG",
    "city traffic light synchronization",
    "smartphone touchpoint heatmap",
]


# ── tag normalisation ────────────────────────────────────────────────────────
def _bucket_tag(t: str) -> tuple[str | None, str | None]:
    """Return (algo_canon, struct_canon) bucket for a leetcode tag."""
    algo: str | None = None
    struct: str | None = None
    if t in ALGO_TAGS_CANON:
        algo = t
    if t in STRUCT_TAGS_CANON:
        struct = t
    # Loose mapping for near-synonyms
    if t == "DFS" or t == "Depth-First Search":
        algo = "Recursion"
        struct = struct or None
    if t == "BFS" or t == "Breadth-First Search":
        algo = "Shortest Path"  # most BFS leetcode use it for shortest path
    if t == "Memoization":
        algo = "Dynamic Programming"
    return algo, struct


def main() -> None:
    if not LC_SRC.exists():
        raise SystemExit(f"Missing source: {LC_SRC}")

    diffs: Counter = Counter()
    algo_counts: Counter = Counter()
    struct_counts: Counter = Counter()
    visual_aspects_counts: Counter = Counter({v: 0 for v in VISUAL_ASPECTS})
    n_records = 0

    with LC_SRC.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_records += 1
            d = (r.get("difficulty") or "Unknown").strip()
            diffs[d] += 1
            for t in r.get("tags") or []:
                algo, struct = _bucket_tag(t)
                if algo:
                    algo_counts[algo] += 1
                if struct:
                    struct_counts[struct] += 1

    # Make sure every canonical tag has at least 1 entry so sampling never
    # returns an empty bucket.
    for t in ALGO_TAGS_CANON:
        algo_counts.setdefault(t, 0)
    for t in STRUCT_TAGS_CANON:
        struct_counts.setdefault(t, 0)

    out = {
        "n_records":      n_records,
        "difficulties":   dict(diffs),
        "algo_tags":      dict(algo_counts),
        "struct_tags":    dict(struct_counts),
        "visual_aspects": dict(visual_aspects_counts),
        "domains":        DOMAINS_LC,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(BASE)}  ({n_records} source records)")
    print(f"  algo_tags:   {len(algo_counts)} canonical")
    print(f"  struct_tags: {len(struct_counts)} canonical")
    print(f"  visual:      {len(VISUAL_ASPECTS)} aspects")
    print(f"  domains:     {len(DOMAINS_LC)} hints")


if __name__ == "__main__":
    main()
