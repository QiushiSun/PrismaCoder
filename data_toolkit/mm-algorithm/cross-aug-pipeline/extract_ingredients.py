"""
Cross-source ingredient pool — biased toward HARDER algorithmic content.

Output (ingredients.json):
  difficulties        target distribution (5 / 25 / 70 — easy / medium / hard)
  algo_tags           extended algorithm tags including advanced ones
  struct_tags         data-structure tags including segment trees, fenwick, etc.
  visual_aspects      same 5 + 1 advanced (algorithm-state visualisation)
  domains             scenario nudges drawn from CP / production-engineering

Usage:
  conda run -n vpb python extract_ingredients.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "ingredients.json"

# ── canonical pools ──────────────────────────────────────────────────────────
ALGO_TAGS_CANON: list[str] = [
    # core
    "Dynamic Programming",
    "Greedy",
    "Two Pointers",
    "Binary Search",
    "Backtracking",
    "Sliding Window",
    "Bit Manipulation",
    "Bitmask DP",
    "Prefix Sum",
    "Math",
    "Recursion",
    # advanced
    "Topological Sort",
    "Strongly Connected Components",
    "Articulation Points / Bridges",
    "Shortest Path (Dijkstra/SPFA)",
    "Minimum Spanning Tree",
    "Network Flow",
    "Tree DP",
    "Digit DP",
    "Game Theory / Sprague-Grundy",
    "Convex Hull",
    "Sweep Line",
    "KMP / Z-algorithm",
    "Suffix Array / Automaton",
    "FFT / NTT",
    "Matrix Exponentiation",
    "Segment Tree (Lazy / Persistent)",
    "Fenwick Tree (BIT)",
    "Sparse Table / RMQ",
    "Heavy-Light Decomposition",
    "LCA",
    "Mo's Algorithm",
    "Offline Queries",
    "Modular Inverse / CRT",
    "Combinatorics (Stirling/Catalan/etc.)",
    "Inclusion-Exclusion",
]

STRUCT_TAGS_CANON: list[str] = [
    "Array",
    "String",
    "Tree (rooted)",
    "Binary Tree",
    "Graph (directed)",
    "Graph (undirected)",
    "DAG",
    "Linked List",
    "Matrix / Grid",
    "Hash Table",
    "Stack (incl. monotonic)",
    "Queue (incl. monotonic / deque)",
    "Heap (Priority Queue)",
    "Trie",
    "Disjoint Set Union",
    "Segment Tree",
    "Fenwick Tree",
    "Sparse Table",
]

VISUAL_ASPECTS: list[str] = [
    "tree-diagram (rooted, labeled nodes, possibly weighted edges)",
    "graph-diagram (directed/undirected, weighted/unweighted, with node labels)",
    "grid-matrix (2-D grid; values, walls, color coding, robot path)",
    "geometric-2d (points, segments, polygons, half-planes on a coordinate axis)",
    "interval-timeline (intervals on a number line; events, queries, sweep)",
    "algo-state (visualisation of an algorithm step: e.g. DP table, queue snapshot)",
]

# Cross-source domain hints — competitive programming + real-world engineering
DOMAINS_CROSS: list[str] = [
    "logistics-network: package routing across hub-and-spoke graph",
    "video-game pathfinding with destructible terrain",
    "circuit-board placement & wiring on a grid",
    "shipping container stack with crane operations",
    "bank-fraud detection on directed transaction graph",
    "compiler basic-block control-flow graph",
    "city sewer-pipe drain-flow network",
    "traffic-light optimisation across an intersection grid",
    "DNA / protein sequence alignment with gap penalties",
    "warehouse robot multi-arm scheduling",
    "stadium seating Catalan-style domino fill",
    "molecular-dynamics particle collision",
    "satellite constellation coverage scheduling",
    "subway timetable adversarial routing",
    "elevator dispatch under bursty queue traffic",
    "concert lighting cue-sheet timeline",
    "cybersecurity attack-surface graph analysis",
    "supply-chain perishable cargo with spoilage timers",
    "distributed-ledger consensus tree",
    "medical-trial patient-cohort partition",
    "stock-market triangular-arbitrage cycle detection",
    "music orchestration polyphonic interval merge",
    "weather forecast cellular-automaton evolution",
    "factory robot assembly DAG with reservation",
    "solar farm panel orientation under shadow",
    "racetrack lap-time DP with tyre wear",
    "lift-and-shift cloud migration cost DP",
    "language-model tokenisation suffix-automaton",
    "subway disaster-evacuation BFS with capacity",
    "puzzle Rubik-cube-like state-space BFS",
    "bridges-of-Königsberg revival as Eulerian path",
    "social-network influence-propagation BFS",
    "mountain trail elevation prefix-sum",
    "delivery drone path with battery refuelling",
    "casino dealer card-shuffling permutation cycle",
]

# Difficulty target distribution (sums to 1.0)
DIFFICULTY_TARGETS: dict[str, float] = {
    "Easy":   0.05,
    "Medium": 0.25,
    "Hard":   0.70,
}

# Number of examples in each problem (1–3)
N_EXAMPLES_TARGETS: dict[int, float] = {
    1: 0.10,
    2: 0.70,
    3: 0.20,
}


def main() -> None:
    out = {
        "difficulties":      DIFFICULTY_TARGETS,
        "n_examples":        N_EXAMPLES_TARGETS,
        "algo_tags":         ALGO_TAGS_CANON,
        "struct_tags":       STRUCT_TAGS_CANON,
        "visual_aspects":    VISUAL_ASPECTS,
        "domains":           DOMAINS_CROSS,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.name}")
    print(f"  difficulties:    {DIFFICULTY_TARGETS}")
    print(f"  algo_tags:       {len(ALGO_TAGS_CANON)} canonical (incl advanced)")
    print(f"  struct_tags:     {len(STRUCT_TAGS_CANON)}")
    print(f"  visual_aspects:  {len(VISUAL_ASPECTS)} (incl algo-state)")
    print(f"  domains:         {len(DOMAINS_CROSS)} hints")


if __name__ == "__main__":
    main()
