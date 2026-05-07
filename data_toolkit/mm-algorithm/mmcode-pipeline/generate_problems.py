"""
MMCode data augmentation pipeline — compositional recipe-based generation
of (problem + solution + test_cases) triples.

Reuses infrastructure from ../generate_queries.py (call_llm, retry, APILogger,
tqdm, resume, --output-dir).

Usage:
  # Dry run
  OPENAI_API_KEY="sk-..." conda run -n vpb python generate_problems.py --dry-run

  # Generate ~2K problems
  OPENAI_API_KEY="sk-..." conda run -n vpb python generate_problems.py \
    --batches 400 --problems-per-batch 5 --output-dir rollout-MMDD-model-mmcode

  # Resume
  ... --resume
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import APILogger

# ── constants ────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent
API_LOG_DIR = BASE / "api-logs"

# ── logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mmcode_aug")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"generation_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
    return logger

# ── data loading ─────────────────────────────────────────────────────────────

def _parse_tags_field(raw) -> list[str]:
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


def _build_example_record(obj: dict, *, require_solution: bool) -> dict | None:
    """Convert a raw MMCode record into the lean example schema used as ICL."""
    pid = obj.get("problem_id", "")

    sols_raw = obj.get("solutions") or "[]"
    try:
        sols = json.loads(sols_raw) if isinstance(sols_raw, str) else (sols_raw or [])
    except json.JSONDecodeError:
        sols = []
    if require_solution and not sols:
        return None

    io_raw = obj.get("input_output") or "{}"
    try:
        io = json.loads(io_raw) if isinstance(io_raw, str) else (io_raw or {})
    except json.JSONDecodeError:
        io = {}

    question = obj.get("question", "")
    question_clean = re.sub(r"!\[image\]\([^)]*\)", "[diagram]", question)

    inputs = [str(x) for x in (io.get("inputs") or [])[:3]]
    outputs = [str(x) for x in (io.get("outputs") or [])[:3]]

    return {
        "problem_id": pid,
        "question_clean": question_clean,
        "solution": sols[0] if sols else "",
        "io_inputs": inputs,
        "io_outputs": outputs,
        "tags": _parse_tags_field(obj.get("tags")),
        "image_tags": obj.get("image_tags") or [],
        "difficulty": str(obj.get("difficulty") or "Unknown"),
        "source": obj.get("source") or "unknown",
    }


def _resolve_seed_path(filename: str) -> Path:
    """Look for `filename` first at the pipeline root, then under seeds_sample/.

    This lets users either drop their own full-corpus mmcode_full.jsonl /
    mmcode_test.jsonl directly into the pipeline directory, or fall back to
    the bundled seeds_sample/ for smoke runs.
    """
    cand_root = BASE / filename
    if cand_root.exists():
        return cand_root
    cand_sample = BASE / "seeds_sample" / filename
    if cand_sample.exists():
        return cand_sample
    return cand_root  # caller will get a clear FileNotFoundError


def load_train_examples() -> list[dict]:
    """Load train examples (excluding test-set ids) that have at least one solution."""
    test_ids: set[str] = set()
    test_path = _resolve_seed_path("mmcode_test.jsonl")
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                test_ids.add(json.loads(line).get("problem_id", ""))

    examples = []
    full_path = _resolve_seed_path("mmcode_full.jsonl")
    with open(full_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("problem_id", "") in test_ids:
                continue
            rec = _build_example_record(obj, require_solution=True)
            if rec is not None:
                examples.append(rec)
    return examples


def load_test_examples() -> list[dict]:
    """Load ALL test examples (solutions optional). Used as ICL for test-anchored mode."""
    examples = []
    test_path = _resolve_seed_path("mmcode_test.jsonl")
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rec = _build_example_record(obj, require_solution=False)
            if rec is not None:
                examples.append(rec)
    return examples


def load_ingredients() -> dict:
    path = BASE / "ingredients.json"
    if not path.exists():
        raise FileNotFoundError("ingredients.json not found — run ingredients.py first")
    return json.loads(path.read_text(encoding="utf-8"))


# ── recipe sampling ──────────────────────────────────────────────────────────

def sample_recipe(ingredients: dict, rng: random.Random) -> dict:
    algo_tags = ingredients["algorithm_tags"]
    img_tags = ingredients["visual_context_types"]
    diff_levels = list(ingredients["difficulty_levels"].keys())
    source_list = ingredients["source_list"]

    n_algo = rng.randint(1, 3)
    chosen_algo = rng.sample(algo_tags, min(n_algo, len(algo_tags)))

    diff_weights = [0.15, 0.25, 0.25, 0.20, 0.15]
    chosen_diff = rng.choices(diff_levels, weights=diff_weights[:len(diff_levels)], k=1)[0]

    return {
        "algorithm_tags": chosen_algo,
        "visual_context": rng.choice(img_tags),
        "difficulty": chosen_diff,
        "source_style": rng.choice(source_list),
    }


def match_icl_examples(
    examples: list[dict],
    recipe: dict,
    rng: random.Random,
    n: int = 2,
) -> list[dict]:
    """Pick ICL examples whose algorithm tags overlap with the recipe."""
    target_tags = set(t.lower() for t in recipe["algorithm_tags"])

    scored = []
    for ex in examples:
        ex_tags = set(t.lower() for t in ex["tags"])
        overlap = len(target_tags & ex_tags)
        diff_match = 1 if ex["difficulty"] == recipe["difficulty"] else 0
        scored.append((overlap * 3 + diff_match, ex))

    scored.sort(key=lambda x: -x[0])
    pool = [s for _, s in scored[:max(n * 4, 15)]]
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


# ── HARD MODE ────────────────────────────────────────────────────────────────
# An opt-in (`--difficulty-mode hard`) variant that:
#   1. Forces recipe.difficulty ∈ {HARD, VERY_HARD} (anchors get up-leveled).
#   2. Forces 2-4 algorithm tags (multi-paradigm).
#   3. Injects one "advanced_technique" hint from the curated pool below.
#   4. Restricts the ICL pool to MEDIUM_HARD+ examples so the LLM never
#      anchors on an EASY/MEDIUM solution as the demo.
#   5. Switches to SYSTEM_PROMPT_HARD which removes the "concise" cap and
#      explicitly targets ICPC/IOI-tier difficulty with tight constraints.
#   6. Uses a larger token budget and smaller per-batch count so each problem
#      has room to breathe.

ADVANCED_TECHNIQUES: list[str] = [
    # Tree / segment-tree heavyweights
    "Heavy-light decomposition",
    "Centroid decomposition",
    "Link-cut tree (Splay-based)",
    "Persistent segment tree",
    "Segment tree beats",
    "Wavelet tree",
    "Euler-tour with segment tree",
    "Sparse table on subtree-flattened tree",
    # DP optimizations
    "Convex hull trick (Li Chao tree)",
    "Divide and conquer DP optimization",
    "Knuth optimization",
    "SOS DP / subset sum DP",
    "DP on broken profile / bitmask DP",
    "Aliens trick (Lagrangian relaxation on convex DP)",
    "Slope trick",
    # Graph theory
    "2-SAT (implication graph + SCC)",
    "Maximum flow (Dinic) / min-cut",
    "Min-cost max-flow (SSP / cycle cancelling)",
    "Bipartite matching (Hopcroft-Karp)",
    "Bridges + biconnected components (Tarjan)",
    "Strongly connected components (Tarjan / Kosaraju)",
    "Articulation points + block-cut tree",
    "Eulerian path / circuit (Hierholzer)",
    # String algorithms
    "Suffix automaton",
    "Suffix array + LCP (Kasai)",
    "Aho-Corasick",
    "Z-function / KMP failure-function tricks",
    "Manacher's algorithm",
    "Hashing with double moduli",
    # Number theory / combinatorics
    "Number theoretic transform (NTT)",
    "Fast Walsh–Hadamard transform",
    "Möbius inversion / Dirichlet convolution",
    "Burnside / Pólya enumeration",
    "Pollard rho factorization + Miller-Rabin",
    "Chinese Remainder Theorem (general)",
    "Lucas' theorem with prime power",
    "Generating functions / formal power series",
    # Geometry
    "Half-plane intersection",
    "Convex hull (Andrew monotone chain) + rotating calipers",
    "Voronoi diagram / Delaunay triangulation",
    "Plane sweep with multiset",
    "Smallest enclosing circle (Welzl)",
    # Sqrt / offline
    "Mo's algorithm (with rollback)",
    "Sqrt decomposition (block + buckets)",
    "Offline DSU rollback",
    # Game theory / advanced misc
    "Sprague-Grundy with mex precomputation",
    "Hackenbush / surreal numbers",
    "Dilworth's theorem reduction",
    "Matroid intersection (greedy)",
]

# Sources whose problems tend to be Hard / Very Hard.
HARD_SOURCE_POOL: list[str] = [
    "codeforces",
    "atcoder",
    "open kattis",
    "codechef",
    "project euler",
    "geekforgeeks",
]


def filter_hard_examples(examples: list[dict]) -> list[dict]:
    """Keep only ICL candidates with HARD/VERY_HARD/MEDIUM_HARD difficulty AND a solution.

    Falls back to all-examples-with-solution if filtering would empty the pool.
    """
    keep = {"HARD", "VERY_HARD", "MEDIUM_HARD"}
    hard = [e for e in examples if e["difficulty"].upper() in keep and e.get("solution")]
    if hard:
        return hard
    return [e for e in examples if e.get("solution")] or examples


def sample_recipe_hard(ingredients: dict, rng: random.Random) -> dict:
    """Hard-mode random recipe (no anchor): always HARD/VERY_HARD, 2-4 algo tags,
    one advanced technique, source from the hard-source pool."""
    algo_pool = ingredients["algorithm_tags"]
    img_tags = ingredients["visual_context_types"]

    n_tags = rng.randint(2, 4)
    chosen_algo = rng.sample(algo_pool, min(n_tags, len(algo_pool)))
    difficulty = rng.choices(["HARD", "VERY_HARD"], weights=[0.4, 0.6], k=1)[0]
    src_pool = [s for s in HARD_SOURCE_POOL if s in ingredients["source_list"]] or ingredients["source_list"]

    return {
        "algorithm_tags": chosen_algo,
        "visual_context": rng.choice(img_tags),
        "difficulty": difficulty,
        "source_style": rng.choice(src_pool),
        "advanced_technique": rng.choice(ADVANCED_TECHNIQUES),
    }


def build_test_anchored_recipe_hard(
    anchor: dict,
    ingredients: dict,
    rng: random.Random,
) -> dict:
    """Hard-mode anchored recipe: keep visual context from anchor for diversity,
    but FORCE difficulty up to HARD/VERY_HARD and grow algo_tags to 2-4."""
    algo_pool = ingredients["algorithm_tags"]
    img_pool = ingredients["visual_context_types"]
    src_pool = [s for s in HARD_SOURCE_POOL if s in ingredients["source_list"]] or ingredients["source_list"]

    base_tags = list(anchor.get("tags") or [])[:3]
    target_n = rng.randint(2, 4)
    if len(base_tags) < target_n and algo_pool:
        extras_pool = [t for t in algo_pool if t not in base_tags]
        rng.shuffle(extras_pool)
        base_tags = base_tags + extras_pool[: target_n - len(base_tags)]
    if not base_tags:
        base_tags = rng.sample(algo_pool, min(2, len(algo_pool)))

    img_tags = anchor.get("image_tags") or []
    visual_ctx = img_tags[0] if img_tags else rng.choice(img_pool)

    difficulty = rng.choices(["HARD", "VERY_HARD"], weights=[0.4, 0.6], k=1)[0]
    if (anchor.get("difficulty") or "").upper() == "VERY_HARD" and rng.random() < 0.7:
        difficulty = "VERY_HARD"

    if rng.random() < 0.4 and (anchor.get("source") in src_pool):
        source_style = anchor["source"]
    else:
        source_style = rng.choice(src_pool)

    return {
        "algorithm_tags": base_tags,
        "visual_context": visual_ctx,
        "difficulty": difficulty,
        "source_style": source_style,
        "advanced_technique": rng.choice(ADVANCED_TECHNIQUES),
        "anchor_test_id": anchor["problem_id"],
    }


# ── test-anchored mode ───────────────────────────────────────────────────────

def get_anchor_for_batch(
    test_examples: list[dict],
    batch_id: int,
) -> dict:
    """Deterministically pick the anchor test example for batch_id.

    Round-robin through `test_examples`; each round (covering all examples) uses
    a different shuffled order so consecutive rounds visit examples in different
    orders. Reproducible (depends only on batch_id and len(test_examples)).
    """
    n = len(test_examples)
    if n == 0:
        raise ValueError("test_examples is empty")
    round_idx = batch_id // n
    pos_in_round = batch_id % n
    indices = list(range(n))
    round_rng = random.Random(round_idx * 7919 + 31)
    round_rng.shuffle(indices)
    return test_examples[indices[pos_in_round]]


def build_test_anchored_recipe(
    anchor: dict,
    ingredients: dict,
    rng: random.Random,
) -> dict:
    """Derive a recipe from an anchor test example, with light perturbation
    so the LLM produces stylistic variations rather than near-duplicates.

    - algorithm_tags: anchor's tags (capped at 3); fallback to random pool if missing.
      With prob 0.4, append one extra random tag for diversity.
    - visual_context: anchor's first image_tag (always present in test set).
    - difficulty: anchor's if known; else weighted random. With prob 0.3, shift +/-1 step.
    - source_style: 40% anchor's source, 60% random — to vary tone across rounds.
    """
    algo_pool = ingredients["algorithm_tags"]
    src_pool = ingredients["source_list"]
    diff_levels = list(ingredients["difficulty_levels"].keys())
    img_pool = ingredients["visual_context_types"]

    base_tags = list(anchor.get("tags") or [])[:3]
    if not base_tags:
        base_tags = rng.sample(algo_pool, min(rng.randint(1, 2), len(algo_pool)))
    elif rng.random() < 0.4 and algo_pool:
        extra = rng.choice(algo_pool)
        if extra not in base_tags:
            base_tags = base_tags + [extra]

    img_tags = anchor.get("image_tags") or []
    visual_ctx = img_tags[0] if img_tags else rng.choice(img_pool)

    anchor_diff = anchor.get("difficulty") or "Unknown"
    if anchor_diff in diff_levels:
        difficulty = anchor_diff
        if rng.random() < 0.3:
            idx = diff_levels.index(difficulty)
            offset = rng.choice([-1, 1])
            difficulty = diff_levels[max(0, min(len(diff_levels) - 1, idx + offset))]
    else:
        weights = [0.15, 0.25, 0.25, 0.20, 0.15][:len(diff_levels)]
        difficulty = rng.choices(diff_levels, weights=weights, k=1)[0]

    if rng.random() < 0.4 and (anchor.get("source") in src_pool):
        source_style = anchor["source"]
    else:
        source_style = rng.choice(src_pool)

    return {
        "algorithm_tags": base_tags,
        "visual_context": visual_ctx,
        "difficulty": difficulty,
        "source_style": source_style,
        "anchor_test_id": anchor["problem_id"],
    }


# ── prompt construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert competitive programming problem designer. Your job is to \
create new, self-contained algorithmic problems with working Python solutions, \
test cases, AND an SVG illustration grounded in the first test case.

Key requirements:
- Problems must be SELF-CONTAINED: describe any visual context (graphs, \
geometric shapes, grids, trees) entirely in words. Do NOT reference any \
external images or diagrams in the problem statement.
- The problem should be one where visual/spatial thinking helps the solver \
reason about the solution.
- Solutions must be correct, complete Python programs that read from stdin \
and write to stdout.
- Test cases must be valid input/output pairs that the solution handles correctly.
- Match the style and difficulty level specified in the recipe.

For the SVG illustration:
- It must depict the FIRST test case input concretely (not abstract/generic).
- For graph problems: draw nodes as labeled circles, edges as lines/arrows.
- For tree problems: hierarchical layout with the root at top.
- For grid/chessboard problems: actual cells with row/column labels.
- For 2D geometry: actual coordinates from the input, with axes if useful.
- For tables/matrices: a grid of cells with values.
- For other types: invent a clear schematic that illustrates the input.
- Use viewBox (e.g. "0 0 600 400"), self-contained (no external CSS/scripts/images).
- Plain SVG primitives (rect, circle, line, polygon, text, g). NO <script>, NO foreignObject."""


SYSTEM_PROMPT_HARD = """\
You are a senior problem-setter for ICPC World Finals and IOI. Your job is to \
craft brand-new, hard, self-contained algorithmic problems together with \
correct Python reference solutions, test cases, and a grounded SVG of the \
first test case.

ABSOLUTE TARGETS:
- Difficulty must be HARD or VERY_HARD as specified in the recipe — do NOT \
silently downgrade. A correct brute-force should TLE on the official constraints; \
the intended solution should require non-obvious reductions or advanced data \
structures / algorithms.
- The problem must combine 2 or more algorithmic ideas, with at least one of \
them being the "advanced technique" hinted in the recipe (it does NOT have to \
be the only technique, but it must materially help).
- Constraints should be TIGHT (e.g. n up to 2·10^5, 5·10^5, 10^6 — pick what \
forces the intended complexity). State a plausible time limit (1-3 seconds).
- The intended complexity (e.g. O(n log n), O(n sqrt n), O((n+q) log^2 n)) \
should be implicitly required by the constraints.

PROBLEM-DESIGN PRINCIPLES (use as many as fit):
- Hidden structure: an obvious greedy/brute force fails on adversarial inputs.
- Multi-step reduction: e.g. reformulate as a graph / segment-tree query / \
matroid intersection / min-cost-max-flow.
- Strong sample inputs that defeat naive heuristics.
- Edge cases: extreme n, all-equal values, degenerate trees (path/star), \
maximum-density graphs, multiple test cases per run, modular arithmetic, etc.

SOLUTION REQUIREMENTS:
- A complete Python program reading from stdin / writing to stdout.
- DO NOT artificially shorten — a Hard solution typically spans 80-300 lines. \
Inline any helper data structures (segment tree, DSU, hashing, NTT, etc.).
- Be careful with recursion depth for tree problems (use iterative or \
sys.setrecursionlimit when needed).
- Brief comments are fine for non-obvious steps; no chain-of-thought, no \
verbose explanations of well-known algorithms.

For the SVG illustration (same rules as standard mode):
- Depict the FIRST test case input concretely (graph nodes / tree shape / grid \
cells / 2D coordinates). Use viewBox; plain SVG primitives only (rect, circle, \
line, polygon, text, g). NO <script>, NO foreignObject, NO external resources.
- It is fine for the SVG to be slightly larger / busier than usual to convey \
the structure of a hard problem."""


def build_prompt(
    recipe: dict,
    icl_examples: list[dict],
    ingredients: dict,
    problems_per_batch: int,
    difficulty_mode: str = "standard",
) -> str:
    parts = []

    diff_desc = ingredients["difficulty_levels"].get(recipe["difficulty"], recipe["difficulty"])
    source_desc = ingredients["source_styles"].get(recipe["source_style"], recipe["source_style"])

    parts.append("**Recipe for this batch:**")
    parts.append(f"- Algorithm focus: {', '.join(recipe['algorithm_tags'])}")
    parts.append(f"- Visual context type: {recipe['visual_context']} "
                 "(the problem should involve spatial/visual reasoning related to this)")
    parts.append(f"- Difficulty: {recipe['difficulty']} — {diff_desc}")
    parts.append(f"- Source style: {recipe['source_style']} — {source_desc}")
    if difficulty_mode == "hard" and recipe.get("advanced_technique"):
        parts.append(f"- Advanced technique to feature: {recipe['advanced_technique']} "
                     "(the intended solution must use this, possibly combined with other ideas)")
    parts.append("")

    if icl_examples:
        parts.append("**Reference examples** (real problems — match this style):\n")
        for i, ex in enumerate(icl_examples, 1):
            q_preview = ex["question_clean"][:600]
            if len(ex["question_clean"]) > 600:
                q_preview += "..."
            parts.append(f"--- Example {i} [{ex['problem_id']}] ---")
            parts.append(f"Tags: {', '.join(ex['tags'])}")
            parts.append(f"Difficulty: {ex['difficulty']}")
            parts.append(f"Problem: {q_preview}")
            if ex["solution"]:
                sol_preview = ex["solution"][:400]
                if len(ex["solution"]) > 400:
                    sol_preview += "\n# ... (truncated)"
                parts.append(f"Solution:\n```python\n{sol_preview}\n```")
            if ex["io_inputs"]:
                parts.append("Sample I/O:")
                for j in range(min(2, len(ex["io_inputs"]))):
                    inp = str(ex["io_inputs"][j])[:200]
                    parts.append(f"  Input:  {inp}")
                    if j < len(ex["io_outputs"]):
                        out = str(ex["io_outputs"][j])[:200]
                        parts.append(f"  Output: {out}")
            parts.append("")

    parts.append(
        f"**Task:** Generate exactly {problems_per_batch} NEW, UNIQUE competitive "
        "programming problems following the recipe above.\n"
    )
    parts.append(
        "For each problem, provide:\n"
        '1. "question": The full problem statement (self-contained, no image references). '
        "Describe any graphs, geometric figures, grids, or spatial layouts in words. "
        "Include input/output format specification and constraints.\n"
        '2. "solution": A complete, correct Python solution that reads from stdin and writes to stdout.\n'
        '3. "input_output": An object with "inputs" (list of input strings) and "outputs" '
        "(list of corresponding expected output strings). Provide 3-5 test cases.\n"
        '4. "visualization_svg": A self-contained SVG (string) that depicts the FIRST '
        "test case input concretely. Must start with <svg ...> and end with </svg>. "
        "Use viewBox; plain primitives only; no scripts, no external resources.\n"
        '5. "visualization_caption": A one-sentence caption describing what the SVG shows.\n'
        '6. "difficulty": The difficulty level (must match the recipe).\n'
        '7. "tags": List of algorithm tags (must include the recipe tags).\n'
    )
    if difficulty_mode == "hard":
        solution_rule = (
            "- Solutions should be COMPLETE and CORRECT — Hard problems typically "
            "need 80-300 lines. Inline any helper data structures (segment tree, "
            "DSU, NTT, suffix automaton, etc.). Brief comments on non-obvious "
            "steps are fine, but NO chain-of-thought, NO verbose derivations.\n"
            "- Constraints in the problem statement must be TIGHT enough to force "
            "the intended complexity (naive brute force should TLE). State a "
            "plausible time limit (e.g. 1-3 s).\n"
            "- For tree / graph problems with large n, watch recursion depth; "
            "use iterative traversals or sys.setrecursionlimit where needed.\n"
        )
    else:
        solution_rule = (
            "- Solutions must be CONCISE: production-style code only. NO inline reasoning, "
            "NO long thought-process comments, NO step-by-step derivations in comments. "
            "A short top-of-function docstring is fine. Aim for under 80 lines per solution.\n"
        )

    parts.append(
        'Return a JSON array of objects. Each object has keys: '
        '"question", "solution", "input_output", "visualization_svg", '
        '"visualization_caption", "difficulty", "tags".\n\n'
        "**Constraints:**\n"
        "- Each problem must be substantially different from the others and the examples\n"
        "- Problems must be solvable and solutions must be correct\n"
        "- Do NOT reference external images in the problem statement — describe in text\n"
        "- The SVG must depict the actual first test case (not a generic illustration)\n"
        f"{solution_rule}"
        "- Inside the SVG string, escape \" as \\\" and use \\n for newlines if needed\n"
        "- Return ONLY the JSON array, no markdown fences, no preamble, no explanation"
    )

    return "\n".join(parts)


# ── LLM client (reused from generate_queries.py) ─────────────────────────────

async def call_llm(
    messages: list[dict],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.9,
    max_tokens: int = 32768,
    max_retries: int = 6,
    logger: logging.Logger | None = None,
    api_logger: APILogger | None = None,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=300.0, headers=headers) as client:
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(60.0, (2 ** attempt) + random.random() * 2)
                    if logger:
                        logger.warning(
                            f"  Retryable status {resp.status_code}, "
                            f"waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})"
                        )
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code == 403:
                    if logger:
                        logger.error(f"  403 Forbidden — check API key or permissions")
                    raise RuntimeError(f"403 Forbidden: {resp.text[:200]}")
                resp.raise_for_status()
                duration_ms = (time.time() - t0) * 1000
                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if api_logger:
                    try:
                        api_logger.log_call(
                            model=model,
                            request_data={
                                "messages": [
                                    {"role": m["role"], "content": m["content"][:200] + "..."}
                                    for m in messages
                                ]
                            },
                            response_data=data,
                            user="mmcode-augmentation",
                            duration_ms=duration_ms,
                            success=True,
                        )
                    except Exception:
                        pass
                return content
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                wait = min(60.0, (2 ** attempt) + random.random() * 2)
                if logger:
                    logger.warning(
                        f"  Network error: {e}, waiting {wait:.1f}s "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                await asyncio.sleep(wait)
            except RuntimeError:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"LLM call failed after {max_retries} retries: {e}"
                    ) from e
                wait = min(60.0, (2 ** attempt) + random.random() * 2)
                if logger:
                    logger.warning(
                        f"  Error: {e}, waiting {wait:.1f}s "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                await asyncio.sleep(wait)
    raise RuntimeError(f"LLM call exhausted {max_retries} retries")


# ── SVG validation ───────────────────────────────────────────────────────────

_SVG_OPEN = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_SVG_CLOSE = re.compile(r"</svg\s*>", re.IGNORECASE)
_SVG_FORBIDDEN = re.compile(r"<(script|foreignObject|iframe|object|embed)\b", re.IGNORECASE)


def validate_svg(svg: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Lightweight structural check, not full XML parse."""
    if not svg or not isinstance(svg, str):
        return False, "empty_or_not_string"
    s = svg.strip()
    if len(s) < 50:
        return False, "too_short"
    if not _SVG_OPEN.search(s):
        return False, "missing_svg_open"
    if not _SVG_CLOSE.search(s):
        return False, "missing_svg_close"
    if _SVG_FORBIDDEN.search(s):
        return False, "forbidden_tag"
    if not s.lower().startswith("<svg") and not s.lower().lstrip().startswith("<svg"):
        return False, "does_not_start_with_svg"
    return True, "ok"


# ── response parsing ─────────────────────────────────────────────────────────

def _recover_objects_from_array(text: str) -> list[dict]:
    """Walk through a (possibly truncated/malformed) JSON array body and salvage
    every fully-formed top-level object. Returns [] if nothing parseable."""
    start = text.find("[")
    if start == -1:
        return []

    objects: list[dict] = []
    i = start + 1
    n = len(text)
    depth = 0
    obj_start = -1
    in_string = False
    escape = False

    while i < n:
        c = text[i]
        if escape:
            escape = False
        elif in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    candidate = text[obj_start:i + 1]
                    try:
                        obj = json.loads(candidate, strict=False)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = -1
        i += 1

    return objects


def parse_problems(raw: str) -> list[dict] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    for tolerant in (False, True):
        try:
            arr = json.loads(text, strict=not tolerant)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        for tolerant in (False, True):
            try:
                arr = json.loads(match.group(), strict=not tolerant)
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                pass

    recovered = _recover_objects_from_array(text)
    if recovered:
        return recovered

    return None


# ── batch processing ─────────────────────────────────────────────────────────

def build_batch_inputs(
    batch_id: int,
    train_examples: list[dict],
    test_examples: list[dict] | None,
    ingredients: dict,
    *,
    icl_source: str = "train",
    difficulty_mode: str = "standard",
) -> tuple[dict, list[dict]]:
    """Deterministically build (recipe, icl_examples) for batch_id.

    Reproducible — used both during generation and post-hoc recovery.
    `difficulty_mode="hard"` switches to HARD/VERY_HARD-forced recipes and a
    HARD/VERY_HARD/MEDIUM_HARD-only ICL pool.
    """
    rng = random.Random(batch_id * 31337 + 42)
    hard = (difficulty_mode == "hard")

    if icl_source == "test":
        if not test_examples:
            raise ValueError("test_examples required when icl_source='test'")
        anchor = get_anchor_for_batch(test_examples, batch_id)
        if hard:
            recipe = build_test_anchored_recipe_hard(anchor, ingredients, rng)
        else:
            recipe = build_test_anchored_recipe(anchor, ingredients, rng)
        others = [t for t in test_examples if t["problem_id"] != anchor["problem_id"]]
        if hard:
            others = filter_hard_examples(others)
        similar = match_icl_examples(others, recipe, rng, n=1)
        icl = [anchor] + similar
    else:
        if hard:
            recipe = sample_recipe_hard(ingredients, rng)
            pool = filter_hard_examples(train_examples)
        else:
            recipe = sample_recipe(ingredients, rng)
            pool = train_examples
        icl = match_icl_examples(pool, recipe, rng, n=2)
    return recipe, icl


async def process_batch(
    batch_id: int,
    train_examples: list[dict],
    ingredients: dict,
    *,
    model: str,
    base_url: str,
    api_key: str,
    problems_per_batch: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    api_logger: APILogger | None = None,
    output_dir: Path | None = None,
    test_examples: list[dict] | None = None,
    icl_source: str = "train",
    difficulty_mode: str = "standard",
) -> list[dict]:
    recipe, icl = build_batch_inputs(
        batch_id, train_examples, test_examples, ingredients,
        icl_source=icl_source, difficulty_mode=difficulty_mode,
    )
    user_msg = build_prompt(
        recipe, icl, ingredients, problems_per_batch, difficulty_mode=difficulty_mode,
    )

    system_msg = SYSTEM_PROMPT_HARD if difficulty_mode == "hard" else SYSTEM_PROMPT
    max_tokens = 49152 if difficulty_mode == "hard" else 32768

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    async with semaphore:
        try:
            raw = await call_llm(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                logger=logger,
                api_logger=api_logger,
            )
        except Exception as e:
            logger.error(f"  batch {batch_id}: LLM call failed: {e}")
            return []

    parsed = parse_problems(raw)
    if parsed is None:
        logger.error(f"  batch {batch_id}: failed to parse response ({len(raw)} chars)")
        fail_dir = (output_dir / "failed_batches") if output_dir else (BASE / "logs")
        fail_dir.mkdir(parents=True, exist_ok=True)
        (fail_dir / f"failed_batch_{batch_id}.txt").write_text(raw, encoding="utf-8")
        return []

    icl_ids = [ex["problem_id"] for ex in icl]
    results = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue

        io = item.get("input_output", {})
        if isinstance(io, str):
            try:
                io = json.loads(io)
            except json.JSONDecodeError:
                io = {}

        svg_str = item.get("visualization_svg", "") or ""
        svg_ok, svg_reason = validate_svg(svg_str)

        record: dict[str, Any] = {
            "id": f"mmcode_aug_{batch_id * problems_per_batch + i:06d}",
            "batch_id": batch_id,
            "recipe": recipe,
            "icl_source": icl_source,
            "difficulty_mode": difficulty_mode,
            "icl_example_ids": icl_ids,
            "model": model,
            "question": item.get("question", ""),
            "solution": item.get("solution", ""),
            "input_output": {
                "inputs": io.get("inputs", []),
                "outputs": io.get("outputs", []),
            },
            "visualization_svg": svg_str if svg_ok else "",
            "visualization_caption": item.get("visualization_caption", ""),
            "svg_valid": svg_ok,
            "svg_reason": svg_reason,
            "difficulty": item.get("difficulty", recipe["difficulty"]),
            "tags": item.get("tags", recipe["algorithm_tags"]),
        }
        results.append(record)

    return results


# ── orchestration ─────────────────────────────────────────────────────────────

async def generate_all(
    *,
    model: str,
    base_url: str,
    api_key: str,
    batches: int,
    problems_per_batch: int,
    concurrency: int,
    temperature: float,
    resume: bool,
    output_dir: Path,
    logger: logging.Logger,
    api_logger: APILogger | None = None,
    icl_source: str = "train",
    difficulty_mode: str = "standard",
) -> int:
    logger.info("Loading train examples...")
    train_examples = load_train_examples()
    logger.info(f"  Loaded {len(train_examples)} train examples with solutions")

    test_examples: list[dict] | None = None
    if icl_source == "test":
        logger.info("Loading test examples (for test-anchored ICL)...")
        test_examples = load_test_examples()
        logger.info(f"  Loaded {len(test_examples)} test examples "
                    f"({sum(1 for t in test_examples if t['solution'])} with solution)")

    ingredients = load_ingredients()
    logger.info(f"  {len(ingredients['algorithm_tags'])} algo tags, "
                f"{len(ingredients['visual_context_types'])} visual types")

    out_path = output_dir / "problems.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_batches: set[int] = set()
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    completed_batches.add(row.get("batch_id", -1))
                except json.JSONDecodeError:
                    continue
        logger.info(f"  Resuming: {len(completed_batches)} batches already done")

    pending = [b for b in range(batches) if b not in completed_batches]
    if not pending:
        logger.info(f"  All {batches} batches already complete")
        return 0

    logger.info(f"  {len(pending)} batches to run")

    semaphore = asyncio.Semaphore(concurrency)
    file_mode = "a" if resume else "w"
    total_generated = 0
    chunk_size = concurrency * 3

    pbar = tqdm(total=len(pending), desc="mmcode", unit="batch", leave=True)

    with open(out_path, file_mode, encoding="utf-8") as fout:
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start:chunk_start + chunk_size]
            tasks = [
                process_batch(
                    bid, train_examples, ingredients,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    problems_per_batch=problems_per_batch,
                    temperature=temperature,
                    semaphore=semaphore,
                    logger=logger,
                    api_logger=api_logger,
                    output_dir=output_dir,
                    test_examples=test_examples,
                    icl_source=icl_source,
                    difficulty_mode=difficulty_mode,
                )
                for bid in chunk
            ]
            results = await asyncio.gather(*tasks)
            for batch_results in results:
                for record in batch_results:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_generated += 1
            fout.flush()
            pbar.update(len(chunk))
            pbar.set_postfix(problems=total_generated)

    pbar.close()
    return total_generated


# ── main ─────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else BASE / "output"
    log_dir = output_dir / "logs"
    logger = setup_logging(log_dir)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    if args.icl_source == "test":
        n_test = len(load_test_examples())
        if args.coverage_rounds is not None:
            implied = n_test * args.coverage_rounds
            if args.batches != implied:
                logger.info(f"  --coverage-rounds={args.coverage_rounds} × test={n_test}"
                            f" → overriding --batches={args.batches} with {implied}")
                args.batches = implied

    logger.info("=" * 60)
    logger.info("MMCode Data Augmentation Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Model:              {args.model}")
    logger.info(f"  Base URL:           {args.base_url}")
    logger.info(f"  ICL source:         {args.icl_source}"
                + (f" (coverage_rounds={args.coverage_rounds})" if args.icl_source == "test" else ""))
    logger.info(f"  Difficulty mode:    {args.difficulty_mode}")
    logger.info(f"  Batches:            {args.batches}")
    logger.info(f"  Problems/batch:     {args.problems_per_batch}")
    logger.info(f"  Target total:       ~{args.batches * args.problems_per_batch}")
    logger.info(f"  Concurrency:        {args.concurrency}")
    logger.info(f"  Temperature:        {args.temperature}")
    logger.info(f"  Resume:             {args.resume}")
    logger.info(f"  Output dir:         {output_dir}")
    logger.info("")

    if args.dry_run:
        train_examples = load_train_examples()
        test_examples = load_test_examples() if args.icl_source == "test" else None
        ingredients = load_ingredients()

        system_msg = SYSTEM_PROMPT_HARD if args.difficulty_mode == "hard" else SYSTEM_PROMPT

        for trial in range(3):
            batch_id = trial * (len(test_examples) if test_examples else 1) + trial
            recipe, icl = build_batch_inputs(
                batch_id, train_examples, test_examples, ingredients,
                icl_source=args.icl_source, difficulty_mode=args.difficulty_mode,
            )
            prompt = build_prompt(
                recipe, icl, ingredients, args.problems_per_batch,
                difficulty_mode=args.difficulty_mode,
            )

            logger.info(f"{'='*60}")
            logger.info(f"DRY RUN — Trial {trial+1} (batch_id={batch_id}, "
                        f"source={args.icl_source}, mode={args.difficulty_mode})")
            logger.info(f"{'='*60}")
            logger.info(f"Recipe: {json.dumps(recipe, indent=2)}")
            logger.info(f"ICL examples: {[ex['problem_id'] for ex in icl]}")
            logger.info(f"\nSystem prompt ({len(system_msg)} chars):")
            logger.info(system_msg)
            logger.info(f"\nUser prompt ({len(prompt)} chars):")
            logger.info(prompt)
            logger.info("")
        return

    api_logger = APILogger(str(API_LOG_DIR))

    t0 = time.time()
    n = await generate_all(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        batches=args.batches,
        problems_per_batch=args.problems_per_batch,
        concurrency=args.concurrency,
        temperature=args.temperature,
        resume=args.resume,
        output_dir=output_dir,
        logger=logger,
        api_logger=api_logger,
        icl_source=args.icl_source,
        difficulty_mode=args.difficulty_mode,
    )
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info(f"Finished: {n} problems generated in {elapsed:.1f}s")
    logger.info("=" * 60)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "icl_source": args.icl_source,
        "difficulty_mode": args.difficulty_mode,
        "coverage_rounds": args.coverage_rounds,
        "batches": args.batches,
        "problems_per_batch": args.problems_per_batch,
        "total_generated": n,
        "elapsed_s": round(elapsed, 1),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate augmented competitive programming problems for MMCode"
    )
    parser.add_argument("--model", default="claude-sonnet-4-6-aws")
    parser.add_argument("--base-url", default="https://az.gptplus5.com/v1")
    parser.add_argument(
        "--batches", type=int, default=670,
        help="Number of batches (x problems_per_batch = total). "
             "When --icl-source=test and --coverage-rounds is set, this is overridden.",
    )
    parser.add_argument("--problems-per-batch", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument(
        "--icl-source", choices=["train", "test"], default="train",
        help="Where to draw ICL examples. 'train' (default): match by tag from train pool. "
             "'test': anchor each batch to a test example, using it as ICL #1 + 1 similar test example.",
    )
    parser.add_argument(
        "--coverage-rounds", type=int, default=None,
        help="When --icl-source=test, target this many anchor passes through the test set "
             "(sets --batches = len(test) * coverage_rounds).",
    )
    parser.add_argument(
        "--difficulty-mode", choices=["standard", "hard"], default="standard",
        help="'standard' (default): full distribution across EASY..VERY_HARD. "
             "'hard': force HARD/VERY_HARD, 2-4 algo tags, one advanced technique injected, "
             "ICL pool restricted to MEDIUM_HARD+, longer solutions allowed, bigger token budget.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Custom output directory (default: ./output)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sample prompts without calling API",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
