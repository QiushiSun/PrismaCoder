"""
Prompts for the cross-source HARD-augmentation pipeline.

Key differences from the leetcode-newgen prompts:
  • Encourage advanced algorithmic content (graph, DP, geometry, etc.)
  • Optionally show 1 "inspiration" image (a competitive-programming problem
    drawn from CodeForces / AtCoder / Aizu / mmcode_full) — without solution —
    to push the model toward harder problem styles.
  • Output schema is identical to leetcode-newgen: a parametric Python
    module with TITLE / DESCRIPTION / METHOD_NAME / SOLUTION_SIGNATURE /
    class Solution / sample_examples / render / make_test_cases.

The build pipeline still substitutes __INPUT_K__ / __OUTPUT_K__ in the
DESCRIPTION when generating concrete variants.

Usage:
  from prompts import build_messages_cross
"""

from __future__ import annotations

from typing import Any

# ────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert competitive-programming problem designer creating NEW
multimodal coding problems for a vision-language model training corpus.
You specialise in HARD algorithmic problems — graph theory, advanced DP,
computational geometry, number theory and combinatorics, segment-tree
queries, network flow, and so on.

You will be given:
  • A "recipe" specifying difficulty (Easy / Medium / Hard, biased toward
    Hard), target tags, visual aspect, target domain, and `n_examples`
    (number of example diagrams shown in the description).
  • 2 reference LEETCODE-style problems — each shown as an image plus its
    prose and the canonical `class Solution`.
  • Optionally, 1 "inspiration" competitive problem (image only, with
    truncated prose, NO solution) — use it ONLY to draw inspiration for the
    DIFFICULTY and STYLE of the new problem; do NOT clone its content.

Your job is to design ONE NEW problem that:
  1. Matches the recipe's difficulty + tag mix and exercises the listed
     algorithmic / data-structure flavour.
  2. Lives in a clearly DIFFERENT story / scenario from BOTH ICL examples
     AND from the inspiration (do not just rename variables).
  3. Has a self-contained `class Solution` with a type-hinted method.
  4. Provides parametric example generation + rendering so the same template
     can be re-instantiated with different seeds for variant generation.
  5. When the recipe says Hard, the solution MUST require multi-step
     reasoning — DP recurrence, graph algorithm, geometry observation, etc.

Output ONE complete Python module — no Markdown commentary, just a single
fenced ```python``` block — with the following top-level symbols:

  TITLE : str
      A short, evocative problem title (LeetCode-style).

  DESCRIPTION : str
      Full LeetCode-style problem prose.  Must contain:
        - one or two paragraphs describing the task,
        - exactly `n_examples` "**Example k:**" sections, each followed by
          one `<image>` placeholder and an Input/Output block using the
          exact placeholders `__INPUT_K__` and `__OUTPUT_K__` (the build
          pipeline will substitute concrete values),
        - a "**Constraints:**" section (markdown bullet list),
        - DO NOT manually compute or guess example outputs; leave the
          `__OUTPUT_K__` placeholder verbatim.

  METHOD_NAME : str
      camelCase method name on `class Solution`.  Pick something
      descriptive (`shortestRepairCycle`, `maxArbitrageDepth`, …).

  SOLUTION_SIGNATURE : str
      Triple-quoted string with the class signature line + the def
      signature line + a TERSE 1-paragraph docstring. The docstring MUST
      declare I/O types only; it MUST NOT spoil the visual rule.

  class Solution:
      Canonical reference solution. Method name MUST equal METHOD_NAME.
      SELF-CONTAINED — no module-level mutable state. Helper functions go
      at module top (or as nested defs inside the method).

  def sample_examples(seed: int, n: int) -> list[dict]:
      Deterministic from `seed`. Returns exactly `n` example dicts:
          [{"kwargs": {"arg1": ..., "arg2": ...}}, ...]
      `kwargs` MUST use only Python literals.  These instances become both
      the visible Example blocks AND the rendered diagrams; pick small,
      illustrative inputs (graphs ≤ 8 vertices, grids ≤ 6×6, arrays ≤ 12).

  def render(example: dict) -> "PIL.Image.Image":
      Render ONE example as a PIL image.
        • Receives the dict produced by sample_examples().
        • Use matplotlib + PIL (savefig → BytesIO → PIL.Image.open).
        • Visual MUST visibly depend on `example["kwargs"]`.
        • Title / legend / cell-text MUST NOT explain the algorithmic rule.
        • Return PIL.Image.Image (NOT a Figure or numpy array).

  def make_test_cases(seed: int) -> list[tuple[dict, "Any"]]:
      Deterministic from `seed`.  Returns 6–10 hidden test cases for
      verification.  Each tuple: ({arg1: ..., arg2: ...}, expected_output).

      CRITICAL: every `expected_output` MUST be obtained by calling
      `Solution().METHOD_NAME(**kwargs)` — NEVER hand-compute or hard-code.
      Recommended pattern:

          inputs = [ {"arg1": v1, "arg2": v2}, ... ]
          sol = Solution()
          return [(kw, getattr(sol, METHOD_NAME)(**kw)) for kw in inputs]

Hard requirements (your code WILL be auto-checked):
  • All typing classes used (List, Tuple, Optional, Dict, …) imported at
    module top.
  • `class Solution` self-contained; no globals other than constants.
  • `render` returns a PIL.Image.Image.
  • All test-case kwargs and example kwargs use ONLY Python literals
    (int / str / float / list / dict / tuple / bool / None).
  • Helper functions MUST be at module top OR nested inside the solving
    method (NEVER at the bottom AFTER make_test_cases — that breaks
    AST-based extraction).
  • DO NOT include `__INPUT_K__` / `__OUTPUT_K__` anywhere except inside
    DESCRIPTION.
  • DO NOT compute example outputs in DESCRIPTION; leave the placeholder
    `__OUTPUT_K__` verbatim.

Matplotlib gotchas (very easy to get wrong — internalise these):
  • Use `mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
    fc="white", ec="black")` — pass `boxstyle=` as a *named* keyword and
    never repeat it.  Do NOT also pass it via a `style="..."` shortcut.
  • Use `ax.annotate("", xy=..., xytext=..., arrowprops=dict(arrowstyle="->"))`
    for arrows; DO NOT pass arrowstyle to `ax.arrow`.
  • Always wrap drawing in a `try/finally: plt.close(fig)` so the figure
    is closed even if matplotlib raises mid-draw.
  • Save through a BytesIO buffer (`fig.savefig(buf, format="png",
    bbox_inches="tight", dpi=120); plt.close(fig); return Image.open(buf)`).
  • `ax.set_aspect("equal")` for geometry; `ax.axis("off")` for clean look.
  • For graph diagrams use `networkx` if helpful — but DO NOT import it
    unless you also use it.

Output format: a single fenced Python block, nothing else.

```python
# your module here
```
"""

# ────────────────────────────────────────────────────────────────────────────
USER_TEMPLATE = """## Recipe

Difficulty:        {difficulty}
Algo tag(s):       {algo_tags}
Struct tag(s):     {struct_tags}
Visual aspect:     {visual_aspect}
Target domain:     {domain}
n_examples:        {n_examples}

## Reference example #1  (do NOT reuse domain or signature)

Title:        {ex1_title}
Tags:         {ex1_tags}
Difficulty:   {ex1_difficulty}

Description (truncated):
{ex1_description}

Canonical class Solution:
```python
{ex1_solution}
```

## Reference example #2  (do NOT reuse domain or signature)

Title:        {ex2_title}
Tags:         {ex2_tags}
Difficulty:   {ex2_difficulty}

Description (truncated):
{ex2_description}

Canonical class Solution:
```python
{ex2_solution}
```
{insp_block}
## Now design ONE new problem

Constraints:
  • Difficulty must match the recipe's `{difficulty}`.  If `Hard`, the
    solution must require multi-step reasoning (DP / graph / careful
    invariant) — not a one-liner.
  • Description must contain EXACTLY {n_examples} "**Example k:**" blocks.
    Each example block contains:
        **Example {{k}}:**
        <image>
        ```
        Input: __INPUT_{{k}}__
        Output: __OUTPUT_{{k}}__
        ```
        (optional brief explanation)
  • Use `__INPUT_K__` / `__OUTPUT_K__` placeholders verbatim — the pipeline
    fills them with `repr(kwargs)` / `repr(Solution().method(**kwargs))`.
  • The story (entities + scenario) must clearly differ from BOTH ICL
    AND the inspiration if shown.
  • The class signature must be self-contained.

Remember:
  - Test-case args + expected MUST be Python literals.
  - solution MUST NOT depend on module-level mutable state.
  - render() MUST visibly depend on `example["kwargs"]`.
  - Output one Python code block, nothing else.
"""

INSP_BLOCK_TEMPLATE = """
## Inspiration  (style/difficulty cue ONLY — do NOT clone, do NOT use the same domain)

Source:       {insp_source}  (competitive programming)
Title:        {insp_title}
Difficulty:   {insp_difficulty}

Description (truncated):
{insp_description}

(No reference solution provided — you must invent your own algorithmic rule.)
"""

# ────────────────────────────────────────────────────────────────────────────
RETRY_HINT_TEMPLATE = """## RETRY CONTEXT

Your previous attempt was rejected due to the following errors. Address them
in your new module — same recipe, same style, just FIX the bugs.

```
{errors_str}
```

Reminders (don't list these, just internalise):
- All typing imports at module top (`from typing import List, ...`).
- Test-case kwargs + expected MUST be Python literals.
- `class Solution` and `METHOD_NAME` MUST be a top-level class + a top-level
  string assignment.  No nested decorators or factories.
- DESCRIPTION MUST contain the exact placeholders `__INPUT_K__` and
  `__OUTPUT_K__` (where K is 1, 2, …) — NEVER compute the outputs yourself.
- arrowstyle '=>' is INVALID; use '->' / '<->' / 'fancy' / 'simple' / 'wedge'.
- render() MUST return a PIL.Image.Image.
- Do not import unused packages (no `import torch`, no `import sklearn`).
"""


# ────────────────────────────────────────────────────────────────────────────
def build_messages_cross(
    *,
    recipe: dict[str, Any],
    icl_examples: list[dict],   # each: {qid, title, description, solution, image_data_url, tags, difficulty}
    inspiration: dict | None = None,    # optional: {source, title, description, image_data_url, difficulty}
    retry_errors: list[dict] | None = None,
) -> list[dict]:
    """Compose the multimodal Chat-Completions message list."""

    def _tags_str(t):
        if isinstance(t, list):
            return ", ".join(t) if t else "(any)"
        return str(t)

    insp_block = ""
    if inspiration:
        insp_block = INSP_BLOCK_TEMPLATE.format(
            insp_source=inspiration.get("source", "?"),
            insp_title=inspiration.get("title", "?"),
            insp_difficulty=inspiration.get("difficulty", "?"),
            insp_description=(inspiration.get("description") or "")[:1000],
        )

    user_text = USER_TEMPLATE.format(
        difficulty=recipe["difficulty"],
        algo_tags=_tags_str(recipe.get("algo_tags") or []),
        struct_tags=_tags_str(recipe.get("struct_tags") or []),
        visual_aspect=recipe["visual_aspect"],
        domain=recipe["domain"],
        n_examples=recipe["n_examples"],
        ex1_title=icl_examples[0].get("title", "?"),
        ex1_tags=_tags_str(icl_examples[0].get("tags") or []),
        ex1_difficulty=icl_examples[0].get("difficulty", "?"),
        ex1_description=icl_examples[0].get("description", "")[:1200],
        ex1_solution=icl_examples[0].get("solution", "")[:3500],
        ex2_title=icl_examples[1].get("title", "?"),
        ex2_tags=_tags_str(icl_examples[1].get("tags") or []),
        ex2_difficulty=icl_examples[1].get("difficulty", "?"),
        ex2_description=icl_examples[1].get("description", "")[:1200],
        ex2_solution=icl_examples[1].get("solution", "")[:3500],
        insp_block=insp_block,
    )

    if retry_errors:
        seen: set[str] = set()
        lines: list[str] = []
        for e in retry_errors:
            tag = f"seed={e.get('seed','?')}: {e.get('error_type','')}: {str(e.get('error',''))[:300]}"
            if tag in seen:
                continue
            seen.add(tag)
            lines.append("- " + tag)
        errors_str = "\n".join(lines[:8])
        user_text = RETRY_HINT_TEMPLATE.format(errors_str=errors_str) + "\n" + user_text

    content: list[dict] = []
    content.append({"type": "text", "text": "**Reference image #1 (LeetCode-style):**"})
    content.append({"type": "image_url", "image_url": {"url": icl_examples[0]["image_data_url"], "detail": "auto"}})
    content.append({"type": "text", "text": "**Reference image #2 (LeetCode-style):**"})
    content.append({"type": "image_url", "image_url": {"url": icl_examples[1]["image_data_url"], "detail": "auto"}})
    if inspiration and inspiration.get("image_data_url"):
        content.append({"type": "text", "text": "**Inspiration image (competitive — STYLE ONLY):**"})
        content.append({"type": "image_url", "image_url": {"url": inspiration["image_data_url"], "detail": "auto"}})
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
