"""
Prompts for ICL-guided synthesis of NEW LeetCode-style multimodal problems.

Differences from HumanEvalV-Data prompts_newgen.py:
  • LeetCode schema: `class Solution: def methodName(...)` + multi-image
    examples + LeetCode-style description with Example 1/2 sections.
  • Each variant shows N example diagrams (N = recipe.n_examples ∈ {1, 2, 3}).
  • Reference solution is the canonical Python implementation; description's
    Example sections are filled with computed I/O at variant-build time, NOT
    by the LLM (LLM may put placeholders).

ICL: 2 leetcode_native records, each contributing image[0] + truncated
prose + the canonical class Solution.

Output: a single fenced Python module, no commentary outside.
"""

from __future__ import annotations

from typing import Any

# ────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert competitive-programming problem designer creating NEW multimodal coding problems for a vision-language model training corpus, in the style of LeetCode.

You will be given:
  • A "recipe" specifying difficulty, target tags, visual aspect, target domain
    and `n_examples` (the number of example diagrams the description shows).
  • 2 reference LeetCode problems — each shown as an image plus its prose and
    the canonical `class Solution`.

Your job is to design ONE NEW problem that:
  1. Matches the recipe's difficulty + tag mix and exercises the listed
     algorithmic / data-structure flavor.
  2. Lives in a clearly DIFFERENT story / scenario from BOTH ICL examples
     (do not just rename variables).
  3. Has a self-contained `class Solution` with type-hinted method.
  4. Provides parametric example generation + rendering so the same template
     can be re-instantiated with different seeds for variant generation.
  5. Is appropriately HARD when the recipe says so: requires multi-step
     algorithmic reasoning, not a one-liner heuristic.

Output ONE complete Python module — no Markdown commentary, just a single
fenced ```python``` block — with the following top-level symbols:

  TITLE : str
      A short, evocative problem title (LeetCode-style).  Example: "Maximum
      Score on a Diamond Lattice".

  DESCRIPTION : str
      The full LeetCode-style problem prose.  Must include:
        - one or two paragraphs describing the task,
        - exactly `n_examples` "**Example k:**" sections, each followed by
          one `<image>` placeholder line and an "Input:" / "Output:" pair
          (use exact placeholders `__INPUT_K__` and `__OUTPUT_K__` — the
          build pipeline will substitute concrete values),
        - a "**Constraints:**" section (markdown bullet list),
        - DO NOT manually compute or guess example outputs; leave the
          `__OUTPUT_K__` placeholder verbatim.

  METHOD_NAME : str
      The camelCase method name on `class Solution`.  Pick something
      descriptive (`shortestSubarray`, `maxScore`, `numIslands`, …).

  SOLUTION_SIGNATURE : str
      A triple-quoted string holding only the class signature line + the
      def signature line + a TERSE 1-paragraph docstring.  The docstring
      MUST declare I/O types only; it MUST NOT spoil the visual rule.
      Format:
        class Solution:
            def methodName(self, args...) -> RetType:
                \"\"\"<terse one-paragraph docstring; no visual leakage>\"\"\"

  class Solution:
      The canonical reference solution.  Method name MUST equal METHOD_NAME.
      SELF-CONTAINED — no module-level mutable state.  All helper functions
      must live at module top OR as nested defs inside the method.

  def sample_examples(seed: int, n: int) -> list[dict]:
      Deterministic from `seed`.  Returns exactly `n` example dicts:
          [{"kwargs": {"arg1": ..., "arg2": ...}}, ...]
      Each dict's `kwargs` MUST be a Python literal mapping argument names
      onto their values.  These instances become both the visible Example
      blocks AND the rendered diagrams; they should be small, illustrative,
      and easy to read at a glance (e.g. trees with ≤ 8 nodes, graphs with
      ≤ 6 nodes, grids ≤ 5×5).

  def render(example: dict) -> "PIL.Image.Image":
      Render ONE example as a PIL image.
        • Receives the dict produced by sample_examples().
        • Use matplotlib + PIL (savefig → buf → PIL.Image.open).
        • The visual MUST visibly depend on `example["kwargs"]` so different
          seeds produce different diagrams (size / layout / labels / …).
        • Title / legend / axis-labels / cell-text MUST NOT explain the
          algorithmic rule.  Use a neutral title like "Example", "Diagram",
          "Network", "Grid Pattern".
        • Return PIL.Image.Image (NOT a Figure or numpy array).

  def make_test_cases(seed: int) -> list[tuple[dict, "Any"]]:
      Deterministic from `seed`.  Returns 5–8 hidden test cases for
      verification.  Each tuple: ({arg1: ..., arg2: ...}, expected_output).

      CRITICAL: every `expected_output` MUST be obtained by calling
      `Solution().METHOD_NAME(**kwargs)` — NEVER hand-compute or hard-code.
      This is a consistency check between the reference solution and the
      tests.

      Recommended pattern:

          inputs = [ {"arg1": v1, "arg2": v2}, ... ]
          sol = Solution()
          return [(kw, getattr(sol, METHOD_NAME)(**kw)) for kw in inputs]

Hard requirements (your code WILL be auto-checked):
  • All typing classes used (List, Tuple, Optional, Dict, …) imported at
    module top.
  • `class Solution` self-contained — no globals other than constants.
  • `render` returns a PIL.Image.Image.
  • All test-case kwargs and example kwargs use ONLY Python literals
    (int / str / float / list / dict / tuple / bool / None).
  • `arrowstyle` (matplotlib FancyArrowPatch) must be one of:
    '->', '<-', '<->', '-', '<|-', '-|>', '<|-|>', 'fancy', 'simple', 'wedge'.
    The token '=>' is INVALID.
  • Helper functions used by the solution must be at module top so AST
    tooling can extract them (or nested inside the method body).
  • If you need a tree / linked-list, use a primitive (level-order list /
    list-of-pairs) and let the method reconstruct it locally.
  • DO NOT include `__INPUT_K__` / `__OUTPUT_K__` anywhere except inside
    the DESCRIPTION string.  In particular, never use them as variables.
  • DO NOT compute example outputs in DESCRIPTION; leave the placeholder
    `__OUTPUT_K__` verbatim and the build pipeline will fill it.

Matplotlib gotchas (very easy to get wrong — internalise these):
  • Use `mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
    fc="white", ec="black")` — pass `boxstyle=` as a *named* keyword and never
    repeat it. Do NOT also pass it via a `style="..."` shortcut.
  • Use `ax.annotate("", xy=..., xytext=..., arrowprops=dict(arrowstyle="->"))`
    for arrows; DO NOT pass arrowstyle to `ax.arrow`.
  • Always wrap drawing in a `try/finally: plt.close(fig)` so the figure is
    closed even if matplotlib raises mid-draw.
  • Save through a BytesIO buffer (`fig.savefig(buf, format="png",
    bbox_inches="tight", dpi=120); plt.close(fig); return Image.open(buf)`).
  • `ax.set_aspect("equal")` for geometry; `ax.axis("off")` for clean look.

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
  • The story (entities + scenario) must clearly differ from BOTH ICL.
  • The class signature must be self-contained (the visual carries the
    final concrete instance, but the docstring + signature must define
    the I/O contract precisely).

Remember:
  - Test-case args + expected MUST be Python literals.
  - solution MUST NOT depend on module-level mutable state.
  - render() MUST visibly depend on `example["kwargs"]`.
  - Output one Python code block, nothing else.
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
def build_messages(
    *,
    recipe: dict[str, Any],
    icl_examples: list[dict],   # each: {qid, title, description, solution, image_data_url, tags, difficulty}
    retry_errors: list[dict] | None = None,
) -> list[dict]:
    """Compose the multimodal Chat-Completions message list."""

    def _tags_str(t):
        if isinstance(t, list):
            return ", ".join(t) if t else "(any)"
        return str(t)

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
        ex1_solution=icl_examples[0].get("solution", ""),
        ex2_title=icl_examples[1].get("title", "?"),
        ex2_tags=_tags_str(icl_examples[1].get("tags") or []),
        ex2_difficulty=icl_examples[1].get("difficulty", "?"),
        ex2_description=icl_examples[1].get("description", "")[:1200],
        ex2_solution=icl_examples[1].get("solution", ""),
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
    content.append({"type": "text", "text": "**Reference image #1:**"})
    content.append({"type": "image_url", "image_url": {"url": icl_examples[0]["image_data_url"], "detail": "auto"}})
    content.append({"type": "text", "text": "**Reference image #2:**"})
    content.append({"type": "image_url", "image_url": {"url": icl_examples[1]["image_data_url"], "detail": "auto"}})
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
