# LeetCode-Newgen Pipeline (Plan-A')

Parametric template-based synthesis of **new** multimodal coding
problems in the LeetCode "class Solution" style.  Designed to grow a
training corpus without scraping copyrighted descriptions.

## What it produces

Each generated record looks like a LeetCode problem:

* a **title**
* a description with one or more `**Example k:**` blocks (each prefixed
  by an `<image>` token referring to a rendered diagram)
* a `class Solution: def methodName(...)` signature
* the canonical reference solution Python code
* 6–10 hidden test cases (auto-derived by calling `Solution()` against
  parametric inputs) — every test is verified before the record is kept

## Pipeline stages

```
extract_ingredients.py    ── recipe pool (difficulty / algo tag / visual)
build_seeds.py            ── ICL seeds from leetcode_records.jsonl (image+code)
generate_templates.py     ── Claude generates parametric Python templates
recover_failed.py         ── (optional) re-extract templates from cached fails
run_templates.py          ── for each template: sample seeds, render images,
                              substitute Input/Output placeholders, sandbox-test
audit_leakage.py          ── Jaccard-similarity against native LC corpus
build_sft.py              ── final SFT jsonl + image folder + summary
```

## Quick smoke test

```bash
export OPENAI_BASE_URL="https://your-gateway/v1"
export OPENAI_API_KEY="sk-..."

cd open-source-pipelines/competition-pipeline/newgen-pipeline
bash run_pipeline.sh
```

The smoke run generates 10 templates × 3 variants ≈ 30 SFT records
in ~5 minutes, < $1.

## Full run

For a 3 500-record SFT pack (~$25 / ~90 minutes at concurrency=32):

```bash
python extract_ingredients.py
python build_seeds.py
python generate_templates.py --target 800
python run_templates.py --seeds 5 --workers 8
python audit_leakage.py --sim-cut 0.5
python build_sft.py
```

## Anatomy of a template

The LLM emits one Python module per template (saved to
`templates/<id>.py`) with these required top-level symbols:

| Symbol | Type | Purpose |
|---|---|---|
| `TITLE` | `str` | Problem title shown in the SFT query. |
| `DESCRIPTION` | `str` | Markdown body with `__INPUT_K__` / `__OUTPUT_K__` placeholders. |
| `METHOD_NAME` | `str` | camelCase method name on `class Solution`. |
| `SOLUTION_SIGNATURE` | `str` | Class + def header + terse docstring. |
| `class Solution` | class | Reference solution; `MethodName` accepts kwargs. |
| `sample_examples(seed, n)` | function | Deterministic example kwargs. |
| `render(example)` | function | Returns a `PIL.Image.Image` for ONE example. |
| `make_test_cases(seed)` | function | Returns 6–10 (kwargs, expected) hidden tests. |

`run_templates.py` instantiates each template at multiple seeds, renders
the example diagrams to PNGs, substitutes the `__INPUT_K__` placeholders
with `repr(kwargs)`, computes outputs by calling
`Solution().METHOD_NAME(**kwargs)`, and **rejects** any variant whose
hidden tests don't all pass in a sandboxed subprocess.

## Output

```
templates/<id>.py                 — generated Python templates
audit/<id>.json                   — per-template metadata
records.jsonl                     — synthetic seed records
variants/records.jsonl            — verified variant records (kwargs / outputs)
variants/images/<id>_v<seed>_<k>.png
sft/train.sft.jsonl               — full SFT pack (post-audit drop)
sft/train.sft.clean.jsonl         — leakage-audited (recommended)
sft/train.sft.hard.jsonl          — Hard-difficulty subset
sft/images/                       — copied diagrams
sft/build_summary.json
```
