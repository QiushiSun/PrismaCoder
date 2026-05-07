# Competition-Solve Pipeline

End-to-end pipeline that **translates, multimodal-solves and sandbox-
verifies** competitive-programming problems pulled from CodeForces /
AtCoder / Aizu (or any source with the same JSON schema), and packages
them into multimodal SFT records.

## Pipeline stages

```
┌──────────────────────┐
│ raw OJ JSON dumps    │ <── CodeForces / AtCoder / Aizu
│ + image folders      │
└──────────┬───────────┘
           ▼
   parse_seeds.py            ── unify schema, dedupe, classify "parseable" tests
           │
           ▼
   translate_jp.py           ── Japanese → English (skipped for English seeds)
           │
           ▼
   solve_with_claude.py      ── multimodal Claude solve + extra-test augmentation
           │
           ▼
   recover_solutions.py      ── re-extract code from cached responses (fence-fix)
           │
           ▼
   verify_sandbox.py         ── run in subprocess, must pass 100% of tests
           │
           ▼
   retry_failed.py           ── stronger prompt + lower temperature for failures
           │                     (loop verify+retry until happy)
           ▼
   verify_combined.py        ── final verify combining originals + retries
           │
           ▼
   build_sft.py              ── SFT-format records (per-source jsonl + aggregate)
```

## Inputs

We expect the upstream JSON dumps to follow this schema (one record per
problem):

```json
{
  "problem_id": "abc123_d",
  "title": "...",
  "url": "https://...",
  "images": ["images/problem_id_xxx.png", ...],
  "messages": [
    {"role": "user", "content_en": "<problem in English>",
                      "content_ja": "<problem in Japanese, optional>"},
    {"role": "assistant", "content_en": ""}
  ],
  "test_cases": [
    {"input": "...", "output": "..."},
    ...
  ]
}
```

A small CodeForces / AtCoder / Aizu sample is bundled at
`seeds_sample/` so you can smoke-test the whole flow without a download.

## Quick smoke test

```bash
export OPENAI_BASE_URL="https://your-gateway/v1"
export OPENAI_API_KEY="sk-..."

cd open-source-pipelines/competition-pipeline/solve-pipeline
bash run_pipeline.sh
```

The smoke run uses 5 seeds per source (15 total).  Expect ~2 minutes
wall time and < $0.50 of API spend.

## Full-scale run

For a real ingest (e.g. all 1 936 OJ problems we used) the script takes
~80 minutes wall and ~$55 of Claude-4.6 tokens.  Knobs you typically
want to tune:

| Flag | Where | Default | Notes |
|---|---|---|---|
| `--concurrency` | `solve_with_claude.py`, `translate_jp.py`, `retry_failed.py` | 32 | Lower if your gateway has tight per-IP RPM. |
| `--max-tokens` | `solve_with_claude.py` | 12 288 | Bump to 16 384 for hard CF problems with long reasoning. |
| `--temperature` | `solve_with_claude.py` | 0.5 | Drop to 0.2 on the retry pass. |
| `--workers` | `verify_sandbox.py`, `verify_combined.py` | 8 | Sandbox is CPU-bound; one worker per core. |
| `--timeout` | `verify_sandbox.py` | 8 (seconds) | Per-test sandbox timeout. |

## Output

```
seeds/records.jsonl                — parsed seeds (1 line per problem)
translations/<sid>.txt             — EN translations of JP problems
solutions/<sid>.json               — Claude's first-pass response
retry_solutions/<sid>.json         — second-pass retries (only failures)
verified/<sid>.json                — solutions that passed 100% of tests
failed/<sid>.json                  — solutions that failed at least one test
sft/codeforces_solved.jsonl        — final SFT subset, per source
sft/atcoder_solved.jsonl
sft/aizu_solved.jsonl
sft/comp_solved_train.jsonl        — aggregated SFT subset
sft/images/<sid>_<i>.png           — copied problem diagrams
sft/build_summary.json             — counts + distributions
```

Each SFT record carries:

```json
{
  "id":          "codeforces_1244D",
  "source":      "codeforces",
  "title":       "...",
  "lang":        "en",
  "image":       "images/codeforces_1244D_0.png",
  "images":      ["images/...png", ...],
  "query":       "<image>\n\nYou are an exceptionally intelligent ...",
  "code":        "import sys\n...",
  "response":    "```python\n...\n```",
  "n_existing_tests": 3,
  "n_extra_tests":    0,
  "n_total_tests":    3,
  "test_cases":  [...]
}
```
