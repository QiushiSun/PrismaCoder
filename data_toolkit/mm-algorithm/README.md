# Multimodal-Code Data Augmentation Pipelines

This folder packages the **three LLM-driven data-augmentation pipelines** we
used to grow the multimodal SFT corpus for VisPlotBench.

| Pipeline | What it does | Source data |
|---|---|---|
| `mmcode-pipeline/` | ICL-guided synthesis of new "MMCode-style" problems with rendered SVG diagrams. | Builds on top of [MMCode](https://huggingface.co/datasets/likaixin/MMCode); needs `mmcode_test.jsonl` or `mmcode_full.jsonl`. |
| `competition-pipeline/` | Two flows for online-judge competitive-programming data: <br>• `solve-pipeline/` translates JP problems → EN, multimodal-solves them with Claude, sandbox-verifies the solution, and packages SFT records.<br>• `newgen-pipeline/` parametric-template generator (Plan-A') for *new* LeetCode-style problems. | Online-judge dumps (CodeForces / AtCoder / Aizu) and an existing LeetCode SFT export. |
| `cross-aug-pipeline/` | HARD-biased multimodal problem synthesis that draws **inspiration from a mixed pool** (LeetCode / CF / AC / Aizu / MMCode) — class-Solution output style, advanced algorithmic content (network flow, segment tree, suffix array, geometry, …). | Re-uses ICL examples from the LC pools and "inspiration" examples from the competitive pools (no solutions exposed). |

Each pipeline is **self-contained** and can be invoked end-to-end through
its top-level `run_pipeline.sh`.  A small subsample of the upstream seed
data is pre-bundled under `seeds_sample/` so you can do a smoke run before
trying it on a larger ingest.

## High-level requirements

These pipelines are designed for an OpenAI-compatible chat-completions
gateway with multimodal (image_url) support.  We used Anthropic Claude
4.6 (`claude-sonnet-4-6-aws`) for the heavy generation steps and `gpt-5`
or `gemini-3.1-pro` for spot consensus checks.  Set the credentials via
environment variables:

```bash
export OPENAI_BASE_URL="https://your-gateway/v1"
export OPENAI_API_KEY="sk-..."
```

Python ≥ 3.10 with `conda` is recommended:

```bash
conda create -n vpb python=3.10 -y
conda activate vpb
pip install -r mmcode-pipeline/requirements.txt
pip install -r competition-pipeline/solve-pipeline/requirements.txt
pip install -r competition-pipeline/newgen-pipeline/requirements.txt
pip install -r cross-aug-pipeline/requirements.txt
```

(All four `requirements.txt` files share the same wheel set; install
once or per-pipeline as you prefer.)

## Folder layout

```
open-source-pipelines/
├── README.md                       — this file
├── mmcode-pipeline/                — MMCode-style ICL generator
│   ├── README.md
│   ├── requirements.txt
│   ├── run_pipeline.sh             — end-to-end orchestrator
│   ├── seeds_sample/               — 30 sampled MMCode seeds for smoke runs
│   ├── ingredients.{py,json}
│   ├── generate_problems.py
│   ├── build_training_set.py
│   ├── extract_mmcode_full_sft.py  — packs upstream MMCode train.jsonl into SFT
│   ├── llm_client.py / logger.py / sandbox.py
│   └── ...
├── competition-pipeline/
│   ├── solve-pipeline/             — translate + multimodal-solve + verify
│   │   ├── README.md
│   │   ├── run_pipeline.sh
│   │   ├── seeds_sample/           — small CodeForces / AtCoder / Aizu ingest
│   │   ├── parse_seeds.py
│   │   ├── translate_jp.py
│   │   ├── solve_with_claude.py
│   │   ├── recover_solutions.py
│   │   ├── retry_failed.py
│   │   ├── verify_sandbox.py
│   │   ├── verify_combined.py
│   │   ├── build_sft.py
│   │   └── llm_client.py / logger.py / sandbox.py
│   └── newgen-pipeline/            — parametric template synthesis (Plan-A')
│       ├── README.md
│       ├── run_pipeline.sh
│       ├── seeds_sample/           — 30 LeetCode multimodal records
│       ├── extract_ingredients.py
│       ├── build_seeds.py
│       ├── prompts.py
│       ├── generate_templates.py
│       ├── run_templates.py
│       ├── audit_leakage.py
│       ├── build_sft.py
│       ├── recover_failed.py
│       └── llm_client.py / logger.py / sandbox.py
└── cross-aug-pipeline/             — HARD-biased multi-source synthesis
    ├── README.md
    ├── run_pipeline.sh
    ├── seeds_sample/               — small mix of all four upstream sources
    ├── extract_ingredients.py
    ├── build_seeds.py
    ├── prompts.py
    ├── generate_templates.py
    ├── run_templates.py
    ├── audit_leakage.py
    ├── build_sft.py
    ├── recover_failed.py
    └── llm_client.py / logger.py / sandbox.py
```

## Cost & runtime guidance

For a full run, you will burn API tokens.  Approximate Claude-4.6
spending we observed:

| Pipeline | Records produced | $ cost | Wall time @ 32-way concurrency |
|---|---|---|---|
| mmcode-pipeline (svg train + hard) | ~3 500 | $40 | ~2 hours |
| solve-pipeline (CF + AC + Aizu solve + retry) | 528 → ~900 after retry | $55 | ~80 minutes |
| newgen-pipeline (LeetCode-style, 800 templates × 5 variants) | ~3 500 verified-clean | $25 | ~90 minutes |
| cross-aug-pipeline (500 templates × 5 variants, hard-biased) | ~2 200 verified-clean post-audit | $50 | ~75 minutes |

You can keep token usage to a few cents by capping `--target` to 50–100
on smoke runs.

## License

The pipeline source is released under the MIT license.  The seed data
under `seeds_sample/` is small re-distributable subsamples of upstream
public sources (LeetCode / CodeForces / AtCoder / Aizu / MMCode) and
retains its original upstream licensing — see each `seeds_sample/README`
for attribution.
