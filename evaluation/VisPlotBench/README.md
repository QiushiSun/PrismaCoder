# VisPlotBench Evaluation Pipeline

An optimized execution and eval pipeline for [VisPlotBench](https://huggingface.co/datasets/TIGER-Lab/VisPlotBench) implemented by PrismaCoder suthors.

This pipeline is built on top of the original VisPlotBench evaluation protocol and introduces several improvements in execution robustness, parallelism, and reporting.

## Setup

You need to install dependencies **once** before running the pipeline. We do NOT
auto-install anything when you call `run.sh` — your environment, your call.

### Option A: one-shot installer (recommended)

```bash
./setup.sh              # python deps + playwright chromium + tool check
./setup.sh --py-only    # only pip install -r requirements.txt
./setup.sh --check      # don't install anything, just report what's missing
```

### Option B: manual

```bash
# 1. Python deps (manifest is the source of truth)
pip install -r requirements.txt

# 2. Playwright browser (needed for html / svg samples)
playwright install chromium

# 3. External CLI tools (only the ones for the languages you care about)
#    macOS examples:
brew install asymptote lilypond node
brew install --cask mactex      # for latex / lualatex
npm i -g @mermaid-js/mermaid-cli  # for mermaid

# 4. Judge API key
export OPENAI_API_KEY="sk-..."
```

> `run.sh` runs a quick preflight check on every invocation and prints a
> warning (not an error) for any missing tool, so you'll know exactly which
> install step you skipped. Pass `--skip-preflight` to silence it.

## Usage

### Quick Start

```bash
# Run execution + evaluation for run-1 (all languages)
./run.sh --run-id 1

# Only re-run HTML execution and evaluation
./run.sh --run-id 7 --lang html

# Re-run multiple languages
./run.sh --run-id 7 --lang html,mermaid

# Skip execution, only re-evaluate
./run.sh --run-id 7 --skip-exec

# Skip evaluation, only re-execute
./run.sh --run-id 7 --skip-eval

# Force re-score (ignore cached scores)
./run.sh --run-id 7 --force
```

### Individual Components

**Execution only:**

```bash
python execution.py \
    --work-dir runs/runs-1/runs-1-python/ \
    --workers 4 \
    --shard-size 8 \
    --timeout 600
```

**Evaluation only:**

```bash
python evaluation.py \
    --bundle-dir runs/runs-1/runs-1-python/ \
    --judge-model gpt-4.1 \
    --base-url https://api.openai.com/v1 \
    --api-key-env OPENAI_API_KEY \
    --concurrency 16 \
    --resume
```

**Aggregation only:**

```bash
python aggregate_results.py \
    --base-folder runs/runs-1 \
    --experiment-id 1 \
    --run-id runs-1 \
    --languages python html svg vegalite mermaid latex asymptote lilypond \
    --output runs/runs-1/vpb-results.xlsx \
    --agg-output vpb-results-agg.xlsx \
    --agg-md vpb-results-agg.md
```

## Dependencies (reference)

### Python Packages

Authoritative list lives in [`requirements.txt`](./requirements.txt). At a glance:

| Group | Packages | Why |
|---|---|---|
| Harness | `pandas` `openpyxl` `nbformat` `nbconvert` `ipykernel` `omegaconf` `nest-asyncio` `tqdm` `httpx` `aiohttp` | core eval pipeline |
| Rendering | `playwright` `Pillow` | html / svg / image diff |
| Python lane | `matplotlib` `numpy` `scipy` `seaborn` `networkx` `scikit-learn` | needed for the model-generated Python samples to **execute at all** — without them the `python` language reports 100% `ModuleNotFoundError` |

### External Tools

| Tool | Languages | Install |
|---|---|---|
| Chromium (via Playwright) | HTML, SVG | `playwright install chromium` |
| `pdflatex` / `lualatex` | LaTeX | TeX distribution (MacTeX, TeX Live) |
| `asy` (Asymptote) | Asymptote | bundled with TeX, or `brew install asymptote` |
| `lilypond` | LilyPond | `brew install lilypond` |
| `mmdc` (Mermaid CLI) | Mermaid | `npm i -g @mermaid-js/mermaid-cli` |
| Node.js | Vega-Lite, Mermaid | `brew install node` |

`run.sh` defensively adds `/Library/TeX/texbin`, `/opt/homebrew/bin`, etc. to
`PATH` if those directories exist, so tools installed via standard Mac/Linux
package managers will be discoverable even from non-login shells.

### API key

```bash
export OPENAI_API_KEY="sk-..."   # required, no default shipped
```

The judge endpoint defaults to `https://api.openai.com/v1`. Override with
`JUDGE_BASE_URL` / `JUDGE_MODEL` env vars if you use a proxy.

## Configuration

Key parameters in `run.sh`:

| Parameter | Default | Description |
|---|---|---|
| `TOTAL_WORKERS` | 32 | Total process pool workers for execution |
| `SHARD_SIZE` | 8 | Samples per Jupyter notebook shard |
| `EXEC_TIMEOUT` | 600 | Per-language execution timeout (seconds) |
| `EXEC_GLOBAL_TIMEOUT` | 900 | Absolute wall-clock limit for execution phase |
| `TOTAL_CONCURRENCY` | 128 | Max concurrent LLM API calls for evaluation |
| `JUDGE_MODEL` | gpt-4.1 | LLM judge model for scoring |

## Supported Languages

| Language | Samples | Rendering Method |
|---|---|---|
| Python | 196 | Jupyter kernel (matplotlib, seaborn, plotly, etc.) |
| HTML | 108 | Playwright Chromium (shared browser per shard) |
| SVG | 65 | XML validation + Playwright rendering |
| Vega-Lite | 129 | JSON validation + vl2png / Playwright |
| Mermaid | 131 | `mmdc` CLI (Mermaid CLI) |
| LaTeX | 112 | `pdflatex` → PDF → PNG conversion |
| Asymptote | 92 | `asy` CLI rendering |
| LilyPond | 55 | `lilypond` CLI → PDF → PNG conversion |

## Improvements over the Original Pipeline

We make some improvements over the original pipeline:

- **Unified entry point** (`run.sh`) that orchestrates execution, evaluation, and aggregation in one command
- **Parallel sharded execution** — samples are split into Jupyter notebook shards and executed across multiple workers via `ProcessPoolExecutor`
- **Browser reuse for HTML** — a single Chromium instance is launched per shard and shared across all samples (instead of one browser per sample), with `networkidle` wait strategy for adaptive rendering
- **Robust code extraction** — whitelist-based stripping of code-fence language tags from model outputs, applied at extraction time, pre-execution patching, and runtime
- **Playwright robustness guard** — each HTML sample cell checks that the shared browser was initialized successfully; if not, a clear `RuntimeError` is raised with the fix command (`playwright install chromium`), preventing silent 0-score failures
- **Per-cell timeout tuning** — language-specific `ExecutePreprocessor` timeouts (HTML: 120s, Mermaid: 60s, LaTeX: 120s, etc.) to avoid unnecessary `KeyboardInterrupt` failures
- **Language filtering** — `--lang html,mermaid` flag to selectively re-run specific languages without re-running the entire benchmark
- **Aggregated reporting** — per-run Excel + global `vpb-results-agg.xlsx` with timestamped rows + lightweight `vpb-results-agg.md` markdown table
