# VisPlotBench Evaluation Pipeline

An optimized execution and eval pipeline for [VisPlotBench](https://huggingface.co/datasets/TIGER-Lab/VisPlotBench) implemented by PrismaCoder suthors.

This pipeline is built on top of the original VisPlotBench evaluation protocol and introduces several improvements in execution robustness, parallelism, and reporting.

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

## Dependencies

### Python Packages

| Package | Purpose |
|---|---|
| `pandas` | DataFrame operations for dataset handling |
| `openpyxl` | Excel report generation |
| `nbformat` | Jupyter notebook creation and parsing |
| `nbconvert` | Notebook execution (`ExecutePreprocessor`) |
| `playwright` | HTML / SVG rendering (Chromium headless) |
| `Pillow` | Image processing and format conversion |
| `omegaconf` | Configuration management |
| `nest-asyncio` | Nested async event loop support (for Jupyter) |
| `tqdm` | Progress bars |
| `httpx` | Async HTTP client for LLM API calls |
| `aiohttp` | Async HTTP (alternative client) |

Install Python dependencies:

```bash
pip install pandas openpyxl nbformat nbconvert playwright Pillow omegaconf nest-asyncio tqdm httpx aiohttp
playwright install chromium
```

### External Tools

| Tool | Languages | Install |
|---|---|---|
| Chromium (via Playwright) | HTML, SVG | `playwright install chromium` |
| `pdflatex` | LaTeX | TeX distribution (e.g. MacTeX, TeX Live) |
| `asy` (Asymptote) | Asymptote | Bundled with TeX or `brew install asymptote` |
| `lilypond` | LilyPond | `brew install lilypond` |
| `mmdc` (Mermaid CLI) | Mermaid | `npm install -g @mermaid-js/mermaid-cli` |
| Node.js | Vega-Lite, Mermaid | `brew install node` |

### Environment Variable

```bash
export OPENAI_API_KEY="your-api-key"
```

The judge model endpoint and API key are configured in `run.sh` (variables `JUDGE_MODEL`, `JUDGE_BASE_URL`, `OPENAI_API_KEY`).

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
- **Per-cell timeout tuning** — language-specific `ExecutePreprocessor` timeouts (HTML: 120s, Mermaid: 60s, LaTeX: 120s, etc.) to avoid unnecessary `KeyboardInterrupt` failures
- **Language filtering** — `--lang html,mermaid` flag to selectively re-run specific languages without re-running the entire benchmark
- **Aggregated reporting** — per-run Excel + global `vpb-results-agg.xlsx` with timestamped rows + lightweight `vpb-results-agg.md` markdown table