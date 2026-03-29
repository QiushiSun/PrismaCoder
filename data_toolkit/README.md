# Data Toolkit — Code Rendering Quality Checker

A suite of tools for validating and rendering code outputs (SVG, Mermaid, R, Asymptote) from JSONL datasets. Each checker extracts code from the `output` field, renders it to PNG, and classifies entries as passed or failed.

## Supported Languages

| Language | Renderer | Approach |
|----------|----------|----------|
| **SVG** | `cairosvg` (Python) | XML validation + rasterization |
| **Mermaid** | `mmdc` (Node.js CLI) | Headless browser rendering |
| **R** | `Rscript` (R) | Execute R code with PNG device; auto-transforms Shiny/plotly/rgl |
| **Asymptote** | `asy` (Asymptote) | Compile `.asy` to PNG; auto-strips outformat/font overrides; PDF→PNG fallback |

## Quick Start

```bash
# SVG
cd svg && python checker.py sample_data/sample.jsonl

# Mermaid (requires: npm install @mermaid-js/mermaid-cli)
cd mermaid && python checker.py sample_data/sample.jsonl

# R (requires: Rscript on PATH)
cd r && python checker.py sample_data/sample.jsonl

# Asymptote (requires: asy + gs on PATH)
cd asymptote && python checker.py sample_data/sample.jsonl
```

## Input Format

All checkers expect JSONL files where each line is a JSON object with at least an `output` field:

```json
{"uuid": "unique-id", "output": "...code here...", "instruction": "...", ...}
```

The code in `output` can be wrapped in markdown code fences (`` ```lang ... ``` ``) or appear as raw text.

## Output Format

Each checker produces two files in the result directory:

- **`passed.jsonl`** — successfully rendered entries (original fields + `png_path`)
- **`failed.jsonl`** — failed entries (original fields + `failure_stage` + `failure_message`)

## Directory Structure

```
data_toolkit/
├── README.md                   # This file
├── svg/
│   ├── checker.py              # SVG checker entry point
│   ├── utils.py                # SVG extraction, validation, rendering
│   ├── sample_data/sample.jsonl
│   └── README.md
├── mermaid/
│   ├── checker.py              # Mermaid checker entry point
│   ├── utils.py                # Mermaid extraction, mmdc rendering
│   ├── sample_data/sample.jsonl
│   └── README.md
├── r/
│   ├── checker.py              # R checker entry point
│   ├── rerun_failed.py         # Re-run failed entries after env upgrades
│   ├── utils.py                # R extraction, interactive transforms, rendering
│   ├── sample_data/sample.jsonl
│   └── README.md
└── asymptote/
    ├── checker.py              # Asymptote checker entry point (parallel + tqdm)
    ├── utils.py                # Asymptote extraction, sanitization, rendering
    ├── sample_data/sample.jsonl
    └── README.md
```

## System Requirements

| Dependency | Required For | Install |
|------------|-------------|---------|
| Python 3.10+ | All | — |
| `cairosvg` (pip) | SVG | `pip install cairosvg Pillow` |
| `libcairo` (system) | SVG | `brew install cairo` / `apt install libcairo2-dev` |
| Node.js 16+ | Mermaid | `brew install node` / `apt install nodejs npm` |
| `@mermaid-js/mermaid-cli` (npm) | Mermaid | `npm install @mermaid-js/mermaid-cli` |
| R 4.0+ | R | `brew install r` / `apt install r-base` |
| R packages (ggplot2, etc.) | R | See `r/README.md` for full list |
| Asymptote (`asy`) | Asymptote | `brew install asymptote` / `apt install asymptote` |
| Ghostscript (`gs`) | Asymptote | `brew install ghostscript` / `apt install ghostscript` |
| TeX Live | Asymptote | Needed for labels; `brew install texlive` / `apt install texlive` |
| `tqdm` (pip) | Asymptote | `pip install tqdm` |

See each language's `README.md` for detailed setup instructions.
