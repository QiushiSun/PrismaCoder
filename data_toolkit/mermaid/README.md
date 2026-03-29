# Mermaid Checker

Extract Mermaid diagram code from JSONL `output` fields, render to PNG via `mmdc` (Mermaid CLI), and split entries into `passed.jsonl` / `failed.jsonl`.

## Prerequisites

### System dependencies

```bash
# Node.js (v16+) is required for mmdc
# macOS
brew install node

# Ubuntu/Debian
sudo apt-get install nodejs npm
```

### Install Mermaid CLI

```bash
# Make sure you're using Node 16+ (e.g. via nvm)
nvm use 20   # if using nvm

# Install locally (checker auto-detects node_modules/.bin/mmdc in parent dirs)
npm install @mermaid-js/mermaid-cli

# Install Puppeteer headless Chrome (required by mmdc)
npx puppeteer browsers install chrome-headless-shell
```

Verify installation:

```bash
npx mmdc --version
```

## Usage

```bash
# Process a single JSONL file
python checker.py path/to/data.jsonl

# Process a directory of JSONL files
python checker.py path/to/jsonl_dir/

# Custom output directories and parallelism
python checker.py data.jsonl -o ./my_pngs -r ./my_results -j 16

# Increase timeout for complex diagrams
python checker.py data.jsonl --timeout 60
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./png_output` | Directory for rendered PNGs |
| `--result`, `-r` | `./results` | Directory for `passed.jsonl` / `failed.jsonl` |
| `--workers`, `-j` | `8` | Number of parallel render workers |
| `--timeout`, `-t` | `30` | Per-diagram render timeout in seconds |

## Input format

Each JSONL line must be a JSON object with an `output` field containing Mermaid code:

```json
{"uuid": "abc123", "output": "```mermaid\ngraph TD\n  A --> B\n```", ...}
```

Supported Mermaid diagram types: `graph`, `flowchart`, `sequenceDiagram`, `classDiagram`, `stateDiagram`, `erDiagram`, `gantt`, `pie`, `journey`, `gitGraph`, `mindmap`, `timeline`, `quadrantChart`, `sankey`, `xychart`, `block-beta`, `C4*`, `requirementDiagram`, `architecture`.

## Output

- **`passed.jsonl`** — entries that rendered successfully, with added `png_path` and `mermaid_extract_method` fields
- **`failed.jsonl`** — entries that failed, with added `failure_stage` and `failure_message` fields

Failure stages: `json_parse`, `extract`, `render`.

## Quick trial

```bash
npm install @mermaid-js/mermaid-cli   # first time only
python checker.py sample_data/sample.jsonl -o /tmp/mermaid_trial_png -r /tmp/mermaid_trial_results
```

## File structure

```
mermaid/
├── checker.py              # Main entry point
├── utils.py                # Mermaid extraction, mmdc discovery, rendering
├── sample_data/
│   └── sample.jsonl        # 8-line sample for trial runs
└── README.md
```
