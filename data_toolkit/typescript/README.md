# TypeScript Checker

Extract TypeScript visualization code from JSONL `output` fields, compile to JavaScript via `esbuild`, render in a headless Puppeteer browser, screenshot to PNG, and split entries into `passed.jsonl` / `failed.jsonl`.

## Prerequisites

### System dependencies

```bash
# Node.js 16+ (via NVM recommended)
nvm install 20
nvm use 20

# Global tools
npm install -g typescript esbuild
```

### npm packages (install in this directory or a parent)

```bash
npm install puppeteer d3 chart.js @types/d3 jsdom @types/jsdom
```

The checker searches upward for `node_modules/`, so packages installed in a parent directory work fine.

### Python packages

```bash
pip install tqdm
```

## Usage

```bash
# Process a single JSONL file
python checker.py path/to/data.jsonl

# Process a directory of JSONL files
python checker.py path/to/jsonl_dir/

# Custom output directories
python checker.py data.jsonl -o ./my_pngs -r ./my_results

# Adjust workers and timeout
python checker.py data.jsonl -j 4 -t 30
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./png_output` | Directory for rendered PNGs |
| `--result`, `-r` | `./results` | Directory for `passed.jsonl` / `failed.jsonl` |
| `--workers`, `-j` | `4` | Parallel render workers (Puppeteer is memory-heavy) |
| `--timeout`, `-t` | `30` | Per-entry render timeout (seconds) |

## Input format

Each JSONL line must be a JSON object with an `output` field containing TypeScript code:

```json
{"uuid": "abc123", "output": "```typescript\nconst svg = d3.select('body')...\n```", ...}
```

The code can be wrapped in markdown code fences (`` ```typescript ... ``` ``, `` ```ts ... ``` ``) or appear as raw text.

## Rendering pipeline

1. **Extract** — find TypeScript code in markdown fences or raw text
2. **Sanitize** — strip `import`/`export` statements (libraries are shimmed at runtime)
3. **Compile** — `esbuild` strips TypeScript types → plain JavaScript
4. **Render** — inject JS into an HTML page with d3/Chart.js loaded, open in Puppeteer, screenshot

### Library shimming

The render script (`ts_render.js`) loads d3 and Chart.js from local `node_modules` and provides a `require()` shim so generated code referencing these libraries works in the browser.

## Output

- **`passed.jsonl`** — entries that rendered successfully, with added `png_path`, `ts_extract_method`, and `ts_compile_method` fields
- **`failed.jsonl`** — entries that failed, with added `failure_stage` and `failure_message` fields

Failure stages: `json_parse`, `extract`, `compile`, `render`.

## Quick trial

```bash
python checker.py sample_data/sample.jsonl -o /tmp/ts_trial_png -r /tmp/ts_trial_results
```

## File structure

```
typescript/
├── checker.py              # Main entry point (parallel processing + tqdm)
├── utils.py                # TS extraction, esbuild compilation, Puppeteer rendering
├── ts_render.js            # Node.js Puppeteer screenshot helper
├── sample_data/
│   └── sample.jsonl        # 8-line sample for trial runs
└── README.md
```
