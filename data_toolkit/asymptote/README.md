# Asymptote Checker

Extract Asymptote code from JSONL `output` fields, render to PNG via `asy`, and split entries into `passed.jsonl` / `failed.jsonl`.

## Prerequisites

### System dependencies

```bash
# macOS
brew install asymptote ghostscript

# Ubuntu/Debian
sudo apt-get install asymptote ghostscript

# Verify installation
asy --version
gs --version
```

- **Asymptote (`asy`)** — required for rendering `.asy` code to PNG
- **Ghostscript (`gs`)** — used as fallback when Asymptote produces PDF/EPS instead of PNG
- **TeX Live** — Asymptote uses LaTeX for labels; a basic TeX Live installation is needed

### Optional: TeX font packages

Some generated code references PostScript fonts (Helvetica, Times). If you have `tlmgr` access:

```bash
sudo tlmgr install psnfss helvetic times
```

The checker auto-strips unavailable font declarations, so this is optional.

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
python checker.py data.jsonl -j 8 -t 60
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./png_output` | Directory for rendered PNGs |
| `--result`, `-r` | `./results` | Directory for `passed.jsonl` / `failed.jsonl` |
| `--workers`, `-j` | `8` | Parallel render workers |
| `--timeout`, `-t` | `60` | Per-script render timeout (seconds) |

## Input format

Each JSONL line must be a JSON object with an `output` field containing Asymptote code:

```json
{"uuid": "abc123", "output": "size(200);\ndraw((0,0)--(1,1));", ...}
```

The code can be wrapped in markdown code fences (`` ```asy ... ``` ``) or appear as raw text.

## Output

- **`passed.jsonl`** — entries that rendered successfully, with added `png_path` and `asy_extract_method` fields
- **`failed.jsonl`** — entries that failed, with added `failure_stage` and `failure_message` fields

Failure stages: `json_parse`, `extract`, `render`.

## Code sanitization

The checker automatically fixes common issues in generated Asymptote code:

- **`settings.outformat` override** — Strips `settings.outformat="pdf"` to force PNG output
- **Missing TeX fonts** — Removes `font("phv",...)`, `font("ptm",...)` and similar unavailable font declarations
- **PDF/EPS fallback** — If `asy` produces PDF or EPS instead of PNG, converts via Ghostscript

## Quick trial

```bash
python checker.py sample_data/sample.jsonl -o /tmp/asy_trial_png -r /tmp/asy_trial_results
```

## File structure

```
asymptote/
├── checker.py              # Main entry point (parallel processing + tqdm)
├── utils.py                # Extraction, sanitization, rendering helpers
├── sample_data/
│   └── sample.jsonl        # 8-line sample for trial runs
└── README.md
```
