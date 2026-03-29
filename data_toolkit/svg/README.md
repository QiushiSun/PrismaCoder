# SVG Checker

Extract SVG code from JSONL `output` fields, validate XML structure, render to PNG, and split entries into `passed.jsonl` / `failed.jsonl`.

## Prerequisites

### System dependencies

```bash
# macOS
brew install cairo

# Ubuntu/Debian
sudo apt-get install libcairo2-dev
```

### Python packages

```bash
pip install cairosvg Pillow
```

## Usage

```bash
# Process a single JSONL file
python checker.py path/to/data.jsonl

# Process a directory of JSONL files
python checker.py path/to/jsonl_dir/

# Custom output directories
python checker.py data.jsonl -o ./my_pngs -r ./my_results

# Adjust render settings
python checker.py data.jsonl --width 2400 --dpi 150
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./png_output` | Directory for rendered PNGs |
| `--result`, `-r` | `./results` | Directory for `passed.jsonl` / `failed.jsonl` |
| `--width`, `-w` | `1600` | PNG width in pixels |
| `--dpi`, `-d` | `96` | Output DPI |

## Input format

Each JSONL line must be a JSON object with an `output` field containing SVG code:

```json
{"uuid": "abc123", "output": "<svg xmlns=\"...\">...</svg>", ...}
```

The SVG can be wrapped in markdown code fences (` ```xml ... ``` `) or appear as raw text.

## Output

- **`passed.jsonl`** — entries that rendered successfully, with added `png_path` and `svg_extract_method` fields
- **`failed.jsonl`** — entries that failed, with added `failure_stage` and `failure_message` fields

Failure stages: `json_parse`, `extract`, `xml_validate`, `render`.

## Quick trial

```bash
python checker.py sample_data/sample.jsonl -o /tmp/svg_trial_png -r /tmp/svg_trial_results
```

## File structure

```
svg/
├── checker.py              # Main entry point
├── utils.py                # SVG extraction, validation, rendering
├── sample_data/
│   └── sample.jsonl        # 8-line sample for trial runs
└── README.md
```
