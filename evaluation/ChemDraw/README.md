# ChemDraw Evaluation

Evaluates a Vision-Language Model's ability to convert 2D chemical structure images into **SMILES** (Simplified Molecular-Input Line-Entry System) strings.

## Task

| Input | Output |
|-------|--------|
| 2D molecular structure image (PNG) | Canonical SMILES string |

The model receives a chemical structure image and must output the corresponding SMILES representation. Outputs are parsed, canonicalized with RDKit, and compared against ground-truth labels.

## Dataset

**128 samples** from [allenai/CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K) (chemical subset, validation split), bundled in `data/`.

```
data/
├── manifest.jsonl        # sample metadata (id, image path, ground-truth SMILES)
└── images/               # 128 chemical structure PNGs
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Exec Rate** | Fraction of samples where the API call succeeded (no error) |
| **Validity Rate** | Fraction of predictions that are valid SMILES (parseable by RDKit) |
| **Exact Match (EM)** | Fraction where predicted canonical SMILES equals ground truth |
| **Avg Tanimoto** | Mean Tanimoto similarity (Morgan/ECFP4 fingerprints, radius=2, 2048 bits) |
| **Tanimoto = 1.0** | Fraction with perfect structural match |
| **Tanimoto >= 0.85** | Fraction with high structural similarity |

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** If installing `rdkit-pypi` fails on your platform, install RDKit via conda:
> `conda install -c conda-forge rdkit`

## Usage

The script uses an **OpenAI-compatible** chat API. You must provide `--model`, `--api-key`, and `--base-url`.

```bash
# Full evaluation (128 samples, 4 threads)
python eval_chemdraw.py \
    --model gpt-4o \
    --api-key sk-your-key \
    --base-url https://api.openai.com/v1

# Quick smoke test (5 samples, sequential)
python eval_chemdraw.py \
    --model gpt-4o \
    --api-key sk-your-key \
    --base-url https://api.openai.com/v1 \
    --subset 5 --max-workers 1

# Higher parallelism
python eval_chemdraw.py \
    --model gemini-2.5-pro \
    --api-key <key> \
    --base-url <url> \
    --max-workers 8
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model name |
| `--api-key` | *(required)* | API key |
| `--base-url` | *(required)* | OpenAI-compatible API base URL |
| `--subset N` | all | Evaluate only the first N samples |
| `--max-workers` | 4 | Number of parallel threads |
| `--max-retries` | 3 | Max retries per API call (exponential backoff) |
| `--temperature` | 0.0 | Sampling temperature |
| `--output-dir` | `./results` | Where to save results |
| `--data-dir` | `./data` | Path to the bundled test set |

## Output

Each run produces three artifacts in `results/`:

| File | Content |
|------|---------|
| `chemdraw_<model>_<tag>_<ts>_detail.jsonl` | Per-sample predictions and metrics |
| `chemdraw_<model>_<tag>_<ts>_summary.json` | Aggregate metrics |
| `leaderboard.csv` | Cumulative results table (appended after each run) |

A formatted **leaderboard table** is also printed to stdout after each run.
