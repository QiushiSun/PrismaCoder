# MMCode Augmentation Pipeline

ICL-driven augmentation of [MMCode](https://huggingface.co/datasets/likaixin/MMCode):
given an MMCode test record (problem prose + diagrams), Claude is shown
two reference problems and asked to generate a *new* problem that
preserves MMCode's visual + algorithmic vocabulary.  The pipeline also
provides an SFT exporter for the upstream MMCode train split.

## Pipeline stages

```
seeds_sample/mmcode_seeds.jsonl  ── small sample (or supply full mmcode_test.jsonl)
            │
            ▼
ingredients.{py,json}            ── tag distribution + scenario hints
            │
            ▼
generate_problems.py             ── Claude rolls out new MMCode-style problems
            │
            ▼
dedup_problems.py                ── drop near-duplicate generations
            │
            ▼
recover_failed_batches.py        ── (optional) salvage partial batches
            │
            ▼
build_training_set.py            ── package into final SFT jsonl + viewer

extract_mmcode_full_sft.py       ── (orthogonal) wrap upstream mmcode_full.jsonl
                                    into the same SFT schema
```

## Quick smoke test

```bash
export OPENAI_BASE_URL="https://your-gateway/v1"
export OPENAI_API_KEY="sk-..."

cd open-source-pipelines/mmcode-pipeline
bash run_pipeline.sh   # uses seeds_sample/, generates ~30 problems
```

## Full run

For our 3 500-record augmented MMCode pack (~$40 / ~2 hours at concurrency=24):

```bash
# 1. Acquire MMCode test split
wget https://huggingface.co/datasets/likaixin/MMCode/resolve/main/mmcode_test.jsonl

# 2. Generate
python generate_problems.py --rollout-dir runs/svg-train --target 4000 \
    --seeds-path mmcode_test.jsonl

# 3. Dedup + finalize
python dedup_problems.py --rollout-dir runs/svg-train
python build_training_set.py --rollout-dir runs/svg-train
```

To create the MMCode-Full SFT (no augmentation, just packaging):

```bash
wget https://huggingface.co/datasets/likaixin/MMCode/resolve/main/mmcode_full.jsonl
python extract_mmcode_full_sft.py --src mmcode_full.jsonl --out sft_full/
```

## Output

```
runs/<rollout>/problems.jsonl              — raw LLM output (one problem/line)
runs/<rollout>/problems.dedup.jsonl        — after deduplication
runs/<rollout>/manifest.json
runs/<rollout>/mmcode-rollout-viewer.html  — quick visual sanity check
training-set/train.jsonl                   — final SFT pack
training-set/images/                       — copied diagrams
```

## Notes

* `ingredients.json` is hand-curated with 26 visual aspects (graph,
  grid, polygon, circuit, etc.) and 60 scenario domains.  Edit it to
  shift the tag mix.
* `generate_problems.py` supports `--retry-failed` to re-prompt only
  the templates that crashed validation.
* Sandbox-verification is INTRA-process (templates are run inside a
  subprocess for safety); see `sandbox.py`.
