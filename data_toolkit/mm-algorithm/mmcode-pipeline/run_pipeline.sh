#!/usr/bin/env bash
# End-to-end runner for the MMCode augmentation pipeline.
# Smoke run: ~6 ICL-generated MMCode problems, < 5 minutes, < $2.
#
# By default it reads the bundled seeds_sample/mmcode_full.jsonl + mmcode_test.jsonl
# (slimmed copies of the upstream MMCode dataset).  To run on the full corpus,
# drop your own mmcode_full.jsonl + mmcode_test.jsonl into this directory.

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

# ── Configuration ───────────────────────────────────────────────────────────
BATCHES="${BATCHES:-2}"               # batches × problems_per_batch = total
PPB="${PPB:-3}"                        # problems per batch
CONCURRENCY="${CONCURRENCY:-4}"
MODEL="${MODEL:-claude-sonnet-4-6-aws}"
DIFFICULTY="${DIFFICULTY:-standard}"  # standard | hard
OUTPUT_DIR="${OUTPUT_DIR:-output}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY env var is required."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "▶ Stage 1/3: extract ingredient pools (algorithm tags, visual tags, etc.)"
python ingredients.py

echo "▶ Stage 2/3: generate $((BATCHES * PPB)) MMCode-style problems"
python generate_problems.py \
    --model "$MODEL" \
    --batches "$BATCHES" \
    --problems-per-batch "$PPB" \
    --concurrency "$CONCURRENCY" \
    --difficulty-mode "$DIFFICULTY" \
    --output-dir "$OUTPUT_DIR"

echo "▶ Stage 3/3: dedup near-duplicates"
python dedup_problems.py \
    --input  "$OUTPUT_DIR/problems.jsonl" \
    --output "$OUTPUT_DIR/problems.dedup.jsonl"

echo "✅ done — see $OUTPUT_DIR/problems.dedup.jsonl"
echo
echo "Optional next step (multi-rollout SFT packaging):"
echo "  python build_training_set.py     # expects 4 rollout dirs as in production"
