#!/usr/bin/env bash
# End-to-end runner for the cross-source HARD-augmentation pipeline.
# Smoke run: target=8 templates × 3 variants ≈ 24 SFT records, < $1.

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

# ── Configuration ───────────────────────────────────────────────────────────
TARGET="${TARGET:-8}"
SEEDS="${SEEDS:-3}"           # variant-seeds per template
CONCURRENCY="${CONCURRENCY:-8}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-8}"
MODEL="${MODEL:-claude-sonnet-4-6-aws}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY env var is required."
  exit 1
fi

echo "▶ Stage 1/6: extract ingredients (advanced algo + struct tags)"
python extract_ingredients.py

echo "▶ Stage 2/6: build cross-source ICL + inspiration seeds"
python build_seeds.py

echo "▶ Stage 3/6: generate $TARGET templates (HARD-biased)"
python generate_templates.py --target "$TARGET" \
    --concurrency "$CONCURRENCY" --model "$MODEL"

echo "▶ Stage 4/6: run templates (variants × seeds=$SEEDS)"
python run_templates.py --seeds "$SEEDS" --workers "$WORKERS" --timeout "$TIMEOUT"

echo "▶ Stage 5/6: leakage audit (vs upstream training pool)"
python audit_leakage.py --sim-cut 0.5

echo "▶ Stage 6/6: build SFT records"
python build_sft.py

echo "✅ done — see sft/train.sft.clean.jsonl"
