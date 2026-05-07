#!/usr/bin/env bash
# End-to-end runner for the competition-solve pipeline.
# Edit the SEEDS_DIR variable below to point at your own raw OJ dump
# (codeforces.json / atcoder.json / aizu.json + an `images/` folder).
#
# This script is intended for SMOKE RUNS — for real ingest you'll want
# to run each stage individually with tuned --concurrency / --workers.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

# ── Configuration ───────────────────────────────────────────────────────────
SEEDS_DIR="${SEEDS_DIR:-$ROOT/seeds_sample}"
CONCURRENCY="${CONCURRENCY:-8}"
SOLVE_LIMIT="${SOLVE_LIMIT:-15}"   # 0 = no limit
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-8}"
MODEL="${MODEL:-claude-sonnet-4-6-aws}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY env var is required."
  exit 1
fi

echo "▶ Stage 1/7: parse seeds  (SEEDS_DIR=$SEEDS_DIR)"
SEEDS_DIR="$SEEDS_DIR" python parse_seeds.py

echo "▶ Stage 2/7: translate JP problems"
python translate_jp.py --concurrency "$CONCURRENCY"

echo "▶ Stage 3/7: solve with Claude  (limit=$SOLVE_LIMIT)"
python solve_with_claude.py \
    --concurrency "$CONCURRENCY" \
    --model "$MODEL" \
    ${SOLVE_LIMIT:+--limit "$SOLVE_LIMIT"}

echo "▶ Stage 4/7: recover code from cached responses"
python recover_solutions.py

echo "▶ Stage 5/7: sandbox-verify"
python verify_sandbox.py --workers "$WORKERS" --timeout "$TIMEOUT"

echo "▶ Stage 6/7: retry failures"
python retry_failed.py --concurrency "$CONCURRENCY" --model "$MODEL"

echo "▶ Stage 7a/7: re-verify with retries"
python verify_combined.py --workers "$WORKERS" --timeout "$TIMEOUT"

echo "▶ Stage 7b/7: build SFT records"
python build_sft.py

echo "✅ done — see sft/comp_solved_train.jsonl"
