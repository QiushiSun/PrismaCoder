#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#  Unified execution + evaluation pipeline
#
#  Usage:
#    ./run.sh                          # run-id 1, exec + eval
#    ./run.sh --run-id 2               # run a specific run
#    ./run.sh --skip-eval              # execution only
#    ./run.sh --skip-exec              # evaluation only (re-score)
#    ./run.sh --run-id 2 --skip-exec   # evaluate run 2 only
#    ./run.sh --force                  # ignore cached scores
#    ./run.sh --run-id 7 --lang html   # only run html
#    ./run.sh --lang html,mermaid      # run multiple selected languages
#
#  Folder layout:
#    runs/
#      runs-1/
#        runs-1-python/   (each language keeps its full name)
#        runs-1-html/
#        ...
#        exec-logs/       (execution logs per language)
#        eval-logs/       (evaluation logs per language)
# ==============================================================================

# ---- Defaults ----------------------------------------------------------------
RUN_ID=1
SKIP_EXEC=false
SKIP_EVAL=false
LANG_FILTER=""

# ---- Evaluation config (edit as needed) --------------------------------------
JUDGE_MODEL=gpt-4.1
JUDGE_BASE_URL=https://az.gptplus5.com/v1
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-HuO5OBUwf8h7wuUncEdyflAVQc2Z8pqpi3WQtpOnoyO0whER}"

# ---- Parallelism config -----------------------------------------------------
TOTAL_WORKERS=32        # execution: total process pool budget
SHARD_SIZE=8            # samples per Jupyter notebook
EXEC_TIMEOUT=600        # per-language execution timeout (seconds)
EXEC_GLOBAL_TIMEOUT=900 # absolute wall-clock limit for execution phase
TOTAL_CONCURRENCY=128   # evaluation: max concurrent API calls
FORCE_RESCORE=false     # set true to ignore cached scores

# ---- Parse CLI args ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)    RUN_ID="$2"; shift 2 ;;
        --skip-exec) SKIP_EXEC=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --force)     FORCE_RESCORE=true; shift ;;
        --lang|-l)   LANG_FILTER="${LANG_FILTER:+${LANG_FILTER},}$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/runs/runs-${RUN_ID}"
AGGREGATE_PATH="${RUN_DIR}/vpb-results.xlsx"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: ${RUN_DIR}"
    echo "  Hint: run 'python setup_run.py --run-id ${RUN_ID}' first to migrate flat folders."
    exit 1
fi

# ---- Auto-detect languages from run folder -----------------------------------
PREFIX="runs-${RUN_ID}-"
LANGUAGES=()
for mf in "${RUN_DIR}"/runs-${RUN_ID}-*/manifest.json; do
    [ -f "$mf" ] || continue
    lang_dir="$(dirname "$mf")"
    folder_name="$(basename "$lang_dir")"
    lang="${folder_name#${PREFIX}}"
    LANGUAGES+=("$lang")
done

if [ ${#LANGUAGES[@]} -eq 0 ]; then
    echo "ERROR: No language subfolders (runs-${RUN_ID}-*/) with manifest.json found in ${RUN_DIR}/"
    exit 1
fi

# ---- Filter languages if --lang was specified --------------------------------
ALL_LANGUAGES=("${LANGUAGES[@]}")

if [ -n "$LANG_FILTER" ]; then
    LANGUAGES=()
    IFS=',' read -ra REQUESTED <<< "$LANG_FILTER"
    for req in "${REQUESTED[@]}"; do
        req="$(echo "$req" | xargs)"  # trim whitespace
        FOUND=false
        for avail in "${ALL_LANGUAGES[@]}"; do
            if [ "$req" = "$avail" ]; then
                LANGUAGES+=("$req")
                FOUND=true
                break
            fi
        done
        if [ "$FOUND" = false ]; then
            echo "ERROR: Language '${req}' not found in run-${RUN_ID}."
            echo "  Available: ${ALL_LANGUAGES[*]}"
            exit 1
        fi
    done
    echo "  [FILTER] Selected languages: ${LANGUAGES[*]}  (from ${#ALL_LANGUAGES[@]} available)"
fi

TOTAL_LANGS=${#LANGUAGES[@]}

# ---- Auto-fix repo_root in all manifests -------------------------------------
for LANG in "${LANGUAGES[@]}"; do
    MF="${RUN_DIR}/runs-${RUN_ID}-${LANG}/manifest.json"
    python3 -c "
import json, sys
p = sys.argv[1]
with open(p) as f: d = json.load(f)
rr = sys.argv[2]
if d.get('repo_root') != rr:
    d['repo_root'] = rr
    with open(p, 'w') as f: json.dump(d, f, indent=2, ensure_ascii=False); f.write('\n')
" "$MF" "${SCRIPT_DIR}/"
done

# ---- Auto-strip language tag prefix from code field in generations ---------
python3 -c "
import json, sys
from pathlib import Path
TAGS = {'python','py','python3','html','javascript','js','typescript','ts',
        'json','xml','css','bash','sh','shell','zsh','sql','yaml','yml',
        'ruby','rb','java','c','cpp','csharp','cs','go','rust','r',
        'swift','kotlin','scala','perl','php','lua','text','plain',
        'plaintext','markdown','md','csv','toml','ini','jsx','tsx'}
run_dir = Path(sys.argv[1])
run_id = sys.argv[2]
fixed = 0
for gf in sorted(run_dir.glob(f'runs-{run_id}-*/generations.jsonl')):
    lines = gf.read_text().strip().split('\n')
    out, n = [], 0
    for line in lines:
        d = json.loads(line)
        code = d.get('code', '')
        parts = code.split('\n', 1)
        if len(parts) > 1 and parts[0].strip().lower() in TAGS:
            d['code'] = parts[1]; n += 1
        out.append(json.dumps(d, ensure_ascii=False))
    if n:
        gf.write_text('\n'.join(out) + '\n')
        fixed += n
if fixed:
    print(f'  [auto-fix] Stripped code fence tags from {fixed} generation(s)')
" "${RUN_DIR}" "${RUN_ID}"

echo ""
echo "======================================================================"
echo "  Run: ${RUN_ID}  |  Languages: ${LANGUAGES[*]}"
echo "  Exec: $( [ "$SKIP_EXEC" = true ] && echo SKIP || echo ON )  |  Eval: $( [ "$SKIP_EVAL" = true ] && echo SKIP || echo ON )"
echo "======================================================================"

# ==============================================================================
#  PHASE 1: EXECUTION
# ==============================================================================
if [ "$SKIP_EXEC" = false ]; then

PER_LANG_W=$((TOTAL_WORKERS / TOTAL_LANGS))
if [ "$PER_LANG_W" -lt 1 ]; then PER_LANG_W=1; fi

EXEC_LOG_DIR="${RUN_DIR}/exec-logs"
mkdir -p "$EXEC_LOG_DIR"

EXEC_START=$(date +%s)

echo ""
echo "  [EXECUTION] ${TOTAL_LANGS} languages  |  Workers: ${TOTAL_WORKERS} (${PER_LANG_W}/lang)  |  Shard: ${SHARD_SIZE}  |  Timeout: ${EXEC_TIMEOUT}s"
echo ""

EXEC_PIDS=()
for LANG in "${LANGUAGES[@]}"; do
    LOG="${EXEC_LOG_DIR}/${LANG}.log"
    python execution.py \
        --work-dir "${RUN_DIR}/runs-${RUN_ID}-${LANG}/" \
        --workers ${PER_LANG_W} \
        --shard-size ${SHARD_SIZE} \
        --timeout ${EXEC_TIMEOUT} > "${LOG}" 2>&1 &
    EXEC_PIDS+=($!)
    echo "    Started ${LANG} (PID $!) -> ${LOG}"
done

echo ""

while true; do
    DONE=0
    ALIVE=0
    for i in "${!LANGUAGES[@]}"; do
        LANG="${LANGUAGES[$i]}"
        PID="${EXEC_PIDS[$i]}"
        if grep -q "Execution summary" "${EXEC_LOG_DIR}/${LANG}.log" 2>/dev/null; then
            DONE=$((DONE + 1))
        elif ! kill -0 "$PID" 2>/dev/null; then
            DONE=$((DONE + 1))
        else
            ALIVE=$((ALIVE + 1))
        fi
    done
    NOW=$(date +%s)
    SEC=$((NOW - EXEC_START))
    printf "\r    Execution progress: %d/%d completed  (%ds)" "$DONE" "$TOTAL_LANGS" "$SEC"
    if [ "$DONE" -eq "$TOTAL_LANGS" ]; then echo ""; break; fi
    if [ "$SEC" -gt "$EXEC_GLOBAL_TIMEOUT" ]; then
        echo ""
        echo "    [TIMEOUT] ${EXEC_GLOBAL_TIMEOUT}s — killing ${ALIVE} remaining"
        for PID in "${EXEC_PIDS[@]}"; do
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null && pkill -9 -P "$PID" 2>/dev/null
        done
        sleep 1; break
    fi
    sleep 2
done

EXEC_FAILED=0
for PID in "${EXEC_PIDS[@]}"; do
    wait "$PID" 2>/dev/null || EXEC_FAILED=$((EXEC_FAILED + 1))
done

EXEC_END=$(date +%s)
EXEC_ELAPSED=$((EXEC_END - EXEC_START))
echo ""
echo "  [EXECUTION] Done in $((EXEC_ELAPSED/60))m $((EXEC_ELAPSED%60))s  ($( [ $EXEC_FAILED -gt 0 ] && echo "${EXEC_FAILED} failed" || echo "all ok" ))"

fi  # SKIP_EXEC

# ==============================================================================
#  PHASE 2: EVALUATION
# ==============================================================================
if [ "$SKIP_EVAL" = false ]; then

PER_LANG_C=$((TOTAL_CONCURRENCY / TOTAL_LANGS))
if [ "$PER_LANG_C" -lt 1 ]; then PER_LANG_C=1; fi

RESUME_FLAG="--resume"
if [ "$FORCE_RESCORE" = true ]; then RESUME_FLAG="--no-resume"; fi

API_LOG_DIR="${SCRIPT_DIR}/api-logs-2604"
mkdir -p "$API_LOG_DIR"
EVAL_LOG_DIR="${RUN_DIR}/eval-logs"
mkdir -p "$EVAL_LOG_DIR"

EVAL_START=$(date +%s)

echo ""
echo "  [EVALUATION] ${TOTAL_LANGS} languages  |  Judge: ${JUDGE_MODEL}  |  Concurrency: ${TOTAL_CONCURRENCY} (${PER_LANG_C}/lang)"
echo ""

EVAL_PIDS=()
for LANG in "${LANGUAGES[@]}"; do
    LOG="${EVAL_LOG_DIR}/${LANG}.log"
    python evaluation.py \
        --bundle-dir "${RUN_DIR}/runs-${RUN_ID}-${LANG}/" \
        --judge-model ${JUDGE_MODEL} \
        --base-url ${JUDGE_BASE_URL} \
        --api-key-env OPENAI_API_KEY \
        --concurrency ${PER_LANG_C} \
        --log-dir ${API_LOG_DIR} \
        ${RESUME_FLAG} > "${LOG}" 2>&1 &
    EVAL_PIDS+=($!)
    echo "    Started ${LANG} (PID $!) -> ${LOG}"
done

echo ""

while true; do
    DONE=0
    for LANG in "${LANGUAGES[@]}"; do
        if grep -q "vis_mean=" "${EVAL_LOG_DIR}/${LANG}.log" 2>/dev/null; then
            DONE=$((DONE + 1))
        fi
    done
    NOW=$(date +%s)
    SEC=$((NOW - EVAL_START))
    printf "\r    Evaluation progress: %d/%d completed  (%ds)" "$DONE" "$TOTAL_LANGS" "$SEC"
    if [ "$DONE" -eq "$TOTAL_LANGS" ]; then echo ""; break; fi
    sleep 2
done

EVAL_FAILED=0
for PID in "${EVAL_PIDS[@]}"; do
    wait "$PID" 2>/dev/null || EVAL_FAILED=$((EVAL_FAILED + 1))
done

EVAL_END=$(date +%s)
EVAL_ELAPSED=$((EVAL_END - EVAL_START))
echo ""
echo "  [EVALUATION] Done in $((EVAL_ELAPSED/60))m $((EVAL_ELAPSED%60))s  ($( [ $EVAL_FAILED -gt 0 ] && echo "${EVAL_FAILED} failed" || echo "all ok" ))"

fi  # SKIP_EVAL

# ==============================================================================
#  PHASE 3: AGGREGATION + SUMMARY
# ==============================================================================
echo ""
echo "  [AGGREGATE] Writing results..."

LANG_STRING="${ALL_LANGUAGES[*]}"

AGG_XLSX="${SCRIPT_DIR}/vpb-results-agg.xlsx"
AGG_MD="${SCRIPT_DIR}/vpb-results-agg.md"

python aggregate_results.py \
    --base-folder "${RUN_DIR}" \
    --experiment-id "${RUN_ID}" \
    --run-id "runs-${RUN_ID}" \
    --languages ${LANG_STRING} \
    --output "${AGGREGATE_PATH}" \
    --agg-output "${AGG_XLSX}" \
    --agg-md "${AGG_MD}"

echo ""
echo "  Per-run results -> ${AGGREGATE_PATH}"
echo "  Aggregated xlsx -> ${AGG_XLSX}"
echo "  Aggregated md   -> ${AGG_MD}"
echo ""
