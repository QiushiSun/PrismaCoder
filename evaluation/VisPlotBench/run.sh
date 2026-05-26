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
#    ./run.sh --python-subproc         # alias for --python-exec-mode subprocess
#                                      # (each python sample = isolated process,
#                                      #  rescues samples that the legacy
#                                      #  ipykernel path would have killed as
#                                      #  "Shard timed out").
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

# ---- Tool PATH (defensive, no installation) ---------------------------------
# Some IDE / cron / nohup shells don't source ~/.profile, so /Library/TeX/texbin
# and /opt/homebrew/bin may be missing → asy / lualatex / lilypond report as
# "binary not found" even though they ARE installed. We just add the directories
# to PATH *if* they already exist on disk; we never install anything here.
for _p in \
    /Library/TeX/texbin \
    /opt/homebrew/bin /opt/homebrew/sbin \
    /usr/local/bin
do
    if [ -d "$_p" ] && [[ ":${PATH}:" != *":${_p}:"* ]]; then
        PATH="${_p}:${PATH}"
    fi
done
export PATH

# ---- Defaults ----------------------------------------------------------------
RUN_ID=1
SKIP_EXEC=false
SKIP_EVAL=false
LANG_FILTER=""
SKIP_PREFLIGHT=false
PYTHON_EXEC_MODE="jupyter"   # 'jupyter' (legacy, default) | 'subprocess'
PYTHON_SUBPROC_TIMEOUT=60     # per-sample timeout, subprocess mode only.
                              # 60s (not 30s) so modern SFT'd models that emit
                              # heavier matplotlib code (more subplots / large
                              # linspace / dense scatter) don't get false-
                              # positive killed before they finish rendering.
PYTHON_SUBPROC_PARALLEL=1     # subprocesses within one shard.  Keep at 1 on
                              # macOS — nesting Popen inside ThreadPool inside
                              # ProcessPool hangs unpredictably.  Outer
                              # --workers handles real parallelism.

# ---- Evaluation config (edit as needed) --------------------------------------
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.openai.com/v1}"
# OPENAI_API_KEY must be exported by the user (e.g. in shell rc or .env).
# We intentionally do NOT ship a default key — open-source safety.
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=\"sk-...\"   # then re-run"
    exit 2
fi

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
        --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
        --python-exec-mode) PYTHON_EXEC_MODE="$2"; shift 2 ;;
        --python-subproc)   PYTHON_EXEC_MODE="subprocess"; shift ;;
        --python-subproc-timeout)  PYTHON_SUBPROC_TIMEOUT="$2"; shift 2 ;;
        --python-subproc-parallel) PYTHON_SUBPROC_PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$PYTHON_EXEC_MODE" in
    jupyter|subprocess) ;;
    *) echo "ERROR: --python-exec-mode must be 'jupyter' or 'subprocess' (got '${PYTHON_EXEC_MODE}')"; exit 1 ;;
esac

# ---- Preflight: warn about missing system tools / Python packages -----------
# We only WARN — installation is the user's responsibility (see setup.sh / README).
# Pass --skip-preflight to bypass.
_warn_missing() {
    local tool="$1"; local langs="$2"; local hint="$3"
    echo "  [WARN] '${tool}' not found on PATH — languages affected: ${langs}"
    echo "         install hint: ${hint}"
    MISSING_TOOLS=$((MISSING_TOOLS + 1))
}
if [ "$SKIP_PREFLIGHT" = false ]; then
    MISSING_TOOLS=0
    command -v asy        >/dev/null 2>&1 || _warn_missing asy      "asymptote" "brew install asymptote   (or install TeX Live)"
    command -v lualatex   >/dev/null 2>&1 || _warn_missing lualatex "latex"     "install MacTeX / TeX Live"
    command -v lilypond   >/dev/null 2>&1 || _warn_missing lilypond "lilypond"  "brew install lilypond"
    command -v node       >/dev/null 2>&1 || _warn_missing node     "mermaid, vegalite" "brew install node"
    # Python-side preflight (uses `python` from current env).
    # `|| true` so a missing python binary doesn't kill us under `set -e`.
    PY_MISSING=$(python - <<'PY' 2>/dev/null || true
import importlib.util
need = {
    "pandas": "core", "nbformat": "notebook", "nbconvert": "notebook",
    "playwright": "html/svg", "PIL": "image", "omegaconf": "config",
    "tqdm": "progress", "httpx": "judge-api",
    "matplotlib": "python plots", "numpy": "python plots",
    "scipy": "python plots", "seaborn": "python plots",
}
miss = [f"{m} ({need[m]})" for m in need if importlib.util.find_spec(m) is None]
print("|".join(miss))
PY
)
    if [ -n "$PY_MISSING" ]; then
        IFS='|' read -ra MISS <<< "$PY_MISSING"
        for m in "${MISS[@]}"; do
            echo "  [WARN] python package missing: ${m}"
        done
        echo "         install hint: pip install -r requirements.txt   (or run ./setup.sh)"
        MISSING_TOOLS=$((MISSING_TOOLS + ${#MISS[@]}))
    fi
    # Chromium for playwright (only if playwright itself is installed)
    if python -c "import playwright" 2>/dev/null; then
        if ! python -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); p.chromium.executable_path; p.stop()" 2>/dev/null; then
            echo "  [WARN] playwright chromium not installed — html samples will all fail"
            echo "         install hint: playwright install chromium"
            MISSING_TOOLS=$((MISSING_TOOLS + 1))
        fi
    fi
    if [ "$MISSING_TOOLS" -gt 0 ]; then
        echo ""
        echo "  ${MISSING_TOOLS} missing dependency warning(s) — affected languages will report all-fail."
        echo "  Run ./setup.sh to install everything, OR pass --skip-preflight to ignore."
        echo ""
    fi
fi

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
if [ "$PYTHON_EXEC_MODE" = "subprocess" ]; then
    echo "  [python-exec-mode] subprocess  |  per-sample timeout=${PYTHON_SUBPROC_TIMEOUT}s  |  intra-shard parallel=${PYTHON_SUBPROC_PARALLEL}"
fi
echo ""

EXEC_PIDS=()
for LANG in "${LANGUAGES[@]}"; do
    LOG="${EXEC_LOG_DIR}/${LANG}.log"
    EXTRA_ARGS=()
    LANG_SHARD_SIZE=${SHARD_SIZE}
    LANG_WORKERS=${PER_LANG_W}

    if [ "$LANG" = "python" ] && [ "$PYTHON_EXEC_MODE" = "subprocess" ]; then
        # In subprocess mode each sample IS its own python process, so a
        # multi-sample shard isn't doing any useful batching anymore.  Force
        # shard-size=1 so one hung sample only blocks that one worker (the
        # whole point of switching modes in the first place).
        #
        # Workers: cap at 16 by default — empirically ~16 concurrent python
        # spawns is the macOS sweet spot before the fork/font-cache lock
        # tank throughput.  Override via SUBPROC_WORKERS=N env var.
        LANG_SHARD_SIZE=1
        LANG_WORKERS="${SUBPROC_WORKERS:-16}"
        EXTRA_ARGS+=(--python-exec-mode subprocess
                     --python-subproc-timeout "${PYTHON_SUBPROC_TIMEOUT}"
                     --python-subproc-parallel "${PYTHON_SUBPROC_PARALLEL}")
    fi

    # ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} expands to nothing when the array
    # is empty, side-stepping the `set -u` "unbound variable" trap on plain
    # ${EXTRA_ARGS[@]} for empty arrays in bash.
    python execution.py \
        --work-dir "${RUN_DIR}/runs-${RUN_ID}-${LANG}/" \
        --workers ${LANG_WORKERS} \
        --shard-size ${LANG_SHARD_SIZE} \
        --timeout ${EXEC_TIMEOUT} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "${LOG}" 2>&1 &
    EXEC_PIDS+=($!)
    echo "    Started ${LANG} (PID $!, shard=${LANG_SHARD_SIZE}, workers=${LANG_WORKERS}) -> ${LOG}"
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
