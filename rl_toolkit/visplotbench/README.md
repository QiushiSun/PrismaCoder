# visplotbench — RL Verifiers (Python / HTML / SVG)

A slim, pure-Python RL verifier toolkit for three of the VisPlotBench
languages. Drop it into your RL trainer to convert policy rollouts into
**per-sample scalar rewards** plus aggregate batch metrics, using the
canonical VisPlotBench execution + judge pipeline under the hood.

This package targets *single-modal* code generation: each verifier
handles exactly one language (`python`, `html`, or `svg`). Adding the
remaining VisPlotBench languages (`mermaid` / `lilypond` / `asymptote` /
`latex` / `vegalite`) is a one-file copy-paste — see the bottom of this
README.

The verifiers are built on top of the canonical VisPlotBench evaluation
pipeline that ships in [`../../evaluation/VisPlotBench/`](../../evaluation/VisPlotBench/);
that folder is the single source of truth for the language handlers,
judge prompts, dataset loaders, and OpenAI-compatible HTTP client.

---

## What you get

```python
result = await PythonRLVerifier(cfg).run_async()
# {
#   "verify_result": {
#       "count": 32, "exec_rate": 0.78,
#       "vis_mean": 64.2, "task_mean": 78.8,
#       "vis_good": 0.587, "task_good": 0.781,
#       "reward_mean": 0.71,
#   },
#   "per_sample": [
#       {"id": "python_3", "exec_ok": True, "has_plot": True,
#        "score_vis": 75, "score_task": 88, "reward": 0.812, "error": ""},
#       ...
#   ],
#   "paths": { ... full work_dir artefact paths ... },
#   "stage_results": {"generation":..., "execution":..., "evaluation":...},
# }
```

`per_sample` is the per-rollout reward signal for your RL loop.
`verify_result` is the batch-level rollup for logging. Everything is
also persisted under `cfg.work_dir` for offline inspection.

---

## Design

### Three composable stages

```
                run_generation        run_execution          run_evaluation
                  (optional)            (optional)             (optional)
                      |                     |                       |
   ----- prepared ----+----- generations ---+----- executions ------+----- scored ---->
     HF dataset                                  (handler renders         (judge:
       OR                                         each sample to            vis score
     prepared_samples.jsonl                       a PNG via Jupyter)        + task score)
                                                                                |
                                                                                v
                                                  reward = w_exec * exec_ok
                                                         + w_vis  * (score_vis  / 100)
                                                         + w_task * (score_task / 100)
                                                                                |
                                                                                v
                          { verify_result, per_sample, paths, stage_results }
```

Each stage is independently toggleable (`run_generation`,
`run_execution`, `run_evaluation`). The most common RL configuration is
`run_generation=False` — your policy server produces rollouts elsewhere,
the verifier only does **execute + judge + reward**.

### Stage details

| Stage | What it does | Backed by |
|-------|--------------|-----------|
| **Generation** (optional) | Calls an OpenAI-compatible chat endpoint for every prepared sample, runs `handler.extract_plotting_code` on each response. | `openai.AsyncOpenAI` (lazy import) + the VisPlotBench language handlers' `extract_plotting_code`. |
| **Execution** | Spawns a `ProcessPoolExecutor`; each shard runs through `handler.build_plots` (which writes + executes a Jupyter notebook) and `handler.parse_plots_notebook` (collects PNG outputs and stack traces). | `evaluation/VisPlotBench/plotting_benchmark/language_utils/<lang>_utils.py` |
| **Evaluation** | Two async judge calls per sample (visual similarity vs gold, task fidelity vs prompt) using an OpenAI-compatible endpoint — both `base_url` and `api_key` are config fields, so the same code works with hosted endpoints **and** a local judge model used in RL. | `evaluation/VisPlotBench/client.py::OpenAICompatClient` (retries on 429 / 5xx). |
| **Reward** | `exec_ok·w_exec + (score_vis/100)·w_vis + (score_task/100)·w_task`; if `reward_zero_on_exec_fail=True` (default), execution failures force reward = 0 instead of awarding partial credit. | Pure Python in `verifier_base._compute_per_sample`. |

### Subclass surface

`PlottingRLVerifier` is the workhorse. Each language subclass overrides
only:

- the class attributes `language` and `csv_export_enabled`,
- `_config_class()` to point at a dataclass adding language-specific
  fields (e.g. `python_exec_mode`, `html_chromium_preflight`),
- `_handler_config(self) -> dict` to forward those fields into the
  plotting handler at runtime,
- `_preflight(self) -> str | None` to probe the env before launching
  worker processes (returns an error string to short-circuit the batch).

That's it — see [`svg_verifier.py`](svg_verifier.py) for the minimal
template (≈20 lines).

---

## Installation

This package re-uses the dependencies of
[`evaluation/VisPlotBench/`](../../evaluation/VisPlotBench/). If you've
already set that up, you're done.

### Recommended: one-shot install via the upstream harness

```bash
cd PrismaCoder/evaluation/VisPlotBench/
./setup.sh                    # pip deps + playwright chromium + tool check
```

This installs everything the harness needs (matplotlib, seaborn,
cairosvg, playwright + chromium, …), which is also everything our
verifiers need at runtime for python/html/svg.

### Manual install (if you can't run the setup script)

The full upstream `requirements.txt` is the source of truth:

```bash
pip install -r PrismaCoder/evaluation/VisPlotBench/requirements.txt
playwright install chromium       # only if you'll run HTML execution
pip install openai                # only if you'll set run_generation=True
```

### Tiny footprint: import-only

If you just want to inspect the API (build configs, browse the source)
without running execution or evaluation yet, the absolute minimum is:

```bash
pip install pandas numpy pillow nbformat nbconvert httpx datasets
```

Everything else can be installed lazily when you turn the corresponding
stage on. Errors are clear and actionable when something is missing
(see "Error handling" below).

### Locating the harness

`verifier_base.py` auto-locates the harness at
`PrismaCoder/evaluation/VisPlotBench/` (two levels up from this file).
If you reorganize the repo or run the package from elsewhere, set:

```bash
export VISPLOTBENCH_HARNESS_ROOT=/abs/path/to/evaluation/VisPlotBench
```

The import will raise a clear `ImportError` with the resolved path if
the harness can't be found.

---

## Use cases

All examples assume you've installed the deps above and that
`import visplotbench` succeeds (either because `PrismaCoder/rl_toolkit/`
is on `sys.path`, or you've `pip install -e`'d this folder).

### A. RL inner loop — judge externally-produced rollouts (most common)

Your policy server has already produced rollouts. Feed them to the
verifier as an in-memory list:

```python
import asyncio
from visplotbench import PythonRLVerifier

# rollouts: list of {"id": <sample_id>, "response_text": <raw policy output>}
rollouts = [
    {"id": "python_3",  "response_text": "```python\nimport matplotlib...\n```"},
    {"id": "python_42", "response_text": "..."},
]

async def step_reward(rollouts, step_idx):
    v = PythonRLVerifier(
        work_dir=f"/tmp/rl/step_{step_idx}/python",
        run_generation=False,
        generations_inline=rollouts,

        # Cache prepared samples + CSVs once across all RL steps — the
        # benchmark doesn't change between steps, so re-downloading from
        # HuggingFace every step is wasteful.
        prepared_samples_path="cache/python/prepared_samples.jsonl",
        csv_dir="cache/python/csv",                 # leaf folder, see Gotchas

        # Local judge endpoint:
        eval_base_url="http://localhost:8001/v1",
        eval_api_key="EMPTY",
        eval_model="my-local-judge",
        eval_concurrency=16,
    )
    result = await v.run_async()
    return {r["id"]: r["reward"] for r in result["per_sample"]}, result["verify_result"]

rewards, batch_stats = asyncio.run(step_reward(rollouts, step_idx=0))
```

If your code is already inside an event loop (e.g. an `async` training
step), use `await v.run_async()` directly — `v.run()` will raise if it
detects a running loop.

`response_text` is the raw model output (may contain code fences /
prose). The verifier calls the language handler's
`extract_plotting_code` automatically. If you already pre-extracted the
code, pass `{"id": ..., "code": <clean code>}` and the extractor is
skipped.

### B. Full pipeline — policy exposed as an OpenAI-compatible server

When the policy is itself an OpenAI-compatible endpoint (e.g. vLLM with
your training checkpoint loaded), let the verifier drive everything:

```python
import asyncio
from visplotbench import HTMLRLVerifier

async def run_one_step():
    v = HTMLRLVerifier(
        work_dir="/tmp/rl/step_0/html",

        # ----- dataset slice -----
        dataset_name="TIGER-Lab/VisPlotBench",
        split="test",
        limit=32,
        # ids=[0, 5, 8, ...],   # or pin specific indices

        # ----- generation: policy model -----
        run_generation=True,
        generation_base_url="http://localhost:8000/v1",
        generation_api_key="EMPTY",
        generation_model="policy-ckpt-step-100",
        generation_temperature=0.7,
        generation_top_p=0.9,
        # vLLM-specific extras:
        generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}},

        # ----- evaluation: judge model -----
        eval_base_url="http://localhost:8001/v1",
        eval_api_key="EMPTY",
        eval_model="my-local-judge",
        eval_concurrency=16,

        html_chromium_preflight=True,   # short-circuits if chromium broken
    )
    return await v.run_async()

print(asyncio.run(run_one_step())["verify_result"])
```

### C. Language-specific tweaks

```python
# Python: switch to per-sample subprocess execution. Recommended for
# SFT'd policies that emit heavy matplotlib code (avoids ipykernel
# hangs eating valid samples).
PythonRLVerifier(
    work_dir="...",
    python_exec_mode="subprocess",
    python_subproc_timeout=60.0,
    python_subproc_parallel=1,
    ...
)

# HTML: skip the preflight probe if you've already verified chromium
# works in the parent process (e.g. inside a baked docker image).
HTMLRLVerifier(work_dir="...", html_chromium_preflight=False, ...)

# SVG: no language-specific knobs.
SVGRLVerifier(work_dir="...", ...)
```

### D. Custom reward weights

`reward_weights` is a plain dict; pass any weights you want. They are
**not** normalised — pick values that sum to 1.0 if you want
reward ∈ [0, 1]:

```python
PythonRLVerifier(
    work_dir="...",
    # Care more about visual quality than task fidelity:
    reward_weights={"exec_ok": 0.1, "vis": 0.6, "task": 0.3},

    # Don't zero out the reward on execution failure — small partial
    # credit for "tried but errored" can help early training.
    reward_zero_on_exec_fail=False,
    ...
)
```

---

## Config reference

All fields live on `PlottingRLVerifierConfig`. Per-language subclasses
extend it with `python_exec_mode` / `html_chromium_preflight`.
Construct via either pattern:

```python
v = PythonRLVerifier(work_dir="...", generation_model="...", ...)   # kwargs

cfg = PythonVerifierConfig(work_dir="...", generation_model="...")
v   = PythonRLVerifier(cfg, eval_concurrency=8)                    # config + overrides
```

| Group | Field | Default | Notes |
|-------|-------|---------|-------|
| **Paths** | `work_dir` | (required) | All artefacts land here. |
|  | `repo_root` | resolved harness root | Used to find `plotting_benchmark/resources/instructs.json`. |
| **Dataset** | `dataset_name` | `"TIGER-Lab/VisPlotBench"` | HF dataset id. |
|  | `split` | `"test"` |  |
|  | `data_descriptor` | `"head"` | Forwarded to `TaskChanger`. |
|  | `ids` | `None` | Optional list of integer indices (post-split). |
|  | `limit` | `None` | Random `seed=42` sample if set. Mutually exclusive with `ids`. |
|  | `prepared_samples_path` | `None` | Skip HF entirely; load this jsonl as-is. |
|  | `csv_dir` | `None` | Leaf folder containing `{id}.csv`. Required for python/html when `prepared_samples_path` is set; auto-populated otherwise. |
| **Stage switches** | `run_generation` / `run_execution` / `run_evaluation` | `True` each | Toggle stages independently. |
| **External rollouts** | `generations_path` / `generations_inline` | `None` | One required when `run_generation=False`. Each row: `{id, response_text \| code}`. |
| **Generation** | `generation_base_url` / `generation_api_key` / `generation_model` | OpenAI-compat |  |
|  | `generation_max_workers` | `32` | Async semaphore. |
|  | `generation_temperature` / `generation_top_p` / `generation_max_tokens` | `0.0` / `None` / `12000` |  |
|  | `generation_timeout` | `600.0` | Seconds. |
|  | `generation_extra_body` | `{}` | Forwarded as the SDK's `extra_body` kwarg (e.g. vLLM `chat_template_kwargs`). |
| **Execution** | `execution_workers` | `4` | `ProcessPoolExecutor` size. |
|  | `execution_shard_size` | `8` | Samples per worker shard. |
|  | `execution_timeout` | `600` | Hard wall-clock for the whole exec stage. |
| **Evaluation** | `eval_base_url` / `eval_api_key` / `eval_model` | OpenAI-compat | Point at a **local judge** in RL. |
|  | `eval_concurrency` | `16` | Async semaphore (vis + task judges count separately). |
|  | `eval_temperature` / `eval_max_tokens` / `eval_timeout` | `0.0` / `8192` / `600.0` |  |
| **Reward** | `reward_weights` | `{"exec_ok": 0.2, "vis": 0.4, "task": 0.4}` | Plain dict; not normalised. |
|  | `reward_zero_on_exec_fail` | `True` | If exec fails (no plot or error), force reward = 0. |
| **Python only** | `python_exec_mode` | `"jupyter"` | `"jupyter"` or `"subprocess"`. |
|  | `python_subproc_timeout` | `60.0` | Per-sample wall-clock in subprocess mode. |
|  | `python_subproc_parallel` | `1` | Intra-shard parallelism; keep at 1 on macOS. |
| **HTML only** | `html_chromium_preflight` | `True` | Probe `playwright.chromium.launch()` in the parent process. |

---

## Error handling

The verifier is built to fail **loudly and actionably** when an env
piece is missing, rather than letting workers silently produce
"Unexpected error" on every sample:

- `run_generation=True` without `openai` installed → `RuntimeError(
  "run_generation=True requires the openai package (pip install
  openai). Alternatively, set run_generation=False …")` at the start
  of the gen stage.
- HTML execution without chromium → `_preflight()` returns a string
  like `"Browser setup failed: Executable doesn't exist at /…/ms-
  playwright/…"` and the verifier marks **every** sample's execution
  row with that error before any worker is spawned. No wasted CPU.
- SVG execution without `libcairo2` system library → workers will fail
  with `OSError: no library called "cairo-2" was found` and surface it
  as the per-sample `error` field.
- Harness not found at the expected path → `ImportError` with the
  resolved path and instructions to set `VISPLOTBENCH_HARNESS_ROOT`.

---

## Gotchas

1. **`csv_dir` is a leaf path.** Both Python and HTML execution read
   `{csv_dir}/{id}.csv`. Point at the *leaf* folder, not the parent.
   `csv_dir="runs/runs-27/runs-27-python/csv/python"`, **not**
   `runs/runs-27/runs-27-python/csv`.

2. **`v.run()` from inside an event loop.** The sync wrapper raises if
   it detects a running asyncio loop. Use `await v.run_async()`.

3. **Generations without `code`.** When passing `generations_inline`
   with only `response_text`, the verifier auto-runs
   `handler.extract_plotting_code(response_text)`. If the policy emits
   clean code (no fences, no prose), pass `{"id": ..., "code": ...}`
   to skip extraction.

4. **HTML chromium binary moves after `pip install --upgrade playwright`.**
   The preflight catches this; the fix is always `playwright install
   chromium`. If you bake chromium into a container differently, set
   `html_chromium_preflight=False`.

5. **`work_dir` accumulates artefacts.** Per-shard Jupyter notebooks
   under `execution_shards/` aren't cleaned up automatically; point
   `work_dir` at `/tmp` (or a tmpfs) and clean between RL steps for
   high-throughput training.

6. **Cache prepared samples between RL steps.** Re-running
   `load_prepared_dataset` for every step is wasteful. Pre-build it
   once (e.g. by running the verifier once with `run_execution=False`
   and `run_evaluation=False`), then pass `prepared_samples_path=` +
   `csv_dir=` on every subsequent step.

---

## Extending to other VisPlotBench languages

The remaining VisPlotBench languages (`mermaid`, `lilypond`,
`asymptote`, `latex`, `vegalite`) all have working handlers under
[`../../evaluation/VisPlotBench/plotting_benchmark/language_utils/`](../../evaluation/VisPlotBench/plotting_benchmark/language_utils/).
To add one, copy [`svg_verifier.py`](svg_verifier.py) (the thinnest
example):

```python
# mermaid_verifier.py
from dataclasses import dataclass
from .verifier_base import PlottingRLVerifier, PlottingRLVerifierConfig

@dataclass
class MermaidVerifierConfig(PlottingRLVerifierConfig):
    language: str = "mermaid"

class MermaidRLVerifier(PlottingRLVerifier):
    language = "mermaid"
    csv_export_enabled = False     # no CSV payload

    @classmethod
    def _config_class(cls):
        return MermaidVerifierConfig
```

Then add the import + re-export in [`__init__.py`](__init__.py).

Languages that need preflight or runtime knobs should follow
[`html_verifier.py`](html_verifier.py) (chromium probe) or
[`python_verifier.py`](python_verifier.py) (forward exec-mode knobs to
the handler) as templates.

---

## Sanity check after install

```bash
python -c "
from visplotbench import (
    PythonRLVerifier, HTMLRLVerifier, SVGRLVerifier,
    PythonVerifierConfig, HTMLVerifierConfig, SVGVerifierConfig,
)
print('visplotbench import ok')
"
```

If that prints `visplotbench import ok`, the harness resolution worked
and you're ready to plug it into your RL trainer.
