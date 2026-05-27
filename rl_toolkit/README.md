# rl_toolkit

PrismaCoder's RL verifier toolkit — pure-Python classes that convert
policy rollouts into **per-sample scalar rewards** plus aggregate batch
metrics, ready to drop into any RL training loop.

The toolkit is organized by **(modality, input-condition) scenario**.
Each subpackage handles one scenario, exposes a single verifier class,
and returns the same shape of output — so swapping in a new scenario
is a one-line change in your trainer.

> **Naming convention.** Subpackage names spell out both the modality
> and the input-condition mode. ``_unimodal`` means the policy is
> conditioned on a text prompt only (no reference image). Multimodal
> scenarios (text + reference image) live in sibling subpackages with
> matching names (e.g. ``python_visualization_multimodal``).

## Currently available

| Subpackage | Scenario | Entry point |
|------------|----------|--------------------------|
| [`rl_toolkit.python_visualization_unimodal`](python_visualization_unimodal/) | Text-only prompt → Python plotting code (matplotlib / seaborn / plotly / bokeh / …) | `PythonRLVerifier` |
| [`rl_toolkit.svg_unimodal`](svg_unimodal/)                                   | Text-only prompt → SVG markup                                                          | `SVGRLVerifier`    |
| [`rl_toolkit.html_unimodal`](html_unimodal/)                                 | Text-only prompt → HTML plotting page (with JS chart libraries)                         | `HTMLRLVerifier`   |

The verifiers share infrastructure via
[`plotting_verifier_base.py`](plotting_verifier_base.py)
(`PlottingRLVerifier` + `PlottingRLVerifierConfig`); each modality
subclass is just a few dozen lines that pin the language and forward
modality-specific knobs (e.g. Python's subprocess execution mode,
HTML's chromium preflight probe).

---

## What you get

Every verifier returns the same dict:

```python
result = await PythonRLVerifier(cfg).run_async()

result["verify_result"]
# {"count": 32, "exec_rate": 0.78,
#  "vis_mean": 64.2, "task_mean": 78.8,
#  "vis_good": 0.587, "task_good": 0.781,
#  "reward_mean": 0.71}

result["per_sample"]
# [{"id": "python_3", "exec_ok": True, "has_plot": True,
#   "score_vis": 75, "score_task": 88, "reward": 0.812, "error": ""},
#  ...]

result["paths"]
# Absolute paths to all artefacts written under cfg.work_dir
# (prepared_samples.jsonl, generations.jsonl, execution_results.jsonl,
#  scored_results.jsonl, stats.json).

result["stage_results"]
# Per-stage summaries: {"generation": ..., "execution": ..., "evaluation": ...}
```

`per_sample` is the per-rollout reward signal for your RL loop.
`verify_result` is the batch-level rollup for logging.

---

## Design

### Three composable stages

```
                run_generation         run_execution           run_evaluation
                  (optional)             (optional)              (optional)
                      |                      |                       |
   --- prepared ------+----- generations ----+------ executions -----+----- scored --->
       HF dataset                              (each sample rendered      (judge:
         OR                                     to a PNG via the           vis score
       prepared_samples.jsonl                   modality handler)          + task score)
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

| Stage | What it does | How |
|-------|--------------|----|
| **Generation** (optional) | Calls an OpenAI-compatible chat endpoint for every prepared sample, then extracts the modality-specific code from the response. | `openai.AsyncOpenAI` (lazy import). |
| **Execution** | Spawns a `ProcessPoolExecutor`; each shard runs the rendering handler for the modality (Jupyter notebook + per-language renderer) and collects PNG outputs + stack traces. | Reuses the rendering pipeline from the canonical evaluation harness (see [Internals](#internals)). |
| **Evaluation** | Two async judge calls per sample (visual similarity vs gold image, task fidelity vs prompt). Both `base_url` and `api_key` are config fields, so the same code works against hosted endpoints **and** a local judge model used in RL. | `httpx`-based OpenAI-compatible client with retry on 429 / 5xx. |
| **Reward** | `exec_ok·w_exec + (score_vis/100)·w_vis + (score_task/100)·w_task`; if `reward_zero_on_exec_fail=True` (default), execution failures force reward = 0 instead of awarding partial credit. | Pure Python in `plotting_verifier_base._compute_per_sample`. |

### Per-modality differences

| Modality | CSV payload | Runtime preflight | Execution backend knob |
|----------|-------------|-------------------|------------------------|
| `python` | exported to `csv/python/{id}.csv`, referenced inside the model's code | none | `python_exec_mode = jupyter \| subprocess` |
| `svg`    | none (markup is self-contained)                              | none | none |
| `html`   | injected into the page as a JS object                          | chromium launch probe in the parent process | none |

---

## Installation

### Step 1 — Core deps (just to import any of the verifiers)

```bash
pip install pandas numpy pillow nbformat nbconvert httpx datasets
```

### Step 2 — Add stage-specific deps for the modality you actually use

| Modality | Pip | System |
|----------|-----|--------|
| `python` execution | `ipykernel jupyter_client matplotlib seaborn` (+ any chart library your policy emits, e.g. `plotly bokeh altair pyecharts holoviews`) | — |
| `svg` execution    | `cairosvg ipykernel jupyter_client` | `libcairo2` (`apt-get install libcairo2`  /  `brew install cairo`) |
| `html` execution   | `playwright nest-asyncio ipykernel jupyter_client`, then `playwright install chromium` | chromium-compatible host |
| Any: `run_generation=True` | `openai>=1.0` | — |
| Any: `run_evaluation=True` | (covered by core) | — |

A one-shot "I want all three modalities + generation" install:

```bash
pip install pandas numpy pillow nbformat nbconvert httpx datasets \
            openai ipykernel jupyter_client \
            matplotlib seaborn \
            cairosvg \
            playwright nest-asyncio \
  && playwright install chromium
```

If you've already installed
[`evaluation/VisPlotBench/requirements.txt`](../evaluation/VisPlotBench/requirements.txt)
for the offline benchmark, you already have everything you need here.

### Step 3 — Sanity check

```bash
python -c "
from rl_toolkit.python_visualization_unimodal import PythonRLVerifier
from rl_toolkit.svg_unimodal                  import SVGRLVerifier
from rl_toolkit.html_unimodal                 import HTMLRLVerifier
print('rl_toolkit import ok')
"
```

(Run from `PrismaCoder/`, or add `PrismaCoder/` to `PYTHONPATH`.)

---

## Use cases

All examples assume `import rl_toolkit.python_visualization_unimodal`
(or `svg_unimodal` / `html_unimodal`) succeeds in your env.

### A. RL inner loop — judge externally-produced rollouts (most common)

Your policy server has already produced rollouts. Feed them to the
verifier as an in-memory list:

```python
import asyncio
from rl_toolkit.python_visualization_unimodal import PythonRLVerifier

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

`response_text` is the raw model output (may contain code fences /
prose). The verifier extracts the code automatically. If you already
pre-extracted clean code, pass `{"id": ..., "code": ...}` to skip
extraction.

### B. Full pipeline — policy exposed as an OpenAI-compatible server

When the policy is itself an OpenAI-compatible endpoint (e.g. vLLM with
your training checkpoint loaded), let the verifier drive everything:

```python
import asyncio
from rl_toolkit.html_unimodal import HTMLRLVerifier

async def run_one_step():
    v = HTMLRLVerifier(
        work_dir="/tmp/rl/step_0/html",

        # ----- dataset slice -----
        dataset_name="TIGER-Lab/VisPlotBench",
        split="test",
        limit=32,

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

        html_chromium_preflight=True,   # short-circuits if chromium broken
    )
    return await v.run_async()

print(asyncio.run(run_one_step())["verify_result"])
```

### C. Modality-specific tweaks

```python
from rl_toolkit.python_visualization_unimodal import PythonRLVerifier
from rl_toolkit.html_unimodal                 import HTMLRLVerifier
from rl_toolkit.svg_unimodal                  import SVGRLVerifier

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

# SVG: no modality-specific knobs.
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

All fields live on `PlottingRLVerifierConfig`. Per-modality subclasses
extend it with their own knobs. Construct via either pattern:

```python
v = PythonRLVerifier(work_dir="...", generation_model="...", ...)    # kwargs

cfg = PythonVerifierConfig(work_dir="...", generation_model="...")
v   = PythonRLVerifier(cfg, eval_concurrency=8)                     # config + overrides
```

| Group | Field | Default | Notes |
|-------|-------|---------|-------|
| **Paths** | `work_dir` | (required) | All artefacts land here. |
|  | `repo_root` | auto-resolved | Used to locate the judge prompts (`plotting_benchmark/resources/instructs.json`). See [Internals](#internals). |
| **Dataset** | `dataset_name` | `"TIGER-Lab/VisPlotBench"` | HF dataset id. |
|  | `split` | `"test"` |  |
|  | `data_descriptor` | `"head"` |  |
|  | `ids` | `None` | Optional list of integer indices (post-split). |
|  | `limit` | `None` | Random `seed=42` sample if set. |
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
| **`python` only** | `python_exec_mode` | `"jupyter"` | `"jupyter"` or `"subprocess"`. |
|  | `python_subproc_timeout` | `60.0` | Per-sample wall-clock in subprocess mode. |
|  | `python_subproc_parallel` | `1` | Intra-shard parallelism; keep at 1 on macOS. |
| **`html` only** | `html_chromium_preflight` | `True` | Probe `playwright.chromium.launch()` in the parent process. |

---

## Error handling

The verifier is built to fail **loudly and actionably** when an env
piece is missing, rather than letting workers silently produce
"Unexpected error" on every sample:

- `run_generation=True` without `openai` installed → `RuntimeError(
  "run_generation=True requires the openai package …")` at the start
  of the gen stage.
- HTML execution without chromium → `_preflight()` returns
  `"Browser setup failed: …"` and the verifier marks every sample's
  execution row with that error before any worker is spawned. No
  wasted CPU.
- SVG execution without `libcairo2` system library → workers surface
  `OSError: no library called "cairo-2" was found` in the per-sample
  `error` field.
- Underlying harness not found → `ImportError` at import time with the
  resolved path and instructions to set `VISPLOTBENCH_HARNESS_ROOT`.

---

## Gotchas

1. **`csv_dir` is a leaf path.** Both Python and HTML execution read
   `{csv_dir}/{id}.csv`. Point at the *leaf* folder, not the parent.
   `csv_dir=".../runs-27-python/csv/python"`, **not**
   `.../runs-27-python/csv`.

2. **`v.run()` from inside an event loop.** The sync wrapper raises if
   it detects a running asyncio loop. Use `await v.run_async()`.

3. **Generations without `code`.** When passing `generations_inline`
   with only `response_text`, the verifier auto-extracts the code. If
   the policy emits clean code (no fences, no prose), pass
   `{"id": ..., "code": ...}` to skip extraction.

4. **HTML chromium binary moves after `pip install --upgrade playwright`.**
   The preflight catches this; the fix is always `playwright install
   chromium`. If you bake chromium into a container differently, set
   `html_chromium_preflight=False`.

5. **`work_dir` accumulates artefacts.** Per-shard Jupyter notebooks
   under `execution_shards/` aren't cleaned up automatically; point
   `work_dir` at `/tmp` (or a tmpfs) and clean between RL steps for
   high-throughput training.

6. **Cache prepared samples between RL steps.** Re-running
   `load_prepared_dataset` every step is wasteful. Pre-build it once,
   then pass `prepared_samples_path=` + `csv_dir=` on every subsequent
   step.

---

## Adding a new modality

If you have a new scenario that fits the same shape ("execute the
sample, get back a PNG, judge it"), adding a verifier is typically a
20-30 line subclass. Follow the existing folder naming convention
(``<modality>_<conditioning>``, e.g. ``mermaid_unimodal``,
``python_visualization_multimodal``) and copy
[`svg_unimodal/verifier.py`](svg_unimodal/verifier.py) (the thinnest
example):

```python
# rl_toolkit/mermaid_unimodal/verifier.py
from dataclasses import dataclass
from ..plotting_verifier_base import PlottingRLVerifier, PlottingRLVerifierConfig

@dataclass
class MermaidVerifierConfig(PlottingRLVerifierConfig):
    language: str = "mermaid"

class MermaidRLVerifier(PlottingRLVerifier):
    language = "mermaid"
    csv_export_enabled = False     # mermaid samples are self-contained

    @classmethod
    def _config_class(cls):
        return MermaidVerifierConfig
```

Add a sibling `__init__.py` that re-exports the two classes, and the
scenario is wired up.

Scenarios that need a runtime env probe should follow
[`html_unimodal/verifier.py`](html_unimodal/verifier.py); scenarios
that forward extra config keys into the language handler should follow
[`python_visualization_unimodal/verifier.py`](python_visualization_unimodal/verifier.py).

---

## Internals

The execution and judge stages reuse the rendering handlers, dataset
loaders, OpenAI-compatible HTTP client, and judge prompts from the
PrismaCoder VisPlotBench evaluation harness under
[`../evaluation/VisPlotBench/`](../evaluation/VisPlotBench/). The
harness is the source of truth for:

- The per-language Jupyter notebook builders + parsers
  (`plotting_benchmark/language_utils/`).
- The HuggingFace dataset loader and prompt-template machinery
  (`dataset_tools.py`, `plotting_benchmark/task_changer.py`).
- The judge system prompt + vis / task instruction templates
  (`plotting_benchmark/resources/instructs.json`).
- The retrying OpenAI-compatible HTTP client (`client.py`).

`plotting_verifier_base.py` auto-locates the harness at
`../evaluation/VisPlotBench/` relative to this folder. Override the
location by setting `VISPLOTBENCH_HARNESS_ROOT` to the absolute path of
an `evaluation/VisPlotBench/` checkout. The import raises a clear
`ImportError` (with the resolved path) if the harness isn't found.
