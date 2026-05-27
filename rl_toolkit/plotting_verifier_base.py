"""Shared base for code-plotting RL verifiers.

Implements the generic three-stage pipeline (generation -> execution ->
evaluation) used to derive per-sample RL rewards and aggregate metrics
for any code-generation modality whose execution amounts to "run the
sample and judge the rendered plot". Each modality (Python / SVG /
HTML / ...) only needs to override a small set of hooks.

This module is the shared internal infrastructure for the per-scenario
subpackages under :mod:`rl_toolkit`. End users normally don't import
from here directly — they go through
``rl_toolkit.python_visualization_unimodal`` /
``rl_toolkit.svg_unimodal`` / ``rl_toolkit.html_unimodal``.

Internals: the execution + judge pipeline reuses the language handlers,
dataset loaders, OpenAI-compatible HTTP client, and judge prompts from
the PrismaCoder VisPlotBench evaluation harness under
``PrismaCoder/evaluation/VisPlotBench/``. The location is auto-resolved
at import time and can be overridden with the
``VISPLOTBENCH_HARNESS_ROOT`` environment variable.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


# --- Evaluation harness resolution -------------------------------------------
# Locate the canonical evaluation pipeline that ships the language handlers
# and judge prompts, prepend its root to sys.path, and surface an actionable
# error if it isn't where we expect.
_HERE = Path(__file__).resolve().parent  # .../rl_toolkit
_DEFAULT_HARNESS_ROOT = _HERE.parent / "evaluation" / "VisPlotBench"
_HARNESS_ROOT = Path(
    os.environ.get("VISPLOTBENCH_HARNESS_ROOT", _DEFAULT_HARNESS_ROOT)
).resolve()

if not (_HARNESS_ROOT / "plotting_benchmark").is_dir():
    raise ImportError(
        f"Cannot locate the evaluation harness at {_HARNESS_ROOT!s}. "
        "Either keep rl_toolkit/ next to PrismaCoder/evaluation/VisPlotBench/, "
        "or set the VISPLOTBENCH_HARNESS_ROOT environment variable to point at "
        "an evaluation/VisPlotBench/ checkout (the folder that contains "
        "plotting_benchmark/, dataset_tools.py, client.py, io_utils.py)."
    )

if str(_HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(_HARNESS_ROOT))

import pandas as pd  # noqa: E402

# ``openai`` is only needed for the optional generation stage. RL trainers
# that hand us rollouts via ``generations_path`` / ``generations_inline``
# never enter that branch, so we import it lazily inside
# ``_stage_generation``. That keeps the import-time dep set minimal.
from client import OpenAICompatClient, image_to_content_part  # noqa: E402
from dataset_tools import (  # noqa: E402
    export_csv_payloads,
    load_instructs,
    load_prepared_dataset,
    row_to_portable_sample,
)
from io_utils import ensure_dir, read_jsonl, write_json, write_jsonl  # noqa: E402
from logger import APILogger  # noqa: E402
from plotting_benchmark.language_utils import get_language_handler  # noqa: E402


# ==============================================================================
#  Config dataclass
# ==============================================================================
@dataclass
class PlottingRLVerifierConfig:
    """All knobs for a single verifier run.

    Field groups:

    * Paths / dataset source
    * Stage switches
    * External rollout injection (when ``run_generation=False``)
    * Generation API config (the policy model being trained)
    * Execution config (process pool, shard size, language-specific knobs)
    * Evaluation API config (judge model, typically a local model in RL)
    * Reward composition weights
    """

    # --- Paths / language --------------------------------------------------
    work_dir: str
    language: str = "python"
    # Forwarded to dataset_tools.load_instructs to locate
    # plotting_benchmark/resources/instructs.json. Defaults to the harness
    # root we resolved at import time.
    repo_root: str = str(_HARNESS_ROOT)

    # --- Dataset source ----------------------------------------------------
    dataset_name: str = "TIGER-Lab/VisPlotBench"
    split: str = "test"
    data_descriptor: str = "head"
    ids: list[int] | None = None
    limit: int | None = None
    # If given, skip HF dataset loading and use this jsonl directly. Each
    # row must contain at least: ``id``, ``task``, ``plots_gt``,
    # ``used_lib`` (python only), ``data`` (python/html only) and the
    # ``task__*`` fields consumed by the judge prompt.
    prepared_samples_path: str | None = None
    # If given, point execution at this CSV directory directly (leaf folder
    # containing ``{id}.csv``). Otherwise the dataset prep step writes into
    # ``work_dir/csv/<lang>/``.
    csv_dir: str | None = None

    # --- Stage switches ----------------------------------------------------
    run_generation: bool = True
    run_execution: bool = True
    run_evaluation: bool = True

    # --- External rollouts (used when ``run_generation=False``) ------------
    # Either point at a jsonl on disk, or hand us a list of dicts directly.
    # Each entry needs ``id`` plus either ``code`` (already extracted) or
    # ``response_text`` (we'll call ``handler.extract_plotting_code`` on it).
    generations_path: str | None = None
    generations_inline: list[dict[str, Any]] | None = None

    # --- Generation (policy model) -----------------------------------------
    generation_base_url: str = "https://api.openai.com/v1"
    generation_api_key: str = "EMPTY"
    generation_model: str = "gpt-4o-mini"
    generation_max_workers: int = 32
    generation_temperature: float = 0.0
    generation_top_p: float | None = None
    generation_max_tokens: int = 12000
    generation_timeout: float = 600.0
    # Extra fields forwarded as the OpenAI SDK's ``extra_body`` kwarg (vLLM
    # ``chat_template_kwargs``, etc.).
    generation_extra_body: dict[str, Any] = field(default_factory=dict)
    generation_enable_logger: bool = False
    generation_api_logs: str = "./api-logs/generation"

    # --- Execution ---------------------------------------------------------
    execution_workers: int = 4
    execution_shard_size: int = 8
    execution_timeout: int = 600

    # --- Evaluation (judge model, typically local in RL) -------------------
    eval_base_url: str = "https://api.openai.com/v1"
    eval_api_key: str = "EMPTY"
    eval_model: str = "gpt-4.1"
    eval_concurrency: int = 16
    eval_temperature: float = 0.0
    eval_max_tokens: int = 8192
    eval_timeout: float = 600.0
    eval_enable_logger: bool = False
    eval_api_logs: str = "./api-logs/judge"

    # --- Reward composition ------------------------------------------------
    # All three components live in [0, 1]; the per-sample reward is a
    # weighted sum (not normalised, but the default weights sum to 1.0).
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"exec_ok": 0.2, "vis": 0.4, "task": 0.4}
    )
    # When execution fails (no plot or runtime error), force reward = 0
    # instead of letting the (zero) score components average up to a small
    # positive value via the exec_ok=False contribution.
    reward_zero_on_exec_fail: bool = True


# ==============================================================================
#  Module-level helper: execute one shard inside a worker process.
#  Must live at module scope to be picklable by ProcessPoolExecutor.
# ==============================================================================
def _execute_chunk(
    language: str,
    csv_dir: str | None,
    shard_rows: list[dict[str, Any]],
    shard_dir: str,
    handler_extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        cfg_dict: dict[str, Any] = {
            "plotting_language": language,
            "run_mode": "normal",
            "debug": {"output_dir": str(Path(shard_dir) / "debug")},
        }
        if handler_extra:
            cfg_dict.update(handler_extra)
        handler = get_language_handler(language, config=cfg_dict)
        df = pd.DataFrame(shard_rows)
        out_nb = Path(shard_dir) / f"plots_{language}.ipynb"
        out_nb.parent.mkdir(parents=True, exist_ok=True)
        handler.build_plots(df, out_nb, Path(csv_dir) if csv_dir else None)
        res = handler.parse_plots_notebook(out_nb)
        merged = df.drop(
            columns=[c for c in res.columns if c in df.columns and c != "id"]
        ).merge(res, on="id", how="left")
        for idx, row in merged.iterrows():
            if row.get("response_text", "") == "" and row.get("error", "") == "":
                merged.at[idx, "error"] = "Unexpected Error"
            if bool(row.get("has_plot", False)) is False and row.get("error", "") == "":
                merged.at[idx, "error"] = "Unexpected Error"
        return merged.to_dict(orient="records")
    except Exception:
        tb = traceback.format_exc()
        error_msg = f"Shard execution failed:\n{tb}"
        return [
            {**row, "error": error_msg, "has_plot": False, "plots_generated": []}
            for row in shard_rows
        ]


def _force_kill_pool(pool: ProcessPoolExecutor) -> None:
    """SIGKILL all workers (used when a shard times out)."""
    import signal as _signal

    for pid in list(getattr(pool, "_processes", {}).keys()):
        try:
            os.kill(pid, _signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _parse_score(text: str) -> int | None:
    """Extract an integer score from a judge response.

    Either a bare integer, or the marker line ``[FINAL SCORE]: <int>``.
    """
    text = (text or "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    m = re.search(r"\[FINAL SCORE\]:?\s*(\d+)", text, re.DOTALL)
    if m:
        return int(m.group(1))
    return None


# ==============================================================================
#  Base verifier
# ==============================================================================
class PlottingRLVerifier:
    """Generic code-plotting RL verifier.

    Subclass and override ``language`` + the ``_handler_config`` /
    ``_preflight`` hooks for per-modality behaviour.
    """

    language: str = "python"
    csv_export_enabled: bool = True

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: PlottingRLVerifierConfig | None = None,
        **overrides: Any,
    ) -> None:
        ConfigCls = type(self)._config_class()
        if config is None:
            config = ConfigCls(**overrides)
        elif overrides:
            values = {**config.__dict__, **overrides}
            config = ConfigCls(**values)
        # Subclasses pin a language; force it onto the config so the
        # caller can leave it unset in overrides.
        config.language = type(self).language
        self.config = config

    @classmethod
    def _config_class(cls) -> type[PlottingRLVerifierConfig]:
        """Hook for subclasses that ship their own config dataclass."""
        return PlottingRLVerifierConfig

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _handler_config(self) -> dict[str, Any]:
        """Extra keys merged into the plotting handler config dict.

        The Python subclass uses this to inject ``python_exec_mode`` /
        ``python_subproc_timeout`` etc.
        """
        return {}

    def _preflight(self) -> str | None:
        """Probe the runtime env. Return None on success, else an error string.

        If a string is returned, every sample's execution result is
        marked with that error and the actual execution is skipped.
        """
        return None

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    async def run_async(self) -> dict[str, Any]:
        cfg = self.config
        work_dir = ensure_dir(cfg.work_dir)

        paths = {
            "work_dir": str(work_dir),
            "prepared_samples": str(work_dir / "prepared_samples.jsonl"),
            "generations": str(work_dir / "generations.jsonl"),
            "executions": str(work_dir / "execution_results.jsonl"),
            "scored_results": str(work_dir / "scored_results.jsonl"),
            "stats": str(work_dir / "stats.json"),
        }
        stage_results: dict[str, Any] = {
            "generation": None,
            "execution": None,
            "evaluation": None,
        }

        # -- Data prep: produce prepared_samples + csv_dir --
        prepared_samples, csv_dir = self._prepare_samples(work_dir)
        write_jsonl(prepared_samples, paths["prepared_samples"])

        # -- Stage 1: generation (policy rollouts) --
        if cfg.run_generation:
            generations, gen_stats = await self._stage_generation(
                prepared_samples, work_dir
            )
            stage_results["generation"] = gen_stats
        else:
            generations = self._load_external_generations(prepared_samples)
            stage_results["generation"] = {
                "count": len(generations),
                "source": "external",
            }
        write_jsonl(generations, paths["generations"])

        # -- Stage 2: execution --
        merged_for_exec = self._merge_prepared_with_generations(
            prepared_samples, generations
        )
        if cfg.run_execution:
            execution_rows, exec_stats = self._stage_execution(
                merged_for_exec, csv_dir, work_dir
            )
            stage_results["execution"] = exec_stats
        else:
            execution_rows = merged_for_exec
        write_jsonl(
            sorted(execution_rows, key=lambda r: r.get("id")), paths["executions"]
        )

        # -- Stage 3: evaluation --
        if cfg.run_evaluation:
            scored_rows, eval_stats = await self._stage_evaluation(
                execution_rows, work_dir
            )
            stage_results["evaluation"] = eval_stats
        else:
            scored_rows = execution_rows
        write_jsonl(scored_rows, paths["scored_results"])

        # -- Reward + aggregate --
        per_sample = self._compute_per_sample(scored_rows)
        verify_result = self._aggregate(per_sample, scored_rows)
        write_json(
            {
                "model": cfg.generation_model if cfg.run_generation else "external",
                "language": cfg.language,
                "judge_model": cfg.eval_model,
                "count": verify_result["count"],
                "execution_error_num": int(verify_result["count"])
                - int(round(verify_result["exec_rate"] * verify_result["count"]))
                if verify_result["count"]
                else 0,
                "no_plots": sum(1 for r in scored_rows if not r.get("has_plot")),
                "verify_result": verify_result,
            },
            paths["stats"],
        )

        return {
            "verify_result": verify_result,
            "per_sample": per_sample,
            "paths": paths,
            "stage_results": stage_results,
        }

    # sync wrappers
    def run(self) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async())
        raise RuntimeError(
            f"{type(self).__name__}.run() cannot be called from an active event "
            "loop; use `await verifier.run_async()` instead."
        )

    def __call__(self) -> dict[str, Any]:
        return self.run()

    # ==================================================================
    # Data-prep helpers
    # ==================================================================
    def _prepare_samples(self, work_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
        """Produce the list of prepared sample dicts + the CSV dir path.

        Path A (no ``prepared_samples_path``): load via HuggingFace dataset.
        Path B (``prepared_samples_path`` given): load that jsonl as-is and
        only honour ``csv_dir`` for execution.
        """
        cfg = self.config
        language = cfg.language

        if cfg.prepared_samples_path:
            samples = read_jsonl(cfg.prepared_samples_path)
            csv_dir = cfg.csv_dir
            return samples, csv_dir

        df = load_prepared_dataset(
            repo_root=cfg.repo_root,
            language=language,
            data_descriptor=cfg.data_descriptor,
            dataset_name=cfg.dataset_name,
            split=cfg.split,
            ids=cfg.ids,
            limit=cfg.limit,
        )

        csv_dir: str | None = None
        if self.csv_export_enabled:
            csv_root = Path(cfg.csv_dir) if cfg.csv_dir else (work_dir / "csv" / language)
            export_csv_payloads(df, csv_root, language)
            csv_dir = str(csv_root)
        else:
            csv_dir = cfg.csv_dir  # may stay None for svg etc.

        instructs = load_instructs(cfg.repo_root, language)
        samples = [
            row_to_portable_sample(row, instructs["plot_instruct"])
            for _, row in df.iterrows()
        ]
        return samples, csv_dir

    # ==================================================================
    # Stage 1: generation
    # ==================================================================
    async def _stage_generation(
        self, prepared_samples: list[dict[str, Any]], work_dir: Path
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cfg = self.config
        instructs = load_instructs(cfg.repo_root, cfg.language)
        handler = get_language_handler(
            cfg.language, config={"plotting_language": cfg.language}
        )

        logger = APILogger(log_dir=cfg.generation_api_logs) if cfg.generation_enable_logger else None

        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover - import-time env probe
            raise RuntimeError(
                "run_generation=True requires the `openai` package "
                "(pip install openai). Alternatively, set run_generation=False "
                "and pass rollouts via generations_path / generations_inline."
            ) from e

        client = AsyncOpenAI(
            api_key=cfg.generation_api_key,
            base_url=cfg.generation_base_url,
            timeout=cfg.generation_timeout,
        )
        sem = asyncio.Semaphore(max(1, cfg.generation_max_workers))

        extra_body_kwargs: dict[str, Any] = {}
        if cfg.generation_extra_body:
            extra_body_kwargs["extra_body"] = cfg.generation_extra_body
        if cfg.generation_top_p is not None:
            extra_body_kwargs["top_p"] = cfg.generation_top_p

        async def _gen_one(sample: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                t0 = time.perf_counter()
                messages = [
                    {"role": "system", "content": instructs["system_prompt"]},
                    {"role": "user", "content": sample["task"]},
                ]
                try:
                    request_starttime = time.time()
                    completion = await client.chat.completions.create(
                        model=cfg.generation_model,
                        messages=messages,
                        temperature=cfg.generation_temperature,
                        max_tokens=cfg.generation_max_tokens,
                        **extra_body_kwargs,
                    )
                    duration = time.time() - request_starttime
                    raw = completion.model_dump()
                    text = (
                        completion.choices[0].message.content or ""
                        if completion.choices
                        else ""
                    )
                    if logger is not None:
                        logger.log_call(
                            model=cfg.generation_model,
                            request_data={
                                "model": cfg.generation_model,
                                "messages": messages,
                                "temperature": cfg.generation_temperature,
                                "max_tokens": cfg.generation_max_tokens,
                                **extra_body_kwargs,
                            },
                            response_data=raw,
                            duration_ms=duration,
                            api_key=cfg.generation_api_key,
                        )
                    code = handler.extract_plotting_code(text)
                    return {
                        "id": sample["id"],
                        "model": cfg.generation_model,
                        "task": sample["task"],
                        "raw_response": raw,
                        "response_text": text,
                        "code": code,
                        "time_used_gen": time.perf_counter() - t0,
                        "error": "",
                    }
                except Exception as e:  # noqa: BLE001
                    return {
                        "id": sample["id"],
                        "model": cfg.generation_model,
                        "task": sample["task"],
                        "raw_response": None,
                        "response_text": "",
                        "code": "",
                        "time_used_gen": time.perf_counter() - t0,
                        "error": f"generation_error: {e}",
                    }

        tasks = [_gen_one(s) for s in prepared_samples]
        results: list[dict[str, Any]] = []
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

        n_errors = sum(1 for r in results if r.get("error"))
        return results, {"count": len(results), "errors": n_errors}

    def _load_external_generations(
        self, prepared_samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Load rollouts produced outside this verifier.

        Accepts either ``generations_path`` (jsonl) or
        ``generations_inline``. Each row must have ``id`` and at least
        one of ``code`` / ``response_text``; missing ``code`` is
        computed via the language handler's ``extract_plotting_code``.
        """
        cfg = self.config
        if cfg.generations_inline is not None:
            raw_rows = list(cfg.generations_inline)
        elif cfg.generations_path:
            raw_rows = read_jsonl(cfg.generations_path)
        else:
            raise ValueError(
                "run_generation=False but neither generations_path nor "
                "generations_inline was provided"
            )

        handler = get_language_handler(
            cfg.language, config={"plotting_language": cfg.language}
        )
        sample_by_id = {str(s["id"]): s for s in prepared_samples}

        out: list[dict[str, Any]] = []
        for r in raw_rows:
            rid = r.get("id")
            response_text = r.get("response_text", "") or ""
            code = r.get("code")
            if not code:
                code = handler.extract_plotting_code(response_text) if response_text else ""
            sample = sample_by_id.get(str(rid), {})
            out.append(
                {
                    "id": rid,
                    "model": r.get("model", "external"),
                    "task": r.get("task", sample.get("task", "")),
                    "raw_response": r.get("raw_response"),
                    "response_text": response_text,
                    "code": code,
                    "time_used_gen": r.get("time_used_gen", 0.0),
                    "error": r.get("error", ""),
                }
            )
        return out

    # ==================================================================
    # Stage 2: execution
    # ==================================================================
    def _merge_prepared_with_generations(
        self,
        prepared_samples: list[dict[str, Any]],
        generations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared_by_id = {str(s["id"]): s for s in prepared_samples}
        rows: list[dict[str, Any]] = []
        for gen in generations:
            base = dict(prepared_by_id.get(str(gen.get("id")), {}))
            base.update(gen)
            rows.append(base)
        return rows

    def _stage_execution(
        self,
        rows: list[dict[str, Any]],
        csv_dir: str | None,
        work_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cfg = self.config
        language = cfg.language

        preflight_err = self._preflight()
        if preflight_err is not None:
            execution_rows = [
                {**row, "error": preflight_err, "has_plot": False, "plots_generated": []}
                for row in rows
            ]
            return execution_rows, {
                "count": len(execution_rows),
                "exec_errors": len(execution_rows),
                "no_plots": len(execution_rows),
                "failed_shards": 0,
                "preflight_error": preflight_err,
            }

        if not rows:
            return [], {"count": 0, "exec_errors": 0, "no_plots": 0, "failed_shards": 0}

        shard_size = (
            cfg.execution_shard_size
            if cfg.execution_shard_size > 0
            else max(1, len(rows))
        )
        shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]
        shard_root = ensure_dir(work_dir / "execution_shards")
        handler_extra = self._handler_config()

        all_rows: list[dict[str, Any]] = []
        failed_shards = 0
        pool = ProcessPoolExecutor(max_workers=max(1, cfg.execution_workers))
        fut_map: dict[Any, int] = {}
        for shard_id, shard in enumerate(shards):
            fut = pool.submit(
                _execute_chunk,
                language,
                csv_dir,
                shard,
                str(shard_root / f"shard_{shard_id}"),
                handler_extra,
            )
            fut_map[fut] = shard_id

        completed: set = set()
        timed_out = False
        try:
            for fut in as_completed(fut_map, timeout=cfg.execution_timeout):
                completed.add(fut)
                shard_id = fut_map[fut]
                try:
                    all_rows.extend(fut.result())
                except Exception as e:  # noqa: BLE001
                    failed_shards += 1
                    err = f"Shard process crashed: {type(e).__name__}: {e}"
                    all_rows.extend(
                        {**row, "error": err, "has_plot": False, "plots_generated": []}
                        for row in shards[shard_id]
                    )
        except TimeoutError:
            timed_out = True

        not_done = [f for f in fut_map if f not in completed]
        for fut in not_done:
            shard_id = fut_map[fut]
            err = f"Shard timed out ({cfg.execution_timeout}s)"
            all_rows.extend(
                {**row, "error": err, "has_plot": False, "plots_generated": []}
                for row in shards[shard_id]
            )
            failed_shards += 1

        if timed_out:
            _force_kill_pool(pool)
        pool.shutdown(wait=not timed_out, cancel_futures=True)

        n_errors = sum(1 for r in all_rows if r.get("error"))
        n_no_plots = sum(1 for r in all_rows if not r.get("has_plot"))
        return all_rows, {
            "count": len(all_rows),
            "exec_errors": n_errors,
            "no_plots": n_no_plots,
            "failed_shards": failed_shards,
        }

    # ==================================================================
    # Stage 3: evaluation (judge)
    # ==================================================================
    async def _stage_evaluation(
        self, execution_rows: list[dict[str, Any]], work_dir: Path
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cfg = self.config
        instructs = load_instructs(cfg.repo_root, cfg.language)
        logger = APILogger(log_dir=cfg.eval_api_logs) if cfg.eval_enable_logger else None
        client = OpenAICompatClient(
            model=cfg.eval_model,
            base_url=cfg.eval_base_url,
            api_key=cfg.eval_api_key,
            timeout=cfg.eval_timeout,
        )
        sem = asyncio.Semaphore(max(1, cfg.eval_concurrency))

        async def _judge_one(messages: list[dict[str, Any]]) -> tuple[int | None, str]:
            async with sem:
                try:
                    t0 = time.time()
                    data = await client.chat_completion(
                        messages,
                        temperature=cfg.eval_temperature,
                        max_tokens=cfg.eval_max_tokens,
                    )
                    duration = time.time() - t0
                    if logger is not None:
                        logger.log_call(
                            model=cfg.eval_model,
                            request_data={"messages": messages},
                            response_data=data,
                            duration_ms=duration,
                            api_key=cfg.eval_api_key,
                        )
                    text = data.get("response_text", "")
                    return _parse_score(text), text
                except Exception as e:  # noqa: BLE001
                    return None, f"judge_error: {e}"

        async def _score_row(row: dict[str, Any]) -> dict[str, Any]:
            out = dict(row)
            gen_plots = row.get("plots_generated") or []
            if not gen_plots:
                out["score_vis"] = 0
                out["score_task"] = 0
                out["scoring_response_vis"] = ""
                out["scoring_response_task"] = ""
                return out

            vis_messages = [
                {"role": "system", "content": instructs["system_prompt"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructs["judge_instruct_vis"]},
                        image_to_content_part(gen_plots[0]),
                        image_to_content_part((row.get("plots_gt") or [""])[0]),
                    ],
                },
            ]
            task_text = "\n".join(
                [
                    instructs["judge_instruct_task"],
                    "[PLOT TASK]:",
                    row.get("task__plot_description", "") or "",
                    row.get("task__plot_style", "") or "",
                ]
            )
            task_parts: list[dict[str, Any]] = [{"type": "text", "text": task_text}]
            task_parts.extend(image_to_content_part(img) for img in gen_plots)
            task_messages = [
                {"role": "system", "content": instructs["system_prompt"]},
                {"role": "user", "content": task_parts},
            ]

            (vis_score, vis_resp), (task_score, task_resp) = await asyncio.gather(
                _judge_one(vis_messages),
                _judge_one(task_messages),
            )
            out["score_vis"] = vis_score
            out["score_task"] = task_score
            out["scoring_response_vis"] = vis_resp
            out["scoring_response_task"] = task_resp
            return out

        results = await asyncio.gather(*[_score_row(r) for r in execution_rows])
        n_errors = sum(
            1 for r in results if r.get("score_vis") is None or r.get("score_task") is None
        )
        return results, {"count": len(results), "judge_errors": n_errors}

    # ==================================================================
    # Reward + aggregate
    # ==================================================================
    def _compute_per_sample(
        self, scored_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        cfg = self.config
        weights = cfg.reward_weights
        w_exec = float(weights.get("exec_ok", 0.0))
        w_vis = float(weights.get("vis", 0.0))
        w_task = float(weights.get("task", 0.0))

        per_sample: list[dict[str, Any]] = []
        for row in scored_rows:
            has_plot = bool(row.get("has_plot"))
            err = row.get("error") or ""
            exec_ok = has_plot and not err

            vis_raw = row.get("score_vis")
            task_raw = row.get("score_task")
            vis_norm = (float(vis_raw) if vis_raw is not None else 0.0) / 100.0
            task_norm = (float(task_raw) if task_raw is not None else 0.0) / 100.0

            if cfg.reward_zero_on_exec_fail and not exec_ok:
                reward = 0.0
            else:
                reward = w_exec * float(exec_ok) + w_vis * vis_norm + w_task * task_norm

            per_sample.append(
                {
                    "id": row.get("id"),
                    "exec_ok": exec_ok,
                    "has_plot": has_plot,
                    "score_vis": vis_raw,
                    "score_task": task_raw,
                    "reward": round(float(reward), 6),
                    "error": err,
                }
            )
        return per_sample

    def _aggregate(
        self,
        per_sample: list[dict[str, Any]],
        scored_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        n = len(per_sample)
        if n == 0:
            return {
                "count": 0,
                "exec_rate": 0.0,
                "vis_mean": 0.0,
                "task_mean": 0.0,
                "vis_good": 0.0,
                "task_good": 0.0,
                "reward_mean": 0.0,
            }

        vis_scores = [float(r.get("score_vis") or 0) for r in scored_rows]
        task_scores = [float(r.get("score_task") or 0) for r in scored_rows]
        rewards = [s["reward"] for s in per_sample]
        exec_oks = [1.0 if s["exec_ok"] else 0.0 for s in per_sample]

        return {
            "count": n,
            "exec_rate": round(mean(exec_oks), 4),
            "vis_mean": round(mean(vis_scores), 2),
            "task_mean": round(mean(task_scores), 2),
            "vis_good": round(sum(1 for s in vis_scores if s >= 75) / n, 4),
            "task_good": round(sum(1 for s in task_scores if s >= 75) / n, 4),
            "reward_mean": round(mean(rewards), 6),
        }


__all__ = [
    "PlottingRLVerifier",
    "PlottingRLVerifierConfig",
]
