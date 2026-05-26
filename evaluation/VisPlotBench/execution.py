import argparse
import math
import os
import signal
import subprocess
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from plotting_benchmark.language_utils import get_language_handler

from io_utils import read_json, read_jsonl, write_jsonl, ensure_dir


_FENCE_LANG_TAGS = frozenset({
    "python", "py", "python3", "html", "javascript", "js", "typescript", "ts",
    "json", "xml", "css", "bash", "sh", "shell", "zsh", "sql", "yaml", "yml",
    "ruby", "rb", "java", "c", "cpp", "csharp", "cs", "go", "rust", "r",
    "swift", "kotlin", "scala", "perl", "php", "lua", "text", "plain",
    "plaintext", "markdown", "md", "csv", "toml", "ini", "dockerfile",
    "makefile", "cmake", "matlab", "octave", "jsx", "tsx",
})


def _strip_code_fence_tag(code: str) -> str:
    """Strip a leftover language tag (e.g. 'python\\n', 'html\\n') from the
    start of a code string.  Only strips known code-fence language tags,
    not domain keywords like 'sequenceDiagram' or 'graph'."""
    parts = code.split("\n", 1)
    if len(parts) > 1 and parts[0].strip().lower() in _FENCE_LANG_TAGS:
        return parts[1]
    return code


def _make_error_records(
    shard_rows: list[dict[str, Any]], error_msg: str,
) -> list[dict[str, Any]]:
    """Return error-marked records for every row in a failed shard."""
    records = []
    for row in shard_rows:
        rec = dict(row)
        rec["error"] = error_msg
        rec["has_plot"] = False
        rec["plots_generated"] = []
        records.append(rec)
    return records


def _execute_chunk(
    repo_root: str,
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
        # handler_extra is plumbed through from main() and lets us flip the
        # Python handler into subprocess mode without touching every other
        # language's contract (they'll just ignore the extra keys).
        if handler_extra:
            cfg_dict.update(handler_extra)
        cfg = OmegaConf.create(cfg_dict)
        handler = get_language_handler(language, config=OmegaConf.to_container(cfg, resolve=True))
        df = pd.DataFrame(shard_rows)
        out_nb = Path(shard_dir) / f"plots_{language}.ipynb"
        out_nb.parent.mkdir(parents=True, exist_ok=True)
        handler.build_plots(df, out_nb, Path(csv_dir) if csv_dir else None)
        res = handler.parse_plots_notebook(out_nb)
        merged = df.drop(columns=[c for c in res.columns if c in df.columns and c != "id"]).merge(res, on="id", how="left")
        for idx, row in merged.iterrows():
            if row.get("response_text", "") == "" and row.get("error", "") == "":
                merged.at[idx, "error"] = "Unexpected Error"
            if bool(row.get("has_plot", False)) is False and row.get("error", "") == "":
                merged.at[idx, "error"] = "Unexpected Error"
        return merged.to_dict(orient="records")
    except Exception:
        tb = traceback.format_exc()
        error_msg = f"Shard execution failed:\n{tb}"
        return _make_error_records(shard_rows, error_msg)


def _descendant_pids(pid: int) -> list[int]:
    """Walk the process tree rooted at ``pid`` (exclusive) and return every
    descendant PID. Uses stdlib + ``pgrep`` so we don't pull in psutil.

    Returns empty list if ``pgrep`` is unavailable (e.g. minimal containers);
    in that case only the direct worker is killed and any orphaned ipykernel
    will be reaped by init at exit. We log a hint in that path.
    """
    descendants: list[int] = []
    stack = [pid]
    while stack:
        cur = stack.pop()
        try:
            res = subprocess.run(
                ["pgrep", "-P", str(cur)],
                capture_output=True, text=True, timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return descendants
        if res.returncode != 0:
            continue
        for line in res.stdout.split():
            try:
                child = int(line)
            except ValueError:
                continue
            descendants.append(child)
            stack.append(child)
    return descendants


def _force_kill_pool(pool: ProcessPoolExecutor) -> None:
    """SIGKILL every worker AND every descendant (ipykernel etc.).

    ``ProcessPoolExecutor`` only tracks its direct workers, so when a shard
    times out the Jupyter kernels spawned via ``nbconvert`` are left as
    orphans (consuming a kernel slot + CPU until the OS reaps them).  We
    walk the tree ourselves before sending SIGKILL so the next ``./run.sh``
    starts from a clean state.
    """
    worker_pids = list(getattr(pool, "_processes", {}).keys())
    all_pids: list[int] = []
    for wpid in worker_pids:
        all_pids.extend(_descendant_pids(wpid))
        all_pids.append(wpid)
    # Kill children before parents so kernel managers can't respawn.
    for pid in sorted(set(all_pids), reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _print_summary(language: str, all_rows: list[dict[str, Any]], failed_shards: int, elapsed: float) -> None:
    total = len(all_rows)
    n_errors = sum(1 for r in all_rows if r.get("error"))
    n_plots = sum(1 for r in all_rows if r.get("has_plot"))
    n_ok = total - n_errors

    print(f"\n  [{language}] Execution summary  ({elapsed:.1f}s)")
    print(f"    Total: {total}  |  OK: {n_ok}  |  With plots: {n_plots}  |  Errors: {n_errors}", end="")
    if failed_shards:
        print(f"  |  Failed shards: {failed_shards}")
    else:
        print()

    if n_errors > 0:
        error_counts: Counter[str] = Counter()
        for r in all_rows:
            err = r.get("error", "")
            if err:
                lines = [l.strip() for l in err.strip().split("\n") if l.strip()]
                key = lines[-1][:120] if lines else err[:120]
                error_counts[key] += 1
        print(f"    Error breakdown ({len(error_counts)} types):")
        for msg, cnt in error_counts.most_common(10):
            print(f"      {cnt:>4}x  {msg}")
        if len(error_counts) > 10:
            print(f"      ... and {len(error_counts) - 10} more error types")


def main(args: argparse.Namespace) -> None:
    bundle = Path(args.work_dir)
    manifest = read_json(bundle / "manifest.json")
    language = manifest["language"]
    model_name = manifest.get("model", "unknown")
    prepared = {row["id"]: row for row in read_jsonl(bundle / manifest["prepared_samples"])}
    gens = read_jsonl(bundle / manifest["generations"])

    rows = []
    for gen in gens:
        base = prepared[gen["id"]].copy()
        base.update(gen)
        if "code" in base:
            base["code"] = _strip_code_fence_tag(base["code"])
        rows.append(base)

    workers = max(1, args.workers)
    shard_size = args.shard_size if args.shard_size > 0 else max(1, math.ceil(len(rows) / workers))
    csv_dir = bundle / manifest["csv_dir"] if manifest.get("csv_dir") else None
    shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]
    shard_root = ensure_dir(bundle / "execution_shards")

    print(f"\n{'='*60}")
    print(f"  Language: {language}  |  Model: {model_name}")
    print(f"  Samples: {len(rows)}  |  Workers: {workers}  |  Shards: {len(shards)}  ({shard_size} samples/shard)")
    print(f"  Timeout: {args.timeout}s")
    print(f"{'='*60}")

    total_timeout = args.timeout
    all_rows: list[dict[str, Any]] = []
    failed_shards = 0
    t0 = time.time()

    # Build per-language handler extras.  Only python+subprocess uses this
    # today; everything else gets ``{}`` which is a no-op in _execute_chunk.
    handler_extra: dict[str, Any] = {}
    if language == "python" and getattr(args, "python_exec_mode", "jupyter") == "subprocess":
        handler_extra["python_exec_mode"] = "subprocess"
        handler_extra["python_subproc_timeout"] = float(args.python_subproc_timeout)
        handler_extra["python_subproc_parallel"] = int(args.python_subproc_parallel)
        print(
            f"  [python-exec-mode] subprocess  "
            f"per-sample timeout={args.python_subproc_timeout}s  "
            f"intra-shard parallel={args.python_subproc_parallel}"
        )

    pool = ProcessPoolExecutor(max_workers=workers)
    fut_map: dict[Any, int] = {}
    for shard_id, shard in enumerate(shards):
        fut = pool.submit(
            _execute_chunk,
            manifest["repo_root"],
            manifest["language"],
            str(csv_dir) if csv_dir else None,
            shard,
            str(shard_root / f"shard_{shard_id}"),
            handler_extra,
        )
        fut_map[fut] = shard_id

    pbar = tqdm(
        total=len(shards),
        desc=f"Executing [{language}]",
        unit="shard",
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )
    completed_futs: set = set()
    timed_out = False

    try:
        for fut in as_completed(fut_map, timeout=total_timeout):
            completed_futs.add(fut)
            shard_id = fut_map[fut]
            try:
                results = fut.result()
                all_rows.extend(results)
                shard_errs = sum(1 for r in results if r.get("error"))
                if shard_errs:
                    pbar.set_postfix_str(f"shard {shard_id}: {shard_errs} err")
            except Exception as e:
                failed_shards += 1
                pbar.write(f"  [!] Shard {shard_id} crashed: {type(e).__name__}: {e}")
                all_rows.extend(
                    _make_error_records(shards[shard_id], f"Shard process crashed: {type(e).__name__}: {e}")
                )
            pbar.update(1)
    except TimeoutError:
        timed_out = True

    not_done = [f for f in fut_map if f not in completed_futs]
    if not_done:
        pbar.write(f"  [!] {len(not_done)} shard(s) timed out after {total_timeout}s — marking as errors")
        for fut in not_done:
            shard_id = fut_map[fut]
            all_rows.extend(
                _make_error_records(shards[shard_id], f"Shard timed out ({total_timeout}s)")
            )
            failed_shards += 1
            pbar.update(1)

    pbar.close()

    if timed_out:
        _force_kill_pool(pool)
    pool.shutdown(wait=not timed_out, cancel_futures=True)

    elapsed = time.time() - t0

    write_jsonl(sorted(all_rows, key=lambda x: x["id"]), bundle / manifest["executions"])
    _print_summary(language, all_rows, failed_shards, elapsed)
    print(f"  -> {bundle / manifest['executions']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=0,
                        help="Samples per Jupyter notebook. 0 = auto (rows/workers). "
                             "Smaller = more parallelism but more kernel startups.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Total timeout in seconds for all shards. "
                             "Shards still running after this are marked as errors. "
                             "Default: 600 (10 min).")
    parser.add_argument("--python-exec-mode", choices=["jupyter", "subprocess"],
                        default="jupyter",
                        help="Python execution backend. "
                             "'jupyter' (default): legacy nbconvert + ipykernel. "
                             "'subprocess': run each sample in its own python "
                             "process with monkey-patched plt.show / Figure.show / "
                             "etc.  Avoids ipykernel hangs + orphan kernels.  "
                             "Only affects language=python; other languages ignore it.")
    parser.add_argument("--python-subproc-timeout", type=float, default=60.0,
                        help="Per-sample wall-clock timeout (seconds) when "
                             "--python-exec-mode=subprocess. Default: 60s "
                             "(bumped from 30s — SFT'd models emit heavier "
                             "matplotlib code that legitimately needs ~30-50s).")
    parser.add_argument("--python-subproc-parallel", type=int, default=1,
                        help="How many subprocesses to run in parallel within "
                             "a single shard when --python-exec-mode=subprocess. "
                             "Default 1 (sequential within shard); rely on the "
                             "outer ProcessPoolExecutor (--workers) for "
                             "parallelism.  Nesting subprocess.Popen inside "
                             "ThreadPool inside ProcessPool causes hangs on "
                             "macOS — keep at 1 unless you've validated >1 "
                             "in your env.")
    main(parser.parse_args())
