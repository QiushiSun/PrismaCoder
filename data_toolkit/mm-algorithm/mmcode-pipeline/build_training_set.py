"""Assemble a training-ready package from the four MMCode augmentation rollouts.

Pipeline (each stage is idempotent and resumable):

  stage1_filter   load 4 rollouts → drop empty solution / svg_valid=False / len(io.inputs)<3
  stage2_xml      strict XML parse via xml.etree.ElementTree.fromstring → drop malformed
  stage3_iocheck  run each solution against its own test cases in a subprocess sandbox
  stage4_render   rasterize kept SVGs to PNG via cairosvg, write under images/{svg,png}/
  stage5_package  emit train.jsonl, stats.json, manifest.json, README.md

Per-record `id` fields collide across rollouts, so we mint a `global_id` of the form
`r{1,2,3,4}_<original_id>` and use it as the filename stem for assets.

Usage:
  conda run -n vpb python build_training_set.py all          # run every stage
  conda run -n vpb python build_training_set.py stage3_iocheck --workers 8 --timeout 10
  conda run -n vpb python build_training_set.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

BASE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BASE / "training-set-v1"

# ── rollouts ─────────────────────────────────────────────────────────────────

# (rollout_dir, short_tag, jsonl_filename)
ROLLOUTS: list[tuple[str, str, str]] = [
    ("rollout-0420-claude46-mmcode-svg",            "r1", "problems.dedup.jsonl"),
    ("rollout-0420-claude46-mmcode-test-icl",       "r2", "problems.dedup.jsonl"),
    ("rollout-0420-claude46-mmcode-hard",           "r3", "problems.dedup.jsonl"),
    ("rollout-0421-claude46-mmcode-hard-test-icl",  "r4", "problems.dedup.jsonl"),
]

INTERMEDIATE = {
    "stage1": "_stage1_filtered.jsonl",
    "stage2": "_stage2_xml_ok.jsonl",
    "stage3": "_stage3_with_io_check.jsonl",
    "stage4": "_stage4_with_assets.jsonl",
}

# ── small helpers ────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _resolve_io(rec: dict) -> dict:
    """Return a dict with `inputs`/`outputs` lists (best-effort)."""
    io = rec.get("input_output") or {}
    if isinstance(io, str):
        try:
            io = json.loads(io)
        except json.JSONDecodeError:
            io = {}
    if not isinstance(io, dict):
        io = {}
    return io


def _normalize_output(s: str) -> str:
    """Whitespace-tolerant normalization for stdout comparison."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# ── stage 1: filter ──────────────────────────────────────────────────────────

def stage1_filter(out_dir: Path) -> dict:
    """Load all 4 rollouts; keep only records that satisfy the basic invariants."""
    out_path = out_dir / INTERMEDIATE["stage1"]
    funnel: dict[str, Any] = {"per_rollout": {}, "kept": 0, "dropped": 0}
    kept: list[dict] = []
    drop_reasons: dict[str, int] = {
        "missing_or_empty_solution": 0,
        "svg_invalid": 0,
        "lt3_inputs": 0,
        "io_outputs_mismatch": 0,
    }

    for rollout_dir, tag, jsonl_name in ROLLOUTS:
        path = BASE / rollout_dir / jsonl_name
        if not path.exists():
            print(f"  [WARN] missing {path}; skipping rollout {tag}")
            continue
        records = _read_jsonl(path)
        local_kept = 0
        for r in records:
            sol = r.get("solution") or ""
            if not isinstance(sol, str) or not sol.strip():
                drop_reasons["missing_or_empty_solution"] += 1
                continue
            if not r.get("svg_valid", False):
                drop_reasons["svg_invalid"] += 1
                continue
            io = _resolve_io(r)
            inputs = io.get("inputs") or []
            outputs = io.get("outputs") or []
            if not isinstance(inputs, list) or len(inputs) < 3:
                drop_reasons["lt3_inputs"] += 1
                continue
            if not isinstance(outputs, list) or len(outputs) != len(inputs):
                drop_reasons["io_outputs_mismatch"] += 1
                continue

            new_rec = dict(r)
            new_rec["rollout"] = rollout_dir
            new_rec["rollout_tag"] = tag
            new_rec["global_id"] = f"{tag}_{r.get('id', 'unknown')}"
            new_rec["input_output"] = {"inputs": inputs, "outputs": outputs}
            kept.append(new_rec)
            local_kept += 1
        funnel["per_rollout"][tag] = {
            "rollout": rollout_dir,
            "input": len(records),
            "kept": local_kept,
        }

    n = _write_jsonl(out_path, kept)
    funnel["kept"] = n
    funnel["dropped"] = sum(drop_reasons.values())
    funnel["drop_reasons"] = drop_reasons
    print(f"  stage1: kept {n} / {n + funnel['dropped']} records → {out_path.name}")
    for k, v in drop_reasons.items():
        print(f"    dropped {k}: {v}")
    return funnel


# ── stage 2: strict XML parse ────────────────────────────────────────────────

def stage2_xml(out_dir: Path) -> dict:
    in_path = out_dir / INTERMEDIATE["stage1"]
    out_path = out_dir / INTERMEDIATE["stage2"]
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = _read_jsonl(in_path)
    kept: list[dict] = []
    fail_reasons: dict[str, int] = {}

    for r in tqdm(records, desc="stage2_xml", unit="rec"):
        svg = r.get("visualization_svg") or ""
        try:
            ET.fromstring(svg)
            kept.append(r)
        except ET.ParseError as e:
            msg = str(e).split(":")[0][:60]
            fail_reasons[msg] = fail_reasons.get(msg, 0) + 1
        except Exception as e:
            msg = f"non-ParseError: {type(e).__name__}"
            fail_reasons[msg] = fail_reasons.get(msg, 0) + 1

    n = _write_jsonl(out_path, kept)
    funnel = {
        "input": len(records),
        "kept": n,
        "dropped": len(records) - n,
        "fail_reasons": fail_reasons,
    }
    print(f"  stage2: kept {n} / {len(records)} records → {out_path.name}")
    for k, v in fail_reasons.items():
        print(f"    XML fail '{k}': {v}")
    return funnel


# ── stage 3: code-IO consistency ─────────────────────────────────────────────

def _run_one_solution(
    solution: str,
    test_input: str,
    *,
    timeout: float,
    python_exe: str,
) -> tuple[str, str, str | None]:
    """Run the given solution as a subprocess.

    Returns (status, stdout, error_message_or_None).
    status ∈ {"ok", "timeout", "runtime_error"}.
    """
    with tempfile.TemporaryDirectory(prefix="mmcode_io_") as tmpdir:
        sol_path = os.path.join(tmpdir, "sol.py")
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution)
        try:
            cp = subprocess.run(
                [python_exe, sol_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "", f"timeout after {timeout}s"
        except Exception as e:
            return "runtime_error", "", f"launch failed: {e}"
    if cp.returncode != 0:
        err = (cp.stderr or "").strip().splitlines()[-1:]
        return "runtime_error", cp.stdout or "", (err[0] if err else f"exit {cp.returncode}")[:200]
    return "ok", cp.stdout or "", None


def _check_one_record(record: dict, timeout: float, python_exe: str) -> dict:
    io = record.get("input_output") or {}
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    n = min(len(inputs), len(outputs))
    sol = record.get("solution") or ""

    per_case: list[dict[str, Any]] = []
    for i in range(n):
        inp = inputs[i] if isinstance(inputs[i], str) else json.dumps(inputs[i])
        exp = outputs[i] if isinstance(outputs[i], str) else json.dumps(outputs[i])
        status, got, err = _run_one_solution(
            sol, inp, timeout=timeout, python_exe=python_exe,
        )
        if status != "ok":
            per_case.append({
                "ok": False,
                "status": status,
                "error": err,
            })
            continue
        ok = _normalize_output(got) == _normalize_output(exp)
        per_case.append({
            "ok": ok,
            "status": "match" if ok else "mismatch",
            "error": None,
        })

    n_pass = sum(1 for c in per_case if c["ok"])
    if n_pass == n and n > 0:
        verdict = "pass_all"
    elif n_pass == 0:
        verdict = "fail_all"
    else:
        verdict = "pass_partial"

    return {
        "tested": n,
        "passed": n_pass,
        "status": verdict,
        "per_case": per_case,
    }


def stage3_iocheck(out_dir: Path, *, workers: int, timeout: float, resume: bool) -> dict:
    in_path = out_dir / INTERMEDIATE["stage2"]
    out_path = out_dir / INTERMEDIATE["stage3"]
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = _read_jsonl(in_path)
    done_ids: set[str] = set()
    if resume and out_path.exists():
        for r in _read_jsonl(out_path):
            done_ids.add(r.get("global_id", ""))
        print(f"  stage3: resume — {len(done_ids)} records already checked")

    pending = [r for r in records if r.get("global_id") not in done_ids]
    if not pending:
        print("  stage3: nothing to do")
    else:
        python_exe = sys.executable
        verdicts_lock = threading.Lock()
        with out_path.open("a" if resume else "w", encoding="utf-8") as fout, \
             ThreadPoolExecutor(max_workers=workers) as ex, \
             tqdm(total=len(pending), desc="stage3_iocheck", unit="rec") as pbar:
            futures = {
                ex.submit(_check_one_record, rec, timeout, python_exe): rec
                for rec in pending
            }
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    io_result = fut.result()
                except Exception as e:
                    io_result = {
                        "tested": 0,
                        "passed": 0,
                        "status": "exec_error",
                        "per_case": [],
                        "error": str(e)[:200],
                    }
                new_rec = dict(rec)
                new_rec["io_check"] = io_result
                with verdicts_lock:
                    fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    fout.flush()
                pbar.update(1)
                pbar.set_postfix(verdict=io_result["status"])

    final = _read_jsonl(out_path)
    by_verdict: dict[str, int] = {}
    for r in final:
        v = r.get("io_check", {}).get("status", "unknown")
        by_verdict[v] = by_verdict.get(v, 0) + 1
    print(f"  stage3: {len(final)} records labelled, distribution: {by_verdict}")
    return {"total": len(final), "by_verdict": by_verdict}


# ── stage 4: SVG → PNG ───────────────────────────────────────────────────────

def _render_one(
    global_id: str,
    svg: str,
    out_svg: Path,
    out_png: Path,
    *,
    target_width: int,
) -> tuple[str, str | None]:
    try:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        with out_svg.open("w", encoding="utf-8") as f:
            f.write(svg)
        import cairosvg  # imported lazily so the script remains importable without it
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(out_png),
            output_width=target_width,
        )
        return "ok", None
    except Exception as e:
        return "fail", f"{type(e).__name__}: {str(e)[:200]}"


def stage4_render(out_dir: Path, *, workers: int, target_width: int, resume: bool) -> dict:
    in_path = out_dir / INTERMEDIATE["stage3"]
    out_path = out_dir / INTERMEDIATE["stage4"]
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = _read_jsonl(in_path)
    images_dir = out_dir / "images"
    svg_dir = images_dir / "svg"
    png_dir = images_dir / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict] = []
    fail_reasons: dict[str, int] = {}

    def task(rec: dict) -> tuple[dict, str, str | None]:
        gid = rec["global_id"]
        svg = rec.get("visualization_svg") or ""
        out_svg = svg_dir / f"{gid}.svg"
        out_png = png_dir / f"{gid}.png"
        if resume and out_svg.exists() and out_png.exists() and out_png.stat().st_size > 0:
            new = dict(rec)
            new["image_svg"]  = f"images/svg/{gid}.svg"
            new["image_png"]  = f"images/png/{gid}.png"
            new["render_status"] = "cached"
            return new, "cached", None
        status, err = _render_one(gid, svg, out_svg, out_png, target_width=target_width)
        new = dict(rec)
        if status == "ok":
            new["image_svg"]    = f"images/svg/{gid}.svg"
            new["image_png"]    = f"images/png/{gid}.png"
            new["render_status"] = "ok"
        else:
            new["render_status"] = "failed"
            new["render_error"]  = err
        return new, status, err

    with ThreadPoolExecutor(max_workers=workers) as ex, \
         tqdm(total=len(records), desc="stage4_render", unit="img") as pbar:
        futures = {ex.submit(task, r): r for r in records}
        for fut in as_completed(futures):
            new_rec, status, err = fut.result()
            enriched.append(new_rec)
            if status not in ("ok", "cached"):
                fail_reasons[err or "unknown"] = fail_reasons.get(err or "unknown", 0) + 1
            pbar.update(1)
            pbar.set_postfix(failed=sum(fail_reasons.values()))

    n = _write_jsonl(out_path, enriched)
    n_ok = sum(1 for r in enriched if r.get("render_status") in ("ok", "cached"))
    print(f"  stage4: rendered {n_ok} / {n} → images/{{svg,png}}/")
    for k, v in fail_reasons.items():
        print(f"    render fail '{k}': {v}")
    return {"total": n, "rendered_ok": n_ok, "fail_reasons": fail_reasons}


# ── stage 5: package ─────────────────────────────────────────────────────────

# Fields kept in the public train.jsonl (drops verbose internals like ICL ids).
PUBLIC_FIELDS = [
    "global_id", "id", "rollout", "rollout_tag", "model",
    "icl_source", "difficulty_mode", "recipe", "icl_example_ids",
    "question", "solution",
    "input_output",
    "image_svg", "image_png", "visualization_caption",
    "difficulty", "tags",
    "io_check",
    "original_id", "dup_seq", "recovered_from_failed_batch",
]


def _trim_record(r: dict) -> dict:
    return {k: r[k] for k in PUBLIC_FIELDS if k in r}


def stage5_package(
    out_dir: Path,
    *,
    io_threshold: str,
    funnel_stats: dict[str, Any] | None = None,
    cli_args: dict | None = None,
) -> dict:
    in_path = out_dir / INTERMEDIATE["stage4"]
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = _read_jsonl(in_path)

    train_strict: list[dict] = []
    train_all: list[dict] = []

    threshold_keep = {
        "pass_all":     {"pass_all"},
        "pass_partial": {"pass_all", "pass_partial"},
        "any":          {"pass_all", "pass_partial", "fail_all", "exec_error"},
    }.get(io_threshold, {"pass_all"})

    by_verdict_kept: dict[str, int] = {}
    by_rollout_kept: dict[str, int] = {}
    by_rollout_all:  dict[str, int] = {}

    for r in records:
        verdict = r.get("io_check", {}).get("status", "unknown")
        if r.get("render_status") not in ("ok", "cached"):
            continue
        trimmed = _trim_record(r)
        train_all.append(trimmed)
        by_rollout_all[r["rollout_tag"]] = by_rollout_all.get(r["rollout_tag"], 0) + 1
        if verdict in threshold_keep:
            train_strict.append(trimmed)
            by_verdict_kept[verdict] = by_verdict_kept.get(verdict, 0) + 1
            by_rollout_kept[r["rollout_tag"]] = by_rollout_kept.get(r["rollout_tag"], 0) + 1

    n_strict = _write_jsonl(out_dir / "train.jsonl", train_strict)
    n_all    = _write_jsonl(out_dir / "train_all.jsonl", train_all)

    stats = {
        "build_timestamp": datetime.now().isoformat(),
        "io_threshold_for_train_jsonl": io_threshold,
        "train_jsonl":           {"records": n_strict, "by_rollout": by_rollout_kept,
                                  "by_io_verdict": by_verdict_kept},
        "train_all_jsonl":       {"records": n_all, "by_rollout": by_rollout_all},
        "funnel": funnel_stats or {},
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "build_timestamp": datetime.now().isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "rollouts": [{"dir": d, "tag": t, "jsonl": j} for d, t, j in ROLLOUTS],
        "stages": list(INTERMEDIATE.keys()) + ["stage5_package"],
        "io_threshold_for_train_jsonl": io_threshold,
        "intermediate_files": list(INTERMEDIATE.values()),
        "cli_args": cli_args or {},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _write_readme(out_dir, stats)
    print(f"  stage5: train.jsonl={n_strict}  train_all.jsonl={n_all}")
    return stats


def _write_readme(out_dir: Path, stats: dict) -> None:
    s = stats["train_jsonl"]
    sa = stats["train_all_jsonl"]
    by_v = s.get("by_io_verdict", {})
    by_r = s.get("by_rollout", {})
    by_ra = sa.get("by_rollout", {})
    threshold = stats["io_threshold_for_train_jsonl"]
    funnel = stats.get("funnel", {})
    text = f"""# MMCode-Aug Training Set v1

Built: {stats['build_timestamp']}

## Files

- `train.jsonl`              — strict training set ({s['records']} records, IO threshold = `{threshold}`)
- `train_all.jsonl`          — every record that passed filter + strict-XML + render ({sa['records']} records, with `io_check` field so you can re-pick a threshold)
- `images/svg/<global_id>.svg` — original SVG illustration per record
- `images/png/<global_id>.png` — cairosvg rasterization (output_width=800, RGBA)
- `stats.json`               — per-stage funnel + per-rollout / per-verdict counts
- `manifest.json`            — build config + provenance

## Per-record schema (`train.jsonl`)

```jsonc
{{
  "global_id":  "r1_mmcode_aug_000000",   // unique across rollouts; filename stem for assets
  "id":         "mmcode_aug_000000",      // original within-rollout id
  "rollout":    "rollout-0420-claude46-mmcode-svg",
  "rollout_tag":"r1",
  "model":      "claude-sonnet-4-6-aws",

  "icl_source":     "train" | "test",     // optional, only present in r2/r3/r4
  "difficulty_mode":"standard" | "hard",  // optional, only present in r3/r4
  "recipe":         {{...}},               // algorithm_tags, visual_context, difficulty,
                                          // source_style, [anchor_test_id], [advanced_technique]
  "icl_example_ids":["cf_570_B", ...],

  "question":              "...markdown...",
  "solution":              "...python...",
  "input_output": {{
    "inputs":  ["...stdin string...", ...],
    "outputs": ["...expected stdout...", ...]
  }},

  "image_svg":             "images/svg/r1_mmcode_aug_000000.svg",
  "image_png":             "images/png/r1_mmcode_aug_000000.png",
  "visualization_caption": "...",

  "difficulty": "MEDIUM",
  "tags":       ["Dynamic programming", ...],

  "io_check": {{
    "tested": 4,
    "passed": 4,
    "status": "pass_all" | "pass_partial" | "fail_all" | "exec_error",
    "per_case": [
      {{"ok": true, "status": "match", "error": null}},
      ...
    ]
  }}
}}
```

## Source rollouts

| tag | rollout | mode | passes filter+xml+render | included @ threshold `{threshold}` |
|-----|---------|------|--------------------------|------------------------------------|
| r1 | rollout-0420-claude46-mmcode-svg            | standard·train     | {by_ra.get('r1', 0)} | {by_r.get('r1', 0)} |
| r2 | rollout-0420-claude46-mmcode-test-icl       | standard·test      | {by_ra.get('r2', 0)} | {by_r.get('r2', 0)} |
| r3 | rollout-0420-claude46-mmcode-hard           | hard·train         | {by_ra.get('r3', 0)} | {by_r.get('r3', 0)} |
| r4 | rollout-0421-claude46-mmcode-hard-test-icl  | hard·test          | {by_ra.get('r4', 0)} | {by_r.get('r4', 0)} |

## IO-verdict distribution (in `train.jsonl`)

| verdict | count |
|---|---|
| pass_all     | {by_v.get('pass_all', 0)} |
| pass_partial | {by_v.get('pass_partial', 0)} |
| fail_all     | {by_v.get('fail_all', 0)} |
| exec_error   | {by_v.get('exec_error', 0)} |

## Filter funnel

```
{json.dumps(funnel, indent=2, ensure_ascii=False)}
```

## How to consume

Every image path is **relative to this directory** so you can `cd training-set-v1/` and load directly. To pick a different IO threshold without re-running stages 1-4:

```bash
conda run -n vpb python ../build_training_set.py stage5_package --io-threshold pass_partial
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ── orchestration ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=["all", "stage1_filter", "stage2_xml", "stage3_iocheck",
                 "stage4_render", "stage5_package"],
        help="Run a single stage (or 'all').",
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers for stage3 (subprocess calls) and stage4 (rendering).")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="Per-test-case execution timeout in seconds.")
    ap.add_argument("--target-width", type=int, default=800,
                    help="PNG output width in pixels (height preserves aspect ratio).")
    ap.add_argument("--io-threshold", choices=["pass_all", "pass_partial", "any"],
                    default="pass_all",
                    help="Which io_check verdicts go into train.jsonl (default: pass_all).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip work that's already in the output (stage3/stage4).")
    ap.add_argument("--limit", type=int, default=None,
                    help="(Debug) Only process the first N records of stage1's output.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel: dict[str, Any] = {}

    t0 = time.time()
    print(f"== build_training_set.py — stage = {args.stage} ==")
    print(f"   out_dir = {out_dir}")
    print(f"   workers = {args.workers}, timeout = {args.timeout}s, "
          f"target_width = {args.target_width}, threshold = {args.io_threshold}, "
          f"resume = {args.resume}")

    if args.stage in ("all", "stage1_filter"):
        funnel["stage1"] = stage1_filter(out_dir)
        if args.limit is not None:
            in_path = out_dir / INTERMEDIATE["stage1"]
            recs = _read_jsonl(in_path)[: args.limit]
            _write_jsonl(in_path, recs)
            print(f"  --limit {args.limit} truncated stage1 output")

    if args.stage in ("all", "stage2_xml"):
        funnel["stage2"] = stage2_xml(out_dir)

    if args.stage in ("all", "stage3_iocheck"):
        funnel["stage3"] = stage3_iocheck(
            out_dir, workers=args.workers, timeout=args.timeout, resume=args.resume,
        )

    if args.stage in ("all", "stage4_render"):
        funnel["stage4"] = stage4_render(
            out_dir, workers=args.workers, target_width=args.target_width, resume=args.resume,
        )

    if args.stage in ("all", "stage5_package"):
        existing_funnel: dict[str, Any] = {}
        stats_path = out_dir / "stats.json"
        if stats_path.exists():
            try:
                existing_funnel = json.loads(stats_path.read_text(encoding="utf-8")).get("funnel", {})
            except json.JSONDecodeError:
                pass
        existing_funnel.update(funnel)
        stage5_package(
            out_dir,
            io_threshold=args.io_threshold,
            funnel_stats=existing_funnel,
            cli_args=vars(args),
        )

    print(f"== done in {time.time() - t0:.1f}s ==")


if __name__ == "__main__":
    main()
