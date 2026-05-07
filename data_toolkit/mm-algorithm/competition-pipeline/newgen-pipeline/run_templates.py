"""
Plan-A' runner for LeetCode templates.

For each generated template (rollout/templates/<id>.py):
  1. Import as a module.
  2. For seed in 0..N-1:
       - examples = sample_examples(seed, n=template.recipe.n_examples)
       - Compute outputs: out_k = Solution().METHOD_NAME(**examples[k].kwargs)
       - Render each example to a PNG: variants/images/<id>_v<seed>_<k>.png
       - Substitute __INPUT_K__ / __OUTPUT_K__ in DESCRIPTION with repr() of
         kwargs / outputs.
       - tests = make_test_cases(seed); verify each test in-process AND in
         a subprocess sandbox.
       - Emit one variant record into variants/records.jsonl.

Drops a variant if any of:
  • render fails (raises)
  • Solution raises on example or test inputs
  • In-process tests don't all pass
  • Sandbox test script returns non-zero
  • Two seeds produce identical-byte images (render not parameterised)

Output:
  variants/records.jsonl
  variants/images/<id>_v<seed>_<k>.png
  variants/run_summary.json
  variants/failures.jsonl

Usage:
  conda run -n vpb python run_templates_lc.py --seeds 5 --workers 8
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import io
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

BASE = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE / "templates"
VARIANTS_DIR = BASE / "variants"
IMAGES_DIR = VARIANTS_DIR / "images"
RECORDS_PATH = VARIANTS_DIR / "records.jsonl"
FAILURES_PATH = VARIANTS_DIR / "failures.jsonl"

sys.path.insert(0, str(BASE))
from sandbox import run_program  # type: ignore  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path, name: str = "run") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"lc_run_{name}")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"run_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


def load_records() -> dict[str, dict]:
    """qid → record (from rollout/records.jsonl produced by generate)."""
    p = BASE / "records.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p} — run generate first")
    out: dict[str, dict] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["qid"]] = r
    return out


# ── safe import ─────────────────────────────────────────────────────────────
def _import_template(tpl_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_tpl_{tpl_path.stem}", str(tpl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("spec_from_file_location returned None")
    mod = importlib.util.module_from_spec(spec)
    # Force matplotlib to a non-interactive backend before module import
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec.loader.exec_module(mod)
    return mod


def _read_attr(mod, name: str, required: bool = True):
    if hasattr(mod, name):
        return getattr(mod, name)
    if required:
        raise RuntimeError(f"missing top-level symbol: {name}")
    return None


# ── DESCRIPTION substitution ────────────────────────────────────────────────
def fill_description(
    description: str,
    examples: list[dict],
    outputs: list[Any],
) -> str:
    """Replace __INPUT_k__ / __OUTPUT_k__ placeholders with repr() of values.

    Note: examples and outputs are 0-indexed here; placeholders are 1-indexed.
    """
    out = description
    for i, (ex, expected) in enumerate(zip(examples, outputs), start=1):
        kwargs = ex.get("kwargs", {})
        # Render kwargs as kwarg=value pairs (cleaner than dict repr)
        parts = [f"{k} = {_compact_repr(v)}" for k, v in kwargs.items()]
        in_str = ", ".join(parts)
        out = out.replace(f"__INPUT_{i}__", in_str)
        out = out.replace(f"__OUTPUT_{i}__", _compact_repr(expected))
    return out


def _compact_repr(v: Any, max_inline: int = 80) -> str:
    """Repr that prefers short inline form, falls back to wrapped form."""
    s = repr(v)
    if len(s) <= max_inline:
        return s
    # Long lists / dicts → wrap
    return s


# ── solution → bare source extraction (drop class wrapper for sandbox) ──────
def extract_solution_source(template_src: str) -> str:
    """Pull `class Solution` + helper imports + module-top defs from the
    template source, ready to concatenate with a test_script in sandbox."""
    try:
        tree = ast.parse(template_src)
    except SyntaxError as e:
        raise RuntimeError(f"template source has SyntaxError: {e}") from e

    keep_lines: list[tuple[int, int]] = []  # [(start_lineno, end_lineno)] 1-indexed
    src_lines = template_src.splitlines()

    def _seg_for(node: ast.AST) -> tuple[int, int]:
        return (node.lineno, getattr(node, "end_lineno", node.lineno))

    # Collect: imports + module-level constants with simple lits + class Solution +
    # any module-level function (helpers used by Solution).
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep_lines.append(_seg_for(node))
        elif isinstance(node, ast.ClassDef) and node.name == "Solution":
            keep_lines.append(_seg_for(node))
        elif isinstance(node, ast.FunctionDef):
            # Keep helpers EXCEPT the LLM-generated harness functions
            if node.name in ("sample_examples", "render", "make_test_cases"):
                continue
            keep_lines.append(_seg_for(node))
        elif isinstance(node, ast.Assign):
            # Keep simple constant assignments at top-level (METHOD_NAME, etc).
            # Skip TITLE / DESCRIPTION / SOLUTION_SIGNATURE since they have
            # __INPUT_k__ placeholders that would otherwise pollute the sandbox
            # when imported.
            keep = True
            target_names = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_names.append(t.id)
            if any(n in ("TITLE", "DESCRIPTION", "SOLUTION_SIGNATURE") for n in target_names):
                keep = False
            if keep:
                keep_lines.append(_seg_for(node))

    keep_lines.sort()
    out_chunks: list[str] = []
    for s, e in keep_lines:
        out_chunks.append("\n".join(src_lines[s - 1 : e]))
    return "\n\n".join(out_chunks) + "\n"


def build_test_script(method_name: str, test_cases: list[tuple[dict, Any]]) -> str:
    """Render a sandbox-runnable test_script appended to the bare solution."""
    parts: list[str] = []
    parts.append("def _candidate(**kwargs):")
    parts.append(f"    return Solution().{method_name}(**kwargs)")
    parts.append("")
    parts.append("def test():")
    parts.append("    cases = [")
    for kw, expected in test_cases:
        parts.append(f"        ({_compact_repr(kw)}, {_compact_repr(expected)}),")
    parts.append("    ]")
    parts.append("    for i, (kw, exp) in enumerate(cases):")
    parts.append("        got = _candidate(**kw)")
    parts.append("        assert got == exp, "
                 "f'case #{i} mismatch: got={got!r}, expected={exp!r} (kw={kw!r})'")
    parts.append("    print('all', len(cases), 'cases passed')")
    parts.append("")
    parts.append("test()")
    return "\n".join(parts)


# ── per-variant worker ──────────────────────────────────────────────────────
def process_template(
    qid: str,
    n_seeds: int,
    timeout_per_seed: float,
    out_dir_str: str,
) -> list[dict]:
    """Process all N seed variants for one template. Returns list of dict
    records (variant or failure). Run inside a subprocess via ProcessPool."""
    out_dir = Path(out_dir_str)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tpl_path = TEMPLATES_DIR / f"{qid}.py"
    out: list[dict] = []

    if not tpl_path.exists():
        out.append({"qid": qid, "seed": None, "status": "missing_template"})
        return out

    template_src = tpl_path.read_text(encoding="utf-8")

    try:
        mod = _import_template(tpl_path)
    except Exception as e:
        out.append({
            "qid": qid, "seed": None, "status": "import_fail",
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc()[-1500:],
        })
        return out

    try:
        title = _read_attr(mod, "TITLE")
        description = _read_attr(mod, "DESCRIPTION")
        method_name = _read_attr(mod, "METHOD_NAME")
        signature = _read_attr(mod, "SOLUTION_SIGNATURE")
        Solution_cls = _read_attr(mod, "Solution")
        sample_examples = _read_attr(mod, "sample_examples")
        render = _read_attr(mod, "render")
        make_test_cases = _read_attr(mod, "make_test_cases")
    except Exception as e:
        out.append({
            "qid": qid, "seed": None, "status": "missing_symbol",
            "error": f"{type(e).__name__}: {e}",
        })
        return out

    if not callable(getattr(Solution_cls(), method_name, None)):
        out.append({
            "qid": qid, "seed": None, "status": "method_not_found",
            "error": f"Solution() has no method {method_name!r}",
        })
        return out

    # Pull n_examples from the sample_examples signature default if present,
    # else fall back to the audit file we wrote.
    n_examples_default = 2
    try:
        sig = inspect.signature(sample_examples)
        n_param = sig.parameters.get("n")
        if n_param and n_param.default is not inspect.Parameter.empty:
            n_examples_default = int(n_param.default)
    except (ValueError, TypeError):
        pass
    audit_p = BASE / "audit" / f"{qid}.json"
    if audit_p.exists():
        try:
            n_examples_default = int(json.loads(audit_p.read_text())["n_examples"])
        except Exception:
            pass

    # Try to extract solution source ONCE per template
    try:
        solution_src = extract_solution_source(template_src)
    except Exception as e:
        out.append({
            "qid": qid, "seed": None, "status": "extract_fail",
            "error": f"{type(e).__name__}: {e}",
        })
        return out

    image_byte_hashes: list[bytes] = []  # detect render-not-parameterised

    for seed in range(n_seeds):
        t0 = time.time()
        rec: dict[str, Any] = {
            "qid": qid,
            "variant_id": f"{qid}_v{seed}",
            "seed": seed,
            "title": title,
            "method_name": method_name,
            "signature": signature,
            "n_examples": n_examples_default,
        }

        # Sample
        try:
            examples = sample_examples(seed, n_examples_default)
            if not isinstance(examples, list) or len(examples) != n_examples_default:
                raise RuntimeError(
                    f"sample_examples returned {len(examples) if hasattr(examples, '__len__') else '?'} "
                    f"items, expected {n_examples_default}"
                )
        except Exception as e:
            rec["status"] = "sample_fail"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["tb"] = traceback.format_exc()[-800:]
            out.append(rec)
            continue

        # Compute expected outputs by calling Solution()
        try:
            sol_inst = Solution_cls()
            outputs = []
            for ex in examples:
                outputs.append(getattr(sol_inst, method_name)(**ex["kwargs"]))
        except Exception as e:
            rec["status"] = "solution_fail"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["tb"] = traceback.format_exc()[-800:]
            out.append(rec)
            continue

        # Render each example
        rendered_paths: list[str] = []
        try:
            for k, ex in enumerate(examples):
                img = render(ex)
                # Allow returning a Figure or numpy array, but require PIL.Image
                if hasattr(img, "savefig"):
                    raise RuntimeError("render returned matplotlib.Figure (must return PIL.Image)")
                if hasattr(img, "shape") and not hasattr(img, "save"):
                    raise RuntimeError("render returned numpy array (must return PIL.Image)")
                if not hasattr(img, "save"):
                    raise RuntimeError(f"render returned non-PIL: {type(img).__name__}")
                img_path = images_dir / f"{rec['variant_id']}_{k}.png"
                img.save(img_path, format="PNG")
                rendered_paths.append(str(img_path.relative_to(out_dir)))
                # collect byte hash for parameterisation check
                with open(img_path, "rb") as f:
                    image_byte_hashes.append(f.read()[:1024])  # head bytes is enough
        except Exception as e:
            rec["status"] = "render_fail"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["tb"] = traceback.format_exc()[-800:]
            out.append(rec)
            continue

        # Get hidden test cases
        try:
            tcs = make_test_cases(seed)
            tcs_norm: list[tuple[dict, Any]] = []
            for tc in tcs:
                if not (isinstance(tc, tuple) and len(tc) == 2 and isinstance(tc[0], dict)):
                    raise RuntimeError(f"test case malformed: {tc!r}")
                tcs_norm.append((tc[0], tc[1]))
        except Exception as e:
            rec["status"] = "tests_fail"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["tb"] = traceback.format_exc()[-800:]
            out.append(rec)
            continue

        # In-proc verification: for each tc, recompute via Solution() and compare
        in_proc_passed = True
        in_proc_msg = ""
        try:
            for kw, expected in tcs_norm:
                got = getattr(sol_inst, method_name)(**kw)
                if got != expected:
                    in_proc_passed = False
                    in_proc_msg = (
                        f"in_proc mismatch: kw={kw} expected={expected!r} got={got!r}"
                    )
                    break
        except Exception as e:
            in_proc_passed = False
            in_proc_msg = f"{type(e).__name__}: {e}"

        if not in_proc_passed:
            rec["status"] = "in_proc_fail"
            rec["error"] = in_proc_msg
            out.append(rec)
            continue

        # Sandbox execution
        program = solution_src + "\n\n" + build_test_script(method_name, tcs_norm)
        sandbox_result = run_program(program, timeout=timeout_per_seed)
        rec["sandbox_check"] = sandbox_result

        if not sandbox_result["passed"]:
            rec["status"] = "sandbox_fail"
            rec["error"] = sandbox_result["result"][:800]
            out.append(rec)
            continue

        # Fill description with computed values
        filled_description = fill_description(description, examples, outputs)

        # Persist
        rec.update({
            "status":             "ok",
            "description_filled": filled_description,
            "examples":           [{"kwargs": ex["kwargs"]} for ex in examples],
            "example_outputs":    [_compact_repr(o) for o in outputs],
            "n_test_cases":       len(tcs_norm),
            "test_cases":         [{"kwargs": kw, "expected": _compact_repr(exp)}
                                   for kw, exp in tcs_norm],
            "image_paths":        rendered_paths,
            "in_proc_passed":     True,
            "elapsed_sec":        round(time.time() - t0, 2),
        })
        out.append(rec)

    # Render-parameterisation check: if all images across all seeds are
    # bit-identical, mark all variants as render_static (=> drop later).
    if len(image_byte_hashes) >= 2 and len(set(image_byte_hashes)) == 1:
        for rec in out:
            if rec.get("status") == "ok":
                rec["status"] = "render_static"
                rec["error"] = "render produces identical bytes across all seeds"
    return out


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="seeds per template")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="sandbox timeout per seed")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N templates (0 = all)")
    args = ap.parse_args()

    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = BASE / "logs"
    logger = setup_logging(log_dir, name="run")
    logger.info(f"args: {vars(args)}")

    records = load_records()
    qids = sorted(records.keys())
    if args.limit:
        qids = qids[: args.limit]

    n_total = len(qids)
    logger.info(f"Processing {n_total} templates × {args.seeds} seeds")
    print(f"Processing {n_total} templates × {args.seeds} seeds")

    by_status: dict[str, int] = {}
    n_variants_ok = 0
    out_dir_str = str(VARIANTS_DIR)

    pbar = tqdm(total=n_total, desc="run-templates", unit="tpl")
    # Pass timeout via process pool
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_template, q, args.seeds, args.timeout, out_dir_str): q
            for q in qids
        }

        # Stream output line-by-line so we can resume on crash.
        records_f = RECORDS_PATH.open("w", encoding="utf-8")
        failures_f = FAILURES_PATH.open("w", encoding="utf-8")
        try:
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    variant_recs = fut.result()
                except Exception as e:
                    logger.warning(f"[{qid}] worker crashed: {e}")
                    failures_f.write(json.dumps({
                        "qid": qid, "status": "worker_crash",
                        "error": f"{type(e).__name__}: {e}",
                        "tb": traceback.format_exc()[-1500:],
                    }, ensure_ascii=False) + "\n")
                    pbar.update(1)
                    continue

                ok_in_tpl = 0
                for r in variant_recs:
                    s = r.get("status", "?")
                    by_status[s] = by_status.get(s, 0) + 1
                    if s == "ok":
                        ok_in_tpl += 1
                        records_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    else:
                        failures_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_variants_ok += ok_in_tpl
                pbar.update(1)
                pbar.set_postfix(
                    pass_rate=f"{n_variants_ok}/{(by_status.get('ok',0)+by_status.get('sandbox_fail',0)+by_status.get('solution_fail',0)+by_status.get('render_fail',0)+by_status.get('in_proc_fail',0)+by_status.get('tests_fail',0)+by_status.get('sample_fail',0)+by_status.get('render_static',0))}",
                    ok=by_status.get("ok", 0),
                )
        finally:
            records_f.close()
            failures_f.close()
    pbar.close()

    summary = {
        "ts":           datetime.now().isoformat(timespec="seconds"),
        "n_templates":  n_total,
        "n_seeds":      args.seeds,
        "n_attempted":  n_total * args.seeds,
        "by_status":    by_status,
        "n_ok":         by_status.get("ok", 0),
        "pass_rate":    round(by_status.get("ok", 0) / max(1, n_total * args.seeds), 4),
    }
    (VARIANTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRun complete: {summary}")


if __name__ == "__main__":
    main()
