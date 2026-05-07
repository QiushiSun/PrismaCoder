"""
Step 4 — verify Claude's STDIN/STDOUT solutions in a sandboxed subprocess.

For each solutions/<sid>.json that has syntax_ok and code:
  1. Combine parseable_tests (from seeds) + extra_tests (from solution).
  2. For each test case, write a tiny driver that:
       echo INPUT | python solution.py
     compares the stdout (after rstrip + token-normalisation) to expected.
  3. Pass means ALL test cases match (100%).
  4. Pass → write verified/<sid>.json
     Fail → write failed/<sid>.json (with which test failed and stderr).

Output:
  verified/<sid>.json   — sid, code, all tests, n_passed, n_total
  failed/<sid>.json     — same + failure_details
  verify_summary.json
  variants/run_summary.json (legacy compat is not needed)

Usage:
  conda run -n vpb python verify_sandbox.py --workers 8 --timeout 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

BASE = Path(__file__).resolve().parent
SEEDS_PATH = BASE / "seeds" / "records.jsonl"
SOL_DIR = BASE / "solutions"
VER_DIR = BASE / "verified"
FAIL_DIR = BASE / "failed"


# ────────────────────────── normalise output ──────────────────────────
_WS_RE = re.compile(r"\s+")


def normalise_stdout(s: str) -> str:
    """Collapse whitespace differences & strip trailing newlines for token compare."""
    return _WS_RE.sub(" ", (s or "").strip())


def equal_outputs(got: str, expected: str) -> bool:
    if got is None or expected is None:
        return False
    a = normalise_stdout(got)
    b = normalise_stdout(expected)
    if a == b:
        return True
    # Numeric tolerance — float problems often allow 1e-2 abs error
    a_toks = a.split()
    b_toks = b.split()
    if len(a_toks) != len(b_toks):
        return False
    for at, bt in zip(a_toks, b_toks):
        if at == bt:
            continue
        try:
            af = float(at)
            bf = float(bt)
            if abs(af - bf) <= 1e-2:
                continue
            return False
        except ValueError:
            return False
    return True


# ────────────────────────── per-problem worker ──────────────────────────
def run_one(sid: str, code: str, tests: list[dict], timeout: float) -> dict:
    """Run code against every test case. Returns dict with results."""
    if not code.strip() or not tests:
        return {
            "sid": sid, "status": "skip",
            "reason": "empty_code" if not code.strip() else "no_tests",
            "n_passed": 0, "n_total": len(tests),
        }

    # Write code to temp file once
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    results: list[dict] = []
    n_passed = 0
    try:
        for i, t in enumerate(tests):
            inp = t.get("input") or ""
            expected = t.get("output") or ""
            if not inp.endswith("\n"):
                inp = inp + "\n"
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "MPLBACKEND": "Agg",
                         "PYTHONDONTWRITEBYTECODE": "1"},
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                rc = proc.returncode
                duration = round(time.time() - t0, 3)
                if rc != 0:
                    results.append({
                        "i": i, "passed": False, "reason": "non_zero_rc",
                        "rc": rc, "stderr": stderr[:600], "duration_sec": duration,
                    })
                    continue
                if not equal_outputs(stdout, expected):
                    results.append({
                        "i": i, "passed": False, "reason": "output_mismatch",
                        "expected": expected[:200], "got": stdout[:200],
                        "duration_sec": duration,
                    })
                    continue
                results.append({"i": i, "passed": True, "duration_sec": duration})
                n_passed += 1
            except subprocess.TimeoutExpired:
                results.append({
                    "i": i, "passed": False, "reason": "timeout",
                    "duration_sec": timeout,
                })
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    n_total = len(tests)
    all_passed = (n_passed == n_total)
    return {
        "sid": sid,
        "status": "ok" if all_passed else "fail",
        "n_passed": n_passed,
        "n_total": n_total,
        "results": results,
    }


def setup_logging() -> logging.Logger:
    log_dir = BASE / "logs"
    log_dir.mkdir(exist_ok=True)
    lg = logging.getLogger("comp_verify")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"verify_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=8.0,
                    help="per-test-case sandbox timeout in seconds")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    VER_DIR.mkdir(exist_ok=True)
    FAIL_DIR.mkdir(exist_ok=True)
    logger = setup_logging()

    seeds: dict[str, dict] = {}
    with SEEDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seeds[r["sid"]] = r

    sol_files = sorted(SOL_DIR.glob("*.json"))
    if args.limit:
        sol_files = sol_files[: args.limit]
    print(f"Verifying {len(sol_files)} solutions  workers={args.workers}  timeout={args.timeout}s")

    by_status: dict[str, int] = {}
    n_passed_total = 0
    n_total_tests = 0

    pbar = tqdm(total=len(sol_files), desc="verify", unit="sol")

    # Build futures
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for p in sol_files:
            sid = p.stem
            try:
                sol = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                by_status["load_fail"] = by_status.get("load_fail", 0) + 1
                pbar.update(1)
                continue
            if not sol.get("syntax_ok") or not sol.get("code"):
                by_status["pre_skip_invalid"] = by_status.get("pre_skip_invalid", 0) + 1
                pbar.update(1)
                continue

            seed = seeds.get(sid)
            if not seed:
                by_status["pre_skip_no_seed"] = by_status.get("pre_skip_no_seed", 0) + 1
                pbar.update(1)
                continue
            tests = (seed.get("parseable_tests") or []) + (sol.get("extra_tests") or [])
            if not tests:
                by_status["pre_skip_no_tests"] = by_status.get("pre_skip_no_tests", 0) + 1
                pbar.update(1)
                continue

            futs[pool.submit(run_one, sid, sol["code"], tests, args.timeout)] = (sid, sol, seed, tests)

        for fut in as_completed(futs):
            sid, sol, seed, tests = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                by_status["worker_crash"] = by_status.get("worker_crash", 0) + 1
                pbar.update(1)
                continue

            by_status[res["status"]] = by_status.get(res["status"], 0) + 1
            n_passed_total += res["n_passed"]
            n_total_tests += res["n_total"]

            out = {
                "sid": sid,
                "source": seed["source"],
                "problem_id": seed["problem_id"],
                "title": seed.get("title"),
                "url": seed.get("url"),
                "lang": seed.get("lang"),
                "n_images": seed.get("n_images"),
                "image_paths": seed.get("image_paths"),
                "code": sol["code"],
                "n_existing_tests": seed.get("n_parseable_tests", 0),
                "n_extra_tests": len(sol.get("extra_tests") or []),
                "all_tests": tests,
                "test_results": res["results"],
                "n_passed": res["n_passed"],
                "n_total": res["n_total"],
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            if res["status"] == "ok":
                (VER_DIR / f"{sid}.json").write_text(
                    json.dumps(out, ensure_ascii=False), encoding="utf-8"
                )
            else:
                (FAIL_DIR / f"{sid}.json").write_text(
                    json.dumps(out, ensure_ascii=False), encoding="utf-8"
                )
            pbar.update(1)
            pbar.set_postfix(
                ok=by_status.get("ok", 0),
                fail=by_status.get("fail", 0),
                pass_rate=f"{n_passed_total}/{n_total_tests}",
            )
    pbar.close()

    summary = {
        "ts":             datetime.now().isoformat(timespec="seconds"),
        "n_solutions":    len(sol_files),
        "by_status":      by_status,
        "n_test_total":   n_total_tests,
        "n_test_passed":  n_passed_total,
        "test_pass_rate": round(n_passed_total / max(1, n_total_tests), 4),
        "solution_pass_rate": round(by_status.get("ok", 0) / max(1, len(sol_files)), 4),
    }
    (BASE / "verify_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
