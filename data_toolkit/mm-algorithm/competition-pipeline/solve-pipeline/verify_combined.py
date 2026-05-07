"""
Step 4b — verify the combined set of (original solutions ∪ retry solutions).

For each sid:
  • prefer retry_solutions/<sid>.json if it exists
  • else solutions/<sid>.json

Same sandbox+timeout logic as verify_sandbox.py. Writes to verified/ and
failed/ (overwriting the previous v2 state).

Usage:
  conda run -n vpb python verify_combined.py --workers 8 --timeout 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Reuse run_one + helpers from verify_sandbox
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_sandbox import run_one  # noqa: E402

BASE = Path(__file__).resolve().parent
SEEDS_PATH = BASE / "seeds" / "records.jsonl"
SOL_DIR = BASE / "solutions"
RETRY_DIR = BASE / "retry_solutions"
VER_DIR = BASE / "verified"
FAIL_DIR = BASE / "failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    VER_DIR.mkdir(exist_ok=True)
    FAIL_DIR.mkdir(exist_ok=True)
    # Clean prior verify state — we want a fresh aggregate
    for sub in (VER_DIR, FAIL_DIR):
        for p in sub.glob("*.json"):
            p.unlink()

    seeds: dict[str, dict] = {}
    with SEEDS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seeds[r["sid"]] = r

    # Collect all sids that have at least one solution file
    all_sids = sorted(set(p.stem for p in SOL_DIR.glob("*.json")) |
                      set(p.stem for p in RETRY_DIR.glob("*.json")))
    if args.limit:
        all_sids = all_sids[: args.limit]

    print(f"Verifying {len(all_sids)} unique sids (retry > original)  workers={args.workers}")

    by_status: Counter = Counter()
    n_passed_total = 0
    n_total_tests = 0
    src_used: Counter = Counter()

    pbar = tqdm(total=len(all_sids), desc="verify", unit="sol")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for sid in all_sids:
            retry_path = RETRY_DIR / f"{sid}.json"
            orig_path = SOL_DIR / f"{sid}.json"
            if retry_path.exists():
                src = "retry"
                sol = json.loads(retry_path.read_text(encoding="utf-8"))
            else:
                src = "orig"
                sol = json.loads(orig_path.read_text(encoding="utf-8"))
            src_used[src] += 1
            if not sol.get("syntax_ok") or not sol.get("code"):
                by_status["pre_skip_invalid"] += 1
                pbar.update(1)
                continue
            seed = seeds.get(sid)
            if not seed:
                by_status["pre_skip_no_seed"] += 1
                pbar.update(1)
                continue
            tests = (seed.get("parseable_tests") or []) + (sol.get("extra_tests") or [])
            if not tests:
                by_status["pre_skip_no_tests"] += 1
                pbar.update(1)
                continue
            futs[pool.submit(run_one, sid, sol["code"], tests, args.timeout)] = (sid, sol, seed, tests, src)

        for fut in as_completed(futs):
            sid, sol, seed, tests, src = futs[fut]
            try:
                res = fut.result()
            except Exception:
                by_status["worker_crash"] += 1
                pbar.update(1)
                continue
            by_status[res["status"]] += 1
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
                "solution_source": src,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            target_dir = VER_DIR if res["status"] == "ok" else FAIL_DIR
            (target_dir / f"{sid}.json").write_text(
                json.dumps(out, ensure_ascii=False), encoding="utf-8"
            )
            pbar.update(1)
            pbar.set_postfix(ok=by_status.get("ok", 0), fail=by_status.get("fail", 0),
                             rate=f"{n_passed_total}/{n_total_tests}")
    pbar.close()

    summary = {
        "ts":             datetime.now().isoformat(timespec="seconds"),
        "n_solutions":    len(all_sids),
        "by_status":      dict(by_status),
        "src_used":       dict(src_used),
        "n_test_total":   n_total_tests,
        "n_test_passed":  n_passed_total,
        "test_pass_rate": round(n_passed_total / max(1, n_total_tests), 4),
        "solution_pass_rate": round(by_status.get("ok", 0) / max(1, len(all_sids)), 4),
    }
    (BASE / "verify_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
