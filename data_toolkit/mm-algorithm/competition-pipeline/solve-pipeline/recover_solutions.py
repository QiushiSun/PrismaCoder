"""
Re-extract code + extra_tests from already-saved solutions/<sid>.json files.

Why: the original solve_with_claude.py shipped with a leaky python-fence regex
that returned empty matches when a yaml block immediately followed the python
block.  Fixed in solve_with_claude.py; this script reuses that fix to repair
the cached responses without re-spending API tokens.

Usage:
  conda run -n vpb python recover_solutions.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

# Reuse the fixed extractors from solve_with_claude
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_with_claude import extract_solution, extract_extra_tests  # noqa: E402

BASE = Path(__file__).resolve().parent
SOL_DIR = BASE / "solutions"


def main() -> None:
    fixed: list[dict] = []
    by_status: Counter = Counter()

    for p in sorted(SOL_DIR.glob("*.json")):
        sol = json.loads(p.read_text(encoding="utf-8"))
        prev_code_len = len(sol.get("code") or "")
        prev_extra = len(sol.get("extra_tests") or [])
        prev_syntax_ok = bool(sol.get("syntax_ok"))

        raw = sol.get("raw_text") or ""
        new_code = extract_solution(raw)
        new_extra = extract_extra_tests(raw) if sol.get("augment_tests") else []

        new_syntax_ok = True
        new_syntax_msg = ""
        if new_code:
            try:
                compile(new_code, f"<{sol['sid']}>", "exec")
            except SyntaxError as e:
                new_syntax_ok = False
                new_syntax_msg = f"SyntaxError: {e.msg} (line {e.lineno})"

        # Only rewrite if something actually changed
        changed = (
            len(new_code) != prev_code_len
            or len(new_extra) != prev_extra
            or new_syntax_ok != prev_syntax_ok
        )
        if changed:
            sol["code"] = new_code
            sol["extra_tests"] = new_extra
            sol["syntax_ok"] = new_syntax_ok
            sol["syntax_msg"] = new_syntax_msg
            sol["recovered_ts"] = datetime.now().isoformat(timespec="seconds")
            p.write_text(json.dumps(sol, ensure_ascii=False), encoding="utf-8")
            fixed.append({
                "sid": sol["sid"],
                "prev_code_len": prev_code_len,
                "new_code_len": len(new_code),
                "prev_extra": prev_extra,
                "new_extra": len(new_extra),
                "prev_syntax_ok": prev_syntax_ok,
                "new_syntax_ok": new_syntax_ok,
            })

        # Tally final status
        if not new_code.strip():
            by_status["empty_code"] += 1
        elif not new_syntax_ok:
            by_status["syntax_fail"] += 1
        elif sol.get("augment_tests") and not new_extra:
            by_status["missing_extra"] += 1
        else:
            by_status["ok"] += 1

    print(f"Examined: {sum(by_status.values())} solutions")
    print(f"Recovered (changed): {len(fixed)}")
    print()
    print("Final status breakdown:")
    for k, v in by_status.most_common():
        print(f"  {k:20s} {v}")

    # Print recover examples
    if fixed:
        print()
        print("Sample recoveries:")
        for f in fixed[:5]:
            print(f"  {f['sid']}: code {f['prev_code_len']}→{f['new_code_len']}  "
                  f"extra {f['prev_extra']}→{f['new_extra']}  "
                  f"syntax {f['prev_syntax_ok']}→{f['new_syntax_ok']}")


if __name__ == "__main__":
    main()
