"""
Cross-platform Python sandbox using subprocess.

The HumanEval-V repo's `execution.py` uses a closure inside
`multiprocessing.Process(target=…)` which can't be pickled under macOS's
default `spawn` start-method. Here we use `subprocess.run` instead — slightly
slower but rock-solid, identical semantics.

Convention:
    program = ground_truth_solution + "\\n\\n" + test_script
    test_script ends with `test()` so running the program is the test.

Returns: dict(passed: bool, result: str, duration_sec: float)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_program(program_src: str, timeout: float = 10.0) -> dict:
    t0 = time.time()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(program_src)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "result": f"TimeoutExpired after {timeout}s",
            "duration_sec": round(time.time() - t0, 3),
        }
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    duration = round(time.time() - t0, 3)
    if proc.returncode == 0:
        return {"passed": True, "result": "passed", "duration_sec": duration}

    err = (proc.stderr or "").strip()
    out = (proc.stdout or "").strip()
    msg = err if err else out
    if not msg:
        msg = f"non-zero exit code {proc.returncode}"
    # Truncate very long stack traces
    if len(msg) > 1500:
        msg = msg[:1500] + " …(truncated)"
    return {"passed": False, "result": msg, "duration_sec": duration}


if __name__ == "__main__":
    src = "print('hello')\nassert 1 + 1 == 2\n"
    print(run_program(src))
    src = "x = 1/0\n"
    print(run_program(src))
    src = "while True: pass\n"
    print(run_program(src, timeout=2.0))
