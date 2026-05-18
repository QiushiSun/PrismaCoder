"""Subprocess-based python executor (alternative to nbconvert / ipykernel).

Why this exists
---------------
The default python executor wraps every shard of samples into a Jupyter
notebook and runs it through ``nbconvert``'s ``ExecutePreprocessor``. That has
three painful failure modes that we have observed in production:

1. **Cell-level timeout is not always honoured.** ``interrupt_on_timeout`` sends
   a KeyboardInterrupt to the kernel, but if the cell is stuck inside a C
   extension (PIL, scipy LU, ``fig.show()`` server loops, etc.) the interrupt
   is ignored and the entire shard hits the global 600 s timeout. **Eight
   completely unrelated samples in the same shard are then marked as
   failures**, which is what we kept seeing on run-1 and run-25
   (8 ``Shard timed out (600s)`` per python lane).

2. **Orphaned ``ipykernel`` subprocesses.** When ``ProcessPoolExecutor`` SIGKILLs
   a hung worker, the kernel process it spawned via ``subprocess.Popen`` is
   left running and keeps consuming CPU between runs.

3. **Hard-to-reason-about isolation.** A subtle ``sys.path`` / module-import
   bug in one cell can poison every subsequent cell in the same kernel.

The subprocess executor in this file fixes (1)+(2)+(3) at once:

* Each sample runs as its **own** ``python script.py`` invocation, so a
  single hang only kills that one sample (per-sample SIGKILL on timeout).
* The subprocess is started with ``start_new_session=True`` and we send
  SIGKILL to the whole process group on timeout, so any child threads /
  C-extension subprocesses go down with it - no orphans.
* The user code is preceded by a ``PATCH_HEADER`` that monkey-patches
  ``plt.show()`` / ``Figure.show()`` / ``bokeh.plotting.show()`` to write
  PNG/HTML files into ``$FIG_DIR``, plus taps ``IPython.display.display``
  so that any in-tree ``convert_*`` patches that emit
  ``display(Image(filename=...))`` continue to work transparently.

To stay 100% compatible with the rest of the pipeline (``parse_plots_notebook``
and the downstream judge), this executor synthesises an ``.ipynb`` whose
cell outputs look exactly like a real notebook run.  The file is the
"contract" with downstream code; nothing else needs to change.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nbformat as nbf
import pandas as pd

from .base_utils import (
    deduplicate_lines,
    remove_ansi_escape,
    remove_bytes_literal,
    shrink_png_b64_to_under_kb,
)


# ---------------------------------------------------------------------------
# Monkey-patch header injected before every user script.
# ---------------------------------------------------------------------------
# Design notes:
# * ``matplotlib.use("Agg", force=True)`` runs *before* any user import to
#   guarantee no GUI backend ever gets selected, which is the #1 source of
#   "code hangs forever waiting for a display server" timeouts.
# * Every save target lives under ``$FIG_DIR``. We bump a counter so multiple
#   ``plt.show()`` calls in the same script produce ``figure_001.png``,
#   ``figure_002.png``, ...
# * The IPython.display tap is what makes the existing ``convert_*`` jupyter-
#   centric helpers (which emit ``display(Image(filename=tmp_png))``)
#   continue to work without modification: when ``display(Image(filename=X))``
#   is called, we copy ``X`` into ``$FIG_DIR`` so the post-run scan picks it
#   up.
# * Every patch is wrapped in ``try/except`` so optional libraries (kaleido,
#   plotly, bokeh, ...) being missing never breaks the script — the user
#   code just falls through with whatever lib it actually uses.
PATCH_HEADER = r'''# === VisPlotBench subprocess executor patch (auto-injected) ===
import os as _vpb_os
import shutil as _vpb_shutil
import sys as _vpb_sys

_VPB_FIG_DIR = _vpb_os.environ.get("FIG_DIR", _vpb_os.getcwd())
_vpb_os.makedirs(_VPB_FIG_DIR, exist_ok=True)
_VPB_FIG_COUNT = 0

def _vpb_next_path(ext: str) -> str:
    global _VPB_FIG_COUNT
    _VPB_FIG_COUNT += 1
    return _vpb_os.path.join(_VPB_FIG_DIR, "figure_%03d.%s" % (_VPB_FIG_COUNT, ext))

# --- matplotlib (covers seaborn / missingno / pandas.plot which all delegate to it)
try:
    import matplotlib as _vpb_mpl
    _vpb_mpl.use("Agg", force=True)
    import matplotlib.pyplot as _vpb_plt

    def _vpb_plt_show(*args, **kwargs):
        # Use matplotlib defaults (rcParams figure.dpi, no bbox_inches="tight")
        # so the rendered PNG matches what nbconvert/inline-display would
        # produce.  Prevents a systematic ~1-2 pt drop in judge Vis scores
        # caused by subtly different framing/dpi between modes.
        try:
            for _num in _vpb_plt.get_fignums():
                _f = _vpb_plt.figure(_num)
                _vpb_p = _vpb_next_path("png")
                _f.savefig(_vpb_p)
            _vpb_plt.close("all")
        except Exception as _e:
            _vpb_sys.stderr.write("[vpb-patch] plt.show save failed: %r\n" % (_e,))

    _vpb_plt.show = _vpb_plt_show
except Exception:
    pass

# --- plotly: try kaleido for PNG, fall back to HTML.
try:
    import plotly.graph_objects as _vpb_go
    import plotly.io as _vpb_pio

    def _vpb_plotly_show(self, *a, **k):
        try:
            self.write_image(_vpb_next_path("png"), engine="kaleido")
        except Exception:
            try:
                self.write_html(_vpb_next_path("html"))
            except Exception as _e:
                _vpb_sys.stderr.write("[vpb-patch] plotly save failed: %r\n" % (_e,))
    _vpb_go.Figure.show = _vpb_plotly_show

    def _vpb_pio_show(fig, *a, **k):
        try:
            fig.write_image(_vpb_next_path("png"), engine="kaleido")
        except Exception:
            try:
                fig.write_html(_vpb_next_path("html"))
            except Exception:
                pass
    _vpb_pio.show = _vpb_pio_show
except Exception:
    pass

# --- bokeh: redirect show() to png export, fall back to html save.
try:
    import bokeh.plotting as _vpb_bp
    import bokeh.io as _vpb_bio

    def _vpb_bokeh_show(obj, *a, **k):
        try:
            _vpb_bio.export_png(obj, filename=_vpb_next_path("png"))
        except Exception:
            try:
                _vpb_bp.save(obj, filename=_vpb_next_path("html"))
            except Exception as _e:
                _vpb_sys.stderr.write("[vpb-patch] bokeh save failed: %r\n" % (_e,))

    _vpb_bp.show = _vpb_bokeh_show
    _vpb_bio.show = _vpb_bokeh_show
    _vpb_bio.output_file = lambda *a, **k: None  # no-op, we save explicitly
    _vpb_bio.output_notebook = lambda *a, **k: None
except Exception:
    pass

# --- altair: chart.show() → save("...png") if vl-convert is available.
try:
    import altair as _vpb_alt
    def _vpb_alt_show(self, *a, **k):
        try:
            self.save(_vpb_next_path("png"))
        except Exception:
            try:
                self.save(_vpb_next_path("html"))
            except Exception as _e:
                _vpb_sys.stderr.write("[vpb-patch] altair save failed: %r\n" % (_e,))
    _vpb_alt.Chart.show = _vpb_alt_show
    if hasattr(_vpb_alt, "TopLevelMixin"):
        _vpb_alt.TopLevelMixin.show = _vpb_alt_show
except Exception:
    pass

# --- IPython.display tap: many in-tree ``convert_*`` helpers emit
# ``display(Image(filename=tmp_png))``. We capture those calls so the PNG
# the helper already saved gets copied into FIG_DIR for post-run scan.
try:
    from IPython import display as _vpb_ipydisp
    _vpb_orig_display = _vpb_ipydisp.display

    def _vpb_display_tap(*objs, **kwargs):
        for _o in objs:
            try:
                _fn = getattr(_o, "filename", None)
                if _fn and _vpb_os.path.exists(_fn):
                    _ext = _vpb_os.path.splitext(_fn)[1].lstrip(".") or "png"
                    _dst = _vpb_next_path(_ext)
                    if _vpb_os.path.abspath(_fn) != _vpb_os.path.abspath(_dst):
                        try:
                            _vpb_shutil.copyfile(_fn, _dst)
                        except Exception:
                            pass
                # Image(data=<bytes>) path — write raw bytes.
                _data = getattr(_o, "data", None)
                if _data and isinstance(_data, (bytes, bytearray)):
                    _dst = _vpb_next_path("png")
                    try:
                        with open(_dst, "wb") as _fh:
                            _fh.write(_data)
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            return _vpb_orig_display(*objs, **kwargs)
        except Exception:
            return None
    _vpb_ipydisp.display = _vpb_display_tap
except Exception:
    pass

# === end VisPlotBench patch ===
'''


# ---------------------------------------------------------------------------
# One-sample execution
# ---------------------------------------------------------------------------
def _run_single_sample(
    sample_id: str,
    code: str,
    fig_dir: Path,
    timeout: float,
    python_exe: str,
) -> dict[str, Any]:
    """Run ``code`` in a fresh python subprocess.

    Returns dict with keys: id, error, plots_generated (list[str b64 png]),
    has_plot (bool), stdout (str), stderr (str), exit_code (int|None).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    script_path = fig_dir / "_script.py"
    full_src = PATCH_HEADER + "\n# --- user code ---\n" + code + "\n"
    script_path.write_text(full_src, encoding="utf-8")

    env = dict(os.environ)
    env["FIG_DIR"] = str(fig_dir)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    err_msg = ""
    stdout = ""
    stderr = ""
    exit_code: int | None = None

    try:
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            cwd=str(fig_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out_b, err_b = proc.communicate(timeout=timeout)
            stdout = (out_b or b"").decode("utf-8", "replace")
            stderr = (err_b or b"").decode("utf-8", "replace")
            exit_code = proc.returncode
            if exit_code != 0:
                # Surface tail of stderr as the error.
                tail = stderr.strip().splitlines()[-30:] if stderr else []
                err_msg = "\n".join(tail) if tail else f"non-zero exit ({exit_code})"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                out_b, err_b = proc.communicate(timeout=5)
                stdout = (out_b or b"").decode("utf-8", "replace")
                stderr = (err_b or b"").decode("utf-8", "replace")
            except Exception:
                pass
            exit_code = None
            err_msg = f"Subprocess timed out ({timeout}s)"
    except Exception:
        err_msg = f"Subprocess launch failed:\n{traceback.format_exc()}"

    plots = _collect_pngs(fig_dir)

    return {
        "id": sample_id,
        "error": _clean(err_msg),
        "plots_generated": plots,
        "has_plot": len(plots) > 0,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
    }


def _collect_pngs(fig_dir: Path) -> list[str]:
    """Scan ``fig_dir`` for PNGs and return shrunk base64 strings.

    Sorted so output ordering is deterministic across runs.
    """
    if not fig_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(fig_dir.glob("*.png")):
        try:
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            b64 = shrink_png_b64_to_under_kb(b64, max_kb=200, step=0.9, min_side=64)
            out.append(b64)
        except Exception:
            continue
    return out


def _clean(err: str) -> str:
    if not err:
        return ""
    err = remove_ansi_escape(err)
    err = remove_bytes_literal(err)
    return err


# ---------------------------------------------------------------------------
# Shard runner — synthesises a notebook so parse_plots_notebook is unchanged.
# ---------------------------------------------------------------------------
def run_shard_subprocess(
    cells: list[tuple[str, str]],
    shard_dir: Path,
    output_path: Path,
    timeout: float = 30.0,
    parallel: int = 1,
) -> None:
    """Run every sample in ``cells`` as its own subprocess and write a
    synthetic notebook compatible with ``PythonHandler.parse_plots_notebook``.

    Args:
        cells: list of ``(sample_id, full_code_with_id_comment)``.  The
            ``full_code_with_id_comment`` should already start with the
            ``# id = <sample_id>\n`` line that ``parse_plots_notebook`` keys on.
        shard_dir: directory where per-sample fig dirs and the synthetic
            notebook live.
        output_path: where to write the synthetic ``.ipynb``.
        timeout: per-sample wall-clock timeout, in seconds.
        parallel: how many subprocesses to run concurrently within this shard.
            **Default 1** (sequential within a shard).  We rely on the outer
            ``ProcessPoolExecutor`` (in execution.py) for parallelism instead;
            empirically, nesting subprocess.Popen inside a ThreadPool inside a
            ProcessPool worker causes Popen to hang (likely fork/spawn lock
            contention on macOS).  Set >1 only if you know your workload
            doesn't trigger that, or you're running in a single-process
            launcher.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable

    results: dict[str, dict[str, Any]] = {}

    def _do(idx_id_code: tuple[int, str, str]) -> tuple[int, dict[str, Any]]:
        idx, sid, code = idx_id_code
        per_dir = shard_dir / f"sample_{idx:03d}_{_safe_id(sid)}"
        res = _run_single_sample(sid, code, per_dir, timeout=timeout, python_exe=python_exe)
        return idx, res

    indexed = [(i, sid, code) for i, (sid, code) in enumerate(cells)]
    if parallel <= 1:
        # Sequential path — most reliable when nested under ProcessPoolExecutor.
        for x in indexed:
            idx, res = _do(x)
            results[res["id"]] = res
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for fut in as_completed([pool.submit(_do, x) for x in indexed]):
                idx, res = fut.result()
                results[res["id"]] = res

    # Build a synthetic notebook with one cell per sample.  Cell outputs
    # mimic what nbconvert emits so parse_plots_notebook works unchanged.
    nb = nbf.v4.new_notebook()
    nb_cells = []
    for sid, code in cells:
        cell = nbf.v4.new_code_cell(code)
        outputs = []
        res = results.get(sid)
        if res is None:
            outputs.append(nbf.v4.new_output(
                "error",
                ename="VPBSubprocessError",
                evalue="No result returned",
                traceback=["No result returned for sample"],
            ))
        else:
            for b64 in res["plots_generated"]:
                outputs.append(nbf.v4.new_output(
                    "display_data",
                    data={"image/png": b64},
                    metadata={},
                ))
            if res["error"]:
                outputs.append(nbf.v4.new_output(
                    "error",
                    ename="VPBSubprocessError",
                    evalue=res["error"][:200],
                    traceback=res["error"].splitlines()[-50:] or [res["error"][:200]],
                ))
        cell["outputs"] = outputs
        nb_cells.append(cell)
    nb["cells"] = nb_cells
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def _safe_id(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", sid)[:60]
