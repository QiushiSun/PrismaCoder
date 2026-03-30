"""
TypeScript utility functions — extraction, compilation, and rendering.

Requires: Node.js 16+, esbuild, puppeteer, d3, chart.js (npm packages).
The companion ts_render.js script must be in the same directory as this file.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# ── Code extraction ───────────────────────────────────────────────────────

CODE_FENCE_RE = re.compile(r"```(?:typescript|ts|tsx|javascript|js)\s*\n?(.*?)```", re.DOTALL)
CODE_FENCE_ANY_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)

TS_INDICATORS = [
    "const ", "let ", "function ", "interface ", "type ",
    "import ", "export ", "=>", ": number", ": string",
    "document.", "createElement", "querySelector",
    "d3.", "Chart(", "svg", "canvas",
]


def _looks_like_ts(text: str) -> bool:
    return sum(1 for ind in TS_INDICATORS if ind in text) >= 3


def extract_ts_code(text: str) -> tuple[str | None, str]:
    """Extract TypeScript code from text.
    1. Look inside TS/JS-tagged markdown code fences.
    2. Try any code fence.
    3. Check raw text.
    Returns (ts_code | None, extraction_method).
    """
    for fence_match in CODE_FENCE_RE.finditer(text):
        inner = fence_match.group(1).strip()
        if inner and _looks_like_ts(inner):
            return inner, "code_fence"

    for fence_match in CODE_FENCE_ANY_RE.finditer(text):
        inner = fence_match.group(1).strip()
        if inner and _looks_like_ts(inner):
            return inner, "code_fence_generic"

    stripped = text.strip()
    if _looks_like_ts(stripped):
        return stripped, "raw"

    return None, "not_found"


# ── Code sanitization ────────────────────────────────────────────────────

def sanitize_ts_code(code: str) -> str:
    """Strip imports/exports so the code can run in a browser shim."""
    code = re.sub(r"import\s+.*?from\s+['\"](?!\.)[^'\"]+['\"];?\s*\n?", "", code)
    code = re.sub(r"export\s+default\s+", "", code)
    code = re.sub(r"export\s+\{[^}]*\};?\s*\n?", "", code)
    code = re.sub(r"export\s+(const|let|var|function|class|interface|type|enum)\s", r"\1 ", code)
    return code


# ── Binary discovery ──────────────────────────────────────────────────────

NODE_PATH: str | None = None
ESBUILD_PATH: str | None = None

_NVM_BIN = os.path.expanduser("~/.nvm/versions/node/v20.20.2/bin")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_SCRIPT = os.path.join(_THIS_DIR, "ts_render.js")


def find_node() -> str:
    global NODE_PATH
    if NODE_PATH:
        return NODE_PATH
    nvm_node = os.path.join(_NVM_BIN, "node")
    if os.path.isfile(nvm_node):
        NODE_PATH = nvm_node
        return NODE_PATH
    path = shutil.which("node")
    if not path:
        raise FileNotFoundError("node (v16+) not found on PATH")
    NODE_PATH = path
    return path


def _find_esbuild() -> str:
    global ESBUILD_PATH
    if ESBUILD_PATH:
        return ESBUILD_PATH
    nvm_esbuild = os.path.join(_NVM_BIN, "esbuild")
    if os.path.isfile(nvm_esbuild):
        ESBUILD_PATH = nvm_esbuild
        return ESBUILD_PATH
    path = shutil.which("esbuild")
    if not path:
        raise FileNotFoundError("esbuild not found on PATH")
    ESBUILD_PATH = path
    return path


def _find_node_modules() -> str:
    """Search upward for a node_modules directory."""
    for parent in [_THIS_DIR, os.path.dirname(_THIS_DIR),
                   os.path.dirname(os.path.dirname(_THIS_DIR))]:
        nm = os.path.join(parent, "node_modules")
        if os.path.isdir(nm):
            return nm
    return os.path.join(_THIS_DIR, "node_modules")


# ── Compilation ───────────────────────────────────────────────────────────

def compile_ts_to_js(ts_code: str, tmp_dir: str) -> tuple[bool, str, str]:
    """Compile TypeScript to JavaScript via esbuild.
    Returns (ok, js_code_or_error, method).
    """
    sanitized = sanitize_ts_code(ts_code)

    ts_path = os.path.join(tmp_dir, "input.ts")
    js_path = os.path.join(tmp_dir, "input.js")
    with open(ts_path, "w", encoding="utf-8") as f:
        f.write(sanitized)

    try:
        esbuild = _find_esbuild()
        result = subprocess.run(
            [esbuild, ts_path, "--bundle=false", "--outfile=" + js_path,
             "--target=es2020", "--format=esm"],
            capture_output=True, text=True, timeout=10,
        )
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as f:
                js_code = f.read()
            if js_code.strip():
                return True, js_code, "esbuild"
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    return True, sanitized, "raw_sanitized"


# ── Rendering ─────────────────────────────────────────────────────────────

def render_js_to_png(
    js_code: str, png_path: str, timeout: int = 30
) -> tuple[bool, str]:
    """Render JS visualization to PNG via Puppeteer headless browser."""
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="ts_")
        js_file = os.path.join(tmp_dir, "code.js")
        with open(js_file, "w", encoding="utf-8") as f:
            f.write(js_code)

        node = find_node()
        env = os.environ.copy()
        env["NODE_PATH"] = _find_node_modules()

        result = subprocess.run(
            [node, RENDER_SCRIPT, js_file, png_path, str(timeout * 1000)],
            capture_output=True, text=True,
            timeout=timeout + 10,
            cwd=_THIS_DIR,
            env=env,
        )

        if result.returncode == 0 and os.path.exists(png_path):
            sz = os.path.getsize(png_path)
            if sz > 500:
                return True, "OK"
            return False, f"PNG too small ({sz} bytes), likely blank"

        stderr = result.stderr.strip()
        if "RENDER_ERROR:" in stderr:
            msg = stderr.split("RENDER_ERROR:")[1][:300]
            return False, f"Render error: {msg}"
        if "PUPPETEER_ERROR:" in stderr:
            msg = stderr.split("PUPPETEER_ERROR:")[1][:300]
            return False, f"Puppeteer error: {msg}"

        msg = stderr[-400:] if len(stderr) > 400 else stderr
        return False, f"Render failed (exit {result.returncode}): {msg}"

    except subprocess.TimeoutExpired:
        return False, f"Render timed out ({timeout}s)"
    except Exception as e:
        return False, f"Render error: {e}"
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
