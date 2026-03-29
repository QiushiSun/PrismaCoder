"""
Asymptote utility functions — extraction, sanitization, and rendering.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# ── Code extraction ───────────────────────────────────────────────────────

CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)

ASY_INDICATORS = [
    "import ", "draw(", "size(", "label(", "filldraw(",
    "fill(", "path ", "pair ", "real ", "pen ",
    "currentprojection", "Arrow", "Circle",
]


def _looks_like_asy(text: str) -> bool:
    return sum(1 for ind in ASY_INDICATORS if ind in text) >= 2


def extract_asy_code(text: str) -> tuple[str | None, str]:
    """Extract Asymptote code from text.
    1. Look inside markdown code fences for asy-like content.
    2. Check if the raw text itself looks like Asymptote code.
    Returns (asy_code | None, extraction_method).
    """
    for fence_match in CODE_FENCE_RE.finditer(text):
        inner = fence_match.group(1).strip()
        if inner and _looks_like_asy(inner):
            return inner, "code_fence"

    stripped = text.strip()
    if _looks_like_asy(stripped):
        return stripped, "raw"

    return None, "not_found"


# ── Code sanitization ─────────────────────────────────────────────────────

_UNAVAILABLE_FONTS_RE = re.compile(
    r'font\s*\(\s*"(?:phv|ptm|Helvetica|helvetica|Times|serif|sans|rm|sf|sffamily|Roman)"\s*'
    r'(?:,\s*"[^"]*")*\s*\)',
)


def sanitize_asy_code(code: str) -> str:
    """Fix common issues in generated Asymptote code before rendering."""
    code = re.sub(r'settings\.outformat\s*=\s*"[^"]*"\s*;', '', code)
    code = re.sub(r'settings\.render\s*=\s*\d+\s*;', '', code)

    code = _UNAVAILABLE_FONTS_RE.sub('', code)

    code = re.sub(r'\+\s*\+', '+', code)
    code = re.sub(r'=\s*\+', '=', code)
    code = re.sub(r'\(\s*\+', '(', code)
    code = re.sub(r'\+\s*\)', ')', code)
    code = re.sub(r'\+\s*;', ';', code)

    code = re.sub(r'defaultpen\s*\(\s*\)\s*;', '', code)

    return code


# ── Binary discovery ──────────────────────────────────────────────────────

ASY_PATH: str | None = None
GS_PATH: str | None = None


def find_asy() -> str:
    global ASY_PATH
    if ASY_PATH:
        return ASY_PATH
    path = shutil.which("asy")
    if not path:
        raise FileNotFoundError("asy (Asymptote) not found on PATH")
    ASY_PATH = path
    return path


def _find_gs() -> str | None:
    global GS_PATH
    if GS_PATH is not None:
        return GS_PATH
    path = shutil.which("gs")
    GS_PATH = path or ""
    return path


# ── Rendering ─────────────────────────────────────────────────────────────

def _pdf_to_png(pdf_path: str, png_path: str) -> tuple[bool, str]:
    """Convert a PDF to PNG using Ghostscript."""
    gs = _find_gs()
    if not gs:
        return False, "Ghostscript (gs) not found for PDF→PNG conversion"
    try:
        result = subprocess.run(
            [gs, "-dNOPAUSE", "-dBATCH", "-dSAFER",
             "-sDEVICE=png16m", "-r300",
             f"-sOutputFile={png_path}", pdf_path],
            capture_output=True, text=True, timeout=15,
        )
        if Path(png_path).exists() and Path(png_path).stat().st_size > 100:
            return True, "OK (pdf→png)"
        return False, f"gs conversion produced no output: {result.stderr[:200]}"
    except Exception as e:
        return False, f"gs error: {e}"


def render_asy_to_png(
    asy_code: str, png_path: str, timeout: int = 30
) -> tuple[bool, str]:
    """Write Asymptote code to a temp file, execute via asy, produce PNG."""
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="asy_")
        asy_file = os.path.join(tmp_dir, "input.asy")
        out_stem = os.path.join(tmp_dir, "output")

        sanitized = sanitize_asy_code(asy_code)
        with open(asy_file, "w", encoding="utf-8") as f:
            f.write(sanitized)

        result = subprocess.run(
            [find_asy(), "-f", "png", "-render", "4", "-noprc", "-noView",
             "-o", out_stem, asy_file],
            capture_output=True, text=True, timeout=timeout,
            cwd=tmp_dir,
        )

        candidates = sorted(Path(tmp_dir).glob("output*.png"))
        if candidates:
            src = candidates[0]
            if src.stat().st_size >= 100:
                shutil.copy2(str(src), png_path)
                return True, "OK"

        pdf_candidates = sorted(Path(tmp_dir).glob("output*.pdf"))
        if pdf_candidates:
            ok, msg = _pdf_to_png(str(pdf_candidates[0]), png_path)
            if ok:
                return True, msg
            return False, f"PDF produced but conversion failed: {msg}"

        eps_candidates = sorted(Path(tmp_dir).glob("output*.eps"))
        if eps_candidates:
            ok, msg = _pdf_to_png(str(eps_candidates[0]), png_path)
            if ok:
                return True, msg

        if result.returncode != 0:
            stderr = result.stderr.strip()
            msg = stderr[-500:] if len(stderr) > 500 else stderr
            return False, f"asy error: {msg}"
        return False, "PNG was not created"

    except subprocess.TimeoutExpired:
        return False, f"Render timed out ({timeout}s)"
    except Exception as e:
        return False, f"Render error: {e}"
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
