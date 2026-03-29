"""
Mermaid utility functions — extraction, mmdc discovery, and rendering.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)

MERMAID_KEYWORDS = [
    "graph ", "graph\n",
    "flowchart ", "flowchart\n",
    "sequenceDiagram", "classDiagram", "stateDiagram",
    "erDiagram", "gantt", "pie ", "pie\n",
    "journey", "gitGraph", "mindmap", "timeline",
    "quadrantChart", "sankey", "xychart", "block-beta",
    "C4Context", "C4Container", "C4Component", "C4Deployment",
    "requirementDiagram", "architecture",
]


def _looks_like_mermaid(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(kw) for kw in MERMAID_KEYWORDS)


def extract_mermaid(text: str) -> tuple[str | None, str]:
    """Extract mermaid diagram code from text.
    1. Look inside markdown code fences.
    2. Check if the raw text itself is mermaid code.
    Returns (mermaid_string | None, extraction_method).
    """
    for fence_match in CODE_FENCE_RE.finditer(text):
        inner = fence_match.group(1).strip()
        if inner and _looks_like_mermaid(inner):
            return inner, "code_fence"

    stripped = text.strip()
    if _looks_like_mermaid(stripped):
        return stripped, "raw"

    return None, "not_found"


MMDC_PATH: str | None = None


def find_mmdc() -> str:
    """Locate the mmdc binary. Checks local node_modules, ancestor directories, then PATH."""
    global MMDC_PATH
    if MMDC_PATH is not None:
        return MMDC_PATH

    # Search upward from script directory for node_modules/.bin/mmdc
    search_dir = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = search_dir / "node_modules" / ".bin" / "mmdc"
        if candidate.exists():
            MMDC_PATH = str(candidate)
            return MMDC_PATH
        if search_dir.parent == search_dir:
            break
        search_dir = search_dir.parent

    which = shutil.which("mmdc")
    if which:
        MMDC_PATH = which
        return MMDC_PATH

    raise FileNotFoundError(
        "mmdc not found. Install via: npm install @mermaid-js/mermaid-cli"
    )


def render_mermaid_to_png(
    mermaid_code: str, png_path: str, timeout: int = 30
) -> tuple[bool, str]:
    """Write mermaid code to a temp file and invoke mmdc."""
    tmp_mmd = None
    try:
        fd, tmp_mmd = tempfile.mkstemp(suffix=".mmd")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        result = subprocess.run(
            [find_mmdc(), "-i", tmp_mmd, "-o", png_path, "-b", "white", "-s", "2"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unknown error").strip()
            return False, f"mmdc error: {err[:300]}"
        if not Path(png_path).exists() or Path(png_path).stat().st_size == 0:
            return False, "PNG file is empty or was not created"
        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, f"Render timed out ({timeout}s)"
    except Exception as e:
        return False, f"Render error: {e}"
    finally:
        if tmp_mmd and os.path.exists(tmp_mmd):
            os.unlink(tmp_mmd)
