"""
SVG utility functions — extraction, validation, and rendering.
"""

import ctypes.util
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def ensure_cairo_lib():
    """Make sure the cairo C library is discoverable before importing cairosvg."""
    if ctypes.util.find_library("cairo"):
        return
    brew_lib = None
    try:
        brew_prefix = subprocess.check_output(
            ["brew", "--prefix"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        brew_lib = os.path.join(brew_prefix, "lib")
    except Exception:
        for candidate in ["/opt/homebrew/lib", "/usr/local/lib"]:
            if os.path.isfile(os.path.join(candidate, "libcairo.dylib")):
                brew_lib = candidate
                break
    if brew_lib:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            brew_lib + ":" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        )


ensure_cairo_lib()

import cairosvg  # noqa: E402

CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)
SVG_TAG_RE = re.compile(r"(<svg[\s\S]*?<\/svg>)", re.IGNORECASE)


def extract_svg(text: str) -> tuple[str | None, str]:
    """Extract SVG content from text.
    1. Try inside markdown code fences.
    2. Direct <svg>...</svg> match.
    Returns (svg_string | None, extraction_method).
    """
    fence_match = CODE_FENCE_RE.search(text)
    if fence_match:
        inner = fence_match.group(1).strip()
        svg_match = SVG_TAG_RE.search(inner)
        if svg_match:
            return svg_match.group(1), "code_fence"

    svg_match = SVG_TAG_RE.search(text)
    if svg_match:
        return svg_match.group(1), "raw"

    return None, "not_found"


def validate_svg_string(svg_str: str) -> tuple[bool, str]:
    """Check well-formed XML with <svg> root."""
    try:
        root = ET.fromstring(svg_str)
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag.lower() != "svg":
            return False, f"Root element is <{tag}>, expected <svg>"
        return True, "OK"
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def render_svg_to_png(
    svg_str: str, png_path: Path, width: int = 1600, dpi: int = 96
) -> tuple[bool, str]:
    """Render SVG string to PNG via cairosvg."""
    try:
        cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            write_to=str(png_path),
            output_width=width,
            dpi=dpi,
        )
        if not png_path.exists() or png_path.stat().st_size == 0:
            return False, "PNG file is empty or was not created"
        return True, "OK"
    except Exception as e:
        return False, f"Render error: {e}"
