"""
R code utility functions — extraction, interactive code classification,
Shiny/plotly/rgl transforms, and Rscript rendering.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# ── R code extraction ───────────────────────────────────────────────────────

CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)

R_INDICATORS = [
    "library(", "require(", "ggplot(", "ggplot2", "geom_",
    "aes(", "plot(", "data.frame(", "<-", "print(",
    "theme(", "labs(", "ggsave(",
]


def _looks_like_r(text: str) -> bool:
    return sum(1 for ind in R_INDICATORS if ind in text) >= 2


def extract_r_code(text: str) -> tuple[str | None, str]:
    """Extract R code from text.
    1. Look inside markdown code fences for R-like content.
    2. Check if the raw text itself looks like R code.
    Returns (r_code | None, extraction_method).
    """
    for fence_match in CODE_FENCE_RE.finditer(text):
        inner = fence_match.group(1).strip()
        if inner and _looks_like_r(inner):
            return inner, "code_fence"

    stripped = text.strip()
    if _looks_like_r(stripped):
        lines = []
        in_code = False
        for line in stripped.split("\n"):
            s = line.strip()
            if s.startswith("```"):
                in_code = not in_code
                continue
            if not in_code:
                if any(ind in s for ind in R_INDICATORS) or s == "" or s.startswith("#"):
                    lines.append(line)
                    in_code = True
            else:
                lines.append(line)
        if lines and _looks_like_r("\n".join(lines)):
            return "\n".join(lines), "raw_filtered"

        if _looks_like_r(stripped):
            return stripped, "raw"

    return None, "not_found"


# ── Interactive R code classification and transformation ────────────────────

INTERACTIVE_PKGS = {
    "shiny", "shinydashboard", "shinythemes", "shinyjs", "shinyWidgets",
    "shinyBS", "miniUI",
}


def classify_interactive(r_code: str) -> str | None:
    """Classify what kind of interactive code this is.
    Returns: 'shiny_plot', 'shiny_no_plot', 'plotly_ggplotly', 'plotly_native',
             'rgl', or None if not interactive.
    """
    has_shiny = bool(re.search(r'library\((shiny|shinydashboard|shinythemes)\)', r_code))
    has_plotly = bool(re.search(r'library\(plotly\)', r_code))
    has_rgl = bool(re.search(r'library\(rgl\)', r_code))

    if has_shiny:
        if "renderPlot" in r_code or ("ggplot(" in r_code and "geom_" in r_code) or "plot(" in r_code:
            return "shiny_plot"
        return "shiny_no_plot"
    if has_plotly:
        if "ggplotly(" in r_code:
            return "plotly_ggplotly"
        return "plotly_native"
    if has_rgl:
        return "rgl"
    return None


# ── Shiny transforms ───────────────────────────────────────────────────────

def _extract_balanced_parens(text: str, start: int) -> str | None:
    """Extract text inside balanced parentheses starting from position of (."""
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def _extract_shiny_input_defaults(r_code: str) -> dict[str, str]:
    """Parse Shiny UI input widgets to extract default values."""
    defaults = {}
    for m in re.finditer(r'(?:slider|numeric|text|select|checkbox|radioButtons)Input\s*\(', r_code):
        paren_start = r_code.index("(", m.start())
        inner = _extract_balanced_parens(r_code, paren_start)
        if not inner:
            continue
        name_m = re.match(r'\s*["\'](\w+)["\']', inner)
        if not name_m:
            continue
        name = name_m.group(1)
        val_m = re.search(r'(?:value|selected)\s*=\s*(.+?)(?:,\s*\w+\s*=|\s*$)', inner, re.DOTALL)
        if val_m:
            val = val_m.group(1).strip().rstrip(",").strip()
            defaults[name] = val
        elif "choices" in inner:
            ch_m = re.search(r'choices\s*=\s*c\(([^)]+)\)', inner)
            if ch_m:
                defaults[name] = ch_m.group(1).split(",")[0].strip()
    for m in re.finditer(r'checkboxInput\s*\(\s*["\'](\w+)["\']', r_code):
        if m.group(1) not in defaults:
            defaults[m.group(1)] = "TRUE"
    return defaults


def _find_balanced_brace_block(text: str, start: int) -> tuple[int, int] | None:
    """Find the matching { ... } starting from `start`.
    Returns (open_idx, close_idx) or None.
    """
    idx = text.find("{", start)
    if idx == -1:
        return None
    depth = 0
    in_string = None
    prev_ch = ""
    for i in range(idx, len(text)):
        ch = text[i]
        if in_string:
            if ch == in_string and prev_ch != "\\":
                in_string = None
        else:
            if ch in ('"', "'"):
                in_string = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (idx, i)
        prev_ch = ch
    return None


def _extract_all_render_plot_bodies(r_code: str) -> list[str]:
    """Extract bodies from all renderPlot({ ... }) blocks."""
    bodies = []
    search_start = 0
    while True:
        m = re.search(r'renderPlot\s*\(', r_code[search_start:])
        if not m:
            break
        abs_start = search_start + m.end()
        brace = _find_balanced_brace_block(r_code, abs_start)
        if brace:
            body = r_code[brace[0] + 1 : brace[1]].strip()
            bodies.append(body)
            search_start = brace[1] + 1
        else:
            search_start = abs_start
    return bodies


def _find_top_level_assignment_blocks(r_code: str) -> list[str]:
    """Extract top-level code (data, functions), skipping ui/server/shinyApp blocks."""
    blocks = []
    lines = r_code.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if re.match(r'^(ui|server)\s*<-', stripped) or re.match(r'^shinyApp\s*\(', stripped):
            brace_depth = 0
            paren_depth = 0
            started = False
            while i < len(lines):
                for ch in lines[i]:
                    if ch == "{":
                        brace_depth += 1
                        started = True
                    elif ch == "}":
                        brace_depth -= 1
                    elif ch == "(":
                        paren_depth += 1
                        started = True
                    elif ch == ")":
                        paren_depth -= 1
                i += 1
                if started and brace_depth <= 0 and paren_depth <= 0:
                    break
            continue
        if re.match(r'^\s*(?:library|require)\s*\(', stripped):
            i += 1
            continue
        blocks.append(lines[i])
        i += 1
    return blocks


def transform_shiny_code(r_code: str) -> str:
    """Strip Shiny framework and extract static plotting code."""
    input_defaults = _extract_shiny_input_defaults(r_code)
    libraries = []
    seen = set()
    for m in re.finditer(r'library\((\w+)\)', r_code):
        pkg = m.group(1)
        if pkg not in INTERACTIVE_PKGS and pkg not in seen:
            libraries.append(f"library({pkg})")
            seen.add(pkg)

    plot_bodies = _extract_all_render_plot_bodies(r_code)
    top_blocks = _find_top_level_assignment_blocks(r_code)

    parts = []
    parts.extend(libraries)
    parts.append("")
    parts.extend(top_blocks)
    parts.append("")
    parts.append("input <- list()")
    for k, v in input_defaults.items():
        parts.append(f"input${k} <- {v}")
    parts.append("")
    for body in plot_bodies:
        parts.append(body)
        parts.append("")

    result = "\n".join(parts)
    for m in re.finditer(r'input\$(\w+)', result):
        name = m.group(1)
        if name not in input_defaults:
            result = result.replace(f"input${name}", "NULL")
    result = re.sub(r'^\s*req\([^)]*\)\s*$', '', result, flags=re.MULTILINE)
    result = re.sub(r'(\w+)\s*<-\s*reactive\(\s*\{', r'\1 <- local({', result)
    return result


# ── Plotly transforms ──────────────────────────────────────────────────────

def transform_plotly_ggplotly(r_code: str) -> str:
    """Convert ggplotly() code to plain ggplot by removing the wrapper."""
    result = re.sub(r'library\(plotly\)\s*\n?', '', r_code)
    result = re.sub(r'ggplotly\(\s*(\w+)\s*\)', r'print(\1)', result)
    result = re.sub(r'ggplotly\(\s*(\w+)\s*,\s*[^)]*\)', r'print(\1)', result)
    while "ggplotly(" in result:
        idx = result.find("ggplotly(")
        paren_start = idx + len("ggplotly")
        depth = 0
        end = paren_start
        for j in range(paren_start, len(result)):
            if result[j] == "(":
                depth += 1
            elif result[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        inner = result[paren_start + 1 : end]
        arg_depth = 0
        first_arg_end = len(inner)
        for j, ch in enumerate(inner):
            if ch in ("(", "[", "{"):
                arg_depth += 1
            elif ch in (")", "]", "}"):
                arg_depth -= 1
            elif ch == "," and arg_depth == 0:
                first_arg_end = j
                break
        first_arg = inner[:first_arg_end].strip()
        result = result[:idx] + first_arg + result[end + 1:]
    return result


def transform_plotly_native(r_code: str, png_path: str) -> str:
    """Add plotly save via htmlwidgets + webshot2 at the end."""
    save_block = f'''

.plotly_obj <- tryCatch(last_plot(), error = function(e) NULL)
if (is.null(.plotly_obj)) {{
  .pnames <- ls()
  for (.pn in rev(.pnames)) {{
    .pval <- tryCatch(get(.pn), error = function(e) NULL)
    if (inherits(.pval, "plotly")) {{
      .plotly_obj <- .pval
      break
    }}
  }}
}}
if (!is.null(.plotly_obj) && inherits(.plotly_obj, "plotly")) {{
  .tmp_html <- tempfile(fileext = ".html")
  htmlwidgets::saveWidget(plotly::as_widget(.plotly_obj), .tmp_html, selfcontained = TRUE)
  webshot2::webshot(.tmp_html, file = {json.dumps(png_path)}, vwidth = 1200, vheight = 900, delay = 2)
  unlink(.tmp_html)
}}
'''
    return r_code + save_block


# ── rgl transforms ─────────────────────────────────────────────────────────

def transform_rgl(r_code: str, png_path: str) -> str:
    """Add rgl headless setup and snapshot at the end."""
    preamble = "options(rgl.useNULL = TRUE)\n"
    postamble = f'''

if (requireNamespace("rgl", quietly = TRUE) && rgl::cur3d() > 0) {{
  rgl::snapshot3d({json.dumps(png_path)})
}}
'''
    return preamble + r_code + postamble


# ── Rendering ───────────────────────────────────────────────────────────────

RSCRIPT_PATH: str | None = None


def find_rscript() -> str:
    global RSCRIPT_PATH
    if RSCRIPT_PATH:
        return RSCRIPT_PATH
    path = shutil.which("Rscript")
    if not path:
        raise FileNotFoundError("Rscript not found on PATH")
    RSCRIPT_PATH = path
    return path


def _indent(text: str, n: int) -> str:
    prefix = "  " * n
    return "\n".join(prefix + line for line in text.split("\n"))


def _patch_ggsave(r_code: str, png_path: str) -> str:
    """Redirect any ggsave() call to our target png_path."""
    return re.sub(
        r'ggsave\s*\([^)]*\)',
        f'ggsave({json.dumps(png_path)}, width=10, height=8, dpi=150)',
        r_code,
    )


def _wrap_r_code(r_code: str, png_path: str, mode: str = "static") -> str:
    """Wrap user R code so it renders to a PNG file instead of screen."""
    skip_png_device = mode in ("plotly_native", "rgl")
    wrapper = f'''
options(warn = -1)
suppressMessages(suppressWarnings({{

.gg_available <- tryCatch({{
  library(ggplot2, quietly = TRUE)
  TRUE
}}, error = function(e) FALSE)

.output_png_path <- {json.dumps(png_path)}

.has_ggsave <- grepl("ggsave", {json.dumps(r_code)}, fixed = TRUE)
.skip_png_device <- {'TRUE' if skip_png_device else 'FALSE'}

if (!.has_ggsave && !.skip_png_device) {{
  png(.output_png_path, width = 1600, height = 1200, res = 150)
}}

tryCatch({{
{_indent(r_code, 2)}
}}, error = function(e) {{
  try(dev.off(), silent = TRUE)
  cat("R_RENDER_ERROR:", conditionMessage(e), "\\n", file = stderr())
  quit(status = 1)
}})

if (!.has_ggsave && !.skip_png_device) {{
  if (.gg_available) {{
    .lp <- tryCatch(ggplot2::last_plot(), error = function(e) NULL)
    if (!is.null(.lp)) {{
      tryCatch(print(.lp), error = function(e) {{}})
    }}
  }}
  dev_list <- dev.list()
  if (!is.null(dev_list)) {{
    dev.off()
  }}
}}

if (!file.exists(.output_png_path) || file.info(.output_png_path)$size < 100) {{
  if (.gg_available) {{
    .lp2 <- tryCatch(ggplot2::last_plot(), error = function(e) NULL)
    if (!is.null(.lp2)) {{
      tryCatch(
        ggplot2::ggsave(.output_png_path, plot = .lp2,
                        width = 10, height = 8, dpi = 150),
        error = function(e) {{}}
      )
    }}
  }}
}}

if (!file.exists(.output_png_path) || file.info(.output_png_path)$size < 100) {{
  cat("R_RENDER_ERROR: PNG was not created or is empty\\n", file = stderr())
  quit(status = 1)
}}

}}))
'''
    return wrapper


def render_r_to_png(
    r_code: str, png_path: str, timeout: int = 30, mode: str = "static"
) -> tuple[bool, str]:
    """Write R code to a temp file, execute via Rscript, produce PNG."""
    tmp_r = None
    try:
        r_code_patched = _patch_ggsave(r_code, png_path)
        wrapped = _wrap_r_code(r_code_patched, png_path, mode=mode)

        fd, tmp_r = tempfile.mkstemp(suffix=".R")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(wrapped)

        result = subprocess.run(
            [find_rscript(), "--vanilla", tmp_r],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "LANG": "en_US.UTF-8"},
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            render_errors = [
                l for l in stderr.split("\n") if "R_RENDER_ERROR:" in l
            ]
            if render_errors:
                msg = render_errors[0].split("R_RENDER_ERROR:")[-1].strip()
            else:
                msg = stderr[-500:] if len(stderr) > 500 else stderr
            return False, f"R error: {msg}"

        if not Path(png_path).exists() or Path(png_path).stat().st_size < 100:
            return False, "PNG file is empty or was not created"

        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, f"Render timed out ({timeout}s)"
    except Exception as e:
        return False, f"Render error: {e}"
    finally:
        if tmp_r and os.path.exists(tmp_r):
            os.unlink(tmp_r)
