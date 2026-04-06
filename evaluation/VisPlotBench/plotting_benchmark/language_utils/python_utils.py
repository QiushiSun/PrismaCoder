import csv
import json
import re
import pandas as pd
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Tuple
from pathlib import Path, PurePosixPath
import base64
from collections import defaultdict
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from .base_handler import LanguageHandler
from .base_utils import (
    remove_ansi_escape,
    remove_bytes_literal, 
    shrink_png_b64_to_under_kb,
    deduplicate_lines,
    extract_fenced_code
)

def _ensure_top_import(code: str, import_line: list[str] | str) -> str:
    """
    Ensures required import statements are added at the top of the code.
    If imports are already present, they won't be duplicated.
    """
    if isinstance(import_line, str):
        import_line = [import_line]

    missing_imports = [line for line in import_line if line.strip() not in code]
    if not missing_imports:
        return code

    import_pattern = re.compile(r'^(?:import\s+\S+|from\s+\S+\s+import\s+\S+)', re.MULTILINE)
    matches = list(import_pattern.finditer(code))

    if matches:
        last_import = matches[-1]
        insert_pos = code.find("\n", last_import.end())
        if insert_pos == -1:
            insert_pos = len(code)
        else:
            insert_pos += 1
        return code[:insert_pos] + "\n".join(missing_imports) + "\n" + code[insert_pos:]
    else:
        return "\n".join(missing_imports) + "\n" + code

def convert_bokeh(code: str, source_id: str) -> str:
    """
    Converts Bokeh visualization code to generate PNG outputs.
    Replaces show() calls with SVG export and PNG conversion.
    """
    library_imports = [
        "import os",
        "import tempfile",
        "from IPython.display import Image, display\n",
        "from bokeh.io import export_svgs",
        "import cairosvg"
    ]
    code = _ensure_top_import(code, library_imports)

    # Replace show(...) calls
    def _repl(m):
        inner = m.group(1).strip()
        if not inner:   # If show() is empty, default to variable p
            inner = "p"
        return (
            "tmpdir = tempfile.mkdtemp()\n"
            f"svg_file = os.path.join(tmpdir, '{source_id}.svg')\n"
            f"png_file = os.path.join(tmpdir, '{source_id}.png')\n"
            f"{inner}.output_backend = 'svg'\n"
            f"export_svgs({inner}, filename=svg_file)\n"
            f"cairosvg.svg2png(url=svg_file, write_to=png_file)\n"
            "display(Image(filename=png_file))"
        )

    code = re.sub(
        r"\bshow\s*\(\s*([^\)]*?)\s*\)",  # Match show(...) calls
        _repl,
        code,
        flags=re.DOTALL
    )
    return code


def convert_plotly(code: str, source_id: str) -> str:
    """
    Converts Plotly visualization code to ensure proper rendering.
    Sets the renderer to PNG for consistent output.
    """
    library_imports = ["import pandas as pd",
                        "import plotly.express as px", 
                        "import plotly.io as pio", 
                        "pio.renderers.default = \"png\""]

    code = _ensure_top_import(code, library_imports)
    return code

def convert_altair(code: str, source_id: str) -> str:
    """
    Converts Altair visualization code to generate PNG outputs.
    Replaces show() calls with save() and display() for PNG files.
    """
    library_imports = ["import altair as alt",
                        "import pandas as pd",
                        "from IPython.display import Image, HTML, display\n",
                        f"display(HTML('<div class=\"snap-anchor\" data-source-id=\"{source_id}\"></div>'))\n",]
    code = _ensure_top_import(code, library_imports)
    png = f"{source_id}.png"
    pattern = r"(\b[A-Za-z_]\w*)\s*\.\s*show\s*\([^)]*\)"

    def _repl(m):
        var = m.group(1)
        return (
            "tmpdir = tempfile.mkdtemp()\n"
            f"png_file = os.path.join(tmpdir, '{source_id}.png')\n"
            f'{var}.save(png_file)\n'
            f'display(Image(filename=png_file))'
        )
    return re.sub(pattern, _repl, code, flags=re.DOTALL)

def convert_pyecharts(code: str, source_id: str) -> str:
    """
    Intelligently instruments pyecharts code:
    - Automatically injects required imports (IPython.display / asyncio / playwright async)
    - Replaces .render(...) / .render_notebook() with .render_embed()
    - Automatically discovers pyecharts chart variables (Bar/Line/Pie/.../Page/Timeline etc.)
    - Uses Playwright for async screenshot to PNG bytes and direct display(Image(...))
    
    Dependencies: pip install playwright && playwright install chromium
    Note: Depends on _ensure_top_import(), consistent with previous versions.
    """
    import ast  # For static analysis to collect variable names

    # 1) Ensure necessary imports (no longer depends on snapshot-selenium)
    library_imports = [
        "import pandas as pd",
        "from IPython.display import Image, display",
        "import asyncio",
        "from playwright.async_api import async_playwright",
    ]
    code = _ensure_top_import(code, library_imports)

    # 2) Collect pyecharts variable names through AST (targets assigned to pyecharts charts)
    chart_class_names = {
        # Common charts
        "Bar","Line","Pie","Scatter","Map","Geo","Radar","HeatMap","EffectScatter",
        "Kline","Boxplot","Funnel","Gauge","Graph","Liquid","Parallel","Polar",
        "Sankey","Sunburst","Themeriver","Tree","Treemap","WordCloud",
        # Container/layout classes, also support render_embed
        "Page","Grid","Timeline","Tab",
    }

    def _is_chart_ctor(call_node: ast.AST) -> bool:
        # Check if it looks like Bar(...) / charts.Bar(...) / pyecharts.charts.Bar(...)
        if not isinstance(call_node, ast.Call):
            return False
        f = call_node.func
        if isinstance(f, ast.Name):
            return f.id in chart_class_names
        if isinstance(f, ast.Attribute):
            # Get rightmost name
            return f.attr in chart_class_names
        return False

    candidate_names = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # a = Bar(...), a: Bar = Bar(...)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.AST):
                if _is_chart_ctor(node.value):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            candidate_names.append(t.id)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if isinstance(node.target, ast.Name) and _is_chart_ctor(node.value):
                    candidate_names.append(node.target.id)
    except Exception:
        # Ignore AST failures, regex fallback below
        pass

    # 3) Fallback with regex: capture X.render(...) / X.render_notebook(...)
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*render(?:_notebook)?\s*\(", code):
        candidate_names.append(m.group(1))

    # Deduplicate while preserving order
    seen = set()
    var_names = []
    for n in candidate_names:
        if n not in seen:
            seen.add(n)
            var_names.append(n)

    # 4) Replace all .render(...) / .render_notebook(...) with .render_embed()
    code = re.sub(r"\.render(?:_notebook)?\s*\([^)]*\)", ".render_embed()", code)

    # 5) Append tail: collect objects -> screenshot with playwright -> display(Image)
    #    Use try blocks to protect against undefined variables
    #    Provide fallback: scan globals() for objects with render_embed() if no variables captured
    tail = f"""
_pyecharts_objs = []
for _name in {var_names!r}:
    try:
        _obj = globals().get(_name, None)
        if _obj is None:
            try:
                _obj = locals()[_name]
            except Exception:
                _obj = None
        if _obj is not None and hasattr(_obj, "render_embed"):
            _pyecharts_objs.append(_obj)
    except Exception:
        pass
if not _pyecharts_objs:
    try:
        for _k, _v in list(globals().items()):
            try:
                if hasattr(_v, "render_embed"):
                    _pyecharts_objs.append(_v)
            except Exception:
                pass
    except Exception:
        pass

async def _pyecharts_display_all(_objs):
    try:
        async with async_playwright() as _p:
            _browser = await _p.chromium.launch()
            _page = await _browser.new_page()
            for _obj in _objs:
                try:
                    _html = _obj.render_embed()
                    await _page.set_content(_html)  
                    await _page.wait_for_timeout(600)
                    _png = await _page.screenshot(full_page=True)
                    display(Image(data=_png))
                except Exception as _e:
                    raise Exception(_e)
            await _browser.close()
    except Exception as _e:
        raise Exception(_e)
await _pyecharts_display_all(_pyecharts_objs)
"""

    return code.rstrip() + tail

def convert_chartify(code: str, source_id: str) -> str:
    """
    Converts Chartify visualization code to generate PNG outputs.
    Handles both .show() and .save() methods.
    """
    library_imports = ["import pandas as pd",
                        "from IPython.display import Image, HTML, display\n",
                        "from bokeh.io.export import export_png",
                        ]

    code = _ensure_top_import(code, library_imports)

    temp_dir_setup = f"""
import os
import tempfile
temp_dir = tempfile.mkdtemp()
png_file = os.path.join(temp_dir, '{source_id}.png')
"""
    code = temp_dir_setup + code

    # 1) Handle .show(...) series
    code = re.sub(
        r"(\b[A-Za-z_]\w*)\s*\.\s*show\s*\([^)]*\)",
        rf"export_png(\1.figure, filename=png_file)\n"
        rf"display(Image(png_file))",
        code,
        flags=re.DOTALL
    )

    # 2) Handle .save('...html'), convert to export_png(...) and display PNG
    code = re.sub(
        r"(\b[A-Za-z_]\w*)\s*\.\s*save\s*\(\s*(['\"])[^'\"]*?\.html\2\s*\)",
        rf"export_png(\1.figure, filename=png_file)\n"
        rf"display(Image(png_file))",
        code,
        flags=re.IGNORECASE
    )

    # 3) Replace subsequent display(HTML(...)) with display(Image(...)) to avoid HTML residue
    code = re.sub(
        r"\bdisplay\s*\(\s*HTML\s*\([^)]*\)\s*\)",
        rf"display(Image(png_file))",
        code
    )

    return code

import re

def convert_holoviews(code: str, source_id: str) -> str:
    """
    Converts HoloViews visualization code to generate PNG outputs.
    Handles different rendering patterns and ensures proper display.
    """
    # Required imports and setup code
    library_imports = [
        "import holoviews as hv",
        "hv.extension('bokeh')",
        "import pandas as pd",
        "from IPython.display import Image, display\n"
    ]
    code = _ensure_top_import(code, library_imports)

    # Set up temporary file paths
    temp_dir_setup = f"""
import os
import tempfile
temp_dir = tempfile.mkdtemp()
png_file = os.path.join(temp_dir, '{source_id}.png')
"""
    code = temp_dir_setup + code

    changed = False

    # 1) show(hv.render(expr)) → hv.save(expr, png_file, fmt="png"); display(Image)
    pattern_show_render = re.compile(
        r"\bshow\s*\(\s*hv\.render\s*\(\s*(?P<expr>.*?)\s*(?:,\s*backend\s*=\s*['\"][^'\"]+['\"])?\s*\)\s*\)",
        flags=re.DOTALL,
    )

    def repl_show_render(m: re.Match) -> str:
        nonlocal changed
        changed = True
        expr = m.group("expr").strip()
        return (
            f"hv.save({expr}, png_file, fmt='png')\n"
            f"display(Image(filename=png_file))"
        )

    code = pattern_show_render.sub(repl_show_render, code)

    # 2) hv.save(expr, 'xxx.png', ...) → hv.save(expr, png_file, fmt="png"); display(Image)
    pattern_hv_save = re.compile(
        r"hv\.save\s*\(\s*(?P<expr>.+?)\s*,\s*(?P<q>['\"])(?P<fname>[^'\"]+?\.png)(?P=q)(?P<rest>[^)]*)\)",
        flags=re.DOTALL,
    )

    def repl_hv_save(m: re.Match) -> str:
        nonlocal changed
        changed = True
        expr = m.group("expr").strip()
        return (
            f"hv.save({expr}, png_file, fmt='png')\n"
            f"display(Image(filename=png_file))"
        )

    code = pattern_hv_save.sub(repl_hv_save, code)

    # 3) If no hv.save/show matched, automatically append save + display at the end
    if not changed:
        lines = code.rstrip("\n").split("\n")
        idx = len(lines) - 1
        while idx >= 0 and (not lines[idx].strip() or lines[idx].lstrip().startswith("#")):
            idx -= 1
        if idx >= 0:
            last_expr = lines[idx].strip()
            append_block = (
                f"\nhv.save({last_expr}, png_file, fmt='png')\n"
                f"display(Image(filename=png_file))"
            )
            code = code.rstrip("\n") + append_block

    return code


def convert_pyvista(code: str, source_id: str) -> str:
    """
    Converts PyVista visualization code to generate PNG outputs.
    Adds timeout handling and ensures off-screen rendering.
    """
    library_imports = [
        "import pyvista as pv",
        "import pandas as pd",
        "import numpy as np",
        "from IPython.display import Image, display",
        "import tempfile, os, signal, contextlib"
    ]
    code = _ensure_top_import(code, library_imports)

    timeout_helper = """
@contextlib.contextmanager
def _pv_timeout(seconds=20):
    def handler(signum, frame):
        raise TimeoutError(f"PyVista operation timeout after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
"""

    lines = code.splitlines()
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import") or line.strip().startswith("from"):
            insert_pos = i
    lines.insert(insert_pos + 1, timeout_helper)
    code = "\n".join(lines)

    code = re.sub(
        r"pv\.Plotter\s*\(\s*\)",
        "pv.Plotter(off_screen=True)",
        code
    )

    replacement = (
        "tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)\n"
        "with _pv_timeout(20):\n"
        "    plotter.screenshot(tmpfile.name)\n"
        "display(Image(filename=tmpfile.name))\n"
        "plotter.close()\n"
        "os.remove(tmpfile.name)"
    )
    code = re.sub(
        r"plotter\.show\s*\(\s*\)",
        replacement,
        code
    )

    return code


def convert_matplotlib(code: str, source_id: str) -> str:
    """
    Converts Matplotlib visualization code.
    Ensures plt.show() is called at the end if not present.
    """
    library_imports = ["import matplotlib.pyplot as plt",
                        "import pandas as pd"
                        ]
    code = _ensure_top_import(code, library_imports)
    if ".show()" not in code:
        code = code + "\nplt.show()"
    return code

def convert_seaborn(code: str, source_id: str) -> str:
    """
    Converts Seaborn visualization code.
    Ensures proper imports and plt.show() call.
    """
    library_imports = ["import pandas as pd",
                        "import seaborn as sns",
                        "import matplotlib.pyplot as plt"
                        ]
    code = _ensure_top_import(code, library_imports)
    if ".show()" not in code:
        code = code + "\nplt.show()"
    return code

class PythonHandler(LanguageHandler):
    """Python language handler supporting all Python visualization libraries"""
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config or {})

    def _get_supported_libraries(self) -> List[str]:
        return [
            "matplotlib", "matplotlib pyplot", 
            "seaborn", 
            "plotly", 
            "bokeh", 
            "altair",
            "pygal",
            "holoviews"
        ]

    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse notebook to collect PNG outputs and errors from each cell.
        
        Args:
            plots_path: Path to the Jupyter notebook file
            
        Returns:
            DataFrame containing plot results and errors
        """
        with open(plots_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        plot_results = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            code = cell["source"].lstrip("\n")
            if not code.startswith("# id = "):
                continue

            id_line = code.split("\n")[0].lstrip("# id = ")
            if (id_line.startswith('"') and id_line.endswith('"')) or \
               (id_line.startswith("'") and id_line.endswith("'")):
                id_value = id_line[1:-1]
            else:
                id_value = id_line

            cell_res = {"id": id_value, "error": "", "plots_generated": []}

            images = []
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    traceback_raw = "\n".join(output.get("traceback", []))
                    err = remove_ansi_escape(traceback_raw)
                    err = remove_bytes_literal(err)
                    cell_res["error"] = err
                elif output.output_type == "display_data" and "image/png" in output.data:
                    b64_png = output.data["image/png"]
                    b64_png = shrink_png_b64_to_under_kb(b64_png, max_kb=200, step=0.9, min_side=64)
                    images.append(b64_png)

            cell_res["plots_generated"] = images
            cell_res["has_plot"] = len(images) > 0
            plot_results.append(cell_res)

        return pd.DataFrame(plot_results)

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract Python plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted Python code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['python', 'json'],
            remove_stop_tokens=True
        )
        try:
            obj = json.loads(code)
            code = json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
        
        return code.strip()
    
    def build_plots(self, dataset: pd.DataFrame, output_path: Path, csv_folder: Path) -> None:
        """
        Build plots as a Jupyter notebook.
        
        Args:
            dataset: DataFrame containing code to visualize
            output_path: Path to save the output notebook
            csv_folder: Path to folder containing CSV data files
        """
        # Generate all code blocks
        cells = dataset.apply(
            self.generate_code, axis=1, args=(csv_folder,)
        ).tolist()
        
        # Build notebook
        self.build_new_nb(cells, output_path)
        
        # Read and execute notebook
        with open(output_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        class DebugExecutePreprocessor(ExecutePreprocessor):
            def preprocess_cell(self, cell, resources, cell_index):
                if cell.cell_type == 'code' and cell.source.strip():
                    first_line = cell.source.split('\n')[0]
                    if first_line.startswith('# id = '):
                        cell_id = first_line.replace('# id = ', '').strip()
                        print(f"[DEBUG] Executing cell {cell_index + 1}/{len(nb.cells)} - ID: {cell_id}")
                    else:
                        print(f"[DEBUG] Executing cell {cell_index + 1}/{len(nb.cells)}")
                
                try:
                    cell, resources = super().preprocess_cell(cell, resources, cell_index)
                    print(f"[DEBUG] Cell {cell_index + 1} completed successfully")
                    return cell, resources
                except Exception as e:
                    print(f"[DEBUG] Cell {cell_index + 1} failed with error: {str(e)[:100]}...")
                    raise

        ep = DebugExecutePreprocessor(
            timeout=30,
            interrupt_on_timeout=True,
            allow_errors=True, 
            kernel_name="python3",
        )

        try:
            ep.preprocess(nb, {"metadata": {"path": "."}})
        except CellExecutionError as e:
            print(f"[WARNING] Execution stopped early: {e}\nNotebook will still be saved.")

        with open(output_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        
        print(f"Python plots notebook saved to {output_path}")

    def build_new_nb(self, blocks: list, plots_nb_path: Path) -> None:
        """
        Save code blocks to a notebook.
        
        Args:
            blocks: List of code blocks to include in the notebook
            plots_nb_path: Path to save the notebook
        """
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_code_cell(block) for block in blocks]

        with open(plots_nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

    def generate_code(self, item: pd.Series, csv_folder: Path) -> str:
        """
        Generate code for a specific visualization item.
        
        Args:
            item: Series containing visualization metadata
            csv_folder: Path to folder containing CSV data files
            
        Returns:
            String containing the generated code
        """
        item_id = item["id"]  # Keep original format
        csv_path = csv_folder / f"{item_id}.csv"
        deduped_code = deduplicate_lines(item["code"], max_repeat=5)
        python_src = deduped_code.strip()
        python_src = python_src.replace("data.csv", str(csv_path))
        used_lib = item["used_lib"]
        if used_lib == "plotly":
            python_src = convert_plotly(python_src, item_id)
        elif used_lib == "pyecharts":
            python_src = convert_pyecharts(python_src, item_id)
        elif used_lib == "altair":
            python_src = convert_altair(python_src, item_id)
        elif used_lib == "chartify":
            python_src = convert_chartify(python_src, item_id)
        elif used_lib == "holoviews":
            python_src = convert_holoviews(python_src, item_id)
        elif used_lib == "bokeh":
            python_src = convert_bokeh(python_src, item_id)
        elif used_lib == "matplotlib":
            python_src = convert_matplotlib(python_src, item_id)
        elif used_lib == "seaborn":
            python_src = convert_seaborn(python_src, item_id)
        elif used_lib == "pyvista":
            python_src = convert_pyvista(python_src, item_id)

        python_src = f"# id = {item_id}\n" + python_src
        return python_src