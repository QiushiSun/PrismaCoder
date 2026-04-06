import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from PIL import Image
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


def _find_node_v14_plus() -> str:
    """Find a Node.js binary >= v14 (required for optional chaining / ESM)."""
    candidates = []

    which_node = shutil.which("node")
    if which_node:
        candidates.append(which_node)

    nvm_dir = Path.home() / ".nvm/versions/node"
    if nvm_dir.is_dir():
        for ver_dir in sorted(nvm_dir.iterdir(), reverse=True):
            p = ver_dir / "bin" / "node"
            if p.exists():
                candidates.append(str(p))

    for p in ("/opt/homebrew/bin/node", "/usr/local/bin/node"):
        if Path(p).exists() and p not in candidates:
            candidates.append(p)

    for c in candidates:
        try:
            r = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=5)
            major = int(r.stdout.strip().lstrip("v").split(".")[0])
            if major >= 14:
                return c
        except Exception:
            continue

    return which_node or "node"

def remove_absolute_paths(text: str) -> str:
    """
    Replace absolute paths in error messages with a placeholder.

    Args:
        text (str): The input text containing absolute paths.

    Returns:
        str: The text with absolute paths replaced by 'file:///.../env/'.
    """
    pattern = r'file://[^)]*?/env/'
    return re.sub(pattern, 'file:///.../env/', text)

class MermaidHandler(LanguageHandler):
    """
    Handler for processing Mermaid code.

    This class provides methods for extracting Mermaid plotting code,
    building and executing Jupyter notebooks, parsing results, and
    generating executable code for rendering Mermaid diagrams.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Mermaid handler.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config=config or {})

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract Mermaid plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted Mermaid code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['mermaid', 'json'],
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
        Build and execute a Jupyter notebook to generate Mermaid diagrams.

        Args:
            dataset (pd.DataFrame): The dataset containing code snippets.
            output_path (Path): Path to save the generated notebook.
            csv_folder (Path): Path to the folder containing CSV files (not used for Mermaid).
        """
        cells = dataset.apply(self.generate_code, axis=1).tolist()
        self.build_new_nb(cells, output_path)

        with open(output_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=60,
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

        print(f"Mermaid plots notebook saved to {output_path}")

    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse the results of a Jupyter notebook containing plot outputs.

        Args:
            plots_path (Path): Path to the notebook file.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed results, including:
                - id: Identifier for the code snippet.
                - error: Error message, if any.
                - plots_generated: List of generated plots.
                - has_plot: Boolean indicating if a plot was generated.
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
                    err = remove_absolute_paths(err)
                    cell_res["error"] = err
                elif output.output_type == "display_data" and "image/png" in output.data:
                    b64_png = output.data["image/png"]
                    b64_png = shrink_png_b64_to_under_kb(b64_png, max_kb=200, step=0.9, min_side=64)
                    images.append(b64_png)

            cell_res["plots_generated"] = images
            cell_res["has_plot"] = len(images) > 0
            plot_results.append(cell_res)

        return pd.DataFrame(plot_results)

    def build_new_nb(self, blocks: list, plots_nb_path: Path) -> None:
        """
        Save code blocks to a Jupyter notebook.

        Args:
            blocks (list): List of code blocks.
            plots_nb_path (Path): Path to save the notebook.
        """
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_code_cell(block) for block in blocks]
        with open(plots_nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

    def generate_code(self, item: pd.Series) -> str:
        """
        Generate executable code for rendering Mermaid diagrams.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.

        Returns:
            str: The generated code block.
        """
        item_id = item["id"]
        mermaid_spec = item["code"]
        mmdc_abs = str(Path("plotting_benchmark/language_utils/env/node_modules/.bin/mmdc").resolve())
        node_abs = _find_node_v14_plus()
        mermaid_spec = deduplicate_lines(mermaid_spec, max_repeat=5, min_length=10)
        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import os",
            "import random",
            "import tempfile",
            "import subprocess",
            "from pathlib import Path",
            "from io import BytesIO",
            "from PIL import Image",
            "from IPython.display import display",
            "",
            "Image.MAX_IMAGE_PIXELS = None",
            "random.seed(42)",
            "",
            """\
def render_mermaid(mermaid_code: str, node_path: str, mmdc_path: str, item_id: str):
    scale = random.choice([2, 3])
    temp_dir = tempfile.mkdtemp()
    input_mmd = os.path.join(temp_dir, f"{item_id}.mmd")
    output_png = os.path.join(temp_dir, f"{item_id}.png")
    try:
        with open(input_mmd, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        cmd = [node_path, mmdc_path, "-i", input_mmd, "-o", output_png, "-s", str(scale)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"{err}")

        with open(output_png, "rb") as f:
            data = f.read()
        return Image.open(BytesIO(data))
    finally:
        for path in [input_mmd, output_png]:
            if os.path.exists(path): os.remove(path)
        if os.path.isdir(temp_dir): os.rmdir(temp_dir)
""",
            "",
            f"node_path = '{node_abs}'",
            f"mmdc_path = '{mmdc_abs}'",
            "",
            f"item_id = '''{item_id}'''",
            f"mermaid_code = '''{mermaid_spec}'''",
            "img = render_mermaid(mermaid_code, node_path, mmdc_path, item_id)",
            "display(img)"
        ])

        return "\n".join(code_blocks)




