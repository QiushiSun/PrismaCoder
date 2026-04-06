import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path
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

class AsymptoteHandler(LanguageHandler):
    """Handler for processing Asymptote code."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AsymptoteHandler.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config=config or {})
    
    def extract_plotting_code(self, response: str) -> str:
        """
        Extract Asymptote plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted Asymptote code.
        """
        code = extract_fenced_code(
            response, 
            language_tags=['asymptote', 'json'],  # Try Asymptote first, then JSON
            remove_stop_tokens=True
        )
        
        # Attempt to parse as JSON for compatibility
        try:
            obj = json.loads(code)
            code = json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
        
        return code.strip()
    
    def build_plots(self, dataset: pd.DataFrame, output_path: Path, csv_folder: Path) -> None:
        """
        Build and execute a Jupyter notebook to generate plots.

        Args:
            dataset (pd.DataFrame): The dataset containing code snippets.
            output_path (Path): Path to save the generated notebook.
            csv_folder (Path): Path to the folder containing CSV files.
        """
        # Generate code cells from the dataset
        cells = dataset.apply(self.generate_code, axis=1).tolist()
        
        # Build the notebook
        self.build_new_nb(cells, output_path)
        
        # Execute the notebook
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

        print(f"Asymptote plots notebook saved to {output_path}")

    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse the results of a Jupyter notebook containing plot outputs.

        Args:
            plots_path (Path): Path to the notebook file.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed results.
        """
        with open(plots_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        plot_results = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            # Extract cell ID
            code = cell["source"].lstrip("\n")
            if not code.startswith("# id = "):
                continue
            
            # Extract the ID value
            id_line = code.split("\n")[0].lstrip("# id = ")
            if (id_line.startswith('"') and id_line.endswith('"')) or \
               (id_line.startswith("'") and id_line.endswith("'")):
                id_value = id_line[1:-1]
            else:
                id_value = id_line
                
            cell_res = {"id": id_value, "error": "", "plots_generated": []}

            images = []
            for output in cell["outputs"]:
                if output.output_type == "error":
                    traceback_raw = "\n".join(output.get("traceback", []))
                    cell_res["error"] = remove_ansi_escape(traceback_raw)
                    cell_res["error"] = remove_bytes_literal(cell_res["error"])
                elif (
                    output.output_type == "display_data" and "image/png" in output.data
                ):
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
        Generate executable code for rendering Asymptote graphics.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.

        Returns:
            str: The generated code block.
        """
        item_id = str(item["id"])
        deduped_code = deduplicate_lines(item["code"], max_repeat=5)
        asymptote_spec = deduped_code
        
        # Build the code block structure with ID header
        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import os",
            "import re",
            "import tempfile",
            "import subprocess",
            "from PIL import Image",
            "from io import BytesIO",
            "",
            '''
def render_asymptote(asymptote_code: str, item_id: str):
    temp_dir = tempfile.mkdtemp()
    input_asy = os.path.join(temp_dir, f"{item_id}.asy")
    output_prefix = os.path.join(temp_dir, f"{item_id}")
    output_png = output_prefix + ".png"

    setting_code = (
        "settings.prc = false;\\nsettings.render = 10;\\n"
        if not any(x in asymptote_code for x in ["import graph3", "import solids", "import three", "import bsp", "import contour"])
        else "settings.prc = false;\\nsettings.render = 0;\\n"
    )
    imports = list(re.finditer(r'^\\s*import .*?;\\s*$', asymptote_code, flags=re.MULTILINE))
    if imports:
        last_import = imports[-1]
        pos = last_import.end()
        asymptote_code = (
            asymptote_code[:pos] + "\\n" + setting_code + "\\n" + asymptote_code[pos:]
        )
    else:
        asymptote_code = setting_code + asymptote_code

    try:
        with open(input_asy, "w", encoding="utf-8") as f:
            f.write(asymptote_code)

        cmd = ["asy", "-f", "png", "-render", "10", "-o", output_prefix, input_asy]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"{err}")

        with open(output_png, "rb") as f:
            image_data = f.read()
        img = Image.open(BytesIO(image_data))
        img.load()
        return img
    finally:
        for p in [input_asy, output_png]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        try:
            if os.path.isdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
    '''.strip(),
            "",
            f"item_id = '''{item_id}'''",
            f"asymptote_code = '''{asymptote_spec}'''",
            "img = render_asymptote(asymptote_code, item_id)",
            "display(img)",
        ])

        return "\n".join(code_blocks)


