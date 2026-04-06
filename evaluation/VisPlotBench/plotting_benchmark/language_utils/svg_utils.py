import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
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
class SVGHandler(LanguageHandler):
    """
    Handler for processing SVG code.

    This class adapts the SVG rendering process to the evaluation framework.
    It provides methods for extracting plotting code, building and executing
    notebooks, parsing results, and generating rendering code.

    Dependencies:
        - CairoSVG must be installed for SVG to PNG conversion.
        - Input DataFrame must contain the columns: 'id' and 'code' (SVG source code).
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SVG handler.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config=config or {})

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract SVG plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted SVG code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['svg', 'json'],
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
        Build and execute a Jupyter notebook to generate SVG plots.

        Args:
            dataset (pd.DataFrame): The dataset containing code snippets.
            output_path (Path): Path to save the generated notebook.
            csv_folder (Path): Path to the folder containing CSV files (not used for SVG).
        """
        cells = dataset.apply(self.generate_code, axis=1).tolist()
        self.build_new_nb(cells, output_path)

        with open(output_path, encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        ep = ExecutePreprocessor(
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

        print(f"SVG plots notebook saved to {output_path}")

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
        Generate executable code for rendering SVG plots.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.

        Returns:
            str: The generated code block.
        """
        item_id = str(item["id"]) 
        deduped_code = deduplicate_lines(item["code"], max_repeat=5)

        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import cairosvg",
            "from io import BytesIO",
            "from PIL import Image",
            "",
            '''
def render_svg(svg_string: str):
    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8")
    )
    img_buffer = BytesIO(png_data)
    return Image.open(img_buffer)
'''.strip(),
            "",
            f"item_id = '''{item_id}'''",
            f"svg_code = '''{deduped_code}'''",
            "img = render_svg(svg_code)",
            "display(img)",
        ])

        return "\n".join(code_blocks)


