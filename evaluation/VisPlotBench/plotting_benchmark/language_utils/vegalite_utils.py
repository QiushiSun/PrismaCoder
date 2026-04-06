import pandas as pd
import json
from typing import Dict, Any
from pathlib import Path, PurePosixPath
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

try:
    import vl_convert as vlc
except ImportError:
    vlc = None
    print("Warning: vl_convert not installed. Vega-Lite rendering will not work.")

class VegaLiteHandler(LanguageHandler):
    """
    Handler for processing Vega-Lite code.

    This class provides methods for extracting Vega-Lite plotting code,
    building and executing Jupyter notebooks, parsing results, and
    generating executable code for rendering Vega-Lite visualizations.

    Dependencies:
        - vl_convert must be installed for Vega-Lite rendering.
        - Input DataFrame must contain the columns: 'id' and 'code' (Vega-Lite specification).
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Vega-Lite handler.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config=config or {})
        if vlc is None:
            print("Warning: vl_convert not available, Vega-Lite rendering may fail")

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract Vega-Lite plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted Vega-Lite code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['vegalite', 'json'],
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
        Build and execute a Jupyter notebook to generate Vega-Lite plots.

        Args:
            dataset (pd.DataFrame): The dataset containing code snippets.
            output_path (Path): Path to save the generated notebook.
            csv_folder (Path): Path to the folder containing CSV files.

        Raises:
            CellExecutionError: If the notebook execution fails.
        """
        cells = dataset.apply(
            self.generate_code, axis=1, args=(csv_folder,)
        ).tolist()
        
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
        
        print(f"Vega-Lite plots notebook saved to {output_path}")

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

    def generate_code(self, item: pd.Series, csv_folder: Path) -> str:
        """
        Generate executable code for rendering Vega-Lite plots.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.
            csv_folder (Path): Path to the folder containing CSV files.

        Returns:
            str: The generated code block.
        """
        item_id = item['id']
        csv_path = csv_folder / f"{item_id}.csv"
        data_path_unix = str(PurePosixPath(csv_path))
        vegalite_spec = deduplicate_lines(item["code"], max_repeat=5)

        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import json",
            "import pandas as pd", 
            "from pathlib import Path",
            "import vl_convert as vlc",
            "from io import BytesIO",
            "from PIL import Image",
            "import random",
            "from wurlitzer import pipes",
            "",
            """
def is_list_of_value_dicts(data):
    if not isinstance(data, list) or len(data) == 0:
        return False
    for item in data:
        if not isinstance(item, dict) or 'value' not in item or len(item) != 1:
            return False
    return True

def load_csv_data(data_config, csv_path):
    if isinstance(data_config, dict) and "url" in data_config:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            csv_dict = df.to_dict(orient="records")
            if is_list_of_value_dicts(csv_dict):
                return {"values": [item['value'] for item in csv_dict]}
            if len(csv_dict) > 0 and 'start' in csv_dict[0] and 'stop' in csv_dict[0] and 'step' in csv_dict[0]:
                return {"sequence": csv_dict[0]}
            else:
                return {"values": csv_dict}
    return data_config

def render_vegalite(vegalite_json, csv_path):
    if "spec" in vegalite_json:
        actual_data = load_csv_data(vegalite_json["spec"].get("data", {}), csv_path)
        vegalite_json["spec"]["data"] = actual_data
    else:
        actual_data = load_csv_data(vegalite_json.get("data", {}), csv_path)
        vegalite_json["data"] = actual_data
    with pipes() as (out, err):
        png_data = vlc.vegalite_to_png(vl_spec=vegalite_json, scale=random.choice([1.5, 2, 2.5, 3]))
    log = err.read()
    if "ERROR" in log.upper():
        raise RuntimeError(f"{log}")
    img_buffer = BytesIO(png_data)
    return Image.open(img_buffer)
""",
            "",
            f"csv_path = Path('{data_path_unix}')",
            "",
            "# Vega-Lite specification",
            f"vegalite_spec = '''{vegalite_spec}'''",
            "",
            "spec_dict = json.loads(vegalite_spec)",
            "img = render_vegalite(spec_dict, csv_path)",
            "display(img)"
        ])
        
        return "\n".join(code_blocks)