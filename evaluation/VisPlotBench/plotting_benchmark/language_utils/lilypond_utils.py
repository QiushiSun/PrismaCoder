import os
import re
import json
import base64
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

class LilyPondHandler(LanguageHandler):
    """
    Handler for processing LilyPond code.

    This class adapts the LilyPond rendering process to the evaluation framework.
    It provides methods for extracting plotting code, building and executing
    notebooks, parsing results, and generating rendering code.

    Dependencies:
        - LilyPond executable must be installed on the system or specified in the config.
        - Input DataFrame must contain the columns: 'id' and 'code' (LilyPond source code).
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LilyPond handler.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config=config or {})

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract LilyPond plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted LilyPond code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['lilypond', 'json'],
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
        Build and execute a Jupyter notebook to generate plots.

        Args:
            dataset (pd.DataFrame): The dataset containing code snippets.
            output_path (Path): Path to save the generated notebook.
            csv_folder (Path): Path to the folder containing CSV files (not used for LilyPond).
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

        print(f"LilyPond plots notebook saved to {output_path}")

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
        Generate executable code for rendering LilyPond plots.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.

        Returns:
            str: The generated code block.
        """
        item_id = str(item["id"])
        lilypond_spec = deduplicate_lines(item["code"], max_repeat=5)

        # Normalize models that emit every LilyPond command with a doubled
        # backslash (``\\version`` instead of ``\version``) -- an artifact of
        # training on JSON-escaped sources. This collapses a doubled backslash
        # ONLY when it precedes a letter (a command name), leaving the
        # legitimate LilyPond ``\\`` voice-separator token (used in
        # ``<< { .. } \\ { .. } >>``) intact, and is a no-op for models that
        # already emit single backslashes.
        #
        # ON by default: the previous ``'''...'''`` embedding silently halved
        # these doubled backslashes when Python parsed the (non-raw) string
        # literal, so historical scores were already computed under this
        # leniency. Keeping it ON preserves backward-compatible numbers; the
        # base64 embedding below no longer does any implicit unescaping. Set
        # config key ``lilypond_normalize_backslashes`` / env var
        # ``LILYPOND_NORMALIZE_BACKSLASHES`` to 0 for strict mode.
        if self._normalize_backslashes():
            lilypond_spec = re.sub(r'\\\\(?=[A-Za-z])', r'\\', lilypond_spec)

        # Embed the source via base64 instead of interpolating it into a Python
        # ``'''...'''`` literal. LilyPond is dense with single quotes (octave
        # marks) and has single-letter note names (b/f/r), so raw interpolation
        # produced accidental ``b'''`` / ``f'''`` / ``r'''`` bytes-literal
        # prefixes and quote-imbalance SyntaxErrors that killed otherwise-valid
        # scores in the Python cell before LilyPond was ever invoked.
        spec_b64 = base64.b64encode(lilypond_spec.encode("utf-8")).decode("ascii")
        id_b64 = base64.b64encode(item_id.encode("utf-8")).decode("ascii")

        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import os",
            "import base64",
            "import tempfile",
            "import subprocess",
            "import random",
            "from io import BytesIO",
            "from PIL import Image, ImageOps",
            "from IPython.display import display",
            "",
            "Image.MAX_IMAGE_PIXELS = None",
            "random.seed(42)",
            "",
            '''
    def crop_background(image):
        bg_color = image.getpixel((0, 0))
        image = image.convert("RGB")
        mask = Image.new('L', image.size, 0)
        for x in range(image.width):
            for y in range(image.height):
                if image.getpixel((x, y)) == bg_color:
                    mask.putpixel((x, y), 255)
        mask = ImageOps.invert(mask)
        bbox = mask.getbbox()
        if bbox:
            buffer_size = random.randint(50, 100)
            left = max(0, bbox[0] - buffer_size)
            upper = max(0, bbox[1] - buffer_size)
            right = min(image.width, bbox[2] + buffer_size)
            lower = min(image.height, bbox[3] + buffer_size)
            return image.crop((left, upper, right, lower))
        return image
    '''.strip(),
            "",
            '''
    def render_lilypond(lilypond_code: str, item_id: str):
        temp_dir = tempfile.mkdtemp()
        input_ly = os.path.join(temp_dir, f"{item_id}.ly")
        output_prefix = os.path.join(temp_dir, f"{item_id}")
        output_png = output_prefix + ".png"
        try:
            with open(input_ly, "w", encoding="utf-8") as f:
                f.write(lilypond_code)

            cmd = ["lilypond", "--png", "-dno-point-and-click", "-o", output_prefix, input_ly]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"{err}")

            img = Image.open(output_png)
            img.load()
            return crop_background(img)
        finally:
            for p in [input_ly, output_png]:
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
            f"item_id = base64.b64decode('{id_b64}').decode('utf-8')",
            f"lilypond_code = base64.b64decode('{spec_b64}').decode('utf-8')",
            "img = render_lilypond(lilypond_code, item_id)",
            "display(img)",
        ])

        return "\n".join(code_blocks)

    def _normalize_backslashes(self) -> bool:
        """Whether to collapse doubled command backslashes (see generate_code)."""
        env = os.environ.get("LILYPOND_NORMALIZE_BACKSLASHES")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        cfg = self.config or {}
        return bool(cfg.get("lilypond_normalize_backslashes", True))

