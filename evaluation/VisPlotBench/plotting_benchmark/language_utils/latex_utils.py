import json
import re
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

def extract_latex_errors(log: str, max_lines: int = 20) -> str:
    """
    Extract relevant LaTeX error messages from the log.

    Args:
        log (str): The LaTeX log file content.
        max_lines (int): Maximum number of lines to extract if no specific errors are found.

    Returns:
        str: Extracted error messages or the first few lines of the log.
    """
    lines = log.splitlines()
    extracted = []
    capture_next = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or "Error" in stripped or "Warning" in stripped:
            extracted.append(stripped)
            capture_next = True
        elif capture_next and stripped.startswith("l."):
            extracted.append(stripped)
            capture_next = False
    if not extracted:
        return "\n".join(lines[:max_lines])
    return "\n".join(extracted[:max_lines])

class LaTexHandler(LanguageHandler):
    """Handler for processing LaTeX code."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LaTeX handler.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config=config or {})
    
    def extract_plotting_code(self, response: str) -> str:
        """
        Extract LaTeX plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted LaTeX code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['latex', 'json'],
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
            timeout=120,
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
        
        print(f"LaTeX plots notebook saved to {output_path}")
    
    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse the results of a Jupyter notebook containing plot outputs.

        Args:
            plots_path (Path): Path to the notebook file.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed results, including:
                - id: Identifier for the code snippet.
                - error: Extracted LaTeX error messages, if any.
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
                    cell_res["error"] = extract_latex_errors(cell_res["error"], max_lines=20)
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

    @staticmethod
    def _insert_csv_into_latex(csv_file_path, latex_source):
        """
        Insert CSV data into a LaTeX source string.

        Args:
            csv_file_path (str): Path to the CSV file.
            latex_source (str): The LaTeX source code.

        Returns:
            str: Modified LaTeX source code with the CSV data inserted.
        """
        df = pd.read_csv(csv_file_path)
        lines = [','.join(df.columns)]
        for _, row in df.iterrows():
            lines.append(','.join(str(x) for x in row))
        csv_string = '\n'.join(lines)
        datatable_definition = f"""\\pgfplotstableread[col sep=comma]{{
    {csv_string}
    }}\\datatable\n
    """
        document_pattern = r'(\\begin\{document\})'
        match = re.search(document_pattern, latex_source)
        if match:
            insert_position = match.start()
            modified_latex = (latex_source[:insert_position]+ '\n' + 
                            datatable_definition + 
                            latex_source[insert_position:])
            return modified_latex
        else:
            return  "\\usepackage{pgfplots}" + datatable_definition  + latex_source   
            
    def generate_code(self, item: pd.Series, csv_folder: Path) -> str:
        """
        Generate executable code for rendering LaTeX plots.

        Args:
            item (pd.Series): A row from the dataset containing code and metadata.
            csv_folder (Path): Path to the folder containing CSV files.

        Returns:
            str: The generated code block.
        """
        item_id = item["id"]
        csv_path = csv_folder / f"{item_id}.csv"

        latex_src = deduplicate_lines(item["code"], max_repeat=5)
        latex_src = latex_src.replace("\\pgfplotstableread{latex.csv}\\datatable", "")
        latex_src = self._insert_csv_into_latex(csv_path, latex_src)

        code_blocks = [f"# id = {item_id}"]
        code_blocks.extend([
            "import os",
            "import tempfile",
            "import subprocess",
            "from shutil import rmtree",
            "import random",
            "import pandas as pd",
            "import re",
            "from PIL import Image, ImageOps",
            "from pdf2image import convert_from_bytes",
            "import locale",
            "",
            r'''def crop_whitespace(image):
    inverted = ImageOps.invert(image.convert("RGB"))
    gray = inverted.convert("L")
    bbox = gray.getbbox()
    if bbox:
        buf = random.randint(50, 100)
        l = max(0, bbox[0] - buf)
        u = max(0, bbox[1] - buf)
        r = min(image.width, bbox[2] + buf)
        b = min(image.height, bbox[3] + buf)
        return image.crop((l, u, r, b))
    return image

def render_latex(latex_source: str, item_id: str):
    def compile_latex(compiler, tex_file, temp_dir):
        proc = subprocess.Popen(
            [compiler, "-interaction=nonstopmode", "-output-directory", temp_dir, tex_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return 1, b"Compilation timed out"
        return proc.returncode, out

    doc_variants = []
    m = re.search(r"\\documentclass(\[.*?\])?\{(standalone|article)\}", latex_source)
    if m:
        current = m.group(2)  # standalone or article
        if current == "standalone":
            doc_variants = [ latex_source, re.sub(r"\\documentclass(\[.*?\])?\{standalone\}", r"\\documentclass{article}", latex_source)]
        else:
            doc_variants = [latex_source, re.sub(r"\\documentclass(\[.*?\])?\{article\}", r"\\documentclass{standalone}", latex_source)]
    else:
        doc_variants = [latex_source]

    temp_dir = tempfile.mkdtemp()
    try:
        last_out = None
        for variant_idx, variant_src in enumerate(doc_variants):
            tex_file = os.path.join(temp_dir, f"{item_id}_{variant_idx}.tex")
            pdf_file = os.path.join(temp_dir, f"{item_id}_{variant_idx}.pdf")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(variant_src)

            for compiler in ["lualatex", "xelatex", "pdflatex"]:
                rc, out = compile_latex(compiler, tex_file, temp_dir)
                if rc == 0 and os.path.exists(pdf_file):
                    with open(pdf_file, "rb") as f:
                        pdf_bytes = f.read()
                    images = convert_from_bytes(pdf_bytes)
                    if images:
                        return crop_whitespace(images[0])
                else:
                    last_out = out

        if last_out is not None:
            err = last_out.decode("utf-8", errors="ignore") if isinstance(last_out, (bytes, bytearray)) else str(last_out)
            raise RuntimeError(f"LaTeX rendering failed:\n{err}")
        else:
            raise RuntimeError("LaTeX rendering failed: no compiler output captured.")
    finally:
        rmtree(temp_dir, ignore_errors=True)

os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"
try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except locale.Error:
    locale.setlocale(locale.LC_ALL, "C")
'''.strip(),
        "",
        f"item_id = '''{item_id}'''",
        f"latex_code = r'''{latex_src}'''",
        "img = render_latex(latex_code, item_id)",
        "display(img)"
        ])

        return "\n".join(code_blocks)
