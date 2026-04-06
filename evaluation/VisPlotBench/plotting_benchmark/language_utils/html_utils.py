import csv
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

def csv_to_list_of_dicts(file_path: str) -> list[dict]:
    """
    Convert a CSV file to a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list[dict]: A list of dictionaries representing the CSV data.
    """
    data = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(dict(row))  # Convert each row to a dictionary and append to the list.
    return data

class HTMLHandler(LanguageHandler):
    """Handler for processing HTML code."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config or {})

    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse the results of a Jupyter notebook containing plot outputs.

        Args:
            plots_path (Path): Path to the notebook file.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed results with columns:
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
                    cell_res["error"] = err  # Extract and clean error messages.
                elif output.output_type == "display_data" and "image/png" in output.data:
                    b64_png = output.data["image/png"]
                    b64_png = shrink_png_b64_to_under_kb(b64_png, max_kb=200, step=0.9, min_side=64)
                    images.append(b64_png)  # Append resized image data.

            cell_res["plots_generated"] = images
            cell_res["has_plot"] = len(images) > 0
            plot_results.append(cell_res)

        return pd.DataFrame(plot_results)

    def extract_plotting_code(self, response: str) -> str:
        """
        Extract HTML plotting code from a response string.

        Args:
            response (str): The response string containing the code.

        Returns:
            str: The extracted HTML code.
        """
        code = extract_fenced_code(
            response,
            language_tags=['html', 'json'],
            remove_stop_tokens=True
        )
        try:
            obj = json.loads(code)
            code = json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
        
        return code.strip()
    
    def _setup_code(self) -> str:
        """Code for the notebook setup cell: imports, helpers, and shared browser launch."""
        return '''
import os, re, json, asyncio
import nest_asyncio
from io import BytesIO
from PIL import Image, ImageOps
from playwright.async_api import async_playwright

nest_asyncio.apply()

def compute_major_px_ratio(image):
    px_count = {}
    for px in image.getdata():
        if px in px_count:
            px_count[px] += 1
        else:
            px_count[px] = 1
    max_px = max(px_count, key=px_count.get)
    return px_count[max_px] / len(image.getdata())

def process_image(image, max_size=(2560, 1440), major_px_threshold=0.98):
    if image.size[0] * image.size[1] > max_size[0] * max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")
    if compute_major_px_ratio(image) >= major_px_threshold:
        raise ValueError(f"Image rejected: dominant pixel ratio >= {major_px_threshold}")
    return Image.frombytes("RGB", image.size, image.tobytes())

def extract_html_width(html):
    for pat in [r'max-width:\\s*(\\d+)px', r'width:\\s*(\\d+)px']:
        m = re.search(pat, html)
        if m:
            return int(m.group(1))
    return None

def extract_html_height(html):
    m = re.search(r'height:\\s*(\\d+)px', html)
    return int(m.group(1)) if m else None

# ── shared browser (launched once per shard) ──
_pw_instance = asyncio.get_event_loop().run_until_complete(async_playwright().start())
_browser = asyncio.get_event_loop().run_until_complete(_pw_instance.chromium.launch())

async def render_html_async(html, full_page=True):
    html = html.replace("initial-scale=1.0", "initial-scale=2.0")
    width = extract_html_width(html) or 800
    height = extract_html_height(html) or (600 if width <= 800 else 800)

    page = await _browser.new_page(viewport={"width": width, "height": height})
    try:
        console_errors, page_errors = [], []
        def _on_console(msg):
            if msg.type == "error" and "WARNING" not in msg.text.upper():
                console_errors.append(f"Console {msg.type}: {msg.text}")
        page.on("console", _on_console)
        page.on("pageerror", lambda e: page_errors.append(f"Page Error: {e}"))
        page.on("requestfailed", lambda r: page_errors.append(f"Request failed: {r.url}"))

        await page.set_content(html)
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
        except Exception:
            pass
        await page.wait_for_timeout(2000)

        screenshot_bytes = await page.screenshot(full_page=full_page)
        all_errors = console_errors + page_errors
        if all_errors:
            raise Exception("HTML rendering errors found:\\n" + "\\n".join(all_errors))
        return Image.open(BytesIO(screenshot_bytes))
    finally:
        await page.close()

def render_html(html, full_page=True):
    return asyncio.get_event_loop().run_until_complete(render_html_async(html, full_page))

print("Browser ready")
'''.strip()

    @staticmethod
    def _teardown_code() -> str:
        """Code for the notebook teardown cell: close browser."""
        return '''
asyncio.get_event_loop().run_until_complete(_browser.close())
asyncio.get_event_loop().run_until_complete(_pw_instance.stop())
print("Browser closed")
'''.strip()

    def build_plots(self, dataset: pd.DataFrame, output_path: Path, csv_folder: Path) -> None:
        """
        Build and execute a Jupyter notebook to generate plots.

        The notebook structure is: [setup] [sample_1] ... [sample_N] [teardown].
        A single Chromium browser is launched in setup and reused for all samples,
        avoiding the ~1-2 s startup cost per cell.
        """
        sample_cells = dataset.apply(
            self.generate_code, axis=1, args=(csv_folder,)
        ).tolist()

        all_cells = [self._setup_code()] + sample_cells + [self._teardown_code()]
        self.build_new_nb(all_cells, output_path)

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

        print(f"HTML plots notebook saved to {output_path}")

    def build_new_nb(self, blocks: list, plots_nb_path: Path) -> None:
        """Save code blocks to a Jupyter notebook."""
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_code_cell(block) for block in blocks]
        with open(plots_nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

    def generate_code(self, item: pd.Series, csv_folder: Path) -> str:
        """Generate a per-sample cell that renders HTML using the shared browser."""
        item_id = item["id"]
        csv_path = csv_folder / f"{item_id}.csv"
        deduped_code = deduplicate_lines(item["code"], max_repeat=5)
        html_src = deduped_code.strip()

        csv_list_dict = csv_to_list_of_dicts(csv_path)
        json_csv_str = json.dumps(csv_list_dict)
        data_str = "const data = " + json_csv_str.encode("unicode_escape").decode("utf-8")
        html_src = re.sub(r"const\s+data\s*=\s*\[\{html\.csv\}\]\s*;?", data_str, html_src)

        return "\n".join([
            f"# id = {item_id}",
            f"html_code = r'''{html_src}'''",
            "img = render_html(html_code)",
            "img = process_image(img)",
            "display(img)",
        ])
