import json
import subprocess
from pathlib import Path, PurePosixPath

import nbformat as nbf
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import nbformat
import re
import os

def truncate_error_message(error: str) -> str:
    head = 1000
    tail = 1000
    if len(error) <= head + tail:
        return error
    return error[:head] + "\n...\n".format(len(error) - head - tail) + error[-tail:]

def read_jsonl(file_path: str | Path) -> list[dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path: str | Path) -> None:
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

def read_responses(
    responses_file: str | Path | None = None, responses: list[dict] | None = None
) -> dict:
    if responses_file is None and responses is None:
        raise ValueError("Either response_file or responses must be provided.")

    if responses_file is not None and responses is not None:
        print(
            "Both responses file and responses list provided. Responses list would be used."
        )

    if responses is None:
        responses = read_jsonl(responses_file)

    responses_dict = dict()

    for entry in responses:
        if "id" in entry:
            responses_dict[entry["id"]] = entry

    return responses_dict

def add_index_to_filename(
    folder: str, filename: str, postfix: str = ""
) -> tuple[Path, Path | None]:
    results_file_base = Path(folder) / filename
    results_file_base = results_file_base.with_stem(results_file_base.stem + postfix)
    results_file = results_file_base.with_stem(results_file_base.stem + "_0")

    i = 0
    last_existing_file = None
    while results_file.exists():
        last_existing_file = results_file
        i += 1
        results_file = results_file_base.with_stem(results_file_base.stem + f"_{i}")

    return results_file, last_existing_file

class VisGenerator:
    """
    Object that runs generated code to build plots.
    At init pass:

    dataset: dataset
    output_file: output file to save results.
    The output is ammendment of the LLM responses logs. So, you can pass same path.
    temp_dir: dir for notebook used for plotting plots.
    """

    def __init__(
        self,
        output_folder: str | Path,
        dataset: Dataset,
        csv_folder: str | Path,
        config: DictConfig | None = None,
        language_handler = None,
    ) -> None:
        self.output_folder = Path(output_folder)
        self.plots_nb_path = self.output_folder / "all_plots.ipynb"
        self.config = config

        self.csv_folder = None if self.config.plotting_language == "mermaid" or \
        self.config.plotting_language == "lilypond" or self.config.plotting_language == "svg" \
        or self.config.plotting_language == "asymptote" else Path(csv_folder)

        # self.csv_folder = Path(csv_folder)
        self.check_csv(dataset, csv_folder)
        if self.config.run_mode == "self_debug":
            self.debug_plots_nb_path = self.config.debug.output_dir

        self.language_handler = language_handler

    def check_csv(self, dataset: Dataset, csv_folder: str) -> None:
        if csv_folder is None:
            print("[IFO] CSV folder is not needed.")
            return
        for item in dataset:
            csv_path = self.csv_folder / f"{item['id']}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Unpacked csv datafile not found on {csv_path}"
                )

    def draw_self_debug_plots(
        self,
        dataset: pd.DataFrame,
        attempt_id: int,
    ) -> pd.DataFrame:
        model_name = dataset["model"].iloc[0].replace("/", "__")
        data_descriptor = dataset["data_descriptor"].iloc[0]
        plotting_language = self.config.plotting_language.split(" ")[0]
        os.makedirs(self.debug_plots_nb_path, exist_ok=True)
        self.plots_nb_path, _ = add_index_to_filename(
            self.debug_plots_nb_path, f"self_debug_attempt_{attempt_id}_{data_descriptor}_{model_name}_{plotting_language}.ipynb"
        )

        self.language_handler.build_plots(dataset, self.plots_nb_path, self.csv_folder)
        response = self.language_handler.parse_plots_notebook(self.plots_nb_path)
        # Removing existing columns from dataset
        common_cols = dataset.columns.intersection(response.columns).drop("id")
        dataset = dataset.drop(columns=common_cols)
        dataset = dataset.merge(response, on="id", how="left")

        return dataset

    def draw_plots(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Test method for new language handler approach.
        Follows the same structure as draw_plots but uses language handlers.
        """
        model_name = dataset["model"].iloc[0].replace("/", "__")
        data_descriptor = dataset["data_descriptor"].iloc[0]
        plotting_language = self.config.plotting_language
        self.plots_nb_path, _ = add_index_to_filename(
            self.output_folder, f"plots_{data_descriptor}_{model_name}_{plotting_language}.ipynb"
        )
        
        # Use language handler to build and parse plots
        self.language_handler.build_plots(dataset, self.plots_nb_path, self.csv_folder)
        response = self.language_handler.parse_plots_notebook(self.plots_nb_path)
        
        # Removing existing columns from dataset (same as original)
        common_cols = dataset.columns.intersection(response.columns).drop("id")
        dataset = dataset.drop(columns=common_cols)
        dataset = dataset.merge(response, on="id", how="left")

        return dataset