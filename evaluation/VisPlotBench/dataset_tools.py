from __future__ import annotations

import base64
import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from plotting_benchmark.task_changer import TaskChanger


def _convert_plot_gt(example: dict[str, Any]) -> dict[str, Any]:
    if example.get("plots_gt"):
        return example
    if example.get("image") is not None:
        buf = BytesIO()
        example["image"].save(buf, format="PNG")
        example["plots_gt"] = [base64.b64encode(buf.getvalue()).decode("utf-8")]
        del example["image"]
    else:
        example["plots_gt"] = []
    return example


def load_instructs(repo_root: str | Path, language: str) -> dict[str, Any]:
    p = Path(repo_root) / "plotting_benchmark" / "resources" / "instructs.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data[language]


def load_prepared_dataset(
    *,
    repo_root: str | Path,
    language: str,
    data_descriptor: str = "head",
    dataset_name: str = "TIGER-Lab/VisPlotBench",
    split: str = "test",
    ids: list[int] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    ds = load_dataset(dataset_name, name=language, split=split)
    ds = ds.map(_convert_plot_gt)
    if ids is not None:
        ds = ds.select(ids)
    elif limit is not None:
        ds = ds.shuffle(seed=42).select(range(limit))

    df = ds.to_pandas()
    if "__index_level_0__" in df.columns:
        del df["__index_level_0__"]

    instructs = load_instructs(repo_root, language)
    changer = TaskChanger()
    changer.init_task_changer(
        data_descriptor_name=data_descriptor,
        data_instruct=instructs["data_instruct"],
        setup_instruct=instructs["setup_instruct"],
        plotting_language=language,
    )
    df = changer.change_task(df)
    return df


def export_csv_payloads(df: pd.DataFrame, csv_dir: str | Path, language: str) -> Path | None:
    csv_dir = Path(csv_dir)
    if language in {"mermaid", "lilypond", "svg", "asymptote"}:
        return None
    csv_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        csv_path = csv_dir / f"{row['id']}.csv"
        csv_path.write_text(row["data"], encoding="utf-8")
    return csv_dir


def prompt_from_row(row: pd.Series, plot_instruct: str) -> str:
    pieces = [plot_instruct]
    for key in [
        "task__setup",
        "task__data_description",
        "task__plot_description",
        "task__plot_style",
    ]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            pieces.append(val.strip())
    return "\n\n".join(pieces)


def row_to_portable_sample(row: pd.Series, plot_instruct: str) -> dict[str, Any]:
    out = row.to_dict()
    out["task"] = prompt_from_row(row, plot_instruct)
    return out
