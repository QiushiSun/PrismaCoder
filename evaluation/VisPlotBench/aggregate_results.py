import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter


def load_stats(stats_path: Path) -> dict[str, Any]:
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def compute_incorrect_code_percent(stats: dict[str, Any]) -> float:
    count = stats.get("count", 0)
    execution_error_num = stats.get("execution_error_num", 0)

    if not count:
        return 0.0

    return round(execution_error_num / count * 100, 2)


def collect_language_stats(base_folder: Path, experiment_id: str, language: str) -> dict[str, Any]:
    # runs/runs-{id}/runs-{id}-{lang}/stats.json  (new layout, base_folder = runs/runs-{id})
    in_run = base_folder / f"runs-{experiment_id}-{language}" / "stats.json"
    # runs-{id}-{lang}/stats.json  (legacy flat layout, base_folder = project root)
    flat = base_folder / f"runs-{experiment_id}-{language}" / "stats.json"
    stats_path = in_run if in_run.exists() else flat

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file (tried {in_run})")

    stats = load_stats(stats_path)

    return {
        "model": stats.get("model", ""),
        "language": stats.get("language", language),
        "count": stats.get("count", 0),
        "execution_error_num": stats.get("execution_error_num", 0),
        "incorrect_code_pct": compute_incorrect_code_percent(stats),
        "vis_mean": safe_get(stats, "scores", "vis", "mean", default=0),
        "task_mean": safe_get(stats, "scores", "task", "mean", default=0),
        "vis_good": safe_get(stats, "scores", "vis", "good", default=0),
        "task_good": safe_get(stats, "scores", "task", "good", default=0),
    }


def build_excel(
    output_path: Path,
    model_name: str,
    run_id: str,
    language_stats: dict[str, dict[str, Any]],
    languages: list[str],
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    bold_font = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")

    # Column A = Model, Column B = Run
    ws["A1"] = ""
    ws["A2"] = ""
    ws["A3"] = ""
    ws["A4"] = model_name

    ws["B1"] = ""
    ws["B2"] = ""
    ws["B3"] = ""
    ws["B4"] = run_id

    current_col = 3  # language data starts at column C

    for lang in languages:
        start_col = current_col
        end_col = current_col + 4

        ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)
        ws.cell(row=1, column=start_col, value=lang.capitalize())

        ws.merge_cells(start_row=2, start_column=start_col, end_row=3, end_column=start_col)
        ws.cell(row=2, column=start_col, value="Incorrect code, %")

        ws.merge_cells(start_row=2, start_column=start_col + 1, end_row=2, end_column=start_col + 2)
        ws.cell(row=2, column=start_col + 1, value="Mean Score")
        ws.cell(row=3, column=start_col + 1, value="Visual")
        ws.cell(row=3, column=start_col + 2, value="Task")

        ws.merge_cells(start_row=2, start_column=start_col + 3, end_row=2, end_column=start_col + 4)
        ws.cell(row=2, column=start_col + 3, value="Good (>=75)")
        ws.cell(row=3, column=start_col + 3, value="Visual")
        ws.cell(row=3, column=start_col + 4, value="Task")

        current_col += 5

    current_col = 3
    for lang in languages:
        stats = language_stats.get(lang)
        if stats is None:
            current_col += 5
            continue

        ws.cell(row=4, column=current_col, value=stats["incorrect_code_pct"])
        ws.cell(row=4, column=current_col + 1, value=stats["vis_mean"])
        ws.cell(row=4, column=current_col + 2, value=stats["task_mean"])
        ws.cell(row=4, column=current_col + 3, value=stats["vis_good"])
        ws.cell(row=4, column=current_col + 4, value=stats["task_good"])

        current_col += 5

    for row in range(1, 4):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = bold_font
            cell.alignment = center

    ws["A4"].alignment = Alignment(vertical="center")

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 14
    for col in range(3, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 16

    ws.freeze_panes = "C4"

    wb.save(output_path)


def _agg_header_columns(languages: list[str]) -> list[str]:
    """Build the flat column header list for the aggregated Excel."""
    cols = ["Model", "Run", "Timestamp"]
    for lang in languages:
        prefix = lang.capitalize()
        cols.extend([
            f"{prefix} Err%",
            f"{prefix} Vis Mean",
            f"{prefix} Task Mean",
            f"{prefix} Vis Good",
            f"{prefix} Task Good",
        ])
    return cols


def append_to_agg_excel(
    agg_path: Path,
    model_name: str,
    run_id: str,
    language_stats: dict[str, dict[str, Any]],
    languages: list[str],
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers = _agg_header_columns(languages)

    if agg_path.exists():
        wb = load_workbook(agg_path)
        ws = wb.active
        for merge in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(merge))
        existing_headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        if existing_headers != headers:
            for c, h in enumerate(headers, 1):
                ws.cell(row=1, column=c, value=h)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Aggregated"
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 14
        ws.column_dimensions["C"].width = 20
        for c in range(4, len(headers) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 16
        ws.freeze_panes = "D2"

    row_num = ws.max_row + 1
    ws.cell(row=row_num, column=1, value=model_name)
    ws.cell(row=row_num, column=2, value=run_id)
    ws.cell(row=row_num, column=3, value=timestamp)

    col = 4
    for lang in languages:
        st = language_stats.get(lang)
        if st is None:
            col += 5
            continue
        ws.cell(row=row_num, column=col, value=st["incorrect_code_pct"])
        ws.cell(row=row_num, column=col + 1, value=st["vis_mean"])
        ws.cell(row=row_num, column=col + 2, value=st["task_mean"])
        ws.cell(row=row_num, column=col + 3, value=st["vis_good"])
        ws.cell(row=row_num, column=col + 4, value=st["task_good"])
        col += 5

    wb.save(agg_path)


def append_to_agg_markdown(
    md_path: Path,
    model_name: str,
    run_id: str,
    language_stats: dict[str, dict[str, Any]],
    languages: list[str],
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    need_header = not md_path.exists() or md_path.stat().st_size == 0

    lang_cols = []
    for lang in languages:
        lang_cols.extend([
            f"{lang} Err%", f"{lang} Vis", f"{lang} Task",
        ])

    if need_header:
        header = "| Model | Run | Timestamp | " + " | ".join(lang_cols) + " |"
        sep = "| --- | --- | --- | " + " | ".join(["---:"] * len(lang_cols)) + " |"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Aggregated Results\n\n{header}\n{sep}\n")

    cells = [model_name, run_id, timestamp]
    for lang in languages:
        st = language_stats.get(lang)
        if st is None:
            cells.extend(["-", "-", "-"])
        else:
            cells.append(f"{st['incorrect_code_pct']:.1f}%")
            cells.append(f"{st['vis_mean']:.2f}" if st["vis_mean"] else "-")
            cells.append(f"{st['task_mean']:.2f}" if st["task_mean"] else "-")

    row = "| " + " | ".join(cells) + " |"
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def print_summary_table(
    run_id: str,
    model_name: str,
    judge_model: str,
    language_stats: dict[str, dict[str, Any]],
    languages: list[str],
) -> None:
    sep = "=" * 70
    print(sep)
    print(f"  Run: {run_id}  |  Model: {model_name}  |  Judge: {judge_model}")
    print(sep)
    header = f"  {'Language':<12} {'Samples':>7} {'OK':>5} {'Err%':>7} {'Vis Mean':>9} {'Task Mean':>10}"
    print(header)
    print(f"  {'-'*12} {'-'*7} {'-'*5} {'-'*7} {'-'*9} {'-'*10}")

    for lang in languages:
        st = language_stats.get(lang)
        if st is None:
            print(f"  {lang:<12}  (no data)")
            continue
        count = st.get("count", 0)
        errs = st.get("execution_error_num", 0)
        ok = count - errs
        err_pct = f"{st['incorrect_code_pct']:.1f}%"
        vis = f"{st['vis_mean']:.2f}" if st["vis_mean"] else "-"
        task = f"{st['task_mean']:.2f}" if st["task_mean"] else "-"
        print(f"  {lang:<12} {count:>7} {ok:>5} {err_pct:>7} {vis:>9} {task:>10}")

    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-folder",
        required=True,
        help="Base folder containing run directories.",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment ID used in folder names.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run label written to the Excel 'Run' column (e.g. 'runs-1').",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of language identifiers, e.g. python latex.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Excel file path. Default: aggregated_exp_<id>.xlsx",
    )
    parser.add_argument(
        "--agg-output",
        default=None,
        help="Path to the global aggregated Excel (append a row each run).",
    )
    parser.add_argument(
        "--agg-md",
        default=None,
        help="Path to the global aggregated Markdown table (append a row each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_folder = Path(args.base_folder).expanduser().resolve()
    experiment_id = str(args.experiment_id)
    run_id = args.run_id or f"runs-{experiment_id}"
    languages = args.languages
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"aggregated_exp_{experiment_id}.xlsx")
    )

    language_stats: dict[str, dict[str, Any]] = {}
    model_name = ""

    for language in languages:
        try:
            stats = collect_language_stats(base_folder, experiment_id, language)
            language_stats[language] = stats
            if not model_name:
                model_name = stats.get("model", "")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")

    if not language_stats:
        print("No stats files found — nothing to aggregate.")
        return

    build_excel(output_path, model_name, run_id, language_stats, languages)
    print(f"Excel file written to: {output_path}")

    if args.agg_output:
        agg_path = Path(args.agg_output)
        append_to_agg_excel(agg_path, model_name, run_id, language_stats, languages)
        print(f"Appended to aggregated Excel: {agg_path}")

    if args.agg_md:
        md_path = Path(args.agg_md)
        append_to_agg_markdown(md_path, model_name, run_id, language_stats, languages)
        print(f"Appended to aggregated Markdown: {md_path}")

    judge_model = "gpt-4.1"
    print()
    print_summary_table(run_id, model_name, judge_model, language_stats, languages)


if __name__ == "__main__":
    main()