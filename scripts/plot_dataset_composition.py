"""Render the README dataset chart from the paper's rounded sample counts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/data/prismacoder-2m.json"
OUTPUT = ROOT / "assets/prismacoder-2m-composition"

COLORS = {
    "Vision-centric": [
        "#4269CB", "#5684D9", "#559DDF", "#5BB4DC",
        "#73C5DE", "#92D2E2", "#B4E1EB",
    ],
    "Text-centric": [
        "#6751BA", "#7B61C9", "#9277D8", "#A38BDF",
        "#B59BE6", "#BCACE9", "#C4B9ED", "#CAA9DF",
        "#D7B9E8", "#DECAED", "#E7DDF3",
    ],
}
INK = "#17244F"
MUTED = "#65708B"
PAPER = "#FCFCFF"


def main():
    rows = json.loads(DATA.read_text())["rows"]
    groups = list(COLORS)
    by_group = {group: [r for r in rows if r["modality"] == group] for group in groups}
    totals = [sum(r["samples"] for r in by_group[group]) for group in groups]
    total = sum(totals)

    # Sans-serif typography follows the README cover and project page.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": INK,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "savefig.facecolor": PAPER,
    })
    fig = plt.figure(figsize=(13.2, 6.4), facecolor=PAPER)
    fig.text(0.045, 0.925, "PrismaCoder-2M", fontsize=27, weight="bold")
    fig.text(0.045, 0.872, "Composition by input modality and task", fontsize=15, color=MUTED)

    ax = fig.add_axes([0.305, 0.115, 0.39, 0.69])
    ax.set_aspect("equal")
    counts = [row["samples"] for group in groups for row in by_group[group]]
    colors = [color for group in groups for color in COLORS[group]]
    ax.pie(
        counts, radius=1.0, startangle=90, counterclock=False, colors=colors,
        wedgeprops={"width": 0.255, "edgecolor": PAPER, "linewidth": 1.7},
    )
    ax.pie(
        totals, radius=0.705, startangle=90, counterclock=False,
        colors=["#5684D9", "#9277D8"],
        wedgeprops={"width": 0.16, "edgecolor": PAPER, "linewidth": 2},
    )
    ax.add_patch(Circle((0, 0), 0.49, color="#F0F1FC", linewidth=0))
    ax.text(0, 0.09, "≈2M", fontsize=34, weight="bold", ha="center", va="center")
    ax.text(0, -0.14, "training samples", fontsize=13.5, color=MUTED, ha="center")
    ax.set_axis_off()

    # Fixed columns keep the smallest categories legible without changing their areas.
    for group, x_left, x_right, y_start in [
        (groups[0], 0.045, 0.298, 0.625),
        (groups[1], 0.715, 0.965, 0.625),
    ]:
        group_total = sum(r["samples"] for r in by_group[group])
        fig.text(x_left, 0.765, group, fontsize=19.5, weight="bold")
        fig.text(
            x_left, 0.712, f"{group_total / total:.1%} of the corpus",
            fontsize=14.5, color=COLORS[group][1],
        )
        fig.text(x_right, 0.667, "Samples", fontsize=11, color=MUTED, ha="right")
        for i, (row, color) in enumerate(zip(by_group[group], COLORS[group])):
            y = y_start - i * 0.047
            fig.add_artist(Rectangle(
                (x_left, y - 0.006), 0.0105, 0.021,
                transform=fig.transFigure, facecolor=color, edgecolor="none",
            ))
            fig.text(x_left + 0.018, y, row["task"], fontsize=12.8, va="center")
            fig.text(
                x_right, y, f"{row['samples'] / 1000:.1f}K",
                fontsize=12.8, color=MUTED, ha="right", va="center",
            )

    for extension in ("svg", "png"):
        fig.savefig(OUTPUT.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    print(f"Saved {OUTPUT.name}.svg and .png from {len(rows)} table rows.")


if __name__ == "__main__":
    main()
