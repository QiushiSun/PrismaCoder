# R Code Checker

Extract R code from JSONL `output` fields, render to PNG via `Rscript`, and split entries into `passed.jsonl` / `failed.jsonl`.

Supports **interactive R code** (Shiny, plotly, rgl) by automatically transforming it into static-renderable code.

## Prerequisites

### System dependencies

```bash
# R (v4.0+) must be installed
# macOS
brew install r

# Ubuntu/Debian
sudo apt-get install r-base r-base-dev
```

Verify installation:

```bash
Rscript --version
```

### R packages

A baseline set of R packages is needed. Install the most common ones:

```r
install.packages(c(
  "ggplot2", "dplyr", "tidyr", "purrr", "readr", "stringr",
  "forcats", "lubridate", "scales", "gridExtra",
  "reshape2", "viridis", "RColorBrewer", "ggrepel",
  "patchwork", "ggridges", "showtext", "ggtext",
  "plotly", "rgl", "webshot2", "chromote", "htmlwidgets"
), repos = "https://cloud.r-project.org")
```

For comprehensive rendering (to minimize `missing_package` failures), also install:

```r
install.packages(c(
  "Cairo", "GGally", "sf", "igraph", "forecast", "survival",
  "survminer", "vegan", "wordcloud", "palmerpenguins",
  "gapminder", "nycflights13", "babynames", "hexbin",
  "latticeExtra", "car", "psych", "randomForest"
), repos = "https://cloud.r-project.org")
```

### Python packages

No additional Python packages required beyond the standard library.

## Usage

### Full run

```bash
# Process a single JSONL file
python checker.py path/to/data.jsonl

# Process a directory of JSONL files
python checker.py path/to/jsonl_dir/

# Custom output directories and parallelism
python checker.py data.jsonl -o ./my_pngs -r ./my_results -j 16

# Increase timeout for heavy plots
python checker.py data.jsonl --timeout 60
```

### Re-run failed entries

After installing additional R packages or upgrading transforms, re-run only the failed entries:

```bash
python rerun_failed.py --result-dir ./results --output-dir ./png_output
```

This reads `results/failed.jsonl`, retries eligible entries, and merges results back.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./png_output` | Directory for rendered PNGs |
| `--result`, `-r` | `./results` | Directory for `passed.jsonl` / `failed.jsonl` |
| `--workers`, `-j` | `8` | Number of parallel render workers |
| `--timeout`, `-t` | `30` | Per-script render timeout in seconds |

## Input format

Each JSONL line must be a JSON object with an `output` field containing R code:

```json
{"uuid": "abc123", "output": "```r\nlibrary(ggplot2)\nggplot(mtcars, aes(mpg, hp)) + geom_point()\n```", ...}
```

## Interactive code handling

The checker automatically detects and transforms interactive R code:

| Type | Detection | Transform |
|------|-----------|-----------|
| **Shiny + plot** | `library(shiny)` + `renderPlot` | Strip Shiny framework, extract plot code, mock `input$` values |
| **Shiny (no plot)** | `library(shiny)` without plot code | Marked as `interactive_no_plot` (not renderable) |
| **plotly (ggplotly)** | `ggplotly(...)` calls | Remove `ggplotly()` wrapper, render underlying ggplot |
| **plotly (native)** | `plot_ly(...)` calls | Save widget to HTML, screenshot via `webshot2` |
| **rgl (3D)** | `library(rgl)` | Enable headless mode, capture via `rgl::snapshot3d()` |

## Output

- **`passed.jsonl`** — entries that rendered successfully, with added `png_path` and `r_extract_method` fields
- **`failed.jsonl`** — entries that failed, with added `failure_stage`, `failure_message`, and `failure_category` fields

### Failure categories

| Category | Description |
|----------|-------------|
| `interactive_no_plot` | Shiny UI/table code with no renderable chart |
| `missing_package` | R package not installed |
| `code_error` | Syntax or logic error in generated code |
| `timeout` | Render exceeded time limit |
| `interactive_*` | Interactive code transform failed |

## Quick trial

```bash
python checker.py sample_data/sample.jsonl -o /tmp/r_trial_png -r /tmp/r_trial_results
```

## File structure

```
r/
├── checker.py              # Main entry point
├── rerun_failed.py         # Re-run failed entries after upgrades
├── utils.py                # Extraction, transforms, rendering
├── sample_data/
│   └── sample.jsonl        # 8-line sample for trial runs
└── README.md
```
