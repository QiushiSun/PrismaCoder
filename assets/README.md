# README assets

## Dataset figures

`prismacoder-2m-composition.svg` and `.png` show dataset composition in the README's blue, violet, and cyan palette. The inner ring represents input modality; the outer ring represents its task groups. The sans-serif type matches the cover and project page.

The source counts in [`data/prismacoder-2m.json`](data/prismacoder-2m.json) were extracted from [Table 1, arXiv v1](https://arxiv.org/html/2610.20538v1#S4.T1) and checked against the [project page data](https://qiushisun.github.io/PrismaCoder-Home/static/js/prismacoder-data.js). Rounded counts sum to 2,004.1K: 1,190.0K vision-centric and 814.1K text-centric. The chart normalizes these counts, giving 59.4% and 40.6%; it does not sum the separately rounded percentages. These are paper statistics, not counts of currently uploaded Hugging Face records.

Regenerate both formats with the existing Matplotlib dependency:

```bash
python scripts/plot_dataset_composition.py
```

`prismacoder-task-overview.png` is the unmodified [task overview from the paper](https://arxiv.org/html/2610.20538v1/task_overview_figure.png), shown in the README's expandable task gallery.

## Cover

`prismacoder-cover.png` is the README banner (2172 × 724 pixels). It was composed with the built-in `image_gen` tool using the [project prism](https://qiushisun.github.io/PrismaCoder-Home/public/logo.png) as the identity reference and the [paper's task overview](https://arxiv.org/html/2610.20538v1/task_overview_figure.png) as visual context.

## Generation prompt

```text
Use case: compositing.
Asset type: a polished, minimal academic GitHub README cover, exactly 3:1 wide landscape, ideally 1920 × 640 pixels.
Primary request: create a professional PrismaCoder banner from the supplied project imagery.

Input image 1 (logo.png): the authoritative project identity, a transparent blue-violet-cyan triangular glass prism. Preserve the same distinctive triangular geometry, facet arrangement and blue-purple-cyan identity. Composite it cleanly; do not redesign the logo.
Input image 2 (task_overview_figure.png): context for the project's visual-code domain only. Do not reproduce its dense cards, captions, input panels, or whole collage. It is optional visual context, not a new research-results source.

Scene/backdrop: a very light off-white background with abundant clean negative space, a barely visible cool blue/lavender illumination near the prism, no visible border.
Composition: a restrained editorial masthead. A large legible title and one smaller subtitle form a clear text block in the left half; the recognizable prism is the single hero on the right half. The prism should be fully visible with generous breathing room, lightly grounded by a soft shadow. Keep all content comfortably within the banner. The center is open and uncluttered.
Style: crisp contemporary academic project branding; refined glass facets, gentle translucency and subtle cyan/violet light; calm and precise, not a glossy promotional poster.
Typography: clean, highly legible modern sans-serif with excellent spacing. Title in deep navy, subtitle in a softer blue-gray. Title visually dominant and easy to read even at GitHub README width.
Text, verbatim, with exact capitalization:
"PrismaCoder"
"Open Multimodal Code Intelligence"
The title must be spelled P-r-i-s-m-a-C-o-d-e-r, as one word. The subtitle appears once on one line if possible.
Constraints: only these two text strings. Keep this a simple banner, with no tiny explanatory content. Do not invent charts, research results, screenshots, statistics, code blocks, badges, extra logos, slogans, or cards. No watermark. Opaque light background. No black background. Aim for a clean final cover suitable for the top of an open-source research repository.
```
