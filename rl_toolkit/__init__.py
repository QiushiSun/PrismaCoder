"""PrismaCoder RL toolkit.

A collection of pure-Python RL verifiers organized by code-generation
modality. Each modality lives in its own subpackage; verifiers share a
common base under :mod:`rl_toolkit.plotting_verifier_base` where it
makes sense.

Folder naming convention: each subpackage encodes both its modality and
its input-condition mode (e.g. ``_unimodal`` = text prompt only, no
reference image). Multimodal variants (text + image prompt) live in
sibling packages alongside these.

Currently available verifiers:

* ``rl_toolkit.python_visualization_unimodal`` — Python plotting code
  (matplotlib / seaborn / plotly / ...) from a text-only prompt.
* ``rl_toolkit.svg_unimodal``  — SVG markup from a text-only prompt.
* ``rl_toolkit.html_unimodal`` — HTML + JS plotting pages from a
  text-only prompt.

See ``README.md`` for design, installation, and use cases.
"""
