"""Single-modal SVG markup RL verifier.

Returns per-sample scalar rewards for policy rollouts that produce raw
SVG source code; samples are rendered to PNG via cairosvg and judged
by the configured LLM judge.

See :mod:`rl_toolkit` README for the full design and use cases.
"""

from .verifier import SVGRLVerifier, SVGVerifierConfig

__all__ = ["SVGRLVerifier", "SVGVerifierConfig"]
