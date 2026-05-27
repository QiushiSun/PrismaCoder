"""SVG-language RL verifier.

SVG samples are self-contained (no external CSV payloads), and rendering
is a simple cairosvg call, so this subclass is almost a pure base
instance with ``language='svg'`` and CSV export disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..plotting_verifier_base import PlottingRLVerifier, PlottingRLVerifierConfig


@dataclass
class SVGVerifierConfig(PlottingRLVerifierConfig):
    language: str = "svg"


class SVGRLVerifier(PlottingRLVerifier):
    language = "svg"
    csv_export_enabled = False

    @classmethod
    def _config_class(cls) -> type[SVGVerifierConfig]:
        return SVGVerifierConfig


__all__ = ["SVGRLVerifier", "SVGVerifierConfig"]
