"""Python-language RL verifier for VisPlotBench.

Thin subclass of ``PlottingRLVerifier``: forces ``language='python'``,
enables CSV payload export (the python handler reads ``data.csv`` from
the configured csv folder), and exposes the optional subprocess
execution mode used to dodge ipykernel hangs on heavier matplotlib code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .verifier_base import PlottingRLVerifier, PlottingRLVerifierConfig


@dataclass
class PythonVerifierConfig(PlottingRLVerifierConfig):
    language: str = "python"

    # Execution backend selection. ``jupyter`` is the legacy nbconvert path;
    # ``subprocess`` runs every sample in an isolated python process,
    # noticeably more robust for SFT'd policies that emit heavy matplotlib
    # code (avoids ipykernel hangs eating valid samples).
    python_exec_mode: str = "jupyter"
    python_subproc_timeout: float = 60.0
    python_subproc_parallel: int = 1


class PythonRLVerifier(PlottingRLVerifier):
    language = "python"
    csv_export_enabled = True

    @classmethod
    def _config_class(cls) -> type[PythonVerifierConfig]:
        return PythonVerifierConfig

    def _handler_config(self) -> dict[str, Any]:
        cfg: PythonVerifierConfig = self.config  # type: ignore[assignment]
        if str(cfg.python_exec_mode).lower() != "subprocess":
            return {}
        return {
            "python_exec_mode": "subprocess",
            "python_subproc_timeout": float(cfg.python_subproc_timeout),
            "python_subproc_parallel": int(cfg.python_subproc_parallel),
        }


__all__ = ["PythonRLVerifier", "PythonVerifierConfig"]
