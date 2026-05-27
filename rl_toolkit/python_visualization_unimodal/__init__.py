"""Single-modal Python plotting code RL verifier.

Returns per-sample scalar rewards for policy rollouts that produce
matplotlib / seaborn / plotly / bokeh / ... visualisation code.

See :mod:`rl_toolkit` README for the full design and use cases.
"""

from .verifier import PythonRLVerifier, PythonVerifierConfig

__all__ = ["PythonRLVerifier", "PythonVerifierConfig"]
