"""Single-modal HTML plotting page RL verifier.

Returns per-sample scalar rewards for policy rollouts that produce a
full HTML page (typically with embedded JavaScript chart libraries);
samples are rendered to PNG via playwright/chromium and judged by the
configured LLM judge.

See :mod:`rl_toolkit` README for the full design and use cases.
"""

from .verifier import HTMLRLVerifier, HTMLVerifierConfig

__all__ = ["HTMLRLVerifier", "HTMLVerifierConfig"]
