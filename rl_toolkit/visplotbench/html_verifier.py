"""HTML-language RL verifier for VisPlotBench.

The HTML handler renders pages via a shared playwright/chromium browser
launched inside the execution notebook. If chromium is missing or
broken, every sample in the shard fails with a single generic ``Browser
setup failed`` message, which is unhelpful as an RL signal. This
subclass adds a preflight probe: we launch chromium once from the
parent process before any worker is spawned. On failure we short-
circuit the whole batch with the verbatim error string, so the trainer
can immediately tell that the env is broken — not that the policy
regressed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .verifier_base import PlottingRLVerifier, PlottingRLVerifierConfig


@dataclass
class HTMLVerifierConfig(PlottingRLVerifierConfig):
    language: str = "html"
    html_chromium_preflight: bool = True


class HTMLRLVerifier(PlottingRLVerifier):
    language = "html"
    # The HTML handler injects CSV payloads into the page (csv_to_list_of_dicts),
    # so CSV export is still required.
    csv_export_enabled = True

    @classmethod
    def _config_class(cls) -> type[HTMLVerifierConfig]:
        return HTMLVerifierConfig

    def _preflight(self) -> str | None:
        cfg: HTMLVerifierConfig = self.config  # type: ignore[assignment]
        if not cfg.html_chromium_preflight:
            return None
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:  # noqa: BLE001
            return (
                f"Browser setup failed: playwright import error ({e!r}). "
                "Fix: pip install playwright && playwright install chromium"
            )
        try:
            pw = sync_playwright().start()
            try:
                browser = pw.chromium.launch()
                browser.close()
            finally:
                pw.stop()
        except Exception as e:  # noqa: BLE001
            return (
                f"Browser setup failed: {e!r}. "
                "Fix: playwright install chromium"
            )
        return None


__all__ = ["HTMLRLVerifier", "HTMLVerifierConfig"]
