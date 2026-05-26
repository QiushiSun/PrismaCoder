"""PrismaCoder VisPlotBench RL verifiers — single-modal Python / HTML / SVG.

Per-language RL verifier classes that wrap the VisPlotBench evaluation
pipeline (under ``PrismaCoder/evaluation/VisPlotBench/``) into pure-
Python callables returning per-sample rewards + aggregate metrics for
RL training.

Quick start::

    from visplotbench import PythonRLVerifier, PythonVerifierConfig

    cfg = PythonVerifierConfig(
        work_dir="/tmp/rl/step_0/python",
        run_generation=False,                            # rollouts come from outside
        generations_inline=rollouts,                     # [{"id":..., "response_text":...}]
        prepared_samples_path="cache/python/prepared.jsonl",
        csv_dir="cache/python/csv",
        eval_base_url="http://localhost:8001/v1",         # local judge
        eval_api_key="EMPTY",
        eval_model="my-local-judge",
    )
    result = await PythonRLVerifier(cfg).run_async()
    reward_per_id = {r["id"]: r["reward"] for r in result["per_sample"]}

Use ``HTMLRLVerifier`` / ``HTMLVerifierConfig`` and ``SVGRLVerifier`` /
``SVGVerifierConfig`` analogously. See README.md for the full design,
installation, and use cases.
"""

from .html_verifier import HTMLRLVerifier, HTMLVerifierConfig
from .python_verifier import PythonRLVerifier, PythonVerifierConfig
from .svg_verifier import SVGRLVerifier, SVGVerifierConfig
from .verifier_base import PlottingRLVerifier, PlottingRLVerifierConfig

__all__ = [
    "PlottingRLVerifier",
    "PlottingRLVerifierConfig",
    "PythonRLVerifier",
    "PythonVerifierConfig",
    "HTMLRLVerifier",
    "HTMLVerifierConfig",
    "SVGRLVerifier",
    "SVGVerifierConfig",
]
