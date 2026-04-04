#!/usr/bin/env python3
"""
ChemDraw Evaluation Pipeline
=============================
Evaluates a VLM's ability to convert 2D chemical structure images into SMILES
(Simplified Molecular-Input Line-Entry System) strings.

Dataset : allenai/CoSyn-400K  (chemical subset, validation split, 128 samples)
Metrics : Exec Rate, Validity, Exact Match, Tanimoto Similarity (Morgan/ECFP4)

Usage:
    python eval_chemdraw.py --model <model> --api-key <key> --base-url <url>

    # quick smoke test
    python eval_chemdraw.py --model gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1 --subset 5

    # full evaluation with 8 threads
    python eval_chemdraw.py --model gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1 --max-workers 8
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "",
    "model_name": "",
    "data_dir": str(SCRIPT_DIR / "data"),
    "max_workers": 4,
    "max_retries": 3,
    "temperature": 0.0,
    "max_tokens": 512,
    "timeout": 60,
    "output_dir": str(SCRIPT_DIR / "results"),
}

SYSTEM_PROMPT = (
    "You are an expert chemist. Given an image of a chemical structure, "
    "your task is to output the corresponding SMILES (Simplified Molecular-Input "
    "Line-Entry System) representation. Output ONLY the SMILES string, "
    "with no explanation, no markdown formatting, and no additional text."
)

USER_PROMPT = (
    "Please provide the SMILES representation for the chemical structure "
    "shown in the image. Output ONLY the SMILES string."
)


# ─────────────────────────────────────────────────────────────────────────────
# SMILES utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_smiles(text: str) -> str:
    """Extract a SMILES string from model response text."""
    if not text:
        return ""
    text = text.strip()

    code_block = re.search(r"```(?:\w*)\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    inline_code = re.match(r"^`([^`]+)`$", text)
    if inline_code:
        text = inline_code.group(1).strip()

    smiles_pat = r"[A-Za-z\[\]()=#@/\\+\-.\d%]+"
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(w in line.lower() for w in
               ["the smiles", "representation", "answer", "here is", "output"]):
            continue
        m = re.fullmatch(smiles_pat, line)
        if m:
            return m.group(0)

    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return text


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def compute_tanimoto(sa: str, sb: str,
                     radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Tanimoto similarity via Morgan (ECFP4) fingerprints."""
    try:
        ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
        if ma is None or mb is None:
            return None
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, radius, nBits=n_bits)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, radius, nBits=n_bits)
        return DataStructs.TanimotoSimilarity(fa, fb)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Image encoding
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_base64_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/%s;base64,%s" % (fmt.lower(), b64)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str) -> List[Dict]:
    """Load the bundled 128-sample test set from manifest.jsonl + images/."""
    data_path = Path(data_dir)
    manifest = data_path / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            "manifest.jsonl not found in %s. "
            "Make sure the data/ directory is intact." % data_dir)

    samples = []
    with open(manifest) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = data_path / entry["image"]
            if not img_path.exists():
                print("  WARNING: missing image %s — skipping" % img_path)
                continue
            samples.append({
                "id": entry["id"],
                "code": entry["code"],
                "image": Image.open(str(img_path)).copy(),
            })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe OpenAI client pool
# ─────────────────────────────────────────────────────────────────────────────

_tls = threading.local()


def _get_client(config: dict) -> OpenAI:
    if not hasattr(_tls, "client") or _tls.client is None:
        _tls.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            http_client=httpx.Client(
                verify=False,
                timeout=httpx.Timeout(
                    connect=30.0, read=config["timeout"],
                    write=60.0, pool=30.0),
            ),
        )
    return _tls.client


# ─────────────────────────────────────────────────────────────────────────────
# Model inference (with retry)
# ─────────────────────────────────────────────────────────────────────────────

def query_model(image: Image.Image, config: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_url",
             "image_url": {"url": pil_to_base64_uri(image)}},
        ]},
    ]
    req = {
        "model": config["model_name"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }

    last_err = None
    for attempt in range(config["max_retries"]):
        try:
            resp = _get_client(config).chat.completions.create(**req)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < config["max_retries"] - 1:
                time.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation logic
# ─────────────────────────────────────────────────────────────────────────────

class _Tracker:
    """Thread-safe running statistics shown in the progress bar."""

    def __init__(self):
        self._lock = threading.Lock()
        self.n = self.n_exec = self.n_valid = self.n_em = 0
        self.tani_sum = 0.0
        self.tani_cnt = 0

    def update(self, r: dict) -> Dict:
        with self._lock:
            self.n += 1
            if not r["error"]:
                self.n_exec += 1
            if r["valid"]:
                self.n_valid += 1
            if r["exact_match"]:
                self.n_em += 1
            if r["tanimoto"] is not None:
                self.tani_sum += r["tanimoto"]
                self.tani_cnt += 1
            return {
                "Exec": "%.1f%%" % (self.n_exec / self.n * 100),
                "EM": "%.1f%%" % (self.n_em / self.n * 100),
                "Valid": "%.1f%%" % (self.n_valid / self.n * 100),
                "Tani": "%.3f" % (self.tani_sum / self.tani_cnt
                                  if self.tani_cnt else 0),
            }


def evaluate_one(idx: int, sample: dict, config: dict) -> dict:
    result = {
        "idx": idx, "id": sample["id"],
        "gt_smiles": sample["code"],
        "pred_raw": None, "pred_smiles": None,
        "gt_canonical": None, "pred_canonical": None,
        "valid": False, "exact_match": False,
        "tanimoto": None, "error": None,
    }
    try:
        raw = query_model(sample["image"], config)
        result["pred_raw"] = raw
        pred = extract_smiles(raw)
        result["pred_smiles"] = pred

        gt_c = canonicalize_smiles(sample["code"])
        pred_c = canonicalize_smiles(pred)
        result["gt_canonical"] = gt_c
        result["pred_canonical"] = pred_c
        result["valid"] = pred_c is not None
        result["exact_match"] = (
            gt_c is not None and pred_c is not None and gt_c == pred_c)
        result["tanimoto"] = (
            compute_tanimoto(sample["code"], pred) if pred_c else None)
    except Exception as e:
        result["error"] = str(e)
    return result


def aggregate(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {}
    n_valid = sum(1 for r in results if r["valid"])
    n_em = sum(1 for r in results if r["exact_match"])
    n_err = sum(1 for r in results if r["error"])
    n_exec = n - n_err

    ts = [r["tanimoto"] for r in results if r["tanimoto"] is not None]
    avg_t = sum(ts) / len(ts) if ts else 0.0

    return {
        "total_samples": n,
        "exec_rate": n_exec / n,
        "validity_rate": n_valid / n,
        "exact_match_rate": n_em / n,
        "avg_tanimoto": avg_t,
        "tanimoto_at_1.0": sum(1 for t in ts if t >= 1.0 - 1e-9) / n,
        "tanimoto_ge_0.85": sum(1 for t in ts if t >= 0.85) / n,
        "n_exec": n_exec, "n_valid": n_valid,
        "n_exact_match": n_em, "n_errors": n_err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

_LB_COLS = [
    "timestamp", "model", "samples", "exec_rate", "validity", "exact_match",
    "avg_tanimoto", "tani_1.0", "tani_>=0.85", "elapsed_s",
]


def _append_leaderboard(metrics: dict, out_dir: Path) -> None:
    lb = out_dir / "leaderboard.csv"
    write_hdr = not lb.exists()
    row = {
        "timestamp":    metrics["timestamp"],
        "model":        metrics["model"],
        "samples":      metrics["total_samples"],
        "exec_rate":    "%.1f" % (metrics["exec_rate"] * 100),
        "validity":     "%.1f" % (metrics["validity_rate"] * 100),
        "exact_match":  "%.1f" % (metrics["exact_match_rate"] * 100),
        "avg_tanimoto": "%.4f" % metrics["avg_tanimoto"],
        "tani_1.0":     "%.1f" % (metrics["tanimoto_at_1.0"] * 100),
        "tani_>=0.85":  "%.1f" % (metrics["tanimoto_ge_0.85"] * 100),
        "elapsed_s":    "%.1f" % metrics["elapsed_seconds"],
    }
    with open(lb, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LB_COLS)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def _print_leaderboard(out_dir: Path) -> None:
    lb = out_dir / "leaderboard.csv"
    if not lb.exists():
        return
    with open(lb, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    cols = [
        ("Model",    "model",        "left"),
        ("N",        "samples",      "right"),
        ("Exec%",    "exec_rate",    "right"),
        ("Valid%",   "validity",     "right"),
        ("EM%",      "exact_match",  "right"),
        ("AvgTani",  "avg_tanimoto", "right"),
        ("T=1.0%",   "tani_1.0",     "right"),
        ("T>=.85%",  "tani_>=0.85",  "right"),
        ("Time(s)",  "elapsed_s",    "right"),
        ("Timestamp","timestamp",    "left"),
    ]
    widths = {}
    for h, k, _ in cols:
        w = len(h)
        for r in rows:
            v = r.get(k, "")
            if k == "timestamp":
                v = v[:19]
            w = max(w, len(str(v)))
        widths[k] = w

    def _row(vals, hdr=False):
        cells = []
        for h, k, a in cols:
            v = h if hdr else vals.get(k, "")
            if k == "timestamp" and not hdr:
                v = v[:19]
            w = widths[k]
            cells.append(v.ljust(w) if a == "left" else str(v).rjust(w))
        return " | ".join(cells)

    total_w = sum(widths.values()) + 3 * (len(cols) - 1)
    print("\n" + "=" * (total_w + 4))
    print("  LEADERBOARD")
    print("=" * (total_w + 4))
    print("  " + _row({}, hdr=True))
    print("  " + "-" * total_w)
    for r in rows:
        print("  " + _row(r))
    print("=" * (total_w + 4))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="ChemDraw Evaluation — Image-to-SMILES benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --model gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1
  %(prog)s --model gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1 --subset 10
  %(prog)s --model gemini-2.5-pro --api-key <key> --base-url <url> --max-workers 8
""",
    )
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument("--api-key", required=True, help="API key")
    p.add_argument("--base-url", required=True, help="API base URL (OpenAI-compatible)")
    p.add_argument("--subset", type=int, default=None, metavar="N",
                   help="Evaluate only the first N samples")
    p.add_argument("--max-workers", type=int, default=DEFAULT_CONFIG["max_workers"],
                   help="Parallel threads (default: %(default)s)")
    p.add_argument("--max-retries", type=int, default=DEFAULT_CONFIG["max_retries"])
    p.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    p.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    args = p.parse_args()

    config = {
        **DEFAULT_CONFIG,
        "model_name": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
    }

    print("=" * 70)
    print("ChemDraw Evaluation Pipeline")
    print("=" * 70)
    print("  Model       : %s" % config["model_name"])
    print("  Base URL    : %s" % config["base_url"])
    print("  Samples     : %s" % (args.subset or "all"))
    print("  Workers     : %d" % config["max_workers"])
    print("  Max retries : %d" % config["max_retries"])
    print("  Temperature : %s" % config["temperature"])
    print()

    # ── Load dataset ──
    print("Loading dataset from %s ..." % config["data_dir"])
    samples = load_dataset(config["data_dir"])
    print("  Loaded %d samples." % len(samples))

    n_total = len(samples)
    if args.subset and args.subset < n_total:
        n_total = args.subset
        print("  >> Subset mode: first %d samples." % n_total)

    # ── Evaluate ──
    print("\nEvaluating %d samples (workers=%d)...\n" % (n_total, config["max_workers"]))
    t0 = time.time()
    results = []
    tracker = _Tracker()
    pbar = tqdm(total=n_total, desc="Evaluating", unit="sample",
                bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"))

    if config["max_workers"] <= 1:
        for i in range(n_total):
            r = evaluate_one(i, samples[i], config)
            results.append(r)
            pbar.set_postfix(tracker.update(r), refresh=False)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=config["max_workers"]) as pool:
            futs = {pool.submit(evaluate_one, i, samples[i], config): i
                    for i in range(n_total)}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                pbar.set_postfix(tracker.update(r), refresh=False)
                pbar.update(1)
    pbar.close()

    results.sort(key=lambda r: r["idx"])
    elapsed = time.time() - t0

    # ── Metrics ──
    metrics = aggregate(results)
    metrics["model"] = config["model_name"]
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["max_workers"] = config["max_workers"]
    metrics["subset"] = args.subset

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("  Model            : %s" % config["model_name"])
    print("  Total samples    : %d" % metrics["total_samples"])
    print("  Exec Rate        : %.1f%%  (%d/%d)" % (
        metrics["exec_rate"] * 100, metrics["n_exec"], metrics["total_samples"]))
    print("  Validity Rate    : %.1f%%  (%d/%d)" % (
        metrics["validity_rate"] * 100, metrics["n_valid"], metrics["total_samples"]))
    print("  Exact Match      : %.1f%%  (%d/%d)" % (
        metrics["exact_match_rate"] * 100, metrics["n_exact_match"], metrics["total_samples"]))
    print("  Avg Tanimoto     : %.4f" % metrics["avg_tanimoto"])
    print("  Tanimoto = 1.0   : %.1f%%" % (metrics["tanimoto_at_1.0"] * 100))
    print("  Tanimoto >= 0.85 : %.1f%%" % (metrics["tanimoto_ge_0.85"] * 100))
    print("  Errors           : %d" % metrics["n_errors"])
    print("  Elapsed          : %.1fs (%.1fs/sample)" % (
        elapsed, elapsed / max(metrics["total_samples"], 1)))
    print("=" * 70)

    # ── Save ──
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = config["model_name"].replace("/", "_")
    tag = "subset%d" % n_total if args.subset else "full"

    detail = out_dir / ("chemdraw_%s_%s_%s_detail.jsonl" % (slug, tag, ts))
    with open(detail, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nDetail  : %s" % detail)

    summary = out_dir / ("chemdraw_%s_%s_%s_summary.json" % (slug, tag, ts))
    with open(summary, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Summary : %s" % summary)

    _append_leaderboard(metrics, out_dir)
    _print_leaderboard(out_dir)


if __name__ == "__main__":
    main()
