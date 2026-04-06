import argparse
import asyncio
import os
import re
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm

from client import OpenAICompatClient, image_to_content_part
from dataset_tools import load_instructs
from io_utils import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from logger import APILogger


def parse_score(text: str) -> int | None:
    text = (text or "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    m = re.search(r"\[FINAL SCORE\]:?\s*(\d+)", text, re.DOTALL)
    if m:
        return int(m.group(1))
    return None


async def _judge_one(
    client: OpenAICompatClient,
    logger: APILogger,
    sem: asyncio.Semaphore,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[int | None, str]:
    async with sem:
        try:
            request_starttime = time.time()

            data = await client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)


            request_endtime = time.time()
            duration = request_endtime - request_starttime

            logger.log_call(
                model=client.model, 
                request_data={'messages': messages},
                response_data=data,
                duration_ms=duration,
                api_key=client.api_key
            )

            text = data.get("response_text", "")
            return parse_score(text), text
        except Exception as e:  # noqa: BLE001
            print(e)
            return None, f"judge_error: {e}"


def _is_row_scored(row: dict[str, Any]) -> bool:
    """A row is considered scored if both vis and task scores are present (not None)."""
    return row.get("score_vis") is not None and row.get("score_task") is not None


def _load_checkpoint(scored_path: Path, checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    """Load already-scored rows from both the final results and the checkpoint file."""
    done: dict[str, dict[str, Any]] = {}
    for p in (scored_path, checkpoint_path):
        if p.exists():
            try:
                for r in read_jsonl(p):
                    rid = r.get("id")
                    if rid is not None and _is_row_scored(r):
                        done[str(rid)] = r
            except Exception:
                pass
    return done


async def main_async(args):
    bundle = Path(args.bundle_dir)
    manifest = read_json(bundle / "manifest.json")
    language = manifest["language"]
    model_name = manifest.get("model", "unknown")
    instructs = load_instructs(manifest["repo_root"], language)

    rows = read_jsonl(bundle / manifest["executions"])
    total = len(rows)
    logger = APILogger(log_dir=args.log_dir)
    client = OpenAICompatClient(
        model=args.judge_model,
        base_url=args.base_url or manifest.get("judge_base_url") or manifest.get("generation_base_url"),
        api_key=os.getenv(args.api_key_env, "EMPTY"),
        timeout=args.timeout,
    )
    sem = asyncio.Semaphore(args.concurrency)

    scored_path = bundle / manifest["scored_results"]
    checkpoint_path = bundle / "_scoring_checkpoint.jsonl"

    # --- Resume: load previously scored rows ---
    done_map: dict[str, dict[str, Any]] = {}
    if args.resume:
        done_map = _load_checkpoint(scored_path, checkpoint_path)

        if done_map:
            exec_lookup = {str(r.get("id")): r for r in rows}
            invalidated = 0
            for rid in list(done_map):
                exec_row = exec_lookup.get(rid)
                scored_row = done_map[rid]
                if exec_row is None:
                    del done_map[rid]
                    invalidated += 1
                elif (bool(exec_row.get("has_plot")) != bool(scored_row.get("has_plot"))
                      or bool(exec_row.get("error")) != bool(scored_row.get("error"))):
                    del done_map[rid]
                    invalidated += 1
            if invalidated:
                print(f"  [!] {invalidated} cached scores invalidated (execution results changed)")

    pending_rows = [r for r in rows if str(r.get("id")) not in done_map]
    n_done = total - len(pending_rows)
    n_pending = len(pending_rows)

    print(f"\n{'='*60}")
    print(f"  Language: {language}  |  Model: {model_name}")
    print(f"  Judge: {args.judge_model}  |  Samples: {total}  |  Concurrency: {args.concurrency}")
    if n_done > 0:
        print(f"  Resuming: {n_done} already scored, {n_pending} remaining")
    print(f"{'='*60}")

    if n_pending == 0:
        print(f"  All {total} samples already scored, skipping API calls.")
        scored = [done_map.get(str(r.get("id")), r) for r in rows]
    else:
        if not args.resume:
            for p in (checkpoint_path, scored_path):
                if p.exists():
                    p.unlink()

        ckpt_lock = asyncio.Lock()

        pbar = tqdm(
            total=n_pending,
            desc=f"Scoring [{language}]",
            unit="sample",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        scored_ok = 0
        skipped = 0
        errors = 0

        def _update_postfix():
            pbar.set_postfix_str(f"ok={scored_ok} skip={skipped} err={errors}")

        async def score_row(row: dict[str, Any]) -> dict[str, Any]:
            nonlocal scored_ok, skipped, errors
            out = dict(row)
            gen_plots = row.get("plots_generated") or []
            if not gen_plots:
                out["score_vis"] = 0
                out["score_task"] = 0
                out["scoring_response_vis"] = ""
                out["scoring_response_task"] = ""
                skipped += 1
                _update_postfix()
                pbar.update(1)
                async with ckpt_lock:
                    append_jsonl(out, checkpoint_path)
                return out

            vis_messages = [
                {"role": "system", "content": instructs["system_prompt"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructs["judge_instruct_vis"]},
                        image_to_content_part(gen_plots[0]),
                        image_to_content_part((row.get("plots_gt") or [""])[0]),
                    ],
                },
            ]
            task_text = "\n".join(
                [
                    instructs["judge_instruct_task"],
                    "[PLOT TASK]:",
                    row.get("task__plot_description", ""),
                    row.get("task__plot_style", ""),
                ]
            )
            task_parts: list[dict[str, Any]] = [{"type": "text", "text": task_text}]
            task_parts.extend(image_to_content_part(img) for img in gen_plots)
            task_messages = [
                {"role": "system", "content": instructs["system_prompt"]},
                {"role": "user", "content": task_parts},
            ]

            (vis_score, vis_resp), (task_score, task_resp) = await asyncio.gather(
                _judge_one(client, logger, sem, vis_messages, temperature=args.temperature, max_tokens=args.max_tokens),
                _judge_one(client, logger, sem, task_messages, temperature=args.temperature, max_tokens=args.max_tokens),
            )
            out["score_vis"] = vis_score
            out["score_task"] = task_score
            out["scoring_response_vis"] = vis_resp
            out["scoring_response_task"] = task_resp

            if vis_score is None or task_score is None:
                errors += 1
            else:
                scored_ok += 1
            _update_postfix()
            pbar.update(1)
            async with ckpt_lock:
                append_jsonl(out, checkpoint_path)
            return out

        t0 = time.time()
        new_results = await asyncio.gather(*[score_row(r) for r in pending_rows])
        pbar.close()
        elapsed = time.time() - t0

        for r in new_results:
            done_map[str(r.get("id"))] = r

        scored = [done_map.get(str(r.get("id")), r) for r in rows]

        print(f"\n  [{language}] Scored {n_pending} samples in {elapsed:.1f}s")

    write_jsonl(scored, scored_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    vis_scores = [r.get("score_vis", 0) or 0 for r in scored]
    task_scores = [r.get("score_task", 0) or 0 for r in scored]
    stats = {
        "model": model_name,
        "language": language,
        "judge_model": args.judge_model,
        "count": len(scored),
        "execution_error_num": sum(1 for r in scored if r.get("error")),
        "no_plots": sum(1 for r in scored if not r.get("has_plot")),
        "scores": {
            "vis": {
                "mean": round(mean(vis_scores), 2) if vis_scores else 0,
                "median": median(vis_scores) if vis_scores else 0,
                "good": round(sum(1 for s in vis_scores if s >= 75) / len(vis_scores), 3) if vis_scores else 0,
                "bad": round(sum(1 for s in vis_scores if s <= 25) / len(vis_scores), 3) if vis_scores else 0,
            },
            "task": {
                "mean": round(mean(task_scores), 2) if task_scores else 0,
                "median": median(task_scores) if task_scores else 0,
                "good": round(sum(1 for s in task_scores if s >= 75) / len(task_scores), 3) if task_scores else 0,
                "bad": round(sum(1 for s in task_scores if s <= 25) / len(task_scores), 3) if task_scores else 0,
            },
        },
    }
    write_json(stats, bundle / manifest["stats"])

    vis_mean = stats["scores"]["vis"]["mean"]
    task_mean = stats["scores"]["task"]["mean"]
    print(f"  [{language}] vis_mean={vis_mean}  task_mean={task_mean}")
    print(f"  -> {bundle / manifest['scored_results']}")
    print(f"  -> {bundle / manifest['stats']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--log-dir", type=str, default="./api-logs")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from previous checkpoint (default: enabled; use --no-resume to force re-score)")
    asyncio.run(main_async(parser.parse_args()))
