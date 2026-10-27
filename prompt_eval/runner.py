"""Run the experiment: inputs x variants x reps, in randomised blocks, resumable.

Design
- Randomised complete block design: every rep is a block containing every (input, variant)
  pair once, shuffled within the block. API latency drifts over time; this spreads that
  drift evenly across variants instead of attributing it to whichever variant ran last.
- Results are appended to runs.jsonl keyed by input|variant|rep. Re-running skips keys that
  already succeeded and retries keys that errored, so a crash or rate-limit never wastes work.
- workers=1 by default because concurrent calls contaminate latency measurements.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import store
from .config import Config
from .data import VARIANTS, build_prompt
from .llm import LLM
from .metrics import cost_usd, count_sentences, count_words


@dataclass(frozen=True)
class PlanItem:
    order: int
    input_id: str
    variant_id: str
    rep: int

    @property
    def key(self) -> str:
        return f"{self.input_id}|{self.variant_id}|{self.rep}"


def build_plan(input_ids: list[str], variant_ids: list[str], reps: int, seed: int) -> list[PlanItem]:
    rng = random.Random(seed)
    plan: list[PlanItem] = []
    for rep in range(reps):
        block = [(i, v) for i in input_ids for v in variant_ids]
        rng.shuffle(block)
        plan.extend(PlanItem(len(plan) + k, i, v, rep) for k, (i, v) in enumerate(block))
    return plan


def _execute(llm: LLM, cfg: Config, item: PlanItem, text: str) -> dict:
    prompt = build_prompt(item.variant_id, text)
    rec = {
        "key": item.key,
        "input_id": item.input_id,
        "variant_id": item.variant_id,
        "rep": item.rep,
        "order": item.order,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg.chat_deployment,
        "error": None,
    }
    try:
        res = llm.invoke(prompt, meta={"variant_id": item.variant_id, "input_id": item.input_id, "rep": item.rep})
        rec.update(
            output=res.text,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            reasoning_tokens=res.reasoning_tokens,
            visible_tokens=max(0, res.output_tokens - res.reasoning_tokens),
            latency_s=round(res.latency_s, 4),
            retries=res.retries,
            words=count_words(res.text),
            sentences=count_sentences(res.text),
            cost_usd=cost_usd(res.input_tokens, res.output_tokens, cfg.price_in_per_m, cfg.price_out_per_m),
        )
    except Exception as exc:  # record and continue; the key is retried on the next run
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return rec


def write_manifest(out_dir: Path, cfg: Config, inputs: list[dict], reps: int, seed: int, dry_run: bool) -> None:
    inputs_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "synthetic_data_warning": "Numbers from a dry run are synthetic mock output." if dry_run else None,
        "seed": seed,
        "reps": reps,
        "n_inputs": len(inputs),
        "inputs_sha256_16": inputs_hash,
        "variants": VARIANTS,
        "config": cfg.snapshot(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_experiment(
    llm: LLM,
    cfg: Config,
    inputs: list[dict],
    out_dir: Path,
    reps: int = 5,
    seed: int = 42,
    workers: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    variant_ids: list[str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    variant_ids = variant_ids or list(VARIANTS)
    texts = {i["id"]: i["text"] for i in inputs}
    plan = build_plan(list(texts), variant_ids, reps, seed)

    done = {k for k, r in store.latest_by_key(runs_path).items() if not r.get("error")}
    pending = [p for p in plan if p.key not in done]
    if limit is not None:
        pending = pending[:limit]
    write_manifest(out_dir, cfg, inputs, reps, seed, dry_run)
    print(f"plan: {len(plan)} calls, {len(done)} already done, {len(pending)} to run (workers={workers})")

    def work(item: PlanItem) -> dict:
        rec = _execute(llm, cfg, item, texts[item.input_id])
        store.append_jsonl(runs_path, rec)
        return rec

    errors = 0
    if workers <= 1:
        results = (work(p) for p in pending)
        for n, rec in enumerate(results, 1):
            errors += bool(rec["error"])
            if n % 10 == 0 or n == len(pending):
                print(f"  {n}/{len(pending)} done, {errors} errors")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, rec in enumerate(pool.map(work, pending), 1):
                errors += bool(rec["error"])
                if n % 10 == 0 or n == len(pending):
                    print(f"  {n}/{len(pending)} done, {errors} errors")
    if errors:
        print(f"{errors} calls failed; re-run the same command to retry them.")
    return runs_path
