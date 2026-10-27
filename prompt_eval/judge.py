"""LLM-as-judge scoring, plus a blind human-audit sample to check the judge itself.

Design choices
- The judge is BLIND: it sees only the source paragraph and the summary, never the variant.
- Length/conciseness is deliberately NOT a rubric item. Length is measured directly, and a
  judge that rewards (or punishes) brevity would double-count it and muddy the quality score.
- Judge calls run in shuffled order and are stored per record, so they are resumable.
- Known limitation: using the same model family to judge its own outputs can inflate scores
  (self-preference). Set JUDGE_DEPLOYMENT to a different model if you have one, and use the
  human audit (`audit` command, then `analyze`) to measure judge/human agreement.
"""

from __future__ import annotations

import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import store
from .config import Config
from .llm import LLM
from .metrics import cost_usd

DIMENSIONS = ("faithfulness", "coverage", "clarity")

JUDGE_TEMPLATE = """You are a careful, impartial evaluator of summaries.

Rate the SUMMARY of the SOURCE on each dimension from 1 (poor) to 5 (excellent):
- faithfulness: every claim in the summary is supported by the source; nothing is invented or distorted.
- coverage: the summary captures the main point(s) of the source.
- clarity: the summary is readable and well-formed.
Do not reward or penalise the summary for being short or long.

SOURCE:
<<<
{source}
>>>

SUMMARY:
<<<
{summary}
>>>

Respond with only JSON, no other text, in exactly this form:
{{"faithfulness": <1-5>, "coverage": <1-5>, "clarity": <1-5>, "rationale": "<one sentence>"}}"""


def parse_judgement(text: str) -> dict:
    """Extract and validate the JSON object; raises ValueError if it is unusable."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    scores = {}
    for d in DIMENSIONS:
        v = obj.get(d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5:
            raise ValueError(f"bad or missing score for {d}: {v!r}")
        scores[d] = int(round(v))
    scores["quality"] = sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS)
    scores["rationale"] = str(obj.get("rationale", ""))[:300]
    return scores


def _record_judge_config(out_dir: Path, cfg: Config) -> None:
    """Add the judge deployment and prices to the manifest (no secrets)."""
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["judge_config"] = {
        "judge_deployment": cfg.judge_deployment,
        "judge_price_in_per_m": cfg.judge_price_in_per_m,
        "judge_price_out_per_m": cfg.judge_price_out_per_m,
        "judge_prices_set": cfg.judge_prices_set,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_judging(llm: LLM, cfg: Config, inputs: list[dict], out_dir: Path, seed: int = 42, workers: int = 1) -> Path:
    runs = store.latest_by_key(out_dir / "runs.jsonl")
    judg_path = out_dir / "judgments.jsonl"
    done = {k for k, r in store.latest_by_key(judg_path).items() if not r.get("error")}
    texts = {i["id"]: i["text"] for i in inputs}
    todo = [r for k, r in runs.items() if not r.get("error") and k not in done]
    random.Random(seed).shuffle(todo)
    _record_judge_config(out_dir, cfg)
    print(f"judging {len(todo)} summaries ({len(done)} already judged)")

    def work(run: dict) -> dict:
        rec = {"key": run["key"], "input_id": run["input_id"], "variant_id": run["variant_id"], "rep": run["rep"], "error": None}
        prompt = JUDGE_TEMPLATE.format(source=texts[run["input_id"]], summary=run["output"].strip())
        try:
            res = llm.invoke(prompt, meta={"role": "judge", "variant_id": "judge"})
            # record spend before parsing, so tokens are accounted for even if the JSON is bad
            rec.update(
                judge_input_tokens=res.input_tokens,
                judge_output_tokens=res.output_tokens,
                judge_reasoning_tokens=res.reasoning_tokens,
                judge_cost_usd=cost_usd(res.input_tokens, res.output_tokens, cfg.judge_price_in_per_m, cfg.judge_price_out_per_m),
            )
            rec.update(parse_judgement(res.text))
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        store.append_jsonl(judg_path, rec)
        return rec

    failures = 0
    if workers <= 1:
        it = map(work, todo)
        for n, rec in enumerate(it, 1):
            failures += bool(rec["error"])
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} judged, {failures} failures")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, rec in enumerate(pool.map(work, todo), 1):
                failures += bool(rec["error"])
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} judged, {failures} failures")
    if failures:
        print(f"{failures} judgements failed to parse or call; re-run to retry them.")
    return judg_path


def write_audit_sample(inputs: list[dict], out_dir: Path, k: int = 30, seed: int = 7) -> tuple[Path, Path]:
    """Blind sample for manual scoring. Rate human_score 1-5 (faithful + covers main point + clear).

    The variant is hidden: rows carry an opaque item_id, and audit_key.csv maps it back.
    """
    runs = [r for r in store.latest_by_key(out_dir / "runs.jsonl").values() if not r.get("error")]
    rng = random.Random(seed)
    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        by_variant.setdefault(r["variant_id"], []).append(r)
    for rs in by_variant.values():
        rng.shuffle(rs)
    sample: list[dict] = []
    while len(sample) < k and any(by_variant.values()):  # round-robin so every variant is covered
        for rs in by_variant.values():
            if rs and len(sample) < k:
                sample.append(rs.pop())
    rng.shuffle(sample)
    texts = {i["id"]: i["text"] for i in inputs}
    audit, key = out_dir / "human_audit.csv", out_dir / "audit_key.csv"
    with open(audit, "w", newline="", encoding="utf-8") as fa, open(key, "w", newline="", encoding="utf-8") as fk:
        wa, wk = csv.writer(fa), csv.writer(fk)
        wa.writerow(["item_id", "source", "summary", "human_score"])
        wk.writerow(["item_id", "key"])
        for n, r in enumerate(sample, 1):
            wa.writerow([n, texts[r["input_id"]], r["output"].strip(), ""])
            wk.writerow([n, r["key"]])
    return audit, key
