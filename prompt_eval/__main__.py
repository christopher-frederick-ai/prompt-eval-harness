"""CLI:  python -m prompt_eval <run|judge|audit|analyze|all> [options]

Add --dry-run to exercise the whole pipeline offline with a mock model (synthetic numbers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import analyze
from .config import Config
from .data import load_inputs
from .judge import run_judging, write_audit_sample
from .llm import AzureLLM, MockLLM
from .runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="prompt_eval", description=__doc__)
    ap.add_argument("command", choices=["run", "judge", "audit", "analyze", "all"])
    ap.add_argument("--out", help="results directory (default: results/run1, or results/dryrun with --dry-run)")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per input and variant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1, help="concurrent calls (>1 distorts latency)")
    ap.add_argument("--limit", type=int, help="run at most N pending calls (for a cheap smoke test)")
    ap.add_argument("--inputs", help="JSONL file of {id, text}; default: built-in synthetic set")
    ap.add_argument("--audit-size", type=int, default=30)
    ap.add_argument("--monthly-calls", type=int, default=1_000_000, help="volume for the projected-cost table")
    ap.add_argument("--dry-run", action="store_true", help="use the offline mock model; no credentials needed")
    a = ap.parse_args(argv)

    out = Path(a.out or ("results/dryrun" if a.dry_run else "results/run1"))
    cfg = Config.from_env(dry_run=a.dry_run)
    inputs = load_inputs(a.inputs)

    def client(deployment: str | None = None):
        return MockLLM(seed=a.seed) if a.dry_run else AzureLLM(cfg, deployment=deployment)

    if a.command in ("run", "all"):
        run_experiment(client(), cfg, inputs, out, a.reps, a.seed, a.workers, a.limit, a.dry_run)
    if a.command in ("judge", "all"):
        run_judging(client(cfg.judge_deployment), cfg, inputs, out, a.seed, a.workers)
    if a.command in ("audit", "all"):
        audit, _ = write_audit_sample(inputs, out, a.audit_size)
        print(f"blind audit sample written: {audit} (fill in human_score 1-5, then re-run analyze)")
    if a.command in ("analyze", "all"):
        print(f"report written: {analyze(out, a.monthly_calls)}")


if __name__ == "__main__":
    main()
