"""End-to-end tests using the offline mock model (no credentials, no network)."""

import csv
import json

from prompt_eval import store
from prompt_eval.analyze import analyze, load_frame
from prompt_eval.config import Config
from prompt_eval.data import INPUTS, VARIANTS
from prompt_eval.judge import run_judging, write_audit_sample
from prompt_eval.llm import MockLLM
from prompt_eval.runner import build_plan, run_experiment

CFG = Config.from_env(dry_run=True)


def test_plan_is_randomised_complete_block_design():
    ids, vids = ["a", "b", "c"], list(VARIANTS)
    plan = build_plan(ids, vids, reps=3, seed=1)
    assert len(plan) == 3 * len(ids) * len(vids)
    assert len({p.key for p in plan}) == len(plan)  # unique keys
    for rep in range(3):  # each rep block holds every pair exactly once
        block = [(p.input_id, p.variant_id) for p in plan if p.rep == rep]
        assert sorted(block) == sorted((i, v) for i in ids for v in vids)
    assert plan == build_plan(ids, vids, reps=3, seed=1)  # reproducible
    assert [p.variant_id for p in plan[:5]] != vids * 1  # shuffled, not in variant order


def test_run_is_resumable_and_retries_errors(tmp_path):
    inputs = INPUTS[:3]
    run_experiment(MockLLM(0), CFG, inputs, tmp_path, reps=2, seed=1, limit=7, dry_run=True)
    assert len(store.read_jsonl(tmp_path / "runs.jsonl")) == 7
    run_experiment(MockLLM(0), CFG, inputs, tmp_path, reps=2, seed=1, dry_run=True)
    latest = store.latest_by_key(tmp_path / "runs.jsonl")
    assert len(latest) == 3 * len(VARIANTS) * 2  # finished the plan without redoing the first 7
    assert len(store.read_jsonl(tmp_path / "runs.jsonl")) == len(latest)

    class Flaky:
        def invoke(self, prompt, meta=None):
            raise RuntimeError("boom")

    fresh = tmp_path / "flaky"
    run_experiment(Flaky(), CFG, inputs, fresh, reps=1, seed=1, dry_run=True)
    assert all(r["error"] for r in store.latest_by_key(fresh / "runs.jsonl").values())
    run_experiment(MockLLM(0), CFG, inputs, fresh, reps=1, seed=1, dry_run=True)  # retries the failures
    assert not any(r["error"] for r in store.latest_by_key(fresh / "runs.jsonl").values())


def test_full_pipeline_outputs_and_no_secrets(tmp_path):
    inputs = INPUTS
    run_experiment(MockLLM(0), CFG, inputs, tmp_path, reps=3, seed=5, dry_run=True)
    run_judging(MockLLM(0), CFG, inputs, tmp_path, seed=5)
    audit, key = write_audit_sample(inputs, tmp_path, k=20)
    report = analyze(tmp_path)

    text = report.read_text()
    assert "DRY RUN" in text and "Paired comparisons" in text
    for name in ("summary.csv", "comparisons.csv", "fig_cost_vs_quality.png", "fig_token_split.png"):
        assert (tmp_path / name).exists()

    df, _ = load_frame(tmp_path)
    assert df["quality"].notna().all()
    # mock encodes: v4 is the long, expensive outlier; v1 is the tight one
    mean_words = df.groupby("variant_id")["words"].mean()
    assert mean_words["v4"] > 3 * mean_words["v1"]

    # audit file is blind: no variant ids leak into what the human sees
    with open(audit, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 and set(rows[0]) == {"item_id", "source", "summary", "human_score"}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "api_key" not in json.dumps(manifest).lower()


def test_env_aliases_accept_existing_dotenv_names(monkeypatch):
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_CHAT_DEPLOYMENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("prompt_eval.config._load_dotenv", lambda: None)  # ignore any real .env
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_KEY", "not-a-real-key")
    monkeypatch.setenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("AZURE_API_VERSION", "2023-05-15")  # must NOT be picked up
    cfg = Config.from_env()
    import os

    assert cfg.endpoint == "https://example.openai.azure.com/"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "not-a-real-key"
    assert cfg.api_version == "2024-12-01-preview"


def test_v6_is_multi_sentence_and_checked_for_word_limit(tmp_path):
    assert VARIANTS["v6"].count(". ") >= 2  # genuinely several sentences of instruction
    run_experiment(MockLLM(0), CFG, INPUTS[:4], tmp_path, reps=2, seed=3, dry_run=True)
    from prompt_eval.analyze import adherence_table

    df, _ = load_frame(tmp_path)
    adh = adherence_table(df).set_index("variant_id")
    assert {"v1", "v5", "v6"} <= set(adh.index)
    assert 0 <= adh.loc["v6", "within_word_limit"] <= 1


def test_report_projects_cost_and_figures_handle_non_reasoning_models(tmp_path):
    """gpt-4o style run: zero reasoning tokens; report must still build and show projected cost."""
    from prompt_eval.plots import make_figures

    run_experiment(MockLLM(0), CFG, INPUTS[:4], tmp_path, reps=2, seed=2, dry_run=True)
    run_judging(MockLLM(0), CFG, INPUTS[:4], tmp_path, seed=2)
    df, _ = load_frame(tmp_path)
    df["visible_tokens"], df["reasoning_tokens"] = df["output_tokens"], 0
    assert (make_figures(df, tmp_path)[1]).exists()

    text = analyze(tmp_path, monthly_calls=500_000).read_text()
    assert "Projected cost at scale" in text and "500,000 calls per month" in text
    assert "per 1M input tokens" in text


def test_judge_spend_is_recorded_priced_separately_and_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_PRICE_IN_PER_M", "1.0")
    monkeypatch.setenv("JUDGE_PRICE_OUT_PER_M", "4.0")
    cfg = Config.from_env(dry_run=True)
    assert cfg.judge_prices_set and cfg.judge_price_out_per_m == 4.0 and cfg.price_out_per_m != 4.0

    run_experiment(MockLLM(0), cfg, INPUTS[:3], tmp_path, reps=2, seed=1, dry_run=True)
    run_judging(MockLLM(0), cfg, INPUTS[:3], tmp_path, seed=1)
    rec = next(iter(store.latest_by_key(tmp_path / "judgments.jsonl").values()))
    expected = (rec["judge_input_tokens"] * 1.0 + rec["judge_output_tokens"] * 4.0) / 1_000_000
    assert rec["judge_cost_usd"] == __import__("pytest").approx(expected)

    text = analyze(tmp_path).read_text()
    assert "Experiment spend" in text and "| Judging" in text and "| Total" in text
    assert "Warning" not in text  # prices were set, so no warning


def test_report_warns_when_judge_differs_but_judge_prices_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_PRICE_IN_PER_M", raising=False)
    monkeypatch.delenv("JUDGE_PRICE_OUT_PER_M", raising=False)
    monkeypatch.setenv("JUDGE_DEPLOYMENT", "other-model")
    cfg = Config.from_env(dry_run=True)
    run_experiment(MockLLM(0), cfg, INPUTS[:3], tmp_path, reps=1, seed=1, dry_run=True)
    run_judging(MockLLM(0), cfg, INPUTS[:3], tmp_path, seed=1)
    assert "JUDGE_PRICE_IN_PER_M" in analyze(tmp_path).read_text()


def test_analyze_survives_smoke_test_and_single_rep_runs(tmp_path):
    """`run --limit 5` then `analyze` (or --reps 1) must produce a report, not crash or warn."""
    import warnings

    smoke = tmp_path / "smoke"
    run_experiment(MockLLM(0), CFG, INPUTS, smoke, reps=1, seed=1, limit=5, dry_run=True)
    single = tmp_path / "single"
    run_experiment(MockLLM(0), CFG, INPUTS[:4], single, reps=1, seed=1, dry_run=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any numpy RuntimeWarning fails the test
        for d in (smoke, single):
            text = analyze(d).read_text()
            assert "Per-variant summary" in text
    assert "n/a" in (single / "report.md").read_text()  # within-input SD needs >= 2 reps
