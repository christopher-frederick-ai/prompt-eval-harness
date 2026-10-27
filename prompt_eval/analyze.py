"""Turn runs.jsonl (+ judgments.jsonl) into tables, comparisons and a markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from . import store
from .data import BASELINE_VARIANT, ONE_SENTENCE_VARIANTS, VARIANTS, WORD_LIMITS
from .stats import bootstrap_mean_ci, holm, paired_compare, spearman

# (column, label, lower_is_better) for paired comparisons against the baseline
COMPARE_METRICS = [
    ("words", "Words", None),
    ("output_tokens", "Output tokens", True),
    ("cost_usd", "Cost per 1k calls (USD)", True),
    ("latency_s", "Latency (s)", True),
    ("quality", "Judge quality (1-5)", False),
    ("words_sd", "Within-input SD of words", True),
]


def load_frame(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    runs = store.read_jsonl(out_dir / "runs.jsonl")
    latest = {r["key"]: r for r in runs}
    all_df = pd.DataFrame(latest.values())
    if all_df.empty:
        raise SystemExit(f"No runs found in {out_dir}. Run the experiment first.")
    errors = all_df.groupby("variant_id")["error"].apply(lambda s: s.notna().mean()).to_dict()
    df = all_df[all_df["error"].isna()].copy()
    df["one_sentence"] = (df["sentences"] == 1).astype(float)
    df["reasoning_share"] = df["reasoning_tokens"] / df["output_tokens"].replace(0, np.nan)

    judg = pd.DataFrame(store.latest_by_key(out_dir / "judgments.jsonl").values())
    if not judg.empty:
        judg = judg[judg["error"].isna()][["key", "faithfulness", "coverage", "clarity", "quality"]]
        df = df.merge(judg, on="key", how="left")
    else:
        df["quality"] = np.nan
    return df, errors


def per_input_matrix(df: pd.DataFrame, col: str, complete: bool = True) -> pd.DataFrame:
    """inputs x variants matrix of the mean over reps.

    complete=True drops inputs with any missing cell (needed for paired comparisons);
    complete=False keeps every variant's own data (used for the per-variant summary).
    """
    if col == "words_sd":
        m = df.groupby(["input_id", "variant_id"])["words"].std(ddof=1).unstack("variant_id")
    else:
        m = df.groupby(["input_id", "variant_id"])[col].mean().unstack("variant_id")
    return m.dropna() if complete else m


def _ci_str(series: pd.Series, nd: int = 1) -> str:
    vals = series.dropna()
    return "n/a" if len(vals) < 2 else _fmt_ci(*bootstrap_mean_ci(vals), nd=nd)


def _fmt_ci(mean: float, lo: float, hi: float, nd: int = 1) -> str:
    return f"{mean:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def summary_table(df: pd.DataFrame, errors: dict) -> pd.DataFrame:
    rows = []
    variants = [v for v in VARIANTS if v in set(df["variant_id"])]
    mats = {c: per_input_matrix(df, c, complete=False) for c in ("words", "words_sd", "output_tokens", "cost_usd", "latency_s")}
    has_q = df["quality"].notna().any()
    qmat = per_input_matrix(df.dropna(subset=["quality"]), "quality", complete=False) if has_q else None
    for v in variants:
        d = df[df["variant_id"] == v]
        row = {
            "variant": v,
            "prompt": VARIANTS[v].strip(),
            "calls": len(d),
            "error_rate": f"{errors.get(v, 0):.1%}",
            "words": _ci_str(mats["words"][v]),
            "words_sd_within_input": "n/a" if mats["words_sd"][v].dropna().empty else f"{mats['words_sd'][v].mean():.1f}",
            "output_tokens": _ci_str(mats["output_tokens"][v], nd=0),
            "reasoning_share": f"{d['reasoning_tokens'].sum() / max(1, d['output_tokens'].sum()):.0%}",
            "cost_per_1k_calls_usd": f"{d['cost_usd'].mean() * 1000:.3f}",
            "latency_mean_s": f"{d['latency_s'].mean():.2f}",
            "latency_p95_s": f"{d['latency_s'].quantile(0.95):.2f}",
            "one_sentence_rate": f"{d['one_sentence'].mean():.0%}",
        }
        if has_q and v in qmat.columns:
            row["quality"] = _ci_str(qmat[v], nd=2)
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(df: pd.DataFrame, baseline: str = BASELINE_VARIANT) -> pd.DataFrame:
    rows = []
    for col, label, _ in COMPARE_METRICS:
        if col == "quality" and not df["quality"].notna().any():
            continue
        sub = df.dropna(subset=["quality"]) if col == "quality" else df
        m = per_input_matrix(sub, col)
        if col == "cost_usd":
            m = m * 1000  # report per 1,000 calls
        if baseline not in m.columns or len(m) < 3:
            continue  # paired comparison needs >= 3 inputs with data for every variant
        others = [v for v in m.columns if v != baseline]
        res = [paired_compare(m[v], m[baseline]) for v in others]
        adj = holm([r["p"] for r in res])
        for v, r, p_adj in zip(others, res, adj):
            rows.append(
                {
                    "metric": label,
                    "variant": v,
                    "vs": baseline,
                    "n_inputs": len(m),
                    "diff": r["diff"],
                    "ci_lo": r["lo"],
                    "ci_hi": r["hi"],
                    "p": r["p"],
                    "p_holm": p_adj,
                    "significant_holm_0.05": bool(p_adj < 0.05),
                }
            )
    return pd.DataFrame(rows)


def adherence_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["variant_id"].isin(ONE_SENTENCE_VARIANTS | set(WORD_LIMITS))].copy()
    limit = sub["variant_id"].map(WORD_LIMITS)
    sub["within_word_limit"] = np.where(limit.notna(), (sub["words"] <= limit).astype(float), np.nan)
    out = sub.groupby("variant_id").agg(
        calls=("one_sentence", "size"), one_sentence_rate=("one_sentence", "mean"), within_word_limit=("within_word_limit", "mean")
    )
    return out.reset_index()


def judge_agreement(df: pd.DataFrame, out_dir: Path) -> dict | None:
    """Spearman correlation between judge quality and human_score, if the audit is filled in."""
    audit, key = out_dir / "human_audit.csv", out_dir / "audit_key.csv"
    if not (audit.exists() and key.exists()) or not df["quality"].notna().any():
        return None
    with open(audit, encoding="utf-8") as f:
        human = {int(r["item_id"]): r["human_score"] for r in csv.DictReader(f) if r["human_score"].strip()}
    with open(key, encoding="utf-8") as f:
        keymap = {int(r["item_id"]): r["key"] for r in csv.DictReader(f)}
    qual = df.set_index("key")["quality"].to_dict()
    pairs = [(float(h), qual[keymap[i]]) for i, h in human.items() if keymap.get(i) in qual and not np.isnan(qual[keymap[i]])]
    if len(pairs) < 10:
        return {"n": len(pairs), "spearman": None, "note": "Fill in at least 10 human_score values."}
    h, j = zip(*pairs)
    return {"n": len(pairs), "spearman": spearman(h, j)}


def projection_table(df: pd.DataFrame, monthly_calls: int, baseline: str = BASELINE_VARIANT) -> pd.DataFrame:
    """Scale measured cost per call to a monthly volume; makes small per-call gaps tangible."""
    per_call = df.groupby("variant_id")["cost_usd"].mean()
    per_call = per_call.reindex([v for v in VARIANTS if v in per_call.index])
    base = per_call.get(baseline, np.nan)
    return pd.DataFrame(
        {
            "variant": per_call.index,
            "cost_per_call_usd": per_call.values,
            "projected_monthly_usd": per_call.values * monthly_calls,
            f"x_vs_{baseline}": per_call.values / base,
        }
    )


def spend_summary(out_dir: Path, df: pd.DataFrame, manifest: dict) -> tuple[list[list], list[str]]:
    """Total spend by stage. Failed attempts that were superseded by a retry are not counted."""
    judg = pd.DataFrame(store.latest_by_key(out_dir / "judgments.jsonl").values())
    has_cost = not judg.empty and "judge_cost_usd" in judg
    judge_total = float(judg["judge_cost_usd"].sum()) if has_cost else 0.0
    n_judge = int(judg["judge_cost_usd"].notna().sum()) if has_cost else 0
    gen_total = float(df["cost_usd"].sum())
    jc = manifest.get("judge_config", {})
    cfg = manifest.get("config", {})
    rows = [
        ["Generation", cfg.get("chat_deployment", "?"), len(df), gen_total],
        ["Judging", jc.get("judge_deployment", "?"), n_judge, judge_total],
        ["Total", "", len(df) + n_judge, gen_total + judge_total],
    ]
    notes = []
    if jc and jc.get("judge_deployment") != cfg.get("chat_deployment") and not jc.get("judge_prices_set"):
        notes.append(
            "**Warning:** the judge uses a different deployment than the generator but JUDGE_PRICE_IN_PER_M / "
            "JUDGE_PRICE_OUT_PER_M were not set, so judge cost used the generator's prices and is probably wrong."
        )
    if not has_cost:
        notes.append("No judge cost recorded (judging has not run, or ran with an older version).")
    return rows, notes


def write_report(out_dir: Path, df: pd.DataFrame, errors: dict, figs: list[Path], monthly_calls: int = 1_000_000) -> Path:
    manifest = json.loads((out_dir / "manifest.json").read_text()) if (out_dir / "manifest.json").exists() else {}
    summ = summary_table(df, errors)
    comp = comparison_table(df)
    adh = adherence_table(df)
    agree = judge_agreement(df, out_dir)
    spend_rows, spend_notes = spend_summary(out_dir, df, manifest)

    summ.to_csv(out_dir / "summary.csv", index=False)
    comp.to_csv(out_dir / "comparisons.csv", index=False)

    n_inputs = df["input_id"].nunique()
    reps = df.groupby(["input_id", "variant_id"]).size().median()
    lines = ["# Results", ""]
    if manifest.get("dry_run"):
        lines += ["> **DRY RUN: all numbers below are synthetic mock output. They only prove the pipeline works.**", ""]
    lines += [
        f"{len(df)} successful calls, {n_inputs} inputs, about {reps:.0f} reps per input and variant, "
        f"model `{manifest.get('config', {}).get('chat_deployment', '?')}`. Intervals are 95% bootstrap CIs "
        "resampling **inputs** (not calls), because repeated runs on one input are correlated.",
        "",
        "## Per-variant summary",
        "",
        tabulate(summ.drop(columns=["prompt"]).values.tolist(), headers=[c for c in summ.columns if c != "prompt"], tablefmt="github"),
        "",
        "Variant prompts: " + "; ".join(f"`{v}` = {VARIANTS[v].strip()!r}" for v in summ["variant"]),
        "",
        "## Projected cost at scale",
        "",
        f"Measured cost per call scaled to {monthly_calls:,} calls per month. Generation calls only; judge calls are excluded. "
        f"Prices used: ${manifest.get('config', {}).get('price_in_per_m', '?')} per 1M input tokens, "
        f"${manifest.get('config', {}).get('price_out_per_m', '?')} per 1M output tokens "
        "(reasoning tokens bill as output; cached-input discounts are not applied).",
        "",
        tabulate(projection_table(df, monthly_calls).values.tolist(), headers=["variant", "cost per call (USD)", "projected monthly (USD)", f"x vs {BASELINE_VARIANT}"], tablefmt="github", floatfmt=("", ".6f", ",.2f", ".2f")),
        "",
        "## Experiment spend",
        "",
        "Actual cost of running this experiment, by stage (reasoning tokens bill as output).",
        "",
        tabulate(spend_rows, headers=["stage", "deployment", "calls", "cost (USD)"], tablefmt="github", floatfmt=("", "", "", ".4f")),
        "",
        *([*spend_notes, ""] if spend_notes else []),
        "## Paired comparisons vs baseline",
        "",
        f"Differences are variant minus `{BASELINE_VARIANT}`, paired by input. `p` is an exact paired "
        "sign-flip permutation test; `p_holm` is Holm-adjusted across variants within each metric.",
        "",
    ]
    if not comp.empty:
        show = comp.copy()
        for c in ("diff", "ci_lo", "ci_hi"):
            show[c] = show[c].map(lambda x: f"{x:.4g}")
        for c in ("p", "p_holm"):
            show[c] = show[c].map(lambda x: f"{x:.4f}")
        lines += [tabulate(show.values.tolist(), headers=list(show.columns), tablefmt="github"), ""]
    lines += ["## Constraint adherence (variants that state a length constraint)", "", tabulate(adh.values.tolist(), headers=list(adh.columns), tablefmt="github"), ""]
    if agree:
        lines += ["## Judge vs human agreement", "", f"n = {agree['n']}, Spearman rho = {agree['spearman'] if agree['spearman'] is None else round(agree['spearman'], 3)}" + (f" ({agree['note']})" if agree.get("note") else ""), ""]
    if figs:
        lines += ["## Figures", ""] + [f"![{f.stem}]({f.name})" for f in figs] + [""]
    lines += [
        "## Caveats",
        "",
        "- Inputs are a small synthetic set; effects can depend on the input type.",
        "- Judge scores are model-generated. Check agreement with the human audit before trusting them.",
        "- Latency depends on service load; trust only large gaps. Costs use assumed prices (see manifest).",
        "- Many comparisons were run; rely on `p_holm`, and treat CIs near zero as inconclusive.",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def analyze(out_dir: Path, monthly_calls: int = 1_000_000) -> Path:
    from .plots import make_figures

    df, errors = load_frame(out_dir)
    figs = make_figures(df, out_dir)
    return write_report(out_dir, df, errors, figs, monthly_calls)
