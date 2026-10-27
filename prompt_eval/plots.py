"""Two figures. Colors follow a fixed variant order (never cycled or re-ranked), marks are
labeled directly so identity is never color-alone, and the grid is recessive."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import VARIANTS
from .stats import bootstrap_mean_ci

# Categorical slots 1-5 of the validated reference palette (light mode), one per variant.
COLORS = {"v1": "#2a78d6", "v2": "#eb6834", "v3": "#1baf7a", "v4": "#eda100", "v5": "#e87ba4", "v6": "#008300"}
LABEL_OFFSET = {"v1": (-24, 9), "v2": (9, -15), "v3": (9, -13), "v4": (9, 7), "v5": (9, 8), "v6": (-24, -16)}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _per_input(df, col, v):
    return df[df["variant_id"] == v].groupby("input_id")[col].mean().dropna().values


def make_figures(df, out_dir: Path) -> list[Path]:
    variants = [v for v in VARIANTS if v in set(df["variant_id"])]
    figs = []

    # 1) Cost vs quality (or cost vs words if no judge scores), with 95% CIs over inputs.
    has_q = df["quality"].notna().any()
    ycol, ylabel = ("quality", "Judge quality (1-5, mean of 3 dimensions)") if has_q else ("words", "Words per summary")
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150, facecolor=SURFACE)
    for v in variants:
        cx, cl, ch = bootstrap_mean_ci(_per_input(df, "cost_usd", v) * 1000)
        qy, ql, qh = bootstrap_mean_ci(_per_input(df, ycol, v))
        ax.errorbar(cx, qy, xerr=[[cx - cl], [ch - cx]], yerr=[[qy - ql], [qh - qy]], fmt="o", ms=9, color=COLORS[v],
                    ecolor=COLORS[v], elinewidth=1.5, capsize=3, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(v, (cx, qy), textcoords="offset points", xytext=LABEL_OFFSET.get(v, (9, 7)), color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("Cost per 1,000 calls (USD)")
    ax.set_ylabel(ylabel)
    ax.set_title("Cost vs quality by prompt variant (95% CI over inputs)", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig_cost_vs_quality.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    figs.append(p)

    # 2) Where the output tokens go: visible answer vs hidden reasoning (stacked, 2px surface gap).
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    vis = np.array([df[df.variant_id == v]["visible_tokens"].mean() for v in variants])
    rea = np.array([df[df.variant_id == v]["reasoning_tokens"].mean() for v in variants])
    x = np.arange(len(variants))
    ax.bar(x, vis, width=0.55, color="#2a78d6", edgecolor=SURFACE, linewidth=2, label="Visible answer")
    has_reasoning = rea.sum() > 0
    if has_reasoning:
        ax.bar(x, rea, width=0.55, bottom=vis, color="#eb6834", edgecolor=SURFACE, linewidth=2, label="Hidden reasoning")
    for i, (a, b) in enumerate(zip(vis, rea)):
        ax.text(i, a + b + max(vis + rea) * 0.015, f"{a + b:.0f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, variants)
    ax.set_ylabel("Mean output tokens per call")
    title = "Output tokens: visible answer vs hidden reasoning" if has_reasoning else "Output tokens per call (no hidden reasoning reported)"
    ax.set_title(title, loc="left", fontsize=11)
    if has_reasoning:
        ax.legend(frameon=False, labelcolor=INK2)
    _style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    p = out_dir / "fig_token_split.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    figs.append(p)
    return figs
