"""Small, dependency-light statistics (numpy only).

Unit of analysis = the INPUT PARAGRAPH, not the individual call. Repeated runs on the same
paragraph are correlated, so treating all calls as independent would overstate the sample
size. We first average over reps within each input, then bootstrap / permute across inputs.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def bootstrap_mean_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI, resampling inputs with replacement."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def signflip_p(d, n_mc: int = 50_000, seed: int = 0) -> float:
    """Two-sided paired permutation (sign-flip) p-value for mean(d) == 0.

    Exact enumeration when there are <= 16 pairs, Monte Carlo otherwise.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(product((-1.0, 1.0), repeat=n)))
        stat = np.abs((signs * d).mean(axis=1))
        return float(np.mean(stat >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_mc, n))
    stat = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(stat >= obs - 1e-12)) / (1 + n_mc))


def paired_compare(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Difference a - b over matched inputs: mean, bootstrap CI, sign-flip p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    return {"diff": mean, "lo": lo, "hi": hi, "p": signflip_p(d, seed=seed)}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def spearman(x, y) -> float:
    """Spearman rank correlation (Pearson on average ranks)."""
    xr = _rank(np.asarray(x, dtype=float))
    yr = _rank(np.asarray(y, dtype=float))
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):  # average ties
        mask = a == v
        ranks[mask] = ranks[mask].mean()
    return ranks
