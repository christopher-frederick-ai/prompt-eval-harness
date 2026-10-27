import math

import numpy as np
import pytest

from prompt_eval.judge import parse_judgement
from prompt_eval.metrics import cost_usd, count_sentences, count_words
from prompt_eval.stats import bootstrap_mean_ci, holm, paired_compare, signflip_p, spearman


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", 0),
        ("One sentence only.", 1),
        ("First one. Second one! Third one?", 3),
        ("Growth was 3.5 percent in the quarter.", 1),  # decimal is not a sentence break
        ("- point one.\n- point two.\n- point three.", 3),  # bullets count separately
        ('He said "stop." Then left.', 2),
    ],
)
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_count_words():
    assert count_words("  a  b\nc ") == 3


def test_cost_bills_reasoning_as_output():
    # 1M input at $0.25 + 1M output at $2.00
    assert cost_usd(1_000_000, 1_000_000, 0.25, 2.00) == pytest.approx(2.25)


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    x = np.arange(1, 13, dtype=float)
    m1, lo1, hi1 = bootstrap_mean_ci(x, seed=1)
    m2, lo2, hi2 = bootstrap_mean_ci(x, seed=1)
    assert lo1 < m1 < hi1 and (m1, lo1, hi1) == (m2, lo2, hi2)


def test_signflip_detects_consistent_effect_and_ignores_noise():
    assert signflip_p(np.full(12, 1.0)) == pytest.approx(2 / 2**12)  # exact: only all-same-sign extremes
    assert signflip_p(np.array([1, -1] * 6, dtype=float)) == pytest.approx(1.0)


def test_paired_compare_identical_inputs():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    r = paired_compare(a, a)
    assert r["diff"] == 0 and r["p"] == 1.0


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert all(0 <= p <= 1 for p in adj)


def test_spearman_handles_ties_and_perfect_order():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_parse_judgement_accepts_wrapped_json_and_rejects_bad_scores():
    good = 'Sure! {"faithfulness": 5, "coverage": 4, "clarity": 3, "rationale": "ok"}'
    out = parse_judgement(good)
    assert out["quality"] == pytest.approx(4.0)
    for bad in ['{"faithfulness": 6, "coverage": 4, "clarity": 3}', "no json here", '{"faithfulness": 5}']:
        with pytest.raises(ValueError):
            parse_judgement(bad)
