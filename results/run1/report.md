# Results

360 successful calls, 12 inputs, about 5 reps per input and variant, model `gpt4o-chat`. Intervals are 95% bootstrap CIs resampling **inputs** (not calls), because repeated runs on one input are correlated.

## Per-variant summary

| variant   |   calls | error_rate   | words             |   words_sd_within_input | output_tokens   | reasoning_share   |   cost_per_1k_calls_usd |   latency_mean_s |   latency_p95_s | one_sentence_rate   | quality           |
|-----------|---------|--------------|-------------------|-------------------------|-----------------|-------------------|-------------------------|------------------|-----------------|---------------------|-------------------|
| v1        |      60 | 0.0%         | 25.7 [23.2, 28.0] |                     3.1 | 31 [28, 34]     | 0%                |                   0.52  |             1.1  |            1.73 | 100%                | 4.82 [4.76, 4.89] |
| v2        |      60 | 0.0%         | 42.3 [35.8, 49.0] |                     3.9 | 51 [44, 59]     | 0%                |                   0.717 |             1.18 |            1.8  | 8%                  | 5.00 [5.00, 5.00] |
| v3        |      60 | 0.0%         | 29.7 [25.5, 34.0] |                     3.9 | 38 [32, 43]     | 0%                |                   0.584 |             1.15 |            2.02 | 87%                 | 4.93 [4.86, 4.98] |
| v4        |      60 | 0.0%         | 46.6 [38.2, 54.9] |                     4.4 | 58 [48, 68]     | 0%                |                   0.779 |             1.75 |            2.55 | 2%                  | 4.98 [4.95, 5.00] |
| v5        |      60 | 0.0%         | 26.2 [23.2, 28.8] |                     2.9 | 33 [29, 37]     | 0%                |                   0.533 |             1.24 |            1.77 | 100%                | 4.86 [4.78, 4.92] |
| v6        |      60 | 0.0%         | 22.3 [20.3, 23.7] |                     1.7 | 28 [25, 30]     | 0%                |                   0.576 |             1.01 |            1.59 | 100%                | 4.81 [4.71, 4.89] |

Variant prompts: `v1` = 'Summarize the following in one sentence:'; `v2` = 'Write a short summary of this paragraph:'; `v3` = 'Condense this into a concise statement:'; `v4` = 'TL;DR:'; `v5` = 'TL;DR in one sentence:'; `v6` = 'You are a careful editor. Summarize the paragraph below in a single sentence of at most 25 words. Keep the key facts, do not add anything that is not in the paragraph, and return only the summary.'

## Projected cost at scale

Measured cost per call scaled to 1,000,000 calls per month. Generation calls only; judge calls are excluded. Prices used: $2.5 per 1M input tokens, $10.0 per 1M output tokens (reasoning tokens bill as output; cached-input discounts are not applied).

| variant   |   cost per call (USD) |   projected monthly (USD) |   x vs v1 |
|-----------|-----------------------|---------------------------|-----------|
| v1        |              0.000520 |                    519.58 |      1.00 |
| v2        |              0.000717 |                    717.42 |      1.38 |
| v3        |              0.000584 |                    583.58 |      1.12 |
| v4        |              0.000779 |                    779.42 |      1.50 |
| v5        |              0.000533 |                    532.75 |      1.03 |
| v6        |              0.000576 |                    576.42 |      1.11 |

## Experiment spend

Actual cost of running this experiment, by stage (reasoning tokens bill as output).

| stage      | deployment   |   calls |   cost (USD) |
|------------|--------------|---------|--------------|
| Generation | gpt4o-chat   |     360 |       0.2226 |
| Judging    | gpt-5-mini   |     360 |       0.3055 |
| Total      |              |     720 |       0.5281 |

## Paired comparisons vs baseline

Differences are variant minus `v1`, paired by input. `p` is an exact paired sign-flip permutation test; `p_holm` is Holm-adjusted across variants within each metric.

| metric                   | variant   | vs   |   n_inputs |     diff |    ci_lo |    ci_hi |      p |   p_holm | significant_holm_0.05   |
|--------------------------|-----------|------|------------|----------|----------|----------|--------|----------|-------------------------|
| Words                    | v2        | v1   |         12 | 16.62    | 11.67    | 22.12    | 0.0005 |   0.0024 | True                    |
| Words                    | v3        | v1   |         12 |  4.05    |  1.367   |  7.267   | 0.0137 |   0.0273 | True                    |
| Words                    | v4        | v1   |         12 | 20.93    | 13.9     | 28.42    | 0.001  |   0.0039 | True                    |
| Words                    | v5        | v1   |         12 |  0.5667  | -0.95    |  2.1     | 0.5059 |   0.5059 | False                   |
| Words                    | v6        | v1   |         12 | -3.4     | -4.85    | -1.95    | 0.0024 |   0.0073 | True                    |
| Output tokens            | v2        | v1   |         12 | 20.03    | 14       | 26.62    | 0.0005 |   0.0024 | True                    |
| Output tokens            | v3        | v1   |         12 |  6.65    |  2.717   | 10.83    | 0.0093 |   0.0205 | True                    |
| Output tokens            | v4        | v1   |         12 | 27.23    | 18.2     | 36.57    | 0.001  |   0.0039 | True                    |
| Output tokens            | v5        | v1   |         12 |  1.817   | -0.6     |  4.55    | 0.2241 |   0.2241 | False                   |
| Output tokens            | v6        | v1   |         12 | -3.317   | -5.183   | -1.483   | 0.0068 |   0.0205 | True                    |
| Cost per 1k calls (USD)  | v2        | v1   |         12 |  0.1978  |  0.1375  |  0.2637  | 0.0005 |   0.0024 | True                    |
| Cost per 1k calls (USD)  | v3        | v1   |         12 |  0.064   |  0.02467 |  0.1058  | 0.0103 |   0.0205 | True                    |
| Cost per 1k calls (USD)  | v4        | v1   |         12 |  0.2598  |  0.1695  |  0.3532  | 0.001  |   0.0039 | True                    |
| Cost per 1k calls (USD)  | v5        | v1   |         12 |  0.01317 | -0.011   |  0.0405  | 0.397  |   0.397  | False                   |
| Cost per 1k calls (USD)  | v6        | v1   |         12 |  0.05683 |  0.03817 |  0.07517 | 0.001  |   0.0039 | True                    |
| Latency (s)              | v2        | v1   |         12 |  0.08042 | -0.1978  |  0.3151  | 0.5767 |   1      | False                   |
| Latency (s)              | v3        | v1   |         12 |  0.04461 | -0.138   |  0.1828  | 0.666  |   1      | False                   |
| Latency (s)              | v4        | v1   |         12 |  0.6543  |  0.09374 |  1.346   | 0.0596 |   0.2979 | False                   |
| Latency (s)              | v5        | v1   |         12 |  0.1363  | -0.2203  |  0.6071  | 0.603  |   1      | False                   |
| Latency (s)              | v6        | v1   |         12 | -0.08591 | -0.3133  |  0.07432 | 0.583  |   1      | False                   |
| Judge quality (1-5)      | v2        | v1   |         12 |  0.1778  |  0.1111  |  0.2444  | 0.002  |   0.0098 | True                    |
| Judge quality (1-5)      | v3        | v1   |         12 |  0.1111  |  0.04444 |  0.1778  | 0.0195 |   0.0586 | False                   |
| Judge quality (1-5)      | v4        | v1   |         12 |  0.1556  |  0.08333 |  0.2278  | 0.0049 |   0.0195 | True                    |
| Judge quality (1-5)      | v5        | v1   |         12 |  0.03333 | -0.01667 |  0.09444 | 0.4219 |   0.8438 | False                   |
| Judge quality (1-5)      | v6        | v1   |         12 | -0.01667 | -0.06667 |  0.03333 | 0.6875 |   0.8438 | False                   |
| Within-input SD of words | v2        | v1   |         12 |  0.7762  | -0.8588  |  2.792   | 0.5103 |   1      | False                   |
| Within-input SD of words | v3        | v1   |         12 |  0.7329  | -1.093   |  2.857   | 0.5425 |   1      | False                   |
| Within-input SD of words | v4        | v1   |         12 |  1.289   | -0.4478  |  3.092   | 0.2021 |   0.8086 | False                   |
| Within-input SD of words | v5        | v1   |         12 | -0.1963  | -1.269   |  0.9715  | 0.7407 |   1      | False                   |
| Within-input SD of words | v6        | v1   |         12 | -1.416   | -2.467   | -0.3134  | 0.0361 |   0.1807 | False                   |

## Constraint adherence (variants that state a length constraint)

| variant_id   |   calls |   one_sentence_rate |   within_word_limit |
|--------------|---------|---------------------|---------------------|
| v1           |      60 |                   1 |          nan        |
| v5           |      60 |                   1 |          nan        |
| v6           |      60 |                   1 |            0.866667 |

## Judge vs human agreement

n = 30, Spearman rho = -0.377

## Figures

![fig_cost_vs_quality](fig_cost_vs_quality.png)
![fig_token_split](fig_token_split.png)

## Caveats

- Inputs are a small synthetic set; effects can depend on the input type.
- Judge scores are model-generated. Check agreement with the human audit before trusting them.
- Latency depends on service load; trust only large gaps. Costs use assumed prices (see manifest).
- Many comparisons were run; rely on `p_holm`, and treat CIs near zero as inconclusive.
