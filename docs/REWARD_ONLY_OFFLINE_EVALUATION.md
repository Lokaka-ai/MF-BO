# Reward-only local BO offline evaluation

## Protocol

- Evaluated 42 independent run logs with a warm-up of 10 valid transitions per run.
- Every prediction is strictly prequential: only earlier transitions from the same run are used.
- No pretraining, task metadata, validity model, or cost model is used by the reward-only model.
- Errors are divided by the run-local robust reward scale before pooling across heterogeneous metrics.
- Only factual outcomes of historically selected actions are scored. The logs do not contain outcomes for unselected candidates, so this analysis cannot establish counterfactual policy regret or final-score uplift.

## Aggregate factual prediction results

| Metric | Current acquisition contribution | Raw reward posterior mean | Previous 49-feature model |
|---|---:|---:|---:|
| Normalized MAE | 0.940 | 1.260 | 199425.650 |
| Median normalized absolute error | 0.184 | 0.292 | 0.364 |
| 90th-percentile normalized absolute error | 1.375 | 1.961 | 2.669 |
| Median per-run normalized MAE | 0.424 | 0.542 | 0.680 |
| Normalized RMSE | 4.426 | 5.974 | 5060650.712 |
| Gain-sign accuracy | 0.590 | 0.590 | 0.547 |
| Pooled Spearman | 0.240 | 0.238 | 0.110 |
| Median per-run Spearman | -0.000 | 0.001 | 0.061 |

The evaluation contains 644 factual next-transition predictions. The reward-only model has lower absolute error on 360 predictions versus 284 for the previous model (paired sign-test p=0.003).

For context, the historical-mean normalized MAE is 1.078, and the zero-gain baseline is 0.907. The observed positive-gain rate is 0.502.

## MCGS signal-proxy comparison

MCGS does not predict continuous gain, so NMAE and uncertainty coverage are not defined for it. The closest factual diagnostic available in these logs is the selected parent's selection-time `mcgs_q`, which is an average of past ordinal search rewards. It is only a signal proxy: the full progressive-MCGS policy also uses UCT exploration, tree traversal, and the time-dependent Top-K switch.

| Metric | Reward-only gain model | Previous gain model | MCGS-Q proxy |
|---|---:|---:|---:|
| Gain direction/proxy accuracy | 0.590 | 0.547 | 0.466 |
| Pooled Spearman with realized gain | 0.238 | 0.110 | -0.033 |
| Median per-run Spearman | 0.001 | 0.061 | 0.054 |
| Mean per-run Spearman | 0.043 | -0.034 | 0.058 |

`mcgs_q` is exactly zero for 0.651 of evaluated selected actions. Consequently, thresholding it at zero is a weak and tie-heavy direction proxy rather than a calibrated gain forecast.

## Findings

- The compact model improves the median and tail factual errors, gain-sign accuracy, and pooled rank correlation relative to the previous 49-feature model. The previous model's pooled mean errors are dominated by catastrophic extrapolation outliers.
- Scaling the model gain by the 0.10-to-0.25 trust schedule improves factual error substantially while preserving its sign and rank ordering. This is the contribution used by the pure BO acquisition; it is not an MCGS mixture weight.
- The raw compact model does not beat the historical-mean or zero-gain baselines on pooled normalized MAE. Its median per-run Spearman correlation is close to zero, so the acquisition deliberately keeps the learned gain contribution small.
- No clipping is applied. The separate candidate replay report quantifies posterior-tail exposure and choice stability under the requested unbounded acquisition.

## History-size slices

| Prior valid transitions | N | Acquisition NMAE | Raw reward NMAE | Previous NMAE | Reward-only sign accuracy |
|---|---:|---:|---:|---:|---:|
| 10-14 | 125 | 1.285 | 1.446 | 12.626 | 0.584 |
| 15-19 | 90 | 1.566 | 2.571 | 7.708 | 0.500 |
| 20-29 | 159 | 0.842 | 1.116 | 15.718 | 0.553 |
| 30+ | 270 | 0.630 | 0.823 | 475649.435 | 0.644 |

## Historical policy outcome comparison

This section compares the actually executed Legacy progressive-MCGS runs with the actually executed **previous** BO runs. It is useful context, but it is not an outcome evaluation of the new reward-only policy, which has not yet been run online.

| Historical outcome | Legacy progressive MCGS | Previous BO |
|---|---:|---:|
| Official private-score wins | 8/21 | 13/21 |
| Any medal | 16/21 | 14/21 |
| Gold medal | 10/21 | 12/21 |
| Candidate launches | 877 | 930 |
| Valid yield | 0.716 | 0.718 |
| Tasks where best candidate arrived later | 7 | 14 |

The median old-BO minus MCGS normalized score was +0.025 gold-gap units; the two-sided score-win sign-test p-value was 0.383. Thus the 13-8 old-BO win count is not statistically decisive from one seed per task.

Within the historical old-BO runs, progressive-MCGS fallback actions and learned-model actions had the following observational outcomes:

| Selected-action subgroup | Progressive-MCGS fallback | Previous learned selector |
|---|---:|---:|
| Actions | 165 | 470 |
| Comparable realized gains | 135 | 356 |
| Positive realized gain | 0.637 | 0.455 |
| Valid action | 0.818 | 0.757 |

The fallback comparison is confounded by warm-up stage, operator mix, parent quality, and remaining headroom; it is descriptive evidence, not a causal estimate of MCGS superiority.

## Interpretation boundary

This is a surrogate-quality gate, not an offline proof that the pure Thompson acquisition improves the final MLEbench score. A policy-level conclusion requires new randomized online runs because each historical decision exposes only the result of the action that the old policy executed.
