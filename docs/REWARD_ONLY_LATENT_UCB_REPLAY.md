# Latent Child-UCB offline evaluation

## Protocol

- Evaluated 42 fixed historical runs after 10 valid transitions.
- Score: `parent_utility + gain_trust * (posterior_mean_gain + beta * latent_std_gain)`.
- Latent std contains epistemic coefficient uncertainty only; observation noise, the predictive 2x multiplier, and the predictive floor are excluded.
- Beta is annealed during the final 10% of either step or wall-clock budget. No gain clipping or MCGS score fusion is used.
- The production Thompson selector remains unchanged; this is an offline experimental comparison.

## Factual upper-bound behavior

`Realized <= upper` compares a noisy realized transition with a latent-mean upper bound. It is a descriptive stress test, not nominal confidence coverage.

| Beta | Mean effective beta | Factual N | Realized <= upper | Upper gain > 0 | Median optimism / scale | Mean optimism / scale | P90 optimism / scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 644 | 0.537 | 0.573 | 0.041 | -0.016 | 0.999 |
| 0.10 | 0.099 | 644 | 0.573 | 0.618 | 0.069 | 0.099 | 1.101 |
| 0.25 | 0.247 | 644 | 0.621 | 0.666 | 0.100 | 0.272 | 1.314 |
| 0.50 | 0.495 | 644 | 0.691 | 0.733 | 0.182 | 0.561 | 1.547 |
| 1.00 | 0.990 | 644 | 0.801 | 0.845 | 0.316 | 1.137 | 2.011 |
| 1.50 | 1.484 | 644 | 0.893 | 0.927 | 0.482 | 1.714 | 2.606 |

## Candidate-choice replay

| Beta | Decisions | Historical agreement | Current TS agreement | Mean-policy agreement | Select max parent | P90 parent sacrifice / scale | P99 selected bonus / scale | P99 largest candidate bonus / scale | Factual matches | Positive gain on matches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 472 | 0.396 | 0.850 | 1.000 | 0.809 | 0.016 | 0.000 | 0.000 | 133 | 0.301 |
| 0.10 | 472 | 0.400 | 0.850 | 0.977 | 0.801 | 0.018 | 0.263 | 0.850 | 134 | 0.299 |
| 0.25 | 472 | 0.394 | 0.845 | 0.953 | 0.788 | 0.019 | 1.582 | 2.124 | 130 | 0.315 |
| 0.50 | 472 | 0.398 | 0.843 | 0.922 | 0.797 | 0.015 | 3.165 | 4.249 | 132 | 0.333 |
| 1.00 | 472 | 0.390 | 0.790 | 0.845 | 0.758 | 0.061 | 6.329 | 8.498 | 128 | 0.336 |
| 1.50 | 472 | 0.375 | 0.758 | 0.792 | 0.739 | 0.086 | 9.494 | 12.747 | 123 | 0.358 |

## Latent uncertainty scale

Across factual predictions, latent std has median / P90 / P99 values of 0.311 / 1.404 / 15.959 run-local scales.

## Behavior by available valid history

| Prior valid transitions | Factual N | Latent std median / P90 / P99 |
|---|---:|---:|
| 10-14 | 125 | 0.358 / 1.229 / 65.866 |
| 15-19 | 90 | 0.325 / 2.321 / 9.953 |
| 20-29 | 159 | 0.291 / 1.807 / 24.629 |
| 30+ | 270 | 0.279 / 0.885 / 4.571 |

| Prior valid transitions | Beta | Decisions | Mean-policy agreement | Select max parent | P90 parent sacrifice | P99 largest bonus |
|---|---:|---:|---:|---:|---:|---:|
| 10-14 | 0.10 | 82 | 0.976 | 0.951 | 0.000 | 1.033 |
| 10-14 | 0.25 | 82 | 0.951 | 0.951 | 0.000 | 2.584 |
| 10-14 | 0.50 | 82 | 0.951 | 0.951 | 0.000 | 5.167 |
| 10-14 | 1.00 | 82 | 0.915 | 0.939 | 0.000 | 10.334 |
| 10-14 | 1.50 | 82 | 0.902 | 0.927 | 0.000 | 15.501 |
| 15-19 | 0.10 | 53 | 0.943 | 0.887 | 0.001 | 0.725 |
| 15-19 | 0.25 | 53 | 0.887 | 0.830 | 0.054 | 1.813 |
| 15-19 | 0.50 | 53 | 0.887 | 0.830 | 0.054 | 3.626 |
| 15-19 | 1.00 | 53 | 0.755 | 0.736 | 0.109 | 7.253 |
| 15-19 | 1.50 | 53 | 0.736 | 0.755 | 0.109 | 10.879 |
| 20-29 | 0.10 | 114 | 0.982 | 0.877 | 0.006 | 0.853 |
| 20-29 | 0.25 | 114 | 0.965 | 0.886 | 0.001 | 2.132 |
| 20-29 | 0.50 | 114 | 0.939 | 0.886 | 0.001 | 4.264 |
| 20-29 | 1.00 | 114 | 0.842 | 0.816 | 0.201 | 8.528 |
| 20-29 | 1.50 | 114 | 0.781 | 0.746 | 0.804 | 12.792 |
| 30+ | 0.10 | 223 | 0.982 | 0.686 | 0.039 | 0.766 |
| 30+ | 0.25 | 223 | 0.964 | 0.668 | 0.039 | 1.915 |
| 30+ | 0.50 | 223 | 0.910 | 0.686 | 0.023 | 3.831 |
| 30+ | 1.00 | 223 | 0.843 | 0.668 | 0.023 | 7.662 |
| 30+ | 1.50 | 223 | 0.771 | 0.664 | 0.039 | 11.493 |

## Offline decision

- Do not replace the production Thompson acquisition with latent Child-UCB based on this replay.
- Beta 0.10 is the only low-risk online ablation: it agrees with the posterior-mean policy on 97.7% of replayed decisions and keeps the P99 largest candidate bonus below one run-local scale.
- Beta 0.25 is a stress-test setting, not a default. Beta 0.50 and above expose multi-scale optimism tails and are rejected for the current small-data posterior.
- The more promising next change is the separately evaluated incumbent-relative BNVS target and compact feature set; acquisition should be isolated as a later ablation.

## Interpretation boundary

Historical agreement exposes an outcome only when the replay chooses the action actually executed. Different beta values match different action subsets, so their matched positive-gain rates are not causally comparable. Final-score conclusions require online paired runs.
