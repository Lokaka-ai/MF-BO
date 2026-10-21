# Pure reward-only acquisition offline replay

## Protocol

- Replayed multi-candidate decisions from 42 fixed historical runs after 10 valid transitions.
- The production acquisition is parent utility plus a gain-trust-scaled joint Thompson gain; no MCGS rank enters the score and no clipping is applied.
- Temperature 0 is the conservative posterior-mean acquisition. Other rows use the same posterior draw seed with increasing exploration.
- Historical agreement exposes a factual outcome only when replay selects the same action as the logged policy; it is not a counterfactual policy score.

## Replay results

| Thompson temperature | Decisions | Historical agreement | Agreement with posterior mean | Select max-utility parent | Median parent sacrifice / scale | P90 sacrifice / scale | P90 / P99 selected abs gain contribution | P99 largest positive / abs candidate contribution | Factual matches | Positive gain on matches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 472 | 0.396 | 1.000 | 0.809 | 0.000 | 0.016 | 0.376 / 0.805 | 3.128 / 21.316 | 133 | 0.301 |
| 0.25 | 472 | 0.364 | 0.850 | 0.771 | 0.000 | 0.018 | 0.388 / 1.504 | 3.549 / 21.277 | 123 | 0.325 |
| 0.50 | 472 | 0.354 | 0.771 | 0.725 | 0.000 | 0.058 | 0.418 / 2.982 | 5.572 / 21.561 | 119 | 0.336 |
| 1.00 | 472 | 0.333 | 0.655 | 0.631 | 0.000 | 0.244 | 0.780 / 8.439 | 12.327 / 26.785 | 111 | 0.360 |

## Stability of the production temperature

Across 16 deterministic posterior seeds per decision, the median modal-choice share is 1.000 and the mean number of unique choices is 1.782.

With no clipping, the production temperature has a P99 / maximum selected absolute gain contribution of 1.504 / 3.159 run-local scales. The P99 / maximum largest positive contribution present in a candidate set is 3.549 / 5.180 scales.

## Interpretation boundary

Replay choice changes cannot be assigned outcomes when they differ from the logged action. The report therefore validates acquisition scale, unbounded-tail exposure, and stochastic stability only. Final-score comparison against progressive MCGS requires online paired runs.
