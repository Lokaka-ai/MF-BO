# BNVS design offline comparison

## Evaluation boundary

- Evaluated 42 historical runs with strictly prequential fitting.
- H=1 target is clipped `(child_utility - pre_action_global_best) / scale`; invalid outcomes are `-1`.
- H=2 uses the best valid utility among the child and its direct children, while training labels are updated only after descendants have actually appeared.
- The 10-feature model uses only exactly reconstructable state. The 12-feature model adds active-node identity and tree distance reconstructed from continuation snapshot ordering and the observed parent graph.
- Root was not a regular candidate in these logs, so root-restart choice quality cannot be replayed.

## Same H=1 target after 10 valid gain observations

| Model | N | MAE | Clipped MAE | Median AE | P90 AE | New-incumbent sign accuracy | Precision | Recall | Pooled Spearman | Median per-run Spearman | Invalid MAE | Out-of-range prediction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current local-gain model, raw child estimate | 846 | 0.996 | 0.373 | 0.368 | 1.557 | 0.779 | 0.214 | 0.420 | 0.368 | 0.302 | 1.221 | 0.281 |
| Current gain-trust mean acquisition | 846 | 0.813 | 0.308 | 0.299 | 1.017 | 0.838 | 0.279 | 0.352 | 0.440 | 0.370 | 1.380 | 0.221 |
| Incumbent target + current 15 features | 846 | 0.371 | 0.352 | 0.319 | 0.749 | 0.874 | 0.212 | 0.080 | 0.358 | 0.347 | 0.557 | 0.066 |
| BNVS H=1, exact 10 features | 846 | 0.353 | 0.341 | 0.303 | 0.720 | 0.883 | 0.238 | 0.057 | 0.390 | 0.305 | 0.510 | 0.060 |
| BNVS H=1, active-aware 12 features | 846 | 0.355 | 0.342 | 0.304 | 0.723 | 0.878 | 0.200 | 0.057 | 0.386 | 0.363 | 0.509 | 0.069 |

## BNVS warm-up sensitivity

| Minimum prior normal observations | Model | N | MAE | Sign accuracy | Pooled Spearman | Median per-run Spearman | Invalid MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | BNVS H=1, exact 10 features | 1140 | 0.379 | 0.842 | 0.402 | 0.401 | 0.542 |
| 3 | BNVS H=1, active-aware 12 features | 1140 | 0.381 | 0.838 | 0.396 | 0.381 | 0.541 |
| 5 | BNVS H=1, exact 10 features | 1060 | 0.368 | 0.854 | 0.406 | 0.332 | 0.531 |
| 5 | BNVS H=1, active-aware 12 features | 1060 | 0.369 | 0.849 | 0.400 | 0.313 | 0.530 |
| 8 | BNVS H=1, exact 10 features | 952 | 0.360 | 0.874 | 0.393 | 0.277 | 0.520 |
| 8 | BNVS H=1, active-aware 12 features | 952 | 0.362 | 0.869 | 0.388 | 0.286 | 0.518 |
| 10 | BNVS H=1, exact 10 features | 887 | 0.357 | 0.882 | 0.387 | 0.262 | 0.514 |
| 10 | BNVS H=1, active-aware 12 features | 887 | 0.358 | 0.876 | 0.384 | 0.255 | 0.513 |
| 15 | BNVS H=1, exact 10 features | 739 | 0.346 | 0.888 | 0.401 | 0.319 | 0.492 |
| 15 | BNVS H=1, active-aware 12 features | 739 | 0.347 | 0.885 | 0.399 | 0.342 | 0.490 |
| 20 | BNVS H=1, exact 10 features | 614 | 0.339 | 0.897 | 0.423 | 0.421 | 0.499 |
| 20 | BNVS H=1, active-aware 12 features | 614 | 0.340 | 0.893 | 0.418 | 0.412 | 0.497 |
| 30 | BNVS H=1, exact 10 features | 439 | 0.337 | 0.916 | 0.419 | 0.368 | 0.491 |
| 30 | BNVS H=1, active-aware 12 features | 439 | 0.338 | 0.913 | 0.411 | 0.353 | 0.488 |

## H=1 versus delayed H=2 credit

| Predictor evaluated on eventual H=2 target | N | MAE | Sign accuracy | Pooled Spearman | Median per-run Spearman |
|---|---:|---:|---:|---:|---:|
| H=1 active-aware model | 846 | 0.365 | 0.857 | 0.371 | 0.323 |
| H=2 active-aware model | 846 | 0.378 | 0.825 | 0.361 | 0.320 |

## Vanilla Thompson candidate replay after 5 observations

| Model | Decisions | Historical agreement | Mean-policy agreement | Current TS agreement | Max-parent rate | Mean parent rank | Continue rate | P99 selected abs score | P99 max candidate abs score | Factual matches | New-incumbent rate on matches | Invalid rate on matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BNVS H=1, exact 10 features | 576 | 0.201 | 0.351 | 0.235 | 0.245 | 0.656 | 0.273 | 1.795 | 5.488 | 116 | 0.190 | 0.267 |
| BNVS H=1, active-aware 12 features | 576 | 0.182 | 0.368 | 0.186 | 0.222 | 0.628 | 0.327 | 2.633 | 5.565 | 105 | 0.200 | 0.219 |
| BNVS H=2, active-aware 12 features | 576 | 0.181 | 0.359 | 0.197 | 0.217 | 0.637 | 0.320 | 2.551 | 5.220 | 104 | 0.192 | 0.279 |

## Interpretation boundary

Factual prediction metrics test the target and surrogate on historically selected actions. Candidate replay tests how the policy would behave, but only matched historical actions expose outcomes. It cannot prove final-score improvement or compare counterfactual continue/switch decisions.
