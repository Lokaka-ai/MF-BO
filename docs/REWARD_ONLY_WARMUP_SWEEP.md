# Reward-only BO warm-up sensitivity

## Protocol

- Evaluated 42 fixed historical runs containing 985 valid continuous-gain transitions.
- Each prediction uses only earlier transitions from the same run; no pretraining or cross-run transfer is used.
- Thresholds change when the model would become active, but do not change the fixed historical action sequence.
- Consequently this measures signal quality and projected policy exposure, not counterfactual final-score uplift.

## Threshold sweep

| Start after valid transitions | Active factual predictions | Runs using BO | Active fraction | Mean gain trust when active | Mean trust over all gains | Acquisition NMAE | Raw-model NMAE | Acquisition median abs. error | Acquisition P90 error | Sign accuracy | Pooled Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 864 | 34 | 0.877 | 0.188 | 0.165 | 0.982 | 1.291 | 0.200 | 1.553 | 0.587 | 0.203 |
| 5 | 797 | 33 | 0.809 | 0.191 | 0.154 | 0.966 | 1.290 | 0.194 | 1.453 | 0.590 | 0.231 |
| 8 | 702 | 29 | 0.713 | 0.196 | 0.139 | 1.011 | 1.338 | 0.190 | 1.434 | 0.591 | 0.229 |
| 10 | 644 | 29 | 0.654 | 0.199 | 0.130 | 0.940 | 1.260 | 0.184 | 1.375 | 0.590 | 0.238 |
| 15 | 519 | 19 | 0.527 | 0.210 | 0.111 | 0.850 | 1.216 | 0.174 | 1.383 | 0.592 | 0.226 |
| 20 | 429 | 17 | 0.436 | 0.219 | 0.095 | 0.706 | 0.931 | 0.149 | 1.150 | 0.611 | 0.246 |
| 30 | 270 | 15 | 0.274 | 0.250 | 0.069 | 0.630 | 0.823 | 0.145 | 0.903 | 0.644 | 0.288 |

The rows are not evaluated on identical suffixes: a later threshold removes early, usually harder predictions. Use the history-bin table below to decide when the signal becomes usable; use this table mainly to quantify coverage and policy exposure.

## Model quality by exact available-history range

| Prior valid transitions | N | NMAE | History-mean NMAE | Zero-gain NMAE | Median abs. error | P90 error | Sign accuracy | Pooled Spearman | 1-sigma coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-4 | 67 | 1.307 | 1.441 | 1.150 | 0.516 | 2.977 | 0.552 | -0.115 | 0.881 |
| 5-7 | 95 | 0.935 | 0.831 | 0.606 | 0.370 | 1.393 | 0.579 | 0.192 | 0.989 |
| 8-9 | 58 | 2.203 | 2.137 | 1.767 | 0.348 | 4.132 | 0.603 | 0.044 | 0.862 |
| 10-14 | 125 | 1.446 | 1.508 | 1.307 | 0.410 | 1.636 | 0.584 | 0.262 | 0.936 |
| 15-19 | 90 | 2.571 | 1.646 | 1.423 | 0.310 | 3.720 | 0.500 | 0.143 | 0.922 |
| 20-29 | 159 | 1.116 | 0.961 | 0.835 | 0.305 | 2.114 | 0.553 | 0.177 | 0.943 |
| 30+ | 270 | 0.823 | 0.759 | 0.593 | 0.196 | 1.419 | 0.644 | 0.288 | 0.952 |

## Decision for the current selector

For these logs, 10 valid transitions is the earliest defensible production threshold. The 3-4 range has negative pooled rank signal, the 8-9 range has a large P90 error, and waiting until 15 does not avoid the weak 15-19 range. Waiting until 20 improves the observed suffix metrics, but reduces model-active factual coverage and excludes many short runs. The current 0.10 starting gain trust limits the effect of noisy 10-19 predictions.

The data therefore support retaining the 10-transition warm-up for the first online pilot. After warm-up, the acquisition is pure BO: gain trust scales the predicted gain and is not a mixture with MCGS. A conservative follow-up ablation can keep trust at 0.10 through 19 valid transitions and ramp from 20 to 30, rather than changing the fit threshold itself.

## Interpretation boundary

Changing the warm-up in a real online run changes which parents are selected and therefore changes all later training data. Those counterfactual outcomes are absent here. A causal threshold comparison requires randomized online runs against pure progressive MCGS.
