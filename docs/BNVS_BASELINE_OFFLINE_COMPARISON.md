# BNVS step-zero baseline comparison

## Protocol

- 42 historical runs were processed strictly prequentially; every prediction used only earlier transitions in the same run.
- BNVS starts from the Bayesian prior at zero observations. No MCGS warm-up or hard observation threshold is used.
- Mean-Greedy and BNVS-TS share the same H=1 target and exact 10 features; only the selection rule differs.
- BNVS-TS-local shares the same 10 features, prior, and Thompson draw seeds, but valid labels use clipped parent-relative gain. Invalid outcomes remain -1.
- Factual value metrics use all 42 runs. Policy replay is restricted to the 21 Legacy runs whose logged selector was Progressive MCGS.

## Baselines

| Method | Value model | Selection rule | Purpose |
|---|---|---|---|
| MCGS | None | Original logged Progressive MCGS | Original baseline |
| Mean-Greedy | Bayesian linear, H=1 | Largest posterior mean | Test whether learned value alone is useful |
| BNVS-TS | Bayesian linear, H=1 | Thompson Sampling | Test uncertainty-driven exploration |
| BNVS-TS-local | Bayesian linear, local gain | Thompson Sampling | Isolate the effect of the target |

## Factual H=1 value prediction on all runs

TS does not change the posterior mean, so Mean-Greedy and BNVS-TS have the same value-model row here; their difference appears in policy replay.

| Method | N | MAE | Clipped MAE | Sign accuracy | Precision | Recall | Spearman | Outside [-1,1] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean-Greedy / BNVS-TS | 1266 | 0.413 | 0.398 | 0.799 | 0.382 | 0.249 | 0.388 | 0.056 |
| BNVS-TS-local | 1266 | 16.252 | 0.335 | 0.803 | 0.413 | 0.312 | 0.484 | 0.275 |

## Factual prediction by available history

| Prior normal observations | N | H=1 clipped MAE | H=1 sign accuracy | Local clipped MAE | Local sign accuracy |
|---|---:|---:|---:|---:|---:|
| 0 | 42 | 0.116 | 0.310 | 0.122 | 0.310 |
| 1-2 | 84 | 0.825 | 0.464 | 0.667 | 0.560 |
| 3-5 | 118 | 0.515 | 0.653 | 0.380 | 0.703 |
| 6-9 | 135 | 0.417 | 0.763 | 0.325 | 0.770 |
| 10+ | 887 | 0.352 | 0.878 | 0.310 | 0.867 |

## Candidate-policy replay on historical MCGS states

| Method | Decisions | Candidates | Agreement with MCGS | Select max parent | Mean parent rank | Stability modal share | Factual matches | New incumbent on matches | Invalid on matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCGS | 111 | 38.180 | 1.000 | 0.243 | 0.633 | N/A | 111 | 0.072 | 0.324 |
| Mean-Greedy | 111 | 38.180 | 0.135 | 0.369 | 0.860 | 1.000 | 15 | 0.067 | 0.533 |
| BNVS-TS | 111 | 38.180 | 0.108 | 0.180 | 0.700 | 0.353 | 12 | 0.083 | 0.250 |
| BNVS-TS-local | 111 | 38.180 | 0.063 | 0.036 | 0.231 | 0.456 | 7 | 0.000 | 0.000 |

The matched-outcome columns are descriptive only: each replay policy exposes outcomes on a different selected subset, so their rates are not a causal policy ranking.

## Agreement with MCGS by available history

| History observations | MCGS | Mean-Greedy | BNVS-TS | BNVS-TS-local |
|---|---:|---:|---:|---:|
| 0 | N/A | N/A | N/A | N/A |
| 1-2 | 1.000 | 0.000 | 0.000 | 0.000 |
| 3-5 | 1.000 | 0.000 | 0.000 | 0.000 |
| 6-9 | 1.000 | 0.400 | 0.000 | 0.000 |
| 10+ | 1.000 | 0.129 | 0.119 | 0.069 |

## Evidence boundary

- Factual prediction evaluates the surrogate only on the action actually taken by the historical policy.
- Candidate replay reveals policy shift, prior behavior, and selection stability, but unselected actions have no outcomes.
- Only MCGS has outcomes for every row in the Legacy histories. Final-score superiority of the three BNVS variants requires randomized online runs.
- Root drafts, debug, aggregation, and other structurally forced paths remain outside normal BNVS ranking; this is not a cold-start fallback.
