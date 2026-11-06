# Reward-only local Bayesian selector

## Scope

`reward_only_bayesian_ts` is the production default for the no-pretraining
ablation. It learns only from transitions completed in the current run. It
does not instantiate or query the validity and cost models.

The old selectors remain available for reproducing earlier experiments, but
must be paired explicitly with `bo_parent_selector.model=bayesian_ridge`.

## Model input and target

The model consumes 15 derived inputs:

- parent rank percentile;
- global and branch gaps divided by the current run's robust reward scale;
- scaled last, recent-mean, and recent-standard-deviation gains;
- log branch stagnation;
- squashed depth;
- log visit count and MCGS Q as contextual state features;
- scaled primary-reference gap and reference-missing indicator;
- one indicator for each normal operator: improve, evolution, and fusion.

The run-local reward scale is the maximum of the observed utility IQR, median
absolute valid gain, and `1e-6`. The target is valid child-minus-parent utility
divided by this scale. Invalid transitions stay in the audit log but do not
train this model.

The surrogate is conjugate Bayesian linear regression with a
Normal-Inverse-Gamma prior. It returns the raw-scale posterior mean and
predictive standard deviation of child-minus-parent gain. Thompson samples are
drawn jointly through one sampled coefficient vector, preserving correlation
between candidates.

## Selection policy

- Fewer than 10 valid transitions: use progressive MCGS only.
- From 10 to 30 valid transitions: linearly increase gain trust from 0.10 to
  0.25.
- At 30 or more valid transitions: cap gain trust at 0.25.
- Draw one joint coefficient-level Thompson sample at temperature 0.25. Because
  the surrogate's uncertainty multiplier is 2.0, this retains half of the raw
  posterior coefficient deviation.
- The remaining 10% of either the step or wall-clock budget linearly anneals
  the Thompson sample to its posterior mean.
- Do not clip the sampled gain.

The final candidate score is:

```text
parent_utility + gain_trust * sampled_gain
```

After warm-up this is a pure BO acquisition. No MCGS rank, UCT score, Top-K
rank, or MCGS mixture weight enters the acquisition. `mcgs_q` and visit count
remain observed state inputs to the reward model; they can help predict gain,
but they are not independently scored or blended at selection time. Progressive
MCGS is used only before 10 valid transitions or if model fitting/selection
raises an exception.

## Configuration

```yaml
parent_selector:
  type: reward_only_bayesian_ts
  decision_scope: all_normal_steps

bo_parent_selector:
  model: conjugate_bayesian_linear
  min_valid_transitions: 10
  min_total_transitions: 10
  min_operator_observations: 0
  acquisition:
    type: conservative_thompson
    use_validity_probability: false
    use_cost: false
    initial_gain_trust: 0.10
    max_gain_trust: 0.25
    full_gain_trust_valid_transitions: 30
    thompson_temperature: 0.25
    anneal_remaining_fraction: 0.10
```

## Offline evaluation

Run the strict prequential evaluator with:

```bash
python tools/reward_only_offline_eval.py \
  --runs-root /path/to/bo-failure-analysis-42/runs \
  --historical-analysis-root /path/to/bo-failure-analysis-42/analysis
```

The checked-in report is `docs/REWARD_ONLY_OFFLINE_EVALUATION.md`, with the
machine-readable aggregate and per-run results in the adjacent JSON file.

Warm-up sensitivity for the current reward-only selector is reproducible with:

```bash
python tools/reward_only_warmup_sweep.py \
  --runs-root /path/to/bo-failure-analysis-42/runs
```

This writes `docs/REWARD_ONLY_WARMUP_SWEEP.md` and the adjacent JSON result.

Candidate-choice scale and Thompson stability are reproducible with:

```bash
python tools/reward_only_acquisition_replay.py \
  --runs-root /path/to/bo-failure-analysis-42/runs
```

This writes `docs/REWARD_ONLY_ACQUISITION_REPLAY.md` and the adjacent JSON
result.

The experimental latent Child-UCB beta sweep is reproducible with:

```bash
python tools/reward_only_latent_ucb_replay.py \
  --runs-root /path/to/bo-failure-analysis-42/runs
```

This writes `docs/REWARD_ONLY_LATENT_UCB_REPLAY.md` and the adjacent JSON. The
experiment does not change the production Thompson configuration.

The evaluator scores only the factual action selected by the historical
policy. It does not claim counterfactual policy uplift because unselected
candidate outcomes do not exist in the logs. The optional historical analysis
input adds two separately labeled MCGS comparisons: `mcgs_q` as an ordinal
signal proxy, and the actual one-seed Legacy progressive-MCGS outcomes versus
the previous BO runs. Neither is presented as an online result for the new
reward-only policy.
