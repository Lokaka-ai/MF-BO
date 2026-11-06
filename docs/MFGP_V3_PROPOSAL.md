# MF-BO v3 proposal: spend the budget on useful evidence and completed solutions

Date: 2026-11-05. Status: **design approved; locally implemented and validated; not approved for launch**.
Reviewed implementation: `14605cb4b068b3acfe6fcf02af085b87ca6f404a`.
The user approved this design on 2026-11-05. Policy v2 remains the historical
reproduction baseline; v3 has a separate executable path. The original review changed documentation only. See MFGP_V3_IMPLEMENTATION.md
for subsequent implementation status; no cluster jobs have been launched.

## 1. Research decision

Keep BO, the two-level discrepancy GP, 20%/100% training data, and full fixed
validation. Change how we prepare, price, and schedule experiments before making
the surrogate more elaborate. The next version should make its final fidelity
decision about an executable program whose workload we have inspected, rather
than commit to fidelity using only a parent/operator context.

The goal is the best **actually completed full-data solution within the budget**.
More programs, more cheap runs, and higher acquisition values are diagnostic
quantities, not success criteria. The present evidence does not show that our
controller outperforms an otherwise matched full-only BO system.

Recommended scope: consistent cost accounting, a bounded execution check,
concrete-program fidelity allocation, stable GP scaling, nonblocking initial
calibration, and individual-action budget checks. Retain full-target knowledge
gradient (KG) as the first acquisition baseline. Evaluate MF-MES as a separate
challenger; do not bundle an unvalidated acquisition replacement into every fix.

## 2. What the experiments establish

Latest evidence directory, outside the source repository:
`/Users/mac/Documents/ChatGPT/AI4AI/deliverables/mila-mfgp14/analysis/mf20_v2_2026-11-04/`.
`evidence.json.gz` holds the frozen trajectories; `analyze.py`, `summary.json`,
`events.csv`, `pairs.csv`, `REPORT.txt`, and `integrity_and_anomaly.json` support
these observations. Source bundle was `mila-mfgp14-14605cb`.

| Observed outcome | Jigsaw | Melanoma | Dog Breed |
|---|---:|---:|---:|
| Official score | 0.98632, higher better | 0.86462, higher better | 0.33191, lower better |
| Generated program nodes / distinct source hashes | 38 / 38 | 12 / 12 | 39 / 38 |
| Valid admitted low / full observations | 6 / 20 | 3 / 4 | 4 / 22 |
| Ordinary acquisition decisions, excluding calibration | 20 | 3 | 23 |
| Ordinary new low / new full / promotion decisions | 1 / 17 / 2 | 0 / 3 / 0 | 1 / 20 / 2 |
| Same-code successful low/full pairs | 4 | 2 | 4 |
| Median candidate-process time saved at 20% | 46% | 66% | 65% |
| Winner completion, active hours | 2.12 | 11.23 | 11.01 |

Ordinary decisions exclude initialization and forced/debug paths. Promotions are
extra evaluations of unchanged source, not additional generated programs. The
paired timing sample is selected, small, and affected by concurrent resource
contention; its savings are not whole-campaign savings or causal performance gains.

The audit reconciled selected winners with valid, budget-eligible full observations,
submission hashes, and official reports. It did not establish enforced filesystem
isolation or exhaustive leakage freedom. Jigsaw's historical AQ05 score is 0.98616,
but generation, evaluation and selection differ; this is not a matched comparison.
Available records did not contain completed original-BO results for the other two.

### Mechanisms worth changing

1. **Cheap actions were often feasible but assigned little value.** Every ordinary
   Jigsaw and Dog decision offered a cheap alternative. For the best cheap action
   by acquisition-per-cost, median estimated cost relative to the chosen action
   was 0.510 and 0.225. Median raw acquisition relative to the chosen action was
   only 0.00499 and 0.00179. Improving cost estimates alone will not necessarily
   reverse decisions. These model values are not evidence that cheap actions
   would actually have produced better outcomes.
2. **Expensive failures consumed opportunities.** Jigsaw had eight syntax failures,
   including conflict-marker text. All three initial Melanoma drafts failed; six
   cheap failures summed to 5.79 worker-hours, and the first valid cheap score
   arrived at 2.51 active hours. Worker-hours overlap on one GPU and must not be
   relabeled allocated GPU-hours.
3. **Calibration and guards restricted expensive-task search.** Melanoma's initial
   full promotions took 177/220 minutes, finishing around 6.07/6.18 hours. Its
   three ordinary decisions were then all terminal full-only. The code also
   pauses ordinary decisions while mandatory calibration is pending. This
   establishes a barrier, not a measured estimate of all idle GPU time it caused.
4. **Runtime predictions did not transfer well between workloads.** Completed
   chosen-action timing ratios had medians 2.37, 1.47 and 1.62 times the estimates.
   These statistics include particular generation/retry scopes and exclude final
   censored attempts; they are diagnostics, not calibrated error distributions.
5. **The statistical model had limited calibration.** Melanoma ended with seven
   valid observations, below the eight-row MAP threshold: it conditioned a GP
   with regularized prior hyperparameters rather than learning them. Dog's
   first-result normalization left an early full observation about 74 times the frozen
   normalization scale away from its center and fitted variance parameters at their upper bounds.

The older 50% pilots strengthen the concern without establishing causality:
RANZCR had only one successful full observation, and a strong cheap result arrived
too late for promotion; Cactus reached its validation ceiling early. Cactus's exact
ceiling behavior was already addressed in policy v2 and must not be reported as an
unfixed v2 defect. Older evidence is under `analysis/lab_meeting_2026-11-02/`.

The new acquisition diagnostic is reproducible with stdlib Python at:
`/Users/mac/Documents/ChatGPT/AI4AI/deliverables/mila-mfgp14/next_version_2026-11-05/analyze_acquisition.py`.
Its JSON includes input SHA256, denominators and decision IDs. It performs no
counterfactual reward inference or new evaluations.

## 3. Code-level findings and their limits

| Implementation path | Verified mechanism | Proposed correction |
|---|---|---|
| `controller.py:observe`, `agent_search.py:_run_single_step`, `mf_gp/execution.py:promote` | Expansion cost includes host scoring/review, while promotion cost uses candidate-process time; both enter the same estimator | Store comparable process timings and separate host costs for every action |
| `model.py:fit` and `controller.py:features` | First nonempty initial reference freezes output scaling; context features depend on that scaling | Recompute one common scale and rebuild features from frozen raw context |
| `controller.py:select`, `policy.py:fill_calibration` | Two priority pairs plus a pending-calibration barrier | One priority full anchor; no global waiting barrier; further pairs compete normally |
| `controller.py:remaining_budget`, `policy.py:estimate_costs` | Global expensive-runtime guard can irreversibly latch all actions into terminal mode | Recompute feasibility for each action and its required completion path |
| `engine/execution.py`, `agents/debug_agent.py` | Syntax/partial-patch defects can reach execution | Compile gate before expensive work, then a bounded runtime probe |
| `controller.py:select` | New-program costs and fidelity are chosen before the generated child exists | Preserve BO origin selection, then rerank fidelity after freezing/probing actual code |

The paths above refer to the reviewed source, not a claim that every observed
failure has a single cause. For example, the reason one Dog program worsened
dramatically on full data remains unresolved; same bytes do not imply the same
optimization trajectory. Do not force monotonic quality improvement with fidelity.

## 4. Proposed workflow

### A. Preserve the BO and Harness boundary

The Harness still determines legal parent/operator options. BO ranks those options;
it does not invent a different operator. Existing immutable programs can also be
promoted. Keep the same three initial drafts and three workers.

At a free worker slot:

1. If there are already prepared programs, compare their low/full executions with
   eligible saved-program promotions. Do not generate more code while a prepared
   program awaits allocation.
2. Otherwise, compare saved promotions with the existing BO's prospective
   parent/operator expansion options. Use the contextual GP and coarse cost prior
   for this origin decision. If promotion wins, execute it directly. If expansion
   wins, generate one program. Its prospective fidelity is advisory, not yet an
   execution commitment.
3. Freeze that generated program's bytes, ID and birth-time context; compile it
   and run the bounded probe described below. Recompute its low/full costs.
4. Compare actual prepared-program executions and saved promotions using the same
   full-target acquisition. The chosen action may differ from the provisional
   decision. Preserve unexecuted prepared programs rather than discarding their
   generation cost. Bound the prepared pool at three programs.
5. Execute in the chosen public data view. Score only with the fixed evaluator,
   record outcome and every cost segment, refit, then repeat.

Persist separate generation and execution reservations: generating → ready →
executing → completed/failed, with a deferred/expired state for unexecuted code.
Advisory fidelity during generation is not an in-flight GP measurement. Only
committed executions enter pending-observation conditioning. Generation, probes
and training all share the same three worker slots and one allocated GPU.

If all prepared actions are unaffordable, defer them out of the bounded ready pool
with an explicit reason, preserve their bytes/costs, and reconsider them only when
cost or pending-state evidence changes. They are unmeasured, not failed quality
labels. This releases the queue so it cannot deadlock; it must not regenerate the
same reserved program or force new generation that itself lacks an affordable
completion path. Expire remaining unexecuted programs at the run deadline.

This is a two-stage allocation approximation, not exact joint optimization over
unknown future source code. The GP initially still knows a new program through
its frozen context and identity effect, not a magically accurate code-to-quality
predictor. No new embedding model or LLM quality judge is proposed. The main new
information before fidelity choice is the actual workload and execution evidence.

### B. A bounded correctness and timing probe

Compile source before executing it. Syntax failure returns to the existing debug
workflow with a recorded failed attempt; never silently repair code or count it
as a measured score. Debugged bytes are a new program identity.

For generated training programs, add a minimal execution contract: a probe mode
that exercises the same preprocessing/model/loss/optimizer/inference path on a
small T-only batch, and emits workload/timing metadata. Use three training steps
(one warm-up, two timed) and one prediction batch as the initial engineering
default, with a 120-second total cap including loading. This probe is not a new
quality fidelity and produces no GP score. It must never read hidden V/test labels.

Unsupported noniterative estimators need an explicit T-only bounded probe path;
do not fabricate neural step timings for them. A probe timeout or unsupported
path means unknown timing, not an invalid ML method. Use the conservative fallback
cost prior or skip the action if its budget cannot be justified. An actual error
on the intended path is a debug event. Log which path ran and its coverage.

The execution contract is a substantive generator change and belongs in the v3
approval. It cannot prevent every late OOM, data-dependent failure, or malicious
branch. Require the same contract in matched full-only controls; otherwise a gain
could be caused by better error checking rather than multi-fidelity allocation.
Probe training state is discarded. Do not turn this into unaccounted warm-starting.

### C. Predict time separately from quality

For a fixed, known program p, estimate remaining work using measured components:

`C_p(s) = setup_p(s) + steps_p(s) * seconds_per_step_p + inference_p(V,test) + host_p`.

For several folds/models, sum their declared workloads. Measure under the actual
CPU/RAM/GPU allocation; log concurrency because three workers share one GPU.
Do not let the LLM supply an unverified runtime number as a measurement. Parsed
workload declarations are predictions, while observed step counts/times are data.

For fixed epochs and similar batch behavior, `a_p + b_p*s` is a useful first
approximation. This explains why 20% data need not cost 20% of total runtime:
model loading, validation and inference remain. Fixed-step schedules, changing
folds, early stopping, preprocessing and superlinear estimators violate that
assumption. When the workload changes by fidelity, log and price that change.

Use completed comparable process measurements to update the estimate. Record
generation, probe, setup, training, prediction, host evaluation/review, failure and
interruption costs separately. Keep full end-to-end cost for reporting. A timeout
is a lower bound on that program's completion cost, not a fast successful sample;
an early crash is a failure expense, not evidence that full training is cheap.

Promotion reruns unchanged code from scratch, so its cost is a full run, not the
increment from 20% to 100%. Decision denominators use future incremental cost:
generation/probe already spent is sunk, while it stays in total-budget accounting.
Origin-selection denominators include still-unspent generation/probe costs.

Retain the existing 1.5 safety factor initially, applied to the individual workload
estimate and comparable evidence. Report it as a heuristic, not a quantile or
guarantee. Without representative measurements, retain a broad conservative prior;
do not replace it with an optimistic linear extrapolation. We do not have enough
data to justify a separate learned GP over arbitrary generated program runtimes.

### D. Keep the discrepancy GP, correct its scaling

Keep the autoregressive idea `H_p = offset + rho * L_p + discrepancy_p` with
context and persistent program-identity effects. Cheap measurements inform the
full posterior through covariance; they never become observed full scores.
Allow rank reversals and negative transfer; do not impose a scaling curve that
guarantees more data improves every program.

Replace first-result-frozen normalization with a single common transformation
of all currently admitted raw low/full utilities. Proposed initial rule: center
at their mean and scale by `max(std(utilities), 0.05*abs(mean), 1e-6)`, using the
existing task prior before any valid observation. This is a starting numerical
rule to validate, not a claim of an optimal noise model. Never normalize each
fidelity independently; that would hide systematic discrepancies.

Because current context features also depend on center/scale, retain immutable
**raw birth-time context** (parent utility, same-fidelity gap, prior child gains,
counts, rating and missingness). Rebuild all observed, pending and proposed
feature vectors consistently with the current transformation. Never reconstruct
historical context using a parent's later results. Low and full observations of
one program share that context and identity. Save transformation, raw context and
model contract in checkpoints and invalidate acquisition caches when it changes.

Keep regularization and the current minimum-evidence threshold initially. Do not
lower the threshold merely to label a sparse fit as learned. Test scale sensitivity,
predictive interval coverage, variance-bound saturation and conditional-versus-
hyperparameter uncertainty explicitly before claiming the model is calibrated.

### E. Obtain a full anchor without stopping other workers

Keep the three initial cheap executions and their approved 40%-budget upper bound.
As soon as a valid cheap initial program has an affordable full path, prioritize
one full promotion (best current cheap utility among eligible initial programs,
deterministic tie rule). While it runs, other workers may continue ordinary work.

Remove the second mandatory random calibration slot and the global pending-pair
barrier. Further pairs compete by the acquisition. If the priority full attempt
fails, preserve it; do not restart it silently or demand unlimited replacement
pairs. Ordinary full actions can provide the first incumbent. Until one full
artifact exists, protect enough time for a feasible full attempt. A single pair
does not identify correlation; retain uncertainty and admit that limitation.
If no valid cheap initial program exists, use the existing legal debug path under
the same budget rather than waiting on nonexistent calibration. The reserve is
a feasibility policy, not a guarantee that a program will succeed.

### F. Budget checks are action-specific

Recalculate feasibility at every decision. A low execution needs time for itself
and a plausible subsequent full execution, unless an already affordable full
candidate can use its information. Prefer the conservative same-program full
reserve for the first implementation. Account for pending allocations and
contention rather than multiplying three-worker time into three GPUs.

An unusually expensive full model must not permanently disable cheap work for
all cheaper models. Remove the irreversible global terminal latch. When no
information-then-full path fits, compare affordable full actions using the existing
terminal improvement objective; if none fit, finalize the best completed eligible
full artifact. Deadlines and same-code lower bounds remain hard constraints.

The 12-hour clock includes generation, probes, GP work, evaluation and all failed
attempts. Do not add a free final training run. Official grading occurs after the
validation-selected winner is frozen and must not feed back into search.

### G. Acquisition: controlled challenger, not an assumed cure

The present KG values information through changes in full posterior means and
also values retaining a measured full artifact. This asymmetry is meaningful:
only a completed full candidate can win. Low information value can reflect weak
cross-fidelity predictiveness, a poorly calibrated model, or a limited one-step
objective. The logs alone cannot distinguish all three explanations.

First implement the corrections above with KG. Add diagnostic exports of raw
information value, incremental cost, pending support, fit status and feasibility
reason for every alternative. Recompute MF-MES on the same finite pool and cost
model as an offline challenger. MF-MES targets information about the best full
value; use it only while a real full completion path remains, with the same
full-artifact finishing rule. Never select an untrained posterior optimum as the
final result.

Do not choose MF-MES because it produces more cheap actions. Require numerically
correct acquisition tests and evidence that it handles both informative and
misleading low-fidelity cases. A logged replay can compare decisions and timing,
but cannot reveal rewards of unexecuted programs. A performance claim requires
a later approved matched experiment.

## 5. Literature: what transfers, and what does not

| Work | Useful part | Boundary in our setting |
|---|---|---|
| [ADMIRE-BayesOpt](https://arxiv.org/html/2508.11551v2) | MF-MES with a fidelity covariance and explicit cost treatment | Its candidates are known data-mixture vectors and fidelities are model scales. Reviewed implementation uses measured means by scale or a linear size proxy in particular experiments; it does not establish a universal runtime or quality law for changing agent code. Borrow the acquisition as a challenger, not its zero-shot winner convention. |
| [FABOLAS](https://proceedings.mlr.press/v54/klein17a.html) | Separate models for quality and dataset-size-dependent runtime; value information about the full objective | Its stable hyperparameter space is easier to model than source generated after selection. Use separate measured cost prediction, not an assumed transferable cost GP or unbudgeted full validation. |
| [MF-MES and asynchronous parallelization](https://proceedings.mlr.press/v119/takeno20a.html) | Information about the high-fidelity maximum; asynchronous handling of pending evidence | Appropriate finite-pool challenger. Pending observations are uncertain, not measured labels; a low-fidelity information gain is not a deployable result. |
| [Multi-Step Budgeted BO with Unknown Evaluation Costs](https://papers.neurips.cc/paper_files/paper/2021/file/a8ecbabae151abacba7dbde04f761c37-Paper.pdf) | Optimize decisions against a remaining budget and uncertain completion costs | A fixed two-action lookahead is not enough: full→full may win whenever it fits, without crediting the additional cheap breadth available in the same time. General time-based lookahead adds substantial complexity and is deferred. |
| [Data Mixture Optimization: A Multi-fidelity Multi-scale Bayesian Framework](https://arxiv.org/html/2503.21023v1) | Structured relationships across mixtures, model sizes and training scales | Relies on a structured training domain and substantial prior run data. Our few selected paired programs do not identify that surface. |
| [In-Context Freeze-Thaw BO](https://arxiv.org/html/2404.16795v2) and [Cost-Sensitive Freeze-thaw BO](https://papers.nips.cc/paper_files/paper/2025/file/f7b75ff5fa544897ed71217de2aad865-Paper-Conference.pdf) | Resume/stop training using partial learning trajectories and cost | Requires reliable per-program learning curves and resumable training. Our current promotions restart different data views; optimizer/epochs as fidelity would be a separate design and comparison. |

Separate **runtime scaling** from **quality scaling**. Two noisy quality levels
per selected program cannot identify a reliable multi-parameter power law, and
the Dog reversal directly cautions against assuming monotonicity. A newer paper
or more complex model is not by itself a reason to replace the present GP.

## 6. Implementation sequence after approval

| Stage | Files / responsibility | Acceptance evidence |
|---|---|---|
| 1. Comparable timings and GP scaling | `mf_gp/controller.py`, `model.py`, `execution.py`, `policy.py`, shared execution timing | Matched timing boundaries; all cost retained; order/scale/resume tests; no cheap labels in full-only targets |
| 2. Compile and bounded probe contract | generation/debug validation, `agent_search.py`, interpreter, small dedicated probe adapter | Malformed patches fail before training; intended code path exercised; unknown timing handled explicitly; no score leakage |
| 3. Deferred allocation and scheduler | `mf_gp/runner.py`, controller, recovery, evidence | Origin and execution decisions logged separately; ≤3 prepared codes and existing worker limit; no starvation/repeated generation; one immutable hash per program; pending/failed states recover exactly |
| 4. Nonblocking anchor and individual guards | `policy.py`, controller | No calibration-wide pause; no stale terminal latch; full-artifact reserve/deadline tests; no repeated failed promotion disguised as new work |
| 5. Acquisition challenger and diagnostics | separate MF-MES evaluator, audit tools | Common candidate support, units, costs and pending treatment; numerical checks; no claim of measured counterfactual improvement |

Use a new policy/model/checkpoint version and a separate manifest. Preserve v1/v2
and all 50% reproductions. Do not resume a v2 scientific run into v3 with a changed
policy, or reinterpret its observations as if generated under v3. Existing completed
trajectories are development evidence, not a v3 performance result.

## 7. Validation and experiment contract

Do local lightweight deterministic checks first. Useful cases: weak/strong/negative
cross-fidelity correlation; heterogeneous setup/training costs; exact ceilings;
failed syntax/probe/generation; censored execution; pending calibration; action
availability near deadline; process interruption at each prepared-state boundary;
one/full-only-valid winner; normalization applied identically after recovery.

Replay historical events in temporal order for accounting and predictive checks.
Fit only on preceding observations. Report pair counts and selection bias; ten
selected v2 pairs are insufficient to certify a general error model. A delayed
outcome must not appear in an earlier fit. Do not assign rewards to untried actions.

After local gates, propose the smallest separate compute preflight that exercises
real program generation → probe → cheap evaluation → immutable full promotion →
fixed validation → resume → final submission. Synthetic tests alone do not certify
Mila execution. This design review does not authorize new Slurm jobs.

Before a future comparison, freeze:

- **Question/estimand:** does the new allocation improve the best eligible full
  solution under the same active-time and resource budget?
- **Arms:** full-only BO with the same generation/checking/evaluation improvements;
  v3 KG; an MF-MES arm only if its offline gates justify the added run. Legacy v2
  is historical context; include a matched v2 arm only if its incremental value
  justifies the compute. Test scheduling changes separately when attribution matters.
- **Support:** identical Harness legality, generation model/prompt contract,
  allowable external data/pretraining, T/V/test boundaries, hardware, concurrency,
  finalization and budget conventions. MF arms alone may choose low observations
  and immutable promotions. Initial-fidelity differences are part of the intervention.
- **Outcome:** official score of the validation-selected completed full candidate;
  secondary time to valid full incumbent, validation anytime curve, valid unique
  programs, paired cost savings, failure time and actual allocated resource cost.
- **Failure handling:** keep all assigned runs and missing submissions visible;
  predeclare no-submission handling rather than average only successes. Preserve
  consumed time on preemption and separately report wasted work/queue downtime.
- **Unit:** complete stochastic runs. Nodes/folds/promotions are not replicates;
  matching seed 42 does not pair unseeded LLM samples.
- **Interpretation:** current tasks and scores have already informed development.
  Replicates can test engineering/performance hypotheses here, but fresh tasks
  and matched controls are needed for generalization claims. Do not tune on official
  test feedback or claim leaderboard-level superiority from one trajectory.

A useful first development comparison can use a shorter predeclared budget if
representative full candidates fit. Do not assert that a short run predicts the
12-hour winner; budget and task choice need their own proposal before launch.

## 8. Deferred scope and known risks

No extra fidelity levels, model-size/epoch search, global power-law performance
model, learning-curve foundation model, increased workers, forced cheap quota,
unbounded candidate queue, automatic retraining outside budget, or alternative
search algorithm in this iteration. Do not enlarge infrastructure for its own sake.

Remaining risks: poor correspondence between cheap/full quality; tiny calibration
samples; Gaussian prior/noise mismatch; workload declarations that omit real costs;
probe overhead or unrepresentative steps; late OOM; throughput changes under
contention; generator changes confounding a comparison; filesystem access that is
cooperative rather than enforced; and weak validation-to-test generalization.
The plan reduces specific failure mechanisms; it does not promise higher scores.

**Approval requested:** implement stages 1–4 as a separately versioned v3, and
stage 5 as diagnostic/offline research. Performance experiments and any decision
to replace KG with MF-MES remain separate decisions after evidence is reviewed.
