# Mathematical sources and reproduction boundary

Task list: MLEvolve, arXiv:2606.06473, section 4.1 and Table 2:
https://arxiv.org/html/2606.06473#S4

Problem statements, mathematical formulas and verification constructions:
Google DeepMind, `alphaevolve_results/mathematical_results.ipynb`, pinned to
commit `4226acbf237ff9ad10ba7673a2af127a2d8a5971`:
https://github.com/google-deepmind/alphaevolve_results/blob/4226acbf237ff9ad10ba7673a2af127a2d8a5971/mathematical_results.ipynb

Notebook SHA256:
`2cce2543e48c89aa3e91614272a698a0147dd2548ea11cf92f1292b7435d38ff`.
The upstream software is Apache-2.0; its license is in `vendor/LICENSE`.
Copyright 2025 Google DeepMind Technologies Limited / Google LLC.
Other upstream materials are CC-BY-4.0, as stated in its README.
The mathematical graders here adapt that notebook's formulas and verification
logic. Candidate serialization, CPU isolation and MLEvolve hooks are local.

The MLEvolve repository does not publish its mathematical-task prompts or
runner. This integration matches the public mathematical definitions and the
local mlebench_lite native agent workflow, with GP6 and CPU defaults. It is not
a claim to recover the authors' unpublished prompts, initialization, numerical
search budget per construction, or reported scores.

Mappings resolve multiple upstream examples using the values in Table 2:
11 hexagons; 26 square circles; 21 rectangle circles; 13 convex-region points;
11 triangle points; dimension-11 kissing; 16 points in two dimensions for the
**squared** max/min distance ratio. Uncertainty uses the three-coefficient
Hermite formulation (not the later Laguerre formulation with a different bound).
Autocorrelation 3 uses `abs(max(convolution))`; its variant uses
`max(abs(convolution))`. Autocorrelation 2 integrates the squared piecewise-linear
autoconvolution exactly.

The two sums/differences entries map to the notebook's `solution_1` and
`solution_2`. Both use `1 + log(|U-U|/|U+U|)/log(2 max(U)+1)` and allow variable
set cardinality. They have different simple initial sets and reference
constructions. No artificial size-dependent reward or alternative inequality
is added. Recomputing official construction 1 gives `1.1479888965092757`, while
MLEvolve Table 2 prints `1.1479889651`; the difference is recorded, not used to
alter the mathematical objective.

Prepared `public/` contains the official statement, interface and a simple
feasible initial construction. It does not contain the AlphaEvolve solution.
Prepared `private/` contains those reference solutions solely for checking the
host grader. These tasks have no supervised train/test data or hidden Kaggle
leaderboard; post-search validation independently regrades the exported object.
The candidate process and host grader are separate processes and never share
candidate-modifiable Python globals. Like the existing local mlebench_lite
execution, this is not a hostile-code filesystem sandbox; a benchmark run with
untrusted adversarial candidates needs OS-level access isolation.

Implementation limits are explicit in every public contract: finite JSON
constructions up to 64 MiB; up to 10,000 step heights; up to 5,000 kissing
vectors; up to 100,000 set elements with maximum at most 1e9. These are resource
limits, not claims about the authors' unpublished search spaces. Geometry uses
absolute tolerance 1e-9 and overlap integration uses a mean tolerance 1e-10.
Reported scores are the raw mathematical objectives, without clipping to [0,1]
or adding task-specific rewards. GP6's existing internal utility transform is
unchanged.
