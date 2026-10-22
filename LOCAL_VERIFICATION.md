# Local package checks — 2026-10-21

- Five offline campaign tests passed via `python3 tests/test_multifidelity_campaign.py`.
- Two existing subset/provenance tests passed by direct invocation with temporary
  directories: frozen uniform half-view/full validation and full-node filtering.
  No dependency stubs were used.
- Changed Python files passed AST syntax checks; `git diff --check` passed.
- The full ML pytest suite could not collect in the available local environment:
  `shutup` was missing. A collection attempt without conftest also found missing
  `backoff`. Dependency installation failed because PyPI DNS was unavailable.
- Full regression tests, including the AQ05 reset forwarding regression, are
  mandatory in the cluster preparation job, before provider calls or searches.
- No real LLM search, CUDA training, dataset download, Slurm submission or
  official grading was run locally. Actual Mila execution remains unverified.

Packaging fixes include forwarding reset's full/full filter through AQ05's
inherited state class, preserving three workers after the fidelity switch,
initializing direct fixed-evaluator instances safely, and accepting long CSV
text cells in the uniform subsampler. These implement the approved behavior.
