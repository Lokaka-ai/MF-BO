# Source provenance

The training program is adapted from the repository's vendored
`EvolveFold/tasks/autoresearch/repo/train.py` at commit
`51b241dc4bedeb963031e051d06baceb5197b0e2`, itself based on
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).

`prepare.py` and `trial_config.py` are copied byte-for-byte from that local
reference. The dataset, BPE training, held-out split, context length, validation
token count and official BPB evaluator are unchanged.

The A100 baseline changes FP32 to BF16 autocast on Ampere or newer GPUs, sets a
16-sequence microbatch and 65,536-token optimizer batch, reports the A100 BF16
peak FLOPs in MFU, and records the maximum step duration for the budget check.
The warmup exclusion and 300-second timer follow the local reference.

The fixed harness boundary, worker, MLEvolve hooks, host metric admission,
resource launcher and provider wrapper are additions for this task.
