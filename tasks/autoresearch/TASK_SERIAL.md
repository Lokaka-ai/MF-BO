# nanoGPT autoresearch: fixed five-minute pretraining

Minimize held-out validation bits per byte (val_bpb, lower is better) after
300 seconds of measured training from random initialization. Propose genuinely
competing architecture/optimization hypotheses and use observed results to improve
learning and throughput. All native MLEvolve planning, stepwise coding, review,
debugging, evolution, fusion, aggregation and global memory remain active.

## Immutable task protocol

- Data: karpathy/climbmix-400b-shuffle, training shards 00000–00009, validation
  shard 06542. The prepared rustbpe tokenizer (8192 tokens) uses training data only.
- Context length 2048; fixed official evaluate_bpb scores 20,971,520 validation
  tokens. It requires unreduced next-token cross entropy with reduction='none'.
  Do not scale, smooth, offset or otherwise alter the evaluation loss. Causal
  predictions must not access future tokens or targets. No validation training.
  Before scoring, the host checks that changing future tokens leaves earlier
  logits unchanged and that unreduced loss equals target-free logits' cross entropy.
- Each candidate is randomly initialized with seed 42 and trains for 300 measured
  seconds, excluding the first 11 warmup/compilation steps. The final step can
  slightly overrun. Compilation, validation and saving take additional wall time.
  The campaign has an eight-hour total wall budget, including all agent work.
- Each setting has one A100 80GB and four CPUs. This comparison uses one
  experiment worker per workflow; each training trial receives all four CPUs
  and the setting's GPU. The native three-stage code generation remains active.
  No pretrained model weights.
- Data loading, tokenization, split, training loop, timing, schedule functions,
  checkpoint saving and official evaluation are supplied by the host. Do not
  reimplement or modify them. No CSV submission, custom dataset files, or Kaggle
  inference pipeline is needed for this task.

## Candidate interface

Return a complete Python **candidate module** containing model definitions,
optimizer implementation and hyperparameter definitions. The host inserts it
between its imports and its immutable training loop. The data preview supplies
a working editable reference module and the read-only host harness.

The following names are supplied by the host: torch, nn, F, math, dataclass,
asdict, TRAIN_DTYPE, cap, TRIAL_CONFIG, MAX_SEQ_LEN. Do not redefine these protocol
controls, import prepare/trial_config, execute a training loop, load data, patch
libraries, set environment variables, or change the timer/evaluator. Normal torch
and model helper imports are allowed. Preserve these required interfaces:

1. GPTConfig is a dataclass accepting sequence_len, vocab_size, n_layer, n_head,
   n_kv_head, n_embd, window_pattern. Additional fields must have defaults.
2. GPT(config) supports construction on the meta device, to_empty(device='cuda'),
   init_weights(), num_scaling_params() returning a dict with 'total', and
   estimate_flops(). Initialize every learned parameter and buffer after to_empty.
3. GPT.forward(idx, targets=None, reduction='mean'): scalar actual next-token
   cross entropy for training; unreduced per-token cross entropy for 'none'.
   Without targets return logits. Preserve causal masking and vocabulary size.
4. GPT.setup_optimizer(unembedding_lr, embedding_lr, scalar_lr, adam_betas,
   matrix_lr, weight_decay) returns a torch optimizer. EVERY parameter group must
   contain 'lr', 'initial_lr', and 'kind' ('adamw' or 'muon'); the host updates lr
   using initial_lr and its fixed schedule. Muon groups also use momentum and
   weight_decay. Changing optimizer type is allowed while retaining this API.
5. Define ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, TOTAL_BATCH_SIZE, EMBEDDING_LR,
   UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY, ADAM_BETAS, WARMUP_RATIO,
   WARMDOWN_RATIO, FINAL_LR_FRAC, DEPTH, DEVICE_BATCH_SIZE. All must be compatible
   with the host harness. DEVICE_BATCH_SIZE must divide 10240; TOTAL_BATCH_SIZE
   must be a positive multiple of DEVICE_BATCH_SIZE * 2048. Keep HEAD_DIM even.

Stepwise generation retains all three native stages: data processing/feature
engineering reasons about the frozen token representation and defines compatible
configuration; model design implements the causal GPT; training/evaluation designs
the optimizer and hyperparameters compatible with the host loop. The merger
returns the complete module. Improve/debug/evolution/fusion/aggregation also return
modules using exactly this interface; SEARCH/REPLACE edits target module source.

The host retains candidate.py, assembled train.py, resource allocation, training
log, official result JSON and checkpoint for every executed node. Scores must come
from the complete CUDA training/evaluation protocol, never from a printed claim.
