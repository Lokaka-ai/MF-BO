# Launch the separate MF-GP campaign

Use the new `mila-mfgp14-COMMIT.tar.gz` bundle. The source bundle includes both launchers. Use separate campaign directories
for carry-over/reset and MF-GP; preserve the earlier failed preparation logs.
The current campaign uses Muse Spark 1.3 Contributor, Codex/OpenCode Go, high
reasoning and Codex CLI 0.164.0. Earlier native OpenCode packages are superseded
for this newly approved model route.
The new campaign has 14 jobs: one `mf_gp` run per retained Lite task, each with
12 hours active search, 500 total actions, 3 initial drafts/workers, seed 42,
one A100 80GB. The Aerial Cactus pilot requests 4 CPUs and 64 GB host RAM after
both staged 24 GB workers reached the cgroup limit; this differs from Xixian's
recorded 24 GB allocation. Other task allocations remain as configured.
Mandatory initial half-data drafts have a 40%-of-active-time execution deadline;
full-data attempts retain the nine-hour candidate cap.

## Before transferring

The bundle includes source and a `SOURCE_SHA256.json` integrity manifest.
It does not contain credentials, environments, downloaded datasets or run outputs.
The accompanying `.sha256` file checks the archive itself. A clean local commit
can be repackaged with:

```bash
python3 tasks/mlebench_lite/package_mfgp.py --output-dir /path/to/deliverables
```

## On Mila

1. Transfer the archive and its `.sha256` file to your scratch directory.
   Verify with `sha256sum -c mila-mfgp14-COMMIT.tar.gz.sha256`, extract the archive,
   and enter its `mila-mfgp14-COMMIT` directory. Substitute its actual commit ID.
2. Make the previously installed tools visible in this login session:

   ```bash
   export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"
   command -v uv codex ffmpeg git unzip 7z
   ```

3. For a matched comparison, wait until the carry-over/reset launcher in this same bundle has prepared all
   14 frozen protocols, then submit preparation with those exact protocols:

   ```bash
   python3 tasks/mlebench_lite/mila_mfgp.py submit-prepare \
     --partition main-cpu \
     --key-file "$HOME/.config/bo-mf14/opencode-api-key" \
     --token-file "$HOME/.config/bo-mf14/kaggle-token" \
     --data-dir /network/scratch/d/duz/mlebench-data \
     --protocol-source /network/scratch/d/duz/mila-mfgp14-COMMIT/runs/mf14_aq05/protocols
   ```

   This reuses the exact T/V membership and host scorer, and therefore the same
   random half-T sample. The script verifies source, seed and protocol identity.
   Protocols are referenced by symlink: retain the original protocols directory.
   Existing prepared datasets are checksum-verified and reused. Omitting
   `--protocol-source` creates a standalone protocol; do not call that a matched
   comparison with separately prepared baseline runs.

4. After preparation succeeds (`runs/mf14_gp/ready.json`), submit the short
   GPU/provider preflight:

   ```bash
   python3 tasks/mlebench_lite/mila_mfgp.py submit-preflight --partition main
   ```

5. Only after that succeeds (`runs/mf14_gp/gpu-ready.json`), submit the runs:

   ```bash
   python3 tasks/mlebench_lite/mila_mfgp.py submit --partition long
   ```

   This submits 14 separate jobs. Slurm determines how many run concurrently;
   it is not a single sequential 168-hour job. Use `--tasks leaf-classification`
   on this command to submit only that task initially. Submission is idempotent.
   Full jobs request 13 hours wall time for startup and final grading; the search
   budget remains 12 active hours. All task resources fit `long`; several exceed
   `main`/`unkillable` limits. See [Mila partition limits](https://docs.mila.quebec/technical_reference/clusters/mila/partitions/).

6. For an interrupted run, after Slurm confirms the previous allocation ended:

   ```bash
   python3 tasks/mlebench_lite/mila_mfgp.py resume --partition long \
     --task leaf-classification --mode mf_gp
   ```

   Replace the task ID as needed. Recovery verifies the old allocation's release,
   preserves remaining active time and each program/fidelity execution cap, and
   reruns interrupted training from scratch. It does not retry a recorded
   evaluator/program infrastructure failure or reuse partially scored output.

7. Collect available results, keeping failed/missing runs in the table:

   ```bash
   python3 tasks/mlebench_lite/mila_mfgp.py collect
   ```

Logs and manifests are under `runs/mf14_gp/`. Official private grading is invoked
only after each search has selected its measured full-data submission. Source
changes invalidate the prepared campaign; never edit the bundle between arms or
resume a campaign using a different source version.

## What still needs Mila

The local audit does not substitute for Slurm allocation, the exact Linux/CUDA
installation, live provider/key access, real competition data preparation and
protocol admission, or the short GPU preflight. No full experiment was launched
by the local audit. See `MF_GP_AUDIT.md` for the completed local checks.
