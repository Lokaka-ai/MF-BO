"""One-time EvaluationSetupAgent, using the existing prompt/compiler/FunctionSpec API.

The parent process owns the total deadline, including inventory, LLM retries,
generated preparation and admission. This agent never creates a SearchNode.
"""

import ast
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

import pandas as pd
from omegaconf import OmegaConf

from llm import FunctionSpec, query, compile_prompt_to_md
from agents import debug_agent
from utils.response import extract_code
from engine.fixed_evaluation import (ROOT, VERSION, EvaluationError, _json, _write_json,
                                     _protocol_identity, _source_identity, admit_prepared)

SETUP_SPEC = FunctionSpec(
    name="submit_evaluation_protocol",
    description="Submit a task-faithful outer evaluation adapter, or explain why preparation is unsupported.",
    json_schema={
        "type": "object",
        "properties": {
            "supported": {"type": "boolean", "description": "False only if the real metric or legal data boundary cannot be implemented."},
            "reasoning": {"type": "string", "description": "Explain sample unit, dependencies, split, target order, official metric and supervision boundary, citing the Evaluation text or official source."},
            "code": {"type": "string", "description": "Complete Python adapter defining prepare(data_dir: Path, output_dir: Path, seed: int). Raw Python or one Python code fence. No training, API calls, or top-level execution."},
        },
        "required": ["supported", "reasoning", "code"],
    },
)


def inventory(data_dir):
    """Bounded discovery, not the candidate preview and never reused by candidates."""
    rows = []
    for base, dirs, files in os.walk(data_dir):
        dirs.sort()
        for name in sorted(files):
            path = Path(base) / name
            item = {"path": str(path.relative_to(data_dir)), "bytes": path.stat().st_size}
            try:
                if path.suffix in (".csv", ".tsv"):
                    item["head"] = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", nrows=4, keep_default_na=False).to_dict("split")
                elif path.suffix == ".zip":
                    with zipfile.ZipFile(path) as archive:
                        item["archive_members"] = archive.namelist()[:40]
                elif path.suffix in (".json", ".jsonl", ".txt"):
                    with path.open() as stream:
                        item["head"] = stream.read(3000)
            except Exception as exc:
                item["inspection_error"] = type(exc).__name__
            rows.append(item)
            if len(rows) >= 100:
                return rows
    return rows


def build_prompt(task_desc, data_inventory, task_id, previous=None):
    expected = _json(ROOT / "config/fixed_evaluation_metrics.json").get(task_id)
    prompt = {
        "Introduction": (
            "You are a Kaggle Grandmaster and Evaluation Engineer preparing a trustworthy, reusable evaluation protocol. "
            "Your expertise includes competition metrics, grouped/temporal validation, multimodal data, and leakage prevention. "
            "This is one initialization job before model search. You do not propose/train a model or create a draft. "
            "After preparation you exit; ordinary programs score every candidate on the same outer validation set."
        ),
        "Task description": task_desc,
        "Data exploration (private setup context only)": json.dumps(data_inventory, ensure_ascii=False, default=str),
        "Official metric constraint": json.dumps(expected) if expected else "Read the task Evaluation section; use a supported exact metric, never a proxy.",
        "Instructions": {
            "Scientific protocol design": [
                "Identify WHAT is predicted (including every target/pixel/token), WHY the split matches the dependency structure, and HOW complete predictions will be scored.",
                "Read actual file contents in prepare(), not only this truncated inventory. Check CSV/TSV/JSON, archives, image/audio names, sample submission, missing values, and actual groups/dates. Stream large tables. Do not train models, install packages, fetch hidden labels, read previous/private scores, or optimize the holdout using model scores.",
                "Prefer provided validation; otherwise use grouped holdout for patients/sentences/recordings, chronological holdout for future prediction (Whale uses recording dates), stratified holdout for independent classification, or random holdout when justified. Record actual memberships. All related samples must remain together. Split seed is supplied by the host. In protocol.json, split.strategy MUST be exactly one of: provided, group, time, stratified, random; describe the rationale separately, not inside this enum value.",
                "Keep sufficient classes for a defined metric. A legal metadata-based split adjustment is allowed during preparation; never delete targets, break groups, or switch macro/micro to make a score defined. If impossible, return supported=false with evidence.",
                "Use the exact official metric. Supported host functions: roc_auc (explicit macro/micro), log_loss (ordered labels and clip_eps), rmse, rmsle (mean of per-target RMSLE), accuracy (exact matching), quadratic_kappa. In multiclass log_loss, labels are strings in prediction-column order. Numeric class predictions for accuracy/kappa must be actual class labels, not continuous values.",
                "Birds' official long Id/Probability submission flattens sample/species pairs into one AUC. Preserve every species. RANZCR/Jigsaw/Plant use the documented macro aggregation. NOMAD scores both targets on the original nonnegative scale, averaging the two RMSLEs. Denoising expands ALL pixels and uses pixel RMSE, not per-image means. Text normalization scores all tokens' exact after strings, grouped by sentence for splitting.",
                "RANZCR's mle-bench CLASSES and prepared official template score nine targets, ending with CVC - Borderline; the original Kaggle description lists eleven. Use all nine template targets in their exact order, without inventing CVC - Normal or Swan Ganz Catheter Present prediction columns. Still treat ALL eleven original labels and annotations as supervision and remove them from public V/test features.",
            ],
            "Program interface and artifacts": [
                "Declare official_sample_submission as {path: original relative file, archive_member: optional CSV member inside an original ZIP}. It MUST point to the actual public official template; the host independently checks all test IDs/columns against it. Never point it at a generated/truncated template. For non-prediction fields such as Insults Date, declare metadata_columns, preserve them exactly (including empty strings) in truth and submissions, and do not include them in the metric. The complete id_columns + metadata_columns combination identifies a scoring row: Insults has duplicate Comment values but unique Comment + Date pairs; retain all these rows. The output columns are exactly id_columns + prediction_columns + metadata_columns.",
                "Return a complete Python module with imports and def prepare(data_dir, output_dir, seed). It may define helpers. Do not add a main guard: the host appends adapter_main(prepare), supporting prepare and score subcommands. Write ONLY under output_dir; never modify original data.",
                "Write protocol.json with: metric (name, maximize, average for AUC, labels for log_loss/kappa, clip_eps for log_loss); id_columns; prediction_columns in official order; truth_columns; prediction_kind ('numeric' or 'text'); source_target_columns (ALL supervised columns and aliases, including label-derived features); value_range for bounded numeric predictions; probability_sum=true for multiclass probabilities; split {unit,strategy,seed}; metric_source (short Evaluation quote or precise local official function reference); candidate_instructions (complete task-specific loading, native asset decoding, ID construction, V/test prediction format, target order, inverse transforms). Host adds identity/version/cost fields.",
                "Write PRIVATE members.csv with unique, nonempty sample_id, source_id, split ('T','V','test'), and group for grouped splits or ISO time for chronological splits. source_id identifies the ACTUAL original sample before renaming (canonical raw file, or source table + original row/key); it must be globally unique and must NOT be partition-prefixed to disguise duplicates. sample_id is the candidate-visible opaque native example ID (image for pixels, token for text with sentence group), not necessarily a scoring row. Original train/test IDs may need source-table namespaces; keep official test submission IDs separately. Never publish source_id or the private members table.",
                "Write exactly four materialized tables under public/: train.csv (only T, features and labels), validation.csv (only V features), test.csv (all test features), sample_submission.csv (official test scoring IDs/target columns). The first three MUST contain sample_id and exactly their members. Preserve useful native fields, e.g. before text, sentence_id, image path or numeric features. JSON/list fields can be serialized in CSV cells with documented decoding. Do not expose label columns, label-derived class metadata or answers in V/test tables.",
                "Write truth.csv OUTSIDE public: id_columns + truth_columns + metadata_columns (if any) + __sample_id. Each scoring row maps to a V sample via __sample_id. IDs must be unique and cover every V sample with every target. Use opaque V scoring IDs so they cannot collide with official test IDs. For class probability outputs truth_columns can contain one class label, while prediction_columns lists classes. For multi-target AUC/RMSLE truth columns are in the SAME target order as prediction columns. For text preserve exact strings including literal NA and empty strings.",
                "The candidate always writes validation.csv under MLEV_EVALUATION_OUTPUT_DIR and ./submission/submission.csv. Both CSVs have exactly id_columns + prediction_columns + metadata_columns (if any). Native structured outputs are losslessly expanded into this CSV interface: pixel coordinates or sample/species pairs become IDs; all tokens remain present. Do not substitute simpler predictions.",
                "Optional assets.csv is PRIVATE, with source, public_path, sample_id, role ('feature' or 'target'). Each row names one individual file, never a directory/archive. source is relative to data_dir, or @prepared/<relative path> for assets decoded under output_dir. The @prepared/ prefix is a marker for output_dir itself, NOT an extra prepared/ subdirectory: @prepared/images/a.png resolves to output_dir/images/a.png. Construct it as '@prepared/' + file_path.relative_to(output_dir).as_posix(), and verify that exact path exists. public_path is relative to public/ and referenced by the public tables. The host creates the asset directories and exposes only these declared assets. Target assets (clean denoising images, masks, labels) may ONLY belong to T. Do not write other files/directories in public/ yourself, including empty image/audio/asset directories.",
            ],
            "Supervision boundary and checks": [
                "Outer V answers are exclusively evaluator-side. Candidates may train, early-stop, cross-validate, stack or calibrate only inside T. Public preview is rebuilt from public tables; your private setup conversation is never candidate memory.",
                "Inspect implicit labels: Dogs/Cats filenames (cat.* / dog.*), Whale filename suffix _0/_1, class directories, target-derived metadata and duplicate label tables. Give V opaque IDs and relative filenames; remove original label-bearing paths/IDs from ALL public V tables and candidate_instructions. V asset exposure uses hard links/copies so readlink cannot expose label-bearing source names. Do not put source path mappings in public data. T may retain its labels.",
                "Do not expose the whole raw dataset via a directory symlink, or the entire clean-image/mask directory. Declare individual feature files and T-only target files. Normal pretrained weights are allowed; old search checkpoints which saw V are not allowed. Same-UID paths are cooperative isolation, not an OS sandbox.",
                "Before returning, verify T/V disjointness (including groups/time), actual V answer coverage, unique IDs, full target order, and no supervised aliases/assets in V. Use exact task semantics; do not replace errors with constants or inner joins. The host validates memberships and scores a synthetic prediction on actual V; repository fixtures independently test the metric mathematics.",
                "Use numpy/pandas/sklearn/Pillow and existing local readers as needed; preparation is CPU data handling. Archive extraction goes under output_dir and must preserve only needed files. Avoid loading huge pixel/table copies when streaming is sufficient.",
                "For each native media format, decode one actual declared feature file with an installed reader during prepare(), and put the verified loading recipe in candidate_instructions. AIFF is not WAV: scipy.io.wavfile.read cannot read it. On Python 3.11, aifc.open plus numpy.frombuffer with the correct sample width and big-endian dtype can read uncompressed AIFF without optional packages. Never advise silently replacing read/decode failures with zero features or prior predictions; raise the error so DEBUG can repair it.",
            ],
            "Response format and bounded repair": [
                "Respond via submit_evaluation_protocol with supported (JSON boolean), reasoning (concise but complete protocol rationale) and code (the entire runnable Python module). No search proposal, candidate score, draft, tool calls or separate evaluator model.",
                "The host executes preparation and checks its outputs. Failures go to the DEBUG agent for repair and re-execution until success, within one hard deadline including API waits and program execution and any explicitly configured attempt cap. On repair use the exact error/output and prior code below; fix the first failed boundary. Return a complete corrected module. Never silently fall back to candidate-reported scoring.",
            ],
        },
    }
    if previous:
        prompt["Previous preparation and observed failure"] = previous
    return prompt


def parse_response(response):
    if not isinstance(response, dict) or type(response.get("supported")) is not bool:
        raise EvaluationError("Setup response must include a boolean supported field")
    if not response["supported"]:
        raise EvaluationError("Unsupported task: " + str(response.get("reasoning", "No reason provided")))
    if not isinstance(response.get("reasoning"), str) or not response["reasoning"].strip():
        raise EvaluationError("Setup response has no protocol rationale")
    code = response.get("code", "")
    if not isinstance(code, str):
        raise EvaluationError("Setup code must be a string")
    if "```" in code:
        code = extract_code(code)
    module = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "prepare" for node in module.body):
        raise EvaluationError("Adapter must define prepare(data_dir, output_dir, seed)")
    return code + "\n\nif __name__ == '__main__':\n    from engine.fixed_evaluation import adapter_main\n    adapter_main(prepare)\n"


def worker(request):
    cfg = OmegaConf.create(request["cfg"])
    directory = Path(request["directory"])
    data_dir = Path(cfg.data_dir).resolve()
    logs = Path(cfg.log_dir) / "evaluation_setup"
    logs.mkdir(parents=True, exist_ok=True)
    started = request.get("started_monotonic", time.monotonic())
    directory.parent.mkdir(parents=True, exist_ok=True)
    # One shared protocol per task/data/seed, including simultaneous method starts.
    with (directory.parent / (directory.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "protocol.json").exists():
            spec = _json(directory / "protocol.json")
            saved_id = spec["protocol_id"]
            identity_spec = {k: v for k, v in spec.items() if k not in ("protocol_id", "setup_cost_seconds")}
            if (spec["version"] != VERSION or spec["task_id"] != cfg.exp_id or spec["split"]["seed"] != cfg.agent.seed
                    or spec["source_dir"] != str(data_dir) or spec["source_identity"] != _source_identity(data_dir)
                    or _protocol_identity(directory, identity_spec) != saved_id
                    or request["expected_protocol_id"] not in (None, saved_id)):
                raise EvaluationError("Prepared protocol/data/seed differs from this run")
            _write_json(logs / "load.json", {"protocol_id": saved_id, "reused": True})
            return
        if request["resuming"]:
            raise EvaluationError("Saved evaluation protocol is missing; refusing to regenerate on resume")
        if directory.exists():
            raise EvaluationError("Protocol directory exists without a completed protocol; preserve it and choose a clean directory")
        data_inventory = inventory(data_dir)
        previous = None
        max_attempts = int(cfg.evaluation.setup_max_attempts)
        attempt = 0
        while True:  # The parent enforces the existing total preparation deadline.
            attempt += 1
            attempt_dir = logs / f"attempt_{uuid.uuid4().hex}"
            prepared = attempt_dir / "prepared"
            prepared.mkdir(parents=True)
            prompt = build_prompt(request["task_desc"], data_inventory, cfg.exp_id, previous)
            code = previous["code"] if previous else ""
            try:
                if previous:
                    response = debug_agent.repair_evaluation_setup(cfg, prompt, SETUP_SPEC, attempt_dir / "prompt.md")
                else:
                    (attempt_dir / "prompt.md").write_text(compile_prompt_to_md(prompt))
                    response = query(system_message=prompt, user_message=None, func_spec=SETUP_SPEC,
                                     model=cfg.agent.code.model, temperature=cfg.agent.code.temp, cfg=cfg)
                _write_json(attempt_dir / "response.json", response)
                if isinstance(response, dict) and isinstance(response.get("code"), str):
                    code = response["code"]  # Preserve even syntax-invalid code for DEBUG.
                (prepared / "evaluate.py").write_text(parse_response(response))
                with (attempt_dir / "prepare.log").open("w") as output:
                    # Same process group as worker, hence parent's deadline kills all descendants.
                    subprocess.run([sys.executable, str(prepared / "evaluate.py"), "prepare", "--data-dir", str(data_dir),
                                    "--output-dir", str(prepared), "--seed", str(cfg.agent.seed)],
                                   stdout=output, stderr=subprocess.STDOUT, check=True)
                spec = admit_prepared(prepared, data_dir, cfg.exp_id, cfg.agent.seed)
                spec["protocol_id"] = _protocol_identity(prepared, spec)
                spec["setup_cost_seconds"] = time.monotonic() - started
                _write_json(prepared / "protocol.json", spec)
                # Rename within the destination filesystem, also for shared external protocol roots.
                staging = directory.parent / (directory.name + ".ready_" + uuid.uuid4().hex)
                shutil.copytree(prepared, staging, symlinks=True)
                spec["setup_cost_seconds"] = time.monotonic() - started
                _write_json(staging / "protocol.json", spec)
                staging.rename(directory)
                _write_json(logs / "success.json", {"attempts": attempt, "protocol_id": spec["protocol_id"],
                                                    "setup_cost_seconds": spec["setup_cost_seconds"]})
                return
            except Exception as exc:
                error = traceback.format_exc()
                (attempt_dir / "error.txt").write_text(error)
                output_path = attempt_dir / "prepare.log"
                previous = {"code": code, "error": error[-6000:],
                            "program_output": output_path.read_text()[-6000:] if output_path.exists() else ""}
                if max_attempts and attempt >= max_attempts:
                    raise EvaluationError(f"Preparation failed after {attempt} total attempts: {exc}") from exc


if __name__ == "__main__":
    worker(json.load(sys.stdin))
