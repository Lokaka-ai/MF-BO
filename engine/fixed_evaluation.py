"""One prepared outer holdout, programmatic scoring and cooperative data views.

The same-UID executor is NOT a security sandbox. Only the prepared public view
is supplied to candidate prompts/cwd; malicious absolute-path access is outside
this boundary. Preparation runs in a killable process group with a total deadline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import numpy as np
import pandas as pd
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1
PUBLIC_TABLES = ("train.csv", "validation.csv", "test.csv", "sample_submission.csv")
MAXIMIZE = {"roc_auc": True, "accuracy": True, "quadratic_kappa": True,
            "log_loss": False, "rmse": False, "rmsle": False}


class EvaluationError(RuntimeError):
    """Shared setup/scorer failure: stop the task, never label a proposal invalid."""


class InvalidPrediction(ValueError):
    """Candidate output violates the frozen contract; eligible for normal debug."""


def enabled(cfg):
    return bool(getattr(getattr(cfg, "evaluation", None), "enabled", False))


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def _inside(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationError("Protocol paths must be relative and stay within their root")
    return path


def _read_table(path):
    # Preserve IDs, empty strings, literal 'NA', Unicode, and exact text targets.
    with open(path, newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise InvalidPrediction("Missing or duplicate column names")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _ids(frame, columns, metadata=()):
    if any(c not in frame for c in columns) or frame[columns].eq("").any().any():
        raise InvalidPrediction("Missing or empty ID columns")
    # Some official formats identify rows by content plus metadata (Insults:
    # Comment + Date). Metadata may be empty, but the complete key is unique.
    keys = pd.MultiIndex.from_frame(frame[columns + list(metadata)])
    if keys.has_duplicates:
        raise InvalidPrediction("Duplicate prediction IDs")
    return keys


def align_predictions(path, reference, spec):
    try:
        pred = _read_table(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise InvalidPrediction("Prediction file is missing or unreadable") from exc
    ids, columns = spec["id_columns"], spec["prediction_columns"]
    metadata = spec.get("metadata_columns", [])
    if set(pred.columns) != set(ids + columns + metadata):
        raise InvalidPrediction("Prediction columns must exactly match the fixed IDs and targets")
    expected, actual = _ids(reference, ids, metadata), _ids(pred, ids, metadata)
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise InvalidPrediction("Missing or extra prediction IDs, or changed submission metadata")
    pred.index = actual
    aligned = pred.reindex(expected)
    if metadata and not np.array_equal(aligned[metadata].to_numpy(), reference[metadata].to_numpy()):
        raise InvalidPrediction("Submission metadata differs from the fixed template")
    values = aligned[columns].to_numpy()
    if spec["prediction_kind"] == "text":
        return values
    try:
        values = values.astype(np.float64)
    except ValueError as exc:
        raise InvalidPrediction("Predictions must be numeric") from exc
    if not np.isfinite(values).all():
        raise InvalidPrediction("Predictions contain NaN or infinity")
    name = spec["metric"]["name"]
    if name in ("roc_auc", "log_loss") and ((values < 0).any() or (values > 1).any()):
        raise InvalidPrediction("Probabilities must be between zero and one")
    if name in ("accuracy", "quadratic_kappa"):
        labels = spec["metric"].get("labels")
        if (values != np.floor(values)).any() or (labels and not np.isin(values, np.asarray(labels, dtype=float)).all()):
            raise InvalidPrediction("Class predictions must be declared discrete class labels")
    bounds = spec.get("value_range")
    if bounds and ((values < bounds[0]).any() or (values > bounds[1]).any()):
        raise InvalidPrediction("Predictions are outside the declared value range")
    if (spec.get("probability_sum") or (name == "log_loss" and values.shape[1] > 1)) and not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise InvalidPrediction("Class probabilities must sum to one")
    return values


def compute_metric(truth, predictions, metric):
    """Controlled standard implementations; no candidate- or LLM-supplied score."""
    name = metric["name"]
    y, p = np.asarray(truth), np.asarray(predictions)
    if name == "accuracy":
        if np.issubdtype(p.dtype, np.number):
            y = y.astype(float)
        return float(metrics.accuracy_score(y.ravel(), p.ravel()))
    if name == "log_loss":
        eps = metric["clip_eps"]
        labels = metric["labels"]
        y = y.astype(float) if all(isinstance(label, (int, float)) for label in labels) else y.astype(str)
        p = np.clip(p.astype(float), eps, 1 - eps)
        if p.shape[1] == 1:
            return float(metrics.log_loss(y.ravel(), p.ravel(), labels=labels))
        # sklearn sorts class labels. Reorder probabilities explicitly from the
        # protocol's target order instead of silently assuming alphabetical order.
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        p = p[:, order]
        p = p / p.sum(axis=1, keepdims=True)
        return float(metrics.log_loss(y.ravel(), p, labels=sorted(labels)))
    y, p = y.astype(float), p.astype(float)
    if name == "roc_auc":
        if metric["average"] == "micro" or y.shape[1] == 1:
            return float(metrics.roc_auc_score(y.ravel(), p.ravel()))
        return float(metrics.roc_auc_score(y, p, average="macro"))
    if name == "quadratic_kappa":
        if not np.isin(p, metric["labels"]).all() or (p != np.floor(p)).any():
            raise InvalidPrediction("Kappa predictions must be declared discrete class labels")
        return float(metrics.cohen_kappa_score(y.ravel(), p.ravel(), weights="quadratic"))
    if name == "rmse":
        return float(np.sqrt(metrics.mean_squared_error(y, p)))
    if name == "rmsle":
        if (y < 0).any() or (p < 0).any():
            raise InvalidPrediction("RMSLE requires nonnegative targets and predictions")
        per_target = np.sqrt(metrics.mean_squared_error(np.log1p(y), np.log1p(p), multioutput="raw_values"))
        return float(np.mean(per_target))
    raise EvaluationError(f"Unsupported official metric: {name}")


def validate_spec(spec, task_id=None):
    for field in ("id_columns", "prediction_columns", "truth_columns", "source_target_columns"):
        value = spec.get(field)
        if not isinstance(value, list) or not value or len(value) != len(set(value)) or not all(isinstance(c, str) and c for c in value):
            raise EvaluationError(f"Invalid protocol {field}")
    if set(spec["id_columns"]) & set(spec["prediction_columns"]):
        raise EvaluationError("IDs and prediction targets must be distinct")
    metric = spec["metric"]
    name = metric["name"]
    if name not in MAXIMIZE or type(metric.get("maximize")) is not bool or metric["maximize"] != MAXIMIZE[name]:
        raise EvaluationError("Unsupported metric or incorrect optimization direction")
    if spec.get("prediction_kind") not in ("numeric", "text"):
        raise EvaluationError("prediction_kind must be numeric or text")
    if spec["prediction_kind"] == "text" and name != "accuracy":
        raise EvaluationError("Text predictions require exact-match accuracy")
    nt, npred = len(spec["truth_columns"]), len(spec["prediction_columns"])
    if (name == "log_loss" and nt != 1) or (name != "log_loss" and nt != npred):
        raise EvaluationError("Truth and prediction target dimensions do not match")
    if name in ("accuracy", "quadratic_kappa") and npred != 1:
        raise EvaluationError("Accuracy/kappa require a single class or token per scoring row")
    if name == "roc_auc" and metric.get("average") not in ("macro", "micro"):
        raise EvaluationError("AUC must explicitly define macro/micro aggregation")
    if name in ("log_loss", "quadratic_kappa") and not metric.get("labels"):
        raise EvaluationError("Ordered class labels are required")
    if name == "log_loss" and not (all(isinstance(x, str) for x in metric["labels"]) or
                                   all(type(x) in (int, float) for x in metric["labels"])):
        raise EvaluationError("Class labels must be uniformly strings or uniformly numbers")
    if name == "log_loss":
        if not 0 < metric.get("clip_eps", 0) < .5:
            raise EvaluationError("log_loss requires an explicit clipping epsilon")
        n = len(spec["prediction_columns"])
        if n != len(metric["labels"]) and not (n == 1 and len(metric["labels"]) == 2):
            raise EvaluationError("Probability columns must match ordered class labels")
    if not spec.get("metric_source") or not spec.get("candidate_instructions"):
        raise EvaluationError("Metric source and task-specific candidate instructions are required")
    expected_path = ROOT / "config" / "fixed_evaluation_metrics.json"
    if task_id and expected_path.exists():
        expected = _json(expected_path).get(task_id)
        if expected and any(metric.get(k) != v for k, v in expected["metric"].items()):
            raise EvaluationError("Metric contradicts the independently recorded mlebench-lite definition")
        if expected and (spec["id_columns"] != expected["id_columns"] or spec.get("metadata_columns", []) != expected["metadata_columns"]):
            raise EvaluationError("ID/metadata columns contradict the official task format")


def _source_identity(data_dir):
    digest = hashlib.sha256()
    for base, dirs, files in os.walk(data_dir):
        dirs.sort()
        for name in sorted(files):
            path = Path(base) / name
            st = path.stat()
            digest.update(json.dumps([str(path.relative_to(data_dir)), st.st_size, st.st_mtime_ns]).encode())
    return digest.hexdigest()


def _protocol_identity(directory, spec):
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for name in ("evaluate.py", "members.csv", "truth.csv", "assets.csv", *["public/" + p for p in PUBLIC_TABLES]):
        path = directory / name
        if path.exists():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def admit_prepared(directory, data_dir, task_id, seed):
    """Once at setup: audit actual memberships, public tables/assets and scorer I/O."""
    spec = _json(directory / "protocol.json")
    validate_spec(spec, task_id)
    split = spec["split"]
    if split["seed"] != seed or not split.get("unit") or split.get("strategy") not in ("stratified", "random", "group", "time", "provided"):
        raise EvaluationError("Invalid split definition: use strategy stratified/random/group/time/provided, a nonempty unit, and the supplied seed")
    members = _read_table(directory / "members.csv")
    _ids(members, ["sample_id"])
    # Private source identity cannot be hidden by renaming a V sample.
    _ids(members, ["source_id"])
    if set(members["split"]) != {"T", "V", "test"}:
        raise EvaluationError("Nonempty T, V and test memberships are required")
    partitions = {part: set(members.loc[members.split == part, "sample_id"]) for part in ("T", "V", "test")}
    all_samples = set(members.sample_id)
    if split["strategy"] == "group":
        if "group" not in members or (members.loc[members.split != "test", "group"] == "").any():
            raise EvaluationError("Group holdout requires actual nonempty group identifiers")
        t = set(members.loc[members.split == "T", "group"])
        v = set(members.loc[members.split == "V", "group"])
        if t & v:
            raise EvaluationError("Training and validation groups overlap")
    if split["strategy"] == "time":
        dates = pd.to_datetime(members.loc[members.split != "test", "time"], utc=True, errors="raise")
        if dates.isna().any() or dates[members.split == "T"].max() >= dates[members.split == "V"].min():
            raise EvaluationError("Time holdout requires a strict chronological boundary")
    public = directory / "public"
    if {p.name for p in public.iterdir()} != set(PUBLIC_TABLES):
        raise EvaluationError("Prepare only the four public tables; the host creates asset directories from assets.csv, so do not create even empty asset directories")
    for part, name in (("T", "train.csv"), ("V", "validation.csv"), ("test", "test.csv")):
        path = public / name
        if path.is_symlink():
            raise EvaluationError("Public tables must be materialized, not links to raw labels")
        frame = _read_table(path)
        _ids(frame, ["sample_id"])
        if set(frame.sample_id) != partitions[part]:
            raise EvaluationError(f"{name} does not match actual {part} membership")
        if part != "T" and set(frame.columns) & set(spec["source_target_columns"]):
            raise EvaluationError("Outer validation/test features contain supervised target columns")
    truth = _read_table(directory / "truth.csv")
    _ids(truth, spec["id_columns"], spec.get("metadata_columns", []))
    if not len(truth) or set(truth["__sample_id"]) != partitions["V"]:
        raise EvaluationError("Truth must cover exactly the fixed outer validation samples")
    y = truth[spec["truth_columns"]].to_numpy()
    # A deterministic smoke prediction exercises the real metric on ALL of V.
    # Mathematical definitions are tested independently in the repository.
    metric = spec["metric"]
    if metric["name"] == "log_loss":
        p = np.full((len(y), len(spec["prediction_columns"])), 1 / len(metric["labels"]))
    elif metric["name"] == "roc_auc":
        p = np.full(y.shape, 1 / y.shape[1] if spec.get("probability_sum") else .5)
    else:
        p = y.copy()
    smoke = truth[spec["id_columns"] + spec.get("metadata_columns", [])].copy()
    for index, column in enumerate(spec["prediction_columns"]):
        smoke[column] = p[:, index]
    smoke_path = directory / "smoke_predictions.csv"
    smoke.to_csv(smoke_path, index=False)
    try:
        aligned = align_predictions(smoke_path, truth, spec)
        score = compute_metric(y, aligned, metric)
    finally:
        smoke_path.unlink(missing_ok=True)
    if not math.isfinite(score):
        raise EvaluationError("The fixed validation metric is undefined")
    template = _read_table(public / "sample_submission.csv")
    if set(template.columns) != set(spec["id_columns"] + spec["prediction_columns"] + spec.get("metadata_columns", [])) or not len(template):
        raise EvaluationError("Submission template must define all test prediction IDs/targets")
    _ids(template, spec["id_columns"], spec.get("metadata_columns", []))
    official_source = spec["official_sample_submission"]
    official_path = _inside(data_dir, official_source["path"])
    if official_source.get("archive_member"):
        with zipfile.ZipFile(official_path) as archive:
            with archive.open(official_source["archive_member"]) as stream:
                official = pd.read_csv(stream, dtype=str, keep_default_na=False)
    else:
        official = _read_table(official_path)
    meta = spec.get("metadata_columns", [])
    official_ids, template_ids = _ids(official, spec["id_columns"], meta), _ids(template, spec["id_columns"], meta)
    expected_targets = [c for c in official.columns if c not in spec["id_columns"] + spec.get("metadata_columns", [])]
    if spec["prediction_columns"] != expected_targets:
        raise EvaluationError(f"Prediction targets/order differ from the complete official template: expected {expected_targets!r}, got {spec['prediction_columns']!r}")
    if set(official.columns) != set(template.columns) or len(official_ids) != len(template_ids) or len(official_ids.difference(template_ids)):
        raise EvaluationError("Prepared template omits/changes official test IDs, target columns or official submission metadata")
    if meta:
        official.index = official_ids
        if not np.array_equal(official.reindex(template_ids)[meta].to_numpy(), template[meta].to_numpy()):
            raise EvaluationError("Prepared template changed official submission metadata")
    if len(template_ids.intersection(_ids(truth, spec["id_columns"], meta))):
        raise EvaluationError("Validation and official test prediction IDs overlap")
    assets_path = directory / "assets.csv"
    if assets_path.exists():
        assets = _read_table(assets_path)
        _ids(assets, ["public_path"])
        sources_with_labels = set(assets.loc[assets.role == "target", "source"])
        source_partitions = {}
        for row in assets.to_dict("records"):
            sample = row["sample_id"]
            if sample not in all_samples or row["role"] not in ("feature", "target"):
                raise EvaluationError("Invalid asset membership or role")
            if row["role"] == "target" and sample not in partitions["T"]:
                raise EvaluationError("Outer validation/test target assets must not be public")
            if row["role"] == "feature" and row["source"] in sources_with_labels:
                raise EvaluationError("A supervised asset cannot also be an input feature")
            # source: raw relative file or @prepared/<path> for decoded archives.
            source = row["source"]
            origin = _inside(directory, source[10:]) if source.startswith("@prepared/") else _inside(data_dir, source)
            destination = _inside(public, row["public_path"])
            if not origin.is_file() or destination.exists() or destination.is_symlink():
                raise EvaluationError(f"Asset source {source!r} must be an existing file at {origin}; public path {row['public_path']!r} must not already exist. @prepared/path means output_dir/path, without an extra prepared/ directory")
            partition = "T" if sample in partitions["T"] else ("V" if sample in partitions["V"] else "test")
            stat = origin.stat()
            canonical_source = (stat.st_dev, stat.st_ino)
            previous_partition = source_partitions.setdefault(canonical_source, partition)
            if previous_partition != partition:
                raise EvaluationError("The same source asset occurs in multiple data partitions")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if sample in partitions["V"]:
                # Filenames such as cat.0.jpg / whale_1.aiff encode answers.
                # Even a readlink() must not reveal their original name.
                try:
                    os.link(origin.resolve(), destination)
                except OSError:
                    shutil.copyfile(origin, destination)
            elif source.startswith("@prepared/"):
                destination.symlink_to(os.path.relpath(origin, destination.parent))
            else:
                destination.symlink_to(origin.resolve())
    spec["version"] = VERSION
    spec["task_id"] = task_id
    spec["source_identity"] = _source_identity(data_dir)
    spec["source_dir"] = str(data_dir.resolve())
    spec["n_expected"] = len(truth)
    return spec


def _run_process(command, *, timeout, log_path, input_text=None):
    """Kill the entire setup/scoring process group, including API/CLI children."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with log_path.open("w") as output:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, env=env, cwd=ROOT)
        try:
            proc.communicate(input=input_text, timeout=max(.001, timeout))
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    if proc.returncode:
        raise EvaluationError(f"Evaluation process failed (exit {proc.returncode}); see {log_path}")


class FixedEvaluation:
    def __init__(self, directory, spec, cfg):
        self.directory, self.spec, self.cfg = Path(directory), spec, cfg
        self.protocol_id = spec["protocol_id"]
        self.failed = threading.Event()
        self._attempts = {}
        self._lock = threading.Lock()

    @classmethod
    def prepare_or_load(cls, cfg, task_desc, journal):
        from omegaconf import OmegaConf
        is_resume = bool(journal.nodes)
        state = getattr(journal, "evaluation_state", None)
        if is_resume and not state:
            raise EvaluationError("Cannot add fixed evaluation to a legacy checkpoint; start a new search")
        if state and state.get("failure"):
            raise EvaluationError("This run stopped at a shared evaluator failure; inspect saved predictions before starting a new search")
        ledger = Path(cfg.log_dir) / "phase1/root_draft_state.json"
        transitions = Path(cfg.log_dir) / "phase1/transitions.jsonl"
        if not is_resume and (ledger.exists() or (transitions.exists() and transitions.stat().st_size)):
            raise EvaluationError("A new fixed search requires a fresh log directory; old feedback cannot be reused")
        if state and not cfg.evaluation.enabled:
            raise EvaluationError("Cannot resume a fixed protocol in legacy mode")
        configured = cfg.evaluation.protocol_dir
        directory = Path(str(configured).format(task_id=cfg.exp_id, seed=cfg.agent.seed) if configured else Path(cfg.log_dir) / "evaluation_protocol").resolve()
        if state and str(directory) != state["protocol_dir"]:
            raise EvaluationError("Resume protocol directory differs from admitted journal")
        started = time.monotonic()
        timeout = min(float(cfg.evaluation.setup_timeout_seconds), float(cfg.agent.time_limit))
        if timeout <= 0 or int(cfg.evaluation.setup_max_attempts) < 0:
            raise EvaluationError("Setup requires positive timeout and a nonnegative attempt cap (0 means deadline only)")
        request = {"cfg": OmegaConf.to_container(cfg, resolve=True), "task_desc": task_desc,
                   "directory": str(directory), "resuming": is_resume,
                   "started_monotonic": started,
                   "expected_protocol_id": state["protocol_id"] if state else None}
        log_path = Path(cfg.log_dir) / "evaluation_setup" / "worker.log"
        try:
            _run_process([sys.executable, "-m", "agents.evaluation_setup_agent"], timeout=timeout,
                         log_path=log_path, input_text=json.dumps(request, default=str))
            spec = _json(directory / "protocol.json")
        except Exception as exc:
            _write_json(Path(cfg.log_dir) / "evaluation_setup" / "failure.json",
                        {"error_type": type(exc).__name__, "log": str(log_path), "elapsed_seconds": time.monotonic() - started})
            raise EvaluationError(f"Fixed evaluation preparation/load failed; see {log_path}") from exc
        elapsed = time.monotonic() - started
        instance = cls(directory, spec, cfg)
        instance.journal = journal
        if state:
            for node in journal.nodes:
                if node.stage != "root" and node.evaluation_protocol_id != instance.protocol_id:
                    raise EvaluationError("Checkpoint contains a node from another/legacy evaluation protocol")
                if node.is_buggy is False and node.stage != "root":
                    result = node.evaluation_result or {}
                    if not result.get("valid") or result.get("metric") != node.metric.value or node.metric.maximize != spec["metric"]["maximize"]:
                        raise EvaluationError("Checkpoint metric is not the admitted evaluator result")
            # The managed runtime can recover nodes newer than journal.json.
            # Admit its ledger/transition snapshots before it appends or fits them.
            if ledger.exists():
                saved = _json(ledger)
                if saved.get("evaluation_protocol_id") != instance.protocol_id:
                    raise EvaluationError("Decision ledger belongs to another/legacy evaluation protocol")
                recovered = json.loads(saved["journal_snapshot"])
                snapshots = list(recovered["nodes"])
                snapshots.extend(p["node_snapshot"]["node"] for p in saved["pending"].values() if p.get("node_snapshot"))
                for node in snapshots:
                    if node["stage"] != "root" and node.get("evaluation_protocol_id") != instance.protocol_id:
                        raise EvaluationError("Recovered decision contains a foreign evaluation protocol")
            if transitions.exists():
                with transitions.open() as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record["valid"]:
                            node = (record.get("metadata") or {}).get("node_snapshot", {}).get("node", {})
                            result = node.get("evaluation_result") or {}
                            if node.get("evaluation_protocol_id") != instance.protocol_id or not result.get("valid") or result.get("metric") != record["child_metric_raw"]:
                                raise EvaluationError("Transition history lacks the matching fixed evaluator result")
        journal.evaluation_state = {"protocol_dir": str(directory), "protocol_id": instance.protocol_id,
                                    "setup_cost_seconds": spec["setup_cost_seconds"], "load_wall_seconds": elapsed}
        instance.budget_charge = elapsed if state else max(elapsed, spec["setup_cost_seconds"])
        instance._install_public_view()
        return instance

    def _install_public_view(self):
        workspace = Path(self.cfg.workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        destination = workspace / "input"
        target = self.directory / "public"
        if destination.is_symlink():
            if destination.resolve() != target:
                raise EvaluationError("Workspace input is bound to another data view")
        else:
            if destination.exists():
                if any(destination.iterdir()):
                    raise EvaluationError("Fixed mode refuses a workspace containing raw input data")
                destination.rmdir()
            destination.symlink_to(target, target_is_directory=True)
        for name in ("working", "submission", "evaluation_outputs", "helpers"):
            (workspace / name).mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "utils/evaluation_helpers.py", workspace / "helpers/evaluation_helpers.py")

    def candidate_contract(self):
        visible = {key: self.spec[key] for key in ("metric", "id_columns", "prediction_columns", "prediction_kind", "split", "candidate_instructions")}
        visible["metadata_columns"] = self.spec.get("metadata_columns", [])
        return ("Fixed outer evaluation protocol " + self.protocol_id + "\n" + json.dumps(visible, ensure_ascii=False) + "\n"
                "Read ./input/train.csv (T only), validation.csv (V features only), test.csv and sample_submission.csv. "
                "Native assets are at the paths in these tables. Never access original datasets or evaluator files. "
                "Actually decode native assets; never swallow loading/decoding errors into zero features or dummy predictions. "
                "AIFF requires an AIFF reader (e.g. aifc for uncompressed AIFF on Python 3.11), not scipy.io.wavfile. "
                "Keep every outer V member/target. Split, early stopping, OOF, calibration and ensemble-weight fitting use T only. "
                "Output validation.csv in os.environ['MLEV_EVALUATION_OUTPUT_DIR']; both CSVs contain exactly "
                "id_columns + prediction_columns + metadata_columns, preserving metadata values including empty strings. "
                "Write test predictions to ./submission/submission.csv. Both must use the SAME trained model, inference, "
                "ensemble order, temperature and clipping. No candidate main score is needed; stdout is diagnostic only.\n"
                "For binary grid calibration import fit_binary_temperature, binary_probabilities from evaluation_helpers. "
                "temperature = fit_binary_temperature(T_calibration_logits, T_calibration_labels, temperatures=grid, clip_eps=eps); "
                "valid_p = binary_probabilities(valid_logits, temperature, eps); "
                "test_p = binary_probabilities(test_logits, temperature, eps). The fitter returns ONLY a temperature.\n"
                "Both helpers require 1-D logits; calibration labels must have the same shape. For multilabel tasks "
                "work per target, or flatten T logits/labels to fit one shared temperature and also flatten V/test "
                "logits before binary_probabilities, then reshape its output to the original target matrix.\n"
                "Optionally save calibration_cache.npz in MLEV_EVALUATION_OUTPUT_DIR with protocol_id, node_id, T calibration "
                "logits/labels, V/test logits and IDs (NO V answers). Debug may reuse the parent's recorded cache only when "
                "protocol and model provenance match; then emit fresh predictions in the new attempt directory.")

    def preview(self):
        return "\n\n".join(name + ":\n" + pd.read_csv(self.directory / "public" / name, nrows=4, keep_default_na=False).to_csv(index=False)
                            for name in PUBLIC_TABLES) + "\n" + self.candidate_contract()

    def begin_attempt(self, node, remaining_seconds):
        if self.failed.is_set():
            raise EvaluationError("Shared evaluator has failed; this search must stop")
        attempt = uuid.uuid4().hex
        output = Path(self.cfg.workspace_dir) / "evaluation_outputs" / node.id / attempt
        output.mkdir(parents=True)
        node.evaluation_protocol_id = self.protocol_id
        node.evaluation_attempt_id = attempt
        node.evaluation_output_dir = str(output)
        with self._lock:
            self._attempts[node.id] = (attempt, str(output))
        # A repeated invocation of the same node cannot reuse a stale test file.
        (Path(self.cfg.workspace_dir) / "submission" / f"submission_{node.id}.csv").unlink(missing_ok=True)
        return {"output_dir": str(output), "protocol_id": self.protocol_id, "node_id": node.id,
                "helpers_dir": str(Path(self.cfg.workspace_dir) / "helpers"),
                "timeout": max(.001, min(float(self.cfg.exec.timeout), remaining_seconds))}

    def score(self, node):
        if self.failed.is_set():
            raise EvaluationError("Shared evaluator failed; no further scores can be admitted")
        with self._lock:
            expected = self._attempts.get(node.id)
        if expected != (node.evaluation_attempt_id, node.evaluation_output_dir) or node.evaluation_protocol_id != self.protocol_id:
            raise EvaluationError("Missing current host-issued evaluation attempt")
        output = Path(node.evaluation_output_dir)
        result_path = Path(self.cfg.log_dir) / "evaluation_results" / f"{node.id}_{node.evaluation_attempt_id}.json"
        try:
            _run_process([sys.executable, "-m", "engine.fixed_evaluation", "score", "--protocol", str(self.directory),
                          "--predictions", str(output / "validation.csv"), "--submission",
                          str(Path(self.cfg.workspace_dir) / "submission" / f"submission_{node.id}.csv"),
                          "--result", str(result_path)], timeout=float(self.cfg.evaluation.score_timeout_seconds),
                         log_path=result_path.with_suffix(".log"))
            result = _json(result_path)
            if type(result.get("valid")) is not bool:
                raise EvaluationError("Scorer returned malformed result")
            if result["valid"] and (not math.isfinite(result["metric"]) or result["n_scored"] != self.spec["n_expected"]):
                raise EvaluationError("Scorer did not evaluate the full fixed validation set")
        except Exception as exc:
            self.failed.set()
            if hasattr(self, "journal"):
                self.journal.evaluation_state["failure"] = {"node_id": node.id, "predictions": str(output), "error_type": type(exc).__name__}
            raise EvaluationError(f"Shared evaluator failure; preserve predictions in {output}") from exc
        result.update(node_id=node.id, protocol_id=self.protocol_id, attempt_id=node.evaluation_attempt_id)
        _write_json(result_path, result)
        return result


def score_files(directory, predictions, submission):
    spec = _json(directory / "protocol.json")
    # Private reference failures are infrastructure errors, not candidate bugs.
    truth = _read_table(directory / "truth.csv")
    template = _read_table(directory / "public/sample_submission.csv")
    y = truth[spec["truth_columns"]].to_numpy()
    try:
        p = align_predictions(predictions, truth, spec)
        align_predictions(submission, template, spec)
        score = compute_metric(y, p, spec["metric"])
        if not math.isfinite(score):
            raise InvalidPrediction("Metric is undefined for these predictions")
    except InvalidPrediction as exc:
        return {"valid": False, "metric": None, "n_expected": len(truth), "n_scored": 0, "error": str(exc)}
    return {"valid": True, "metric": score, "n_expected": len(truth), "n_scored": len(truth), "error": None}


def finalize_submission(cfg, journal=None):
    """Export only the best admitted node, including the legacy ensemble location."""
    from engine.search_node import Journal
    from utils import serialize
    if journal is None:
        journal = serialize.load_json(Path(cfg.log_dir) / "journal.json", Journal)
    state = journal.evaluation_state
    if not state:
        raise EvaluationError("No fixed protocol journal available for final submission")
    candidates = [n for n in journal.nodes if n.stage != "root" and n.is_valid is True and n.is_buggy is False
                  and n.evaluation_protocol_id == state["protocol_id"] and n.evaluation_result
                  and n.evaluation_result.get("valid") and n.metric.value == n.evaluation_result.get("metric")]
    if not candidates:
        raise EvaluationError("No externally scored valid node is available for final submission")
    best = max(candidates, key=lambda n: n.metric)
    source = Path(best.evaluation_result["submission_file"])
    if not source.is_file():
        raise EvaluationError("Best scored node's submission is missing")
    workspace = Path(cfg.workspace_dir)
    for target in (workspace / "best_submission/submission.csv", workspace / "ensembles_csv/top1ens-total_run_time0.0h.csv"):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".csv.tmp")
        shutil.copyfile(source, temp)
        temp.replace(target)
    _write_json(Path(cfg.log_dir) / "final_submission.json", {"node_id": best.id, "protocol_id": state["protocol_id"],
                "metric": best.metric.value, "maximize": best.metric.maximize, "source": str(source),
                "submission": str(workspace / "best_submission/submission.csv"), "selection": "best_fixed_evaluated_node"})
    return best


def adapter_main(prepare):
    """Stable entry point used by the generated evaluate.py, including score replay."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("score")
    for name in ("protocol", "predictions", "submission", "result"):
        s.add_argument("--" + name, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--run-info", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(Path(args.data_dir), Path(args.output_dir), args.seed)
    elif args.command == "score":
        result = score_files(Path(args.protocol), Path(args.predictions), Path(args.submission))
        if result["valid"]:
            snapshot = Path(args.result).with_suffix(".submission.csv")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.submission, snapshot)
            result["submission_file"] = str(snapshot)
        _write_json(args.result, result)
    else:
        from types import SimpleNamespace
        info = _json(args.run_info)
        if not info.get("fixed_evaluation"):
            raise EvaluationError("The selected run did not enable fixed evaluation")
        finalize_submission(SimpleNamespace(log_dir=Path(info["log_dir"]), workspace_dir=Path(info["workspace_dir"])))


if __name__ == "__main__":
    adapter_main(None)
