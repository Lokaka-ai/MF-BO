"""Final submission selection and official grading, separate from search feedback."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


LEGACY_SELECTION_RULE = "latest_structurally_valid_ensemble_else_best_v1"


def _csv_signature(path):
    # Same structural check as the archived Base campaign's submission freezer.
    csv.field_size_limit(sys.maxsize)
    digest = hashlib.sha256()
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            count, width_errors = 0, 0
            for row in reader:
                count += 1
                width_errors += len(row) != len(header)
                value = (row[0] if row else "").encode("utf-8", errors="surrogatepass")
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        return {"parseable": True, "header": header, "row_count": count,
                "width_errors": width_errors, "duplicate_headers": len(header) != len(set(header)),
                "ordered_first_column_sha256": digest.hexdigest()}
    except Exception as exc:
        return {"parseable": False, "error_type": type(exc).__name__}


def finalize_legacy_submission(log_dir, workspace_dir):
    """Record the old score-blind choice without replacing the best node's CSV."""
    workspace = Path(workspace_dir).resolve()
    best = workspace / "best_submission/submission.csv"
    reference = _csv_signature(best) if best.is_file() else {"parseable": False}
    selected = best if best.is_file() else None
    selection = "best_submission_fallback" if selected else "missing"
    if reference.get("parseable"):
        candidates = sorted((workspace / "ensembles_csv").glob("*.csv"),
                            key=lambda p: (p.stat().st_mtime_ns, str(p.resolve())), reverse=True)
        for candidate in candidates:
            signature = _csv_signature(candidate)
            if (signature.get("parseable") and not signature["duplicate_headers"]
                    and signature["width_errors"] == 0
                    and all(signature[key] == reference[key]
                            for key in ("header", "row_count", "ordered_first_column_sha256"))):
                selected = candidate
                selection = "latest_structurally_valid_ensemble"
                break
    record = {"submission": str(selected) if selected else None, "selection": selection,
              "selection_rule": LEGACY_SELECTION_RULE}
    target = Path(log_dir) / "final_submission.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return record


def grade_run(log_dir, data_dir=None):
    """Grade exactly one preselected submission; never rank candidates by private score."""
    log_dir = Path(log_dir)
    cfg = OmegaConf.load(log_dir / "config.yaml")
    fixed = OmegaConf.select(cfg, "evaluation.enabled", default=False)
    final = log_dir / "final_submission.json"
    if not fixed and not final.is_file():
        finalize_legacy_submission(log_dir, cfg.workspace_dir)
    record = json.loads(final.read_text(encoding="utf-8"))
    path = Path(record["submission"]) if record["submission"] else None
    result = {"submission": str(path) if path else None, "selection": record["selection"],
              "report": None}
    if path is not None:
        from mlebench.grade import grade_csv
        from mlebench.registry import registry
        competition = registry.set_data_dir(Path(data_dir or cfg.dataset_dir)).get_competition(cfg.exp_id)
        result["report"] = grade_csv(path, competition).to_dict()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade the recorded final submission using mle-bench")
    parser.add_argument("--log-dir", required=True, help="Exact run log directory containing config.yaml")
    parser.add_argument("--data-dir", help="Override the saved mle-bench dataset root for the grading host")
    args = parser.parse_args()
    print(json.dumps(grade_run(args.log_dir, args.data_dir), allow_nan=False))
