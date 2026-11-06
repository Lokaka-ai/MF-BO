"""Compare saved settings against base mlebench_lite + the existing GP6 preset."""
import json
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}


def audit_settings(config_path, command, *, smoke):
    expected = OmegaConf.merge(OmegaConf.load(ROOT/"config/config.yaml"),
                              OmegaConf.load(ROOT/"config/experiments/gp6.yaml"),
                              OmegaConf.load(ROOT/"tasks/alphaevolve/config.yaml"))
    explicit = OmegaConf.from_dotlist([arg for arg in command if "=" in arg and not arg.startswith("--")])
    allowed = {"agent.code.model", "agent.code.base_url", "agent.feedback.model", "agent.feedback.base_url"}
    if smoke:
        expected.agent.initial_drafts = 1
        expected.agent.steps = 2
        expected.agent.search.parallel_search_num = 1
        expected.exec.timeout = 120
    for key in allowed:
        override = OmegaConf.select(explicit, key)
        if override is not None:
            OmegaConf.update(expected, key, override)
    actual = OmegaConf.load(config_path)
    expected_flat = flatten(OmegaConf.to_container(expected, resolve=True))
    actual_flat = flatten(OmegaConf.to_container(actual, resolve=True))
    domains = ("agent.", "parent_selector.", "bo_parent_selector.", "coldstart.", "exec.", "evaluation.")
    checked = {key: value for key, value in expected_flat.items() if key.startswith(domains)}
    differences = [{"key": key, "expected": value, "actual": actual_flat.get(key)}
                   for key, value in checked.items() if actual_flat.get(key) != value]
    # Never expose credentials in reports, even if a user supplied a config key.
    for item in differences:
        if any(word in item["key"].lower() for word in ("api_key", "token", "password")):
            item["actual"] = item["expected"] = "[redacted]"
    return {"passed": not differences, "checked_fields": len(checked), "differences": differences,
            "agent_seconds": actual.agent.time_limit, "steps": actual.agent.steps,
            "initial_drafts": actual.agent.initial_drafts, "workers": actual.agent.search.parallel_search_num,
            "cpu_cores": int(actual.cpu_number), "memory_device": actual.agent.memory_embedding_device,
            "search_preset": "gp6", "model": actual.agent.code.model, "provider": actual.agent.code.base_url,
            "domain_adaptations": ["mathematical verifier/API", "CPU execution and memory", "disable ML pretrained coldstart"],
            "test_overrides": ["1 draft", "1 improve", "1 worker", "120s candidate cap"] if smoke else []}


def audit_smoke_trace(log_dir, workspace_dir, nodes):
    """Check actual GP6 decisions and CPU allocations, in addition to config."""
    try:
        decisions = [json.loads(line) for line in
                     (Path(log_dir)/"phase1/selection_decisions.jsonl").read_text().splitlines() if line.strip()]
        if len(decisions) != 2 or len(nodes) != 2:
            raise ValueError("Expected exactly two native decisions and completed nodes")
        draft, improve = sorted(decisions, key=lambda x: x["decision_index"])
        if draft["selected_operator"] != "draft" or draft["decision_kind"] != "initialization":
            raise ValueError("The first decision did not initialize a draft")
        if (improve["selected_operator"] != "improve" or improve["decision_kind"] != "model"
                or improve["selected_parent_id"] != nodes[0]["id"]
                or improve["selector_name"] != "gp_child_ucb"
                or not str(improve["model_version"]).startswith("gp6-")
                or improve["fallback_used"]):
            raise ValueError("The improvement was not selected by the native GP6 model")
        snapshots = [json.loads(line) for line in
                     (Path(log_dir)/"phase1/selection_snapshots.jsonl").read_text().splitlines() if line.strip()]
        selected = [x for x in snapshots if x["decision_id"] == improve["decision_id"] and x["selected"]]
        if (len(selected) != 1 or selected[0]["acquisition_beta"] != 1.5
                or selected[0]["acquisition_beta_schedule"] != "fixed"
                or selected[0]["include_root_draft"] is not False):
            raise ValueError("The observed acquisition did not match GP6")
        resources = [json.loads((Path(workspace_dir)/"trials"/node["id"]/"resources.json").read_text())
                     for node in nodes]
        cpus = int(OmegaConf.load(Path(log_dir)/"config.yaml").cpu_number)
        if any(x["cuda_visible_devices"] != "" or len(x["cpus"]) != cpus or x["workers"] != 1 for x in resources):
            raise ValueError("Candidate resource records do not match the CPU smoke allocation")
        return {"passed": True, "decisions": decisions, "resources": resources}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "error": str(exc)}
