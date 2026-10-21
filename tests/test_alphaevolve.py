"""Math validity, independent scoring, native preset and regression contracts."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tasks.alphaevolve.graders import score, sum_difference_counts
from tasks.alphaevolve.prepare import initial
from tasks.alphaevolve.registry import TASKS


def baseline(tid):
    return initial(TASKS[tid], {TASKS[tid]["output_key"]: [1.]*20})


@pytest.mark.parametrize("tid", list(TASKS))
def test_simple_feasible_constructions(tid):
    result = score(tid, baseline(tid))
    assert result["valid"] and math.isfinite(result["metric"])
    assert result["maximize"] == TASKS[tid]["maximize"]


def test_no_candidate_metric_or_wrong_instance():
    construction = baseline("circle_packing_square")
    original = score("circle_packing_square", construction)["metric"]
    assert score("circle_packing_square", dict(construction, metric=1e99))["metric"] == original
    construction["circles"] = construction["circles"][:-1]
    with pytest.raises(ValueError, match="shape"):
        score("circle_packing_square", construction)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1", True])
def test_nonfinite_and_nonnumeric_rejected(bad):
    with pytest.raises(ValueError):
        score("first_autocorrelation", {"heights": [bad, bad]})


def test_geometry_checks_and_invariance():
    circles = baseline("circle_packing_square")
    circles["circles"][1] = circles["circles"][0]
    with pytest.raises(ValueError, match="overlap"):
        score("circle_packing_square", circles)
    points = baseline("max_min_ratio")
    assert score("max_min_ratio", points)["metric"] == pytest.approx(18)
    shifted = {"points": (np.array(points["points"])*7+19).tolist()}
    assert score("max_min_ratio", shifted)["metric"] == pytest.approx(18)
    hexagons = baseline("hexagon_packing")
    hexagons["hexagons"][1] = hexagons["hexagons"][0]
    with pytest.raises(ValueError, match="overlap"):
        score("hexagon_packing", hexagons)


def test_kissing_dimension_and_pairwise_constraints():
    points = baseline("kissing_number_d11")
    assert score("kissing_number_d11", points)["metric"] == 22
    points["points"].append(points["points"][0])
    with pytest.raises(ValueError, match="overlap"):
        score("kissing_number_d11", points)
    with pytest.raises(ValueError, match="dimension"):
        score("kissing_number_d11", {"points": np.eye(10).tolist()})


def test_autoconvolution_formulas_not_discrete_proxy():
    assert score("first_autocorrelation", {"heights": [1, 1]})["metric"] == 2
    assert score("second_autocorrelation", {"heights": [1, 1]})["metric"] == pytest.approx(2/3)
    signed = {"heights": [-3, -1, 2, 3]}
    assert score("third_autocorrelation", signed)["metric"] == pytest.approx(96)
    assert score("third_autocorrelation_variant", signed)["metric"] == pytest.approx(176)
    with pytest.raises(ValueError):
        score("first_autocorrelation", signed)
    with pytest.raises(ValueError):
        score("minimum_overlap", {"heights": [.4, .4]})


def test_sum_difference_fft_matches_exact_enumeration():
    rng = np.random.default_rng(99)
    for _ in range(12):
        u = np.unique(np.r_[0, rng.integers(1, 250, size=35)])
        plus, minus = len({int(a+b) for a in u for b in u}), len({int(a-b) for a in u for b in u})
        assert sum_difference_counts(u) == (plus, minus)
        expected = 1+math.log(minus/plus)/math.log(2*int(u[-1])+1)
        for tid in ("sums_differences_1", "sums_differences_2"):
            assert score(tid, {"integers": u.tolist()})["metric"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        score("sums_differences_1", {"integers": [0, 1.5, 3]})
    with pytest.raises(ValueError):
        score("sums_differences_1", {"integers": [0, 1, 1]})


def test_fresh_grader_ignores_candidate_forged_score(tmp_path):
    # Run actual worker code, with a candidate attempting to write a bogus score.
    tid = "max_min_ratio"
    (tmp_path/"problem.json").write_text(json.dumps({"id": tid}))
    (tmp_path/"initial.json").write_text(json.dumps(baseline(tid)))
    (tmp_path/"candidate.py").write_text('''def solve(problem, initial, time_budget, seed):
    import json
    open("result.json", "w").write(json.dumps({"valid": True, "metric": -999}))
    return initial
''')
    worker = ROOT/"tasks/alphaevolve/worker.py"
    subprocess.run([sys.executable, str(worker), "generate", str(tmp_path), "--seconds", "5"], cwd=tmp_path, check=True, timeout=30)
    subprocess.run([sys.executable, str(worker), "grade", tid, str(tmp_path/"construction.json"), str(tmp_path/"result.json")], check=True, timeout=30)
    assert json.loads((tmp_path/"result.json").read_text())["metric"] == pytest.approx(18)


def test_task_defaults_preserve_gp6_and_native_agent(monkeypatch):
    from config import _load_cfg
    from omegaconf import OmegaConf
    monkeypatch.setattr(sys, "argv", ["run.py", "task_preset=alphaevolve", "exp_id=max_min_ratio"])
    cfg = _load_cfg()
    base = OmegaConf.load(ROOT/"config/config.yaml")
    assert cfg.parent_selector.type == "gp_child_ucb"
    assert cfg.bo_parent_selector.model == "gp_matern32"
    assert cfg.bo_parent_selector.acquisition.beta == 1.5
    assert cfg.get("audited_bo", "baseline") == "baseline"
    assert cfg.agent.time_limit == 43200
    assert cfg.agent.memory_embedding_device == "cpu" and cfg.agent.search.num_gpus == 0
    assert cfg.agent.initial_drafts == base.agent.initial_drafts == 3
    assert cfg.agent.search.parallel_search_num == base.agent.search.parallel_search_num == 3
    for name in ("steps", "use_global_memory", "use_diff_mode", "use_stepwise_generation", "use_evolution", "use_fusion", "use_aggregation"):
        assert cfg.agent[name] == base.agent[name]
    assert cfg.agent.code == base.agent.code and cfg.agent.feedback == base.agent.feedback
    assert cfg.coldstart.use_coldstart is False
    monkeypatch.setattr(sys, "argv", ["run.py"])
    assert _load_cfg().agent == base.agent


def smoke_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(OmegaConf.load(ROOT/"config/config.yaml"),
                          OmegaConf.load(ROOT/"config/experiments/gp6.yaml"),
                          OmegaConf.load(ROOT/"tasks/alphaevolve/config.yaml"))
    cfg.agent.steps = 2
    cfg.agent.initial_drafts = 1
    cfg.agent.search.parallel_search_num = 1
    cfg.exec.timeout = 120
    return cfg


@pytest.mark.parametrize("key,value", [("agent.time_limit", 300),
                                       ("agent.use_global_memory", False),
                                       ("bo_parent_selector.acquisition.beta", 2.0)])
def test_settings_audit_rejects_silent_runtime_changes(tmp_path, key, value):
    from omegaconf import OmegaConf
    from tasks.alphaevolve.audit import audit_settings
    cfg = smoke_config()
    path = tmp_path/"config.yaml"
    OmegaConf.save(cfg, path)
    assert audit_settings(path, [], smoke=True)["passed"]
    OmegaConf.update(cfg, key, value)
    OmegaConf.save(cfg, path)
    audit = audit_settings(path, [], smoke=True)
    assert not audit["passed"]
    assert [item["key"] for item in audit["differences"]] == [key]


@pytest.mark.parametrize("change", ["parent", "debug", "invalid", "same_code", "extra_node"])
def test_smoke_requires_generated_draft_and_distinct_valid_improvement(tmp_path, change):
    from omegaconf import OmegaConf
    from tasks.alphaevolve.run import verify_smoke
    nodes = [{"id": name, "stage": stage, "step": step, "code": name,
              "is_valid": True, "is_buggy": False, "evaluation_result": {"valid": True},
              "metric": {"value": 1., "maximize": True}}
             for step, (name, stage) in enumerate((("first", "draft"), ("second", "improve")))]
    journal = {"nodes": nodes, "node2parent": {"second": "first"}}
    path = tmp_path/"journal.json"
    OmegaConf.save(smoke_config(), tmp_path/"config.yaml")
    path.write_text(json.dumps(journal))
    assert len(verify_smoke(tmp_path)) == 2
    if change == "parent":
        journal["node2parent"]["second"] = "another"
    elif change == "debug":
        nodes[1]["stage"] = "debug"
    elif change == "invalid":
        nodes[1]["evaluation_result"]["valid"] = False
    elif change == "same_code":
        nodes[1]["code"] = nodes[0]["code"]
    else:
        nodes.append(copy.deepcopy(nodes[1]))
    path.write_text(json.dumps(journal))
    with pytest.raises(ValueError):
        verify_smoke(tmp_path)
