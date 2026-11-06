"""Per-attempt data binding and immutable full-data reruns for MF-GP."""

from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import time
from engine.fixed_evaluation import FixedEvaluation, EvaluationError, _write_json
from engine.mf_gp.controller import code_hash


def insert_setup(code, setup):
    """Keep valid future imports/docstrings ahead of host execution setup."""
    import ast

    try:
        module = ast.parse(code)
    except SyntaxError:
        return setup + code  # Let the candidate subprocess report its syntax error.
    after = 0
    for index, statement in enumerate(module.body):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ) or (
            isinstance(statement, ast.ImportFrom) and statement.module == "__future__"
        ):
            after = statement.end_lineno
        else:
            break
    lines = code.splitlines(keepends=True)
    return "".join(lines[:after]).rstrip("\n") + "\n" + setup + "".join(lines[after:])


class MFEvaluation(FixedEvaluation):
    allowed_modes = ("mf_gp",)

    def _install_public_view(self):
        # The public preview is stable; running workers NEVER use this symlink.
        self.multifidelity_phase = "half"
        super()._install_public_view()
        self.multifidelity_phase = "adaptive"
        self.journal.evaluation_state["multifidelity_phase"] = "adaptive"
        self.journal.evaluation_state.pop("multifidelity_switch_fraction", None)

    def switch_to_full(self):
        raise EvaluationError("MF-GP uses per-attempt fidelity, not a global switch")

    def executable(self, code):
        seed = int(self.cfg.agent.seed)
        return insert_setup(
            code,
            f"import random as _mf_random, numpy as _mf_numpy, torch as _mf_torch\n"
            f"_mf_random.seed({seed}); _mf_numpy.random.seed({seed}); _mf_torch.manual_seed({seed})\n",
        )

    def candidate_contract(self):
        from engine.mf_gp.v3.probe import CONTRACT
        probe = CONTRACT if getattr(self.cfg.evaluation, "mf_gp_policy_version", 1) == 3 else ""
        return super().candidate_contract() + probe + (
            f"\nThe host may execute this SAME saved program with a fixed {self.low_fraction:.0%} or 100% T view. "
            "Design the pipeline for the full task; use only the supplied ./input view. "
            "Keep training rules and seeds in the executable. Fidelity-tagged score "
            "differences alone do not demonstrate a code improvement.\n"
        )

    def begin_attempt(self, node, remaining_seconds):
        fraction = getattr(node, "training_fraction", None)
        if fraction not in (self.low_fraction, 1.0):
            raise EvaluationError("MF execution has no reserved fidelity")
        context = super().begin_attempt(node, remaining_seconds)
        node.training_fraction = fraction  # base's global-phase value is inactive
        work = Path(context["output_dir"]) / "work"
        work.mkdir()
        (work / "input").symlink_to(
            self.half_view if fraction == self.low_fraction else self.directory / "public",
            target_is_directory=True,
        )
        for name in ("working", "submission"):
            (work / name).mkdir()
        context["working_dir"] = str(work)
        context["training_fraction"] = fraction
        context["attempt_id"] = node.evaluation_attempt_id
        mf = (getattr(node, "phase1_selection_features", None) or {}).get("mf", {})
        context["estimated_action_seconds"] = mf.get("estimated_action_seconds")
        context["cost_estimate_scope"] = mf.get("cost_estimate_scope")
        context["protocol_id"] = self.protocol_id
        context["mf_preserve_future_imports"] = True
        return context

    def score(self, node):
        work = Path(node.evaluation_output_dir) / "work"
        source = work / "submission" / f"submission_{node.id}.csv"
        if source.is_file():
            destination = Path(self.cfg.workspace_dir) / "submission" / source.name
            shutil.copyfile(source, destination)
        # Host scorer snapshots the submission before the next attempt may run.
        started = time.monotonic()
        result = super().score(node)
        _write_json(Path(node.evaluation_output_dir) / "host_cost.json", dict(
            program_id=node.id, attempt_id=node.evaluation_attempt_id,
            fidelity=node.training_fraction, protocol_id=self.protocol_id,
            host_score_seconds=time.monotonic() - started,
            valid=bool(result.get("valid")),
        ))
        return result


def promote(agent, ticket, exec_callback):
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    with agent.journal_lock:
        reservation = copy.deepcopy(selector.pending_promotions[ticket])
        program_id = reservation["program_id"]
        original = next(n for n in agent.journal.nodes if n.id == program_id)
        if code_hash(original.code) != selector.programs[program_id]["code_sha256"]:
            raise EvaluationError("Promotion code differs from saved executable")
        node = copy.copy(original)
        node.training_fraction = 1.0
        remaining = float(agent.acfg.time_limit) - (time.time() - agent.start_time)
        if remaining <= 0:
            raise EvaluationError("Cannot start promotion after the deadline")
        context = agent.fixed_evaluation.begin_attempt(node, remaining)
        context["estimated_action_seconds"] = reservation["cost"]
        context["cost_estimate_scope"] = "execution_only"
        selector.pending_promotions[ticket]["attempt_id"] = node.evaluation_attempt_id
        runtime._persist_root_state(agent)
    started = time.perf_counter()
    # No code generation, review, mutation of original node, or child counters.
    result = exec_callback(
        agent.fixed_evaluation.executable(node.code),
        node.id,
        True,
        working_dir=context["working_dir"],
        evaluation_context=context,
    )
    if result.exc_type is None:
        grade = agent.fixed_evaluation.score(node)
    else:
        grade = dict(
            valid=False,
            metric=None,
            error=f"Execution failed: {result.exc_type}",
            protocol_id=agent.fixed_evaluation.protocol_id,
            node_id=program_id,
            attempt_id=node.evaluation_attempt_id,
            training_fraction=1.0,
        )
    elapsed = time.perf_counter() - started
    eligible = time.time() - agent.start_time <= float(agent.acfg.time_limit)
    record = dict(
        **reservation["row"],
        valid=bool(grade["valid"]) and eligible,
        utility=(
            (float(grade["metric"]) * (1 if agent.metric_maximize else -1))
            if grade["valid"] and eligible
            else None
        ),
        budget_eligible=eligible,
        result=grade,
        generation_seconds=0.0,
        execution_seconds=float(result.exec_time) if selector.policy_version in (2, 3) else elapsed,
        **(dict(host_seconds=max(0., elapsed-float(result.exec_time))) if selector.policy_version == 3 else {}),
        promotion_total_seconds=elapsed,
        completed_runtime_observation=result.exc_type is None,
        runtime_lower_bound=result.exc_type == "TimeoutError",
        transition_id=ticket,
        kind="promotion",
    )
    with agent.journal_lock:
        selector.observations.append(record)
        del selector.pending_promotions[ticket]
        selector._fit()
        from engine.mf_gp.evidence import update_best
        from engine.solution_manager import update_top_candidates

        update_best(agent)
        update_top_candidates(agent, original)
        runtime._persist_root_state(agent)
    return original


def finalize(cfg, journal):
    """Select only completed valid H artifacts, including separate promotions."""
    state = journal.parent_selector_state["study7_state"]
    selector_state = state
    if state["contract"]["mode"] != "mf_gp":
        raise EvaluationError("MF finalizer received another mode")
    choices = [
        o
        for o in selector_state["observations"]
        if o["valid"] and o["fidelity"] == 1.0 and o.get("budget_eligible", True)
    ]
    if not choices:
        raise EvaluationError("No completed valid full-data MF result")
    best = max(choices, key=lambda o: o["utility"])
    result = best["result"]
    if result.get("protocol_id") != journal.evaluation_state[
        "protocol_id"
    ] or not result.get("valid"):
        raise EvaluationError("MF winner protocol mismatch")
    source = Path(result["submission_file"])
    if not source.is_file():
        raise EvaluationError("MF winner submission is missing")
    for name in (
        "best_submission/submission.csv",
        "ensembles_csv/top1ens-total_run_time0.0h.csv",
    ):
        target = Path(cfg.workspace_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        shutil.copyfile(source, temp)
        temp.replace(target)
    _write_json(
        Path(cfg.log_dir) / "final_submission.json",
        dict(
            node_id=best["program_id"],
            evaluation_id=best["transition_id"],
            protocol_id=result["protocol_id"],
            metric=result["metric"],
            source=str(source),
            submission=str(Path(cfg.workspace_dir) / "best_submission/submission.csv"),
            selection="best_completed_full_mf_gp",
            training_fraction=1.0,
        ),
    )
    return best
