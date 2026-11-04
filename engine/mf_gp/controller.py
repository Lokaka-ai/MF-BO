"""One controller for expansion at either fidelity and exact-code promotion."""

from __future__ import annotations
import copy
from dataclasses import asdict, replace
import hashlib
import math
import random
import time
import uuid
import numpy as np
from engine.parent_selectors.base import (
    SelectionDecision,
    SelectorResult,
    CandidateScore,
    utc_timestamp,
)
from engine.parent_selectors.runtime import (
    Phase1SelectorRuntime,
    SearchExhausted,
    SelectionPending,
)
from engine.mf_gp.model import MultiFidelityGP, CONTRACT, acquisition, digest


class PromotionRequested(Exception):
    def __init__(self, ticket):
        self.ticket = ticket


def code_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()


class MFSelector:
    name = "acq7_gp4"  # native ledger dispatcher; contract/schema identify MF mode
    greedy = False
    full_training_only = False

    def __init__(self, config, records_provider, selection_logger, low_fraction=0.5,
                 policy_version=1, cost_prior=None):
        from engine.mf_gp.policy import validate_policy
        self.policy_version, self.cost_prior = validate_policy(policy_version, cost_prior, low_fraction)
        self.config, self.records_provider, self.logger = (
            config,
            records_provider,
            selection_logger,
        )
        self.gp = MultiFidelityGP(low_fraction)
        self.low_fraction = self.gp.low_fraction
        self.observations = []
        self.programs = {}
        self.seen = set()
        self.pending_promotions = {}
        self.calibration = None
        self.terminal = False
        self.last_prediction_details = {}
        self.decision_index = 0

    def contract(self):
        contract = dict(
            mode="mf_gp",
            schema="mf_gp_v1",
            model=self.gp.contract,
            native=asdict(self.config),
            initialization="best_plus_seeded_random",
        )
        if self.policy_version == 2:
            contract.update(policy_version=2, cost_prior=self.cost_prior,
                            initialization="best_then_seeded_random_async_two_slots",
                            ceiling="exact_known_full_metric_bound")
        return contract

    def observe(self, records):
        for r in records:
            if r.transition_id in self.seen:
                continue
            metadata = r.metadata or {}
            node = (metadata.get("node_snapshot") or {}).get("node") or {}
            if r.child_node_id and node:
                mf = r.selection_features.get("mf", {})
                x = mf.get("x", [0.0] * 8)
                operator = mf.get("operator", r.operator)
                fidelity = node.get("training_fraction")
                if fidelity not in (self.low_fraction, 1.0):
                    # A failed generation has no measurement; retain its cost.
                    fidelity = mf.get("fidelity", self.low_fraction)
                program = dict(
                    program_id=r.child_node_id,
                    x=x,
                    operator=operator,
                    parent_id=r.parent_node_id,
                    code_sha256=code_hash(node.get("code", "")),
                    initial=bool(mf.get("initial", False))
                    or (
                        r.operator == "debug"
                        and self.programs.get(r.parent_node_id, {}).get(
                            "initial", False
                        )
                    ),
                )
                old = self.programs.setdefault(r.child_node_id, program)
                if old != program:
                    raise ValueError("Immutable MF program/context changed")
                valid = bool(r.valid and r.child_utility is not None)
                self.observations.append(
                    dict(
                        **{k: program[k] for k in ("program_id", "x", "operator")},
                        fidelity=fidelity,
                        valid=valid,
                        utility=float(r.child_utility) if valid else None,
                        budget_eligible=metadata.get("budget_eligible", True),
                        result=node.get("evaluation_result"),
                        transition_id=r.transition_id,
                        generation_seconds=r.generation_wall_time_sec,
                        execution_seconds=r.execution_wall_time_sec,
                        completed_runtime_observation=(node.get("exec_time") is not None and node.get("exc_type") is None),
                        runtime_lower_bound=node.get("exc_type") == "TimeoutError",
                        kind="expansion",
                    )
                )
            else:
                self.observations.append(
                    dict(
                        program_id=None,
                        valid=False,
                        utility=None,
                        fidelity=r.selection_features.get("mf", {}).get(
                            "fidelity", self.low_fraction
                        ),
                        generation_seconds=r.generation_wall_time_sec,
                        execution_seconds=0.0,
                        transition_id=r.transition_id,
                        kind="generation_failure",
                    )
                )
            self.seen.add(r.transition_id)
        self._fit()

    def _fit(self, freeze=False):
        initial = [
            o["utility"]
            for o in self.observations
            if o["valid"]
            and (self.policy_version == 1 or o.get("budget_eligible", True))
            and o["fidelity"] == self.low_fraction
            and self.programs[o["program_id"]]["initial"]
        ]
        changed = self.gp.fit(self.observations, initial if freeze else None)
        if changed and self.logger:
            self.logger._append(
                "mf_gp_fits.jsonl",
                dict(
                    model_version=self.gp.model_version,
                    diagnostics=self.gp.fit_diagnostics,
                    observations=[o.get("transition_id") for o in self.observations],
                ),
            )

    def features(self, agent, parent, operator):
        self.observe(self.records_provider())
        self._fit(freeze=True)
        valid = [
            o
            for o in self.observations
            if o["valid"] and o.get("budget_eligible", True)
        ]
        own = [o for o in valid if o["program_id"] == parent.id]
        chosen = max(own, key=lambda o: o["fidelity"]) if own else None
        fidelity = (
            chosen["fidelity"] if chosen else getattr(parent, "training_fraction", self.low_fraction)
        )
        utility = chosen["utility"] if chosen else 0.0
        parent_id = parent.parent.id if parent.parent else None
        ancestor = next(
            (
                o
                for o in valid
                if o["program_id"] == parent_id and o["fidelity"] == fidelity
            ),
            None,
        )
        birth = utility - ancestor["utility"] if ancestor else 0.0
        local = [
            o
            for o in valid
            if o["fidelity"] == fidelity
            and self.programs[o["program_id"]]["parent_id"] == parent.id
            and o["operator"] == operator
            and o["kind"] == "expansion"
        ]
        gains = [o["utility"] - utility for o in local]
        count = sum(
            p["parent_id"] == parent.id and p["operator"] == operator
            for p in self.programs.values()
        )
        rating = getattr(parent, "agent_potential_level", None)
        scale = self.gp.scale
        x = [
            math.tanh((utility - self.gp.center) / scale),
            math.tanh(birth / scale),
            math.tanh(sum(gains) / ((len(gains) + 1) * scale)),
            math.log1p(count) / (1 + math.log1p(count)),
            float(fidelity == 1.0),
            float(ancestor is not None),
            (float(rating) - 1.0) if rating in (0, 1, 2) else 0.0,
            float(rating in (0, 1, 2)),
        ]
        return dict(
            x=x,
            operator=operator,
            parent_id=parent.id,
            parent_utility=utility,
            parent_fidelity=fidelity,
            birth_commit_index=next(
                i for i, n in enumerate(agent.journal.nodes) if n.id == parent.id
            ),
        )

    def costs(self, program_id=None, operator=None):
        if self.policy_version == 2:
            from engine.mf_gp.policy import estimate_costs
            return estimate_costs(self.observations, self.low_fraction, self.cost_prior, program_id, operator)
        measured = [o for o in self.observations if o["execution_seconds"] > 0]
        low = [o["execution_seconds"] for o in measured if o["fidelity"] == self.low_fraction]
        high = [o["execution_seconds"] for o in measured if o["fidelity"] == 1.0]
        ratios = []
        for p in self.programs:
            l = next(
                (
                    o["execution_seconds"]
                    for o in measured
                    if o["program_id"] == p and o["fidelity"] == self.low_fraction
                ),
                None,
            )
            h = next(
                (
                    o["execution_seconds"]
                    for o in measured
                    if o["program_id"] == p and o["fidelity"] == 1.0
                ),
                None,
            )
            if l and h:
                ratios.append(h / l)
        ratio = float(np.median(ratios)) if ratios else 2.0
        own = next(
            (
                o["execution_seconds"]
                for o in measured
                if o["program_id"] == program_id and o["fidelity"] == self.low_fraction
            ),
            None,
        )
        cheap = own or (float(np.median(low)) if low else 60.0)
        full = (
            own * ratio if own else (float(np.median(high)) if high else cheap * ratio)
        )
        gen = [
            o["generation_seconds"]
            for o in self.observations
            if o["generation_seconds"] > 0
            and (operator is None or o.get("operator") == operator)
        ]
        if not gen:
            gen = [
                o["generation_seconds"]
                for o in self.observations
                if o["generation_seconds"] > 0
            ]
        generation = float(np.median(gen)) if gen else 60.0
        # Runtime guard uses upper observed costs, including failed executions.
        guard_full = (
            max(
                full,
                float(np.quantile(high, 0.9)) if high else full,
                own * max(ratios) if own and ratios else 0.0,
            )
            * CONTRACT["cost_margin"]
        )
        return max(1.0, cheap), max(1.0, full), generation, max(1.0, guard_full)

    def pending_rows(self, agent):
        rows = []
        for pending in agent.phase1_runtime.root_draft_state["pending"].values():
            mf = pending["option"]["state_features"].get("mf")
            if mf:
                rows.append(
                    {k: mf[k] for k in ("program_id", "x", "operator", "fidelity")}
                )
        rows += [p["row"] for p in self.pending_promotions.values()]
        return rows

    def remaining_budget(self, agent):
        remaining = float(agent.acfg.time_limit) - (time.time() - agent.start_time)
        cheap, _, generation, full_guard = self.costs()
        if 1.5 * (cheap + generation) + full_guard > remaining:
            self.terminal = True
        return remaining

    def select(self, agent, options):
        self.observe(self.records_provider())
        self._fit(freeze=True)
        remaining = self.remaining_budget(agent)
        pending = self.pending_rows(agent)
        completed_high = {
            o["program_id"] for o in self.observations if o["fidelity"] == 1.0
        }
        pending_high = {
            p["row"]["program_id"] for p in self.pending_promotions.values()
        }
        candidates = [
            p
            for p in self.programs
            if p not in completed_high
            and p not in pending_high
            and any(
                o["program_id"] == p
                and o["valid"]
                and o.get("budget_eligible", True)
                and o["fidelity"] == self.low_fraction
                for o in self.observations
            )
        ]
        valid_high = [
            o["utility"]
            for o in self.observations
            if o["valid"] and o["fidelity"] == 1.0 and o.get("budget_eligible", True)
        ]
        incumbent = max(valid_high, default=None)
        if self.policy_version == 2:
            from engine.mf_gp.policy import fill_calibration
            previous = list(self.calibration or [])
            self.calibration = fill_calibration(previous, candidates, self.programs,
                                                self.observations, int(agent.acfg.seed))
            if self.calibration != previous and self.logger:
                self.logger._append("mf_gp_policy.jsonl", dict(event="calibration_slots",
                    programs=self.calibration, remaining_seconds=remaining))
        else:
            if self.calibration is None:
                initial = [p for p in candidates if self.programs[p]["initial"]]
                if initial:
                    low_score = {
                        o["program_id"]: o["utility"]
                        for o in self.observations
                        if o["valid"] and o["fidelity"] == self.low_fraction
                    }
                    best = max(initial, key=lambda p: (low_score[p], p))
                    others = sorted(p for p in initial if p != best)
                    self.calibration = [best] + (
                        [random.Random(int(agent.acfg.seed)).choice(others)]
                        if others
                        else []
                    )
                else:
                    self.calibration = []
        actions, targets = [], {}
        for p in sorted(candidates):
            saved = self.programs[p]
            row = {k: saved[k] for k in ("program_id", "x", "operator")}
            row["fidelity"] = 1.0
            _, full, _, guard = self.costs(p)
            if guard <= remaining:
                actions.append(
                    dict(kind="promote", row=row, cost=full, guard=guard, program_id=p)
                )
                targets[p] = row
        # Freeze one prospective identity per legal parent/operator, shared by
        # its hypothetical L/H alternatives, distinct from every past child.
        for index, option in enumerate(options):
            context = option.state_features["study7"]
            row = dict(
                program_id=f'fresh:{agent.phase1_runtime.root_draft_state["revision"]}:{index}',
                x=context["x"],
                operator=option.operator,
                fidelity=1.0,
            )
            cheap, full, generation, guard = self.costs(operator=option.operator)
            if generation * 1.5 + guard <= remaining:
                actions.append(
                    dict(
                        kind="expand",
                        row=row,
                        cost=full + generation,
                        guard=guard + 1.5 * generation,
                        option_index=index,
                    )
                )
                targets[row["program_id"]] = row
            if not self.terminal and 1.5 * (cheap + generation) + guard <= remaining:
                actions.append(
                    dict(
                        kind="expand",
                        row={**row, "fidelity": self.low_fraction},
                        cost=cheap + generation,
                        guard=1.5 * (cheap + generation) + guard,
                        option_index=index,
                    )
                )
                # Both actions probe the same prospective program, but the
                # controller's objective is its full-data quality. The cheap
                # action must not replace the full-data target in KG.
        selected = None
        for p in self.calibration:
            if p in completed_high or p in pending_high:
                continue
            selected = next(
                (a for a in actions if a["kind"] == "promote" and a["program_id"] == p),
                None,
            )
            if selected:
                break
        # Finish initial pairs before ordinary acquisitions; pending pair work
        # is real occupied work, not an invitation to spawn more calibration.
        if selected is None and pending_high.intersection(self.calibration or []):
            raise SelectionPending(agent.phase1_runtime.root_draft_state["revision"])
        if not actions:
            if pending:
                raise SelectionPending(
                    agent.phase1_runtime.root_draft_state["revision"]
                )
            raise SearchExhausted("No MF action fits the remaining execution budget")
        scores, details = acquisition(
            self.gp, list(targets.values()), actions, incumbent, pending, self.terminal
        )
        chosen = (
            actions.index(selected)
            if selected is not None
            else min(range(len(actions)), key=lambda i: (-scores[i], i))
        )
        action = actions[chosen]
        ticket = uuid.uuid4().hex
        self.logger._append(
            "mf_gp_decisions.jsonl",
            dict(
                decision_id=ticket,
                remaining_seconds=remaining,
                model_version=self.gp.model_version,
                incumbent=incumbent,
                actions=actions,
                scores=scores,
                chosen=chosen,
                calibration=selected is not None,
                **(dict(policy_version=2, cost_prior=self.cost_prior,
                        completed_runtime_count=sum(o.get("completed_runtime_observation") is True for o in self.observations),
                        censored_runtime_count=sum(bool(o.get("runtime_lower_bound")) for o in self.observations))
                   if self.policy_version == 2 else {}),
                **details,
            ),
        )
        if action["kind"] == "promote":
            self.pending_promotions[ticket] = dict(
                action,
                decision_id=ticket,
                started_active=time.time() - agent.start_time,
            )
            agent.phase1_runtime._persist_root_state(agent)
            raise PromotionRequested(ticket)
        option = options[action["option_index"]]
        mf = copy.deepcopy(action["row"])
        mf["estimated_action_seconds"] = action["cost"]
        mf["cost_estimate_scope"] = "generation_plus_execution"
        mf["program_id"] = "fresh:" + ticket
        option = replace(option, state_features={**option.state_features, "mf": mf})
        decision = SelectionDecision(
            self.name,
            option.parent_node_id,
            option.operator,
            scores[chosen],
            self.gp.model_version,
            False,
            ticket,
            utc_timestamp(),
        )
        score = CandidateScore(
            option.parent_node_id,
            option.operator,
            option.reference_node_ids,
            option.parent_utility,
            None,
            None,
            None,
            action["cost"],
            None,
            scores[chosen],
        )
        return SelectorResult(decision, option, (score,), tuple(options))

    def validate_reservation(self, agent, result):
        return True  # native journal/selection locks cover selection + reservation

    def reserve(self, decision, option, commit_index):
        if "mf" not in option.state_features:
            # Forced initial drafts/debug remain native operations. Debug is a
            # NEW executable and inherits the failed attempt's fidelity.
            parent = option.parent_node
            fidelity = (
                self.low_fraction
                if option.operator == "draft"
                else (getattr(parent, "training_fraction", None) or self.low_fraction)
            )
            if self.terminal:
                fidelity = 1.0
            context = option.state_features.get("study7", {})
            option.state_features["mf"] = dict(
                program_id="fresh:" + decision.decision_id,
                x=context.get("x", [0.0] * 8),
                operator=option.operator,
                fidelity=fidelity,
            )
        option.state_features["mf"]["initial"] = (
            decision.selection_reason == "initial_draft_bypass"
        )
        return {
            "root_lineage_id": str(option.parent_node.branch_id),
            "program_identity_pending": True,
        }

    def publish_committed(self, decision_id=None):
        pass

    def state_dict(self):
        return dict(
            contract=self.contract(),
            observations=self.observations,
            programs=self.programs,
            seen=sorted(self.seen),
            pending_promotions=self.pending_promotions,
            calibration=self.calibration,
            terminal=self.terminal,
            gp=self.gp.state_dict(),
        )

    def load_state_dict(self, state):
        if state["contract"] != self.contract():
            raise ValueError("MF controller contract changed on resume")
        for key in (
            "observations",
            "programs",
            "pending_promotions",
            "calibration",
            "terminal",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
        self.seen = set(state["seen"])
        self.gp.load_state_dict(state["gp"])
        self._fit()


class MFRuntime(Phase1SelectorRuntime):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.acq7_enabled or self.study7_settings.method_id != "AQ05":
            raise ValueError("MF-GP requires the AQ05 native expansion workflow")
        from engine.mf_gp.settings import low_fraction
        from engine.mf_gp.policy import load_cost_prior
        self.study7_selector = MFSelector(
            self.study7_settings, self.transition_store.records, self.selection_logger,
            low_fraction=low_fraction(cfg),
            policy_version=getattr(cfg.evaluation, "mf_gp_policy_version", 1),
            cost_prior=load_cost_prior(
                getattr(cfg.evaluation, "mf_gp_cost_profile", None),
                getattr(cfg.evaluation, "protocol_dir", None), low_fraction(cfg)),
        )
        self.surrogate = self.study7_selector.gp
        self.selectors[self.selector_type] = self.study7_selector
        self.feature_schema_version = "mf_gp_v1"

    def _forced_result(self, agent, parent_node, **kwargs):
        # Native repair/aggregation bypasses still obey MF fidelity and cost
        # guards. An unaffordable bypass must not launch cheap work at deadline.
        if kwargs.get("reason") != "initial_draft_bypass":
            selector = self.study7_selector
            remaining = selector.remaining_budget(agent)
            cheap, _, generation, full_guard = selector.costs()
            fidelity = (
                1.0
                if selector.terminal
                else (getattr(parent_node, "training_fraction", None) or selector.low_fraction)
            )
            guard = 1.5 * generation + full_guard
            if fidelity == selector.low_fraction:
                guard += 1.5 * cheap
            if guard > remaining:
                return selector.select(agent, self._normal_options(agent))
        return super()._forced_result(agent, parent_node, **kwargs)

    def _rank_options(self, agent, options):
        # Promotions are controller actions, not native selector failures.
        return self.study7_selector.select(agent, options)

    def bind_agent(self, agent, *, resuming=False):
        if resuming:
            import json

            path = self.artifact_root / "root_draft_state.json"
            saved = (
                json.loads(path.read_text())
                if path.exists()
                else agent.journal.parent_selector_state
            )
            if saved.get("study7_state", {}).get(
                "pending_promotions"
            ) and self._owner_alive(saved):
                raise RuntimeError(
                    "Cannot resume promotions before the old allocation is released"
                )
        return super().bind_agent(agent, resuming=resuming)

    def _select_global_with_root(self, agent):
        state = self.root_draft_state
        selector = self.study7_selector
        selector.observe(self.transition_store.records())
        if selector.policy_version == 2:
            from engine.mf_gp.policy import exact_ceiling_reached
            metric = agent.fixed_evaluation.spec["metric"]
            if exact_ceiling_reached(selector.observations, metric):
                selector.logger._append("mf_gp_policy.jsonl", dict(event="exact_metric_ceiling", metric=metric))
                raise SearchExhausted("Valid full-data incumbent reached exact metric ceiling")
        if state["initial_terminal"] < state["initial_goal"]:
            return super()._select_global_with_root(agent)
        if not self._budget_available(agent):
            raise SearchExhausted("MF search budget exhausted")
        # The native worker would debug its failed continuation next. The MF
        # runner returns to global selection, so expose that same forced repair
        # before normal acquisition; never reserve two repairs of one program.
        from engine.operator_preview import preview_next_operator

        pending_parents = self._pending_parent_ids()
        for node in agent.journal.nodes:
            if (
                node.stage != "root"
                and node.id not in pending_parents
                and not node.is_terminal
                and not node.lock
                and not node.reached_child_limit(agent.scfg)
            ):
                preview = preview_next_operator(agent, node)
                if preview.operator == "debug":
                    return self._forced_result(
                        agent, node, preview=preview, reason="special_path_bypass"
                    )
        # Keep native normal-operation previews, feasibility/debug and
        # aggregation bypasses. Promotions can still run when parents expire.
        options = self._normal_options(agent)
        if (
            options
            and all(
                p
                in {
                    o["program_id"]
                    for o in selector.observations
                    if o["fidelity"] == 1.0
                }
                for p in (selector.calibration or [])
            )
            and selector.calibration is not None
        ):
            return super()._select_global_with_root(agent)
        if not options:
            has_cheap = any(
                o["valid"]
                and o["fidelity"] == selector.low_fraction
                and not any(
                    h["program_id"] == o["program_id"] and h["fidelity"] == 1.0
                    for h in selector.observations
                )
                for o in selector.observations
            )
            if not has_cheap and not selector.pending_promotions:
                return super()._select_global_with_root(agent)
        return selector.select(agent, options)

    def _budget_available(self, agent):
        if time.time() - agent.start_time >= float(agent.acfg.time_limit):
            return False
        if not super()._budget_available(agent):
            return False
        selector = self.study7_selector
        promotions = sum(o["kind"] == "promotion" for o in selector.observations) + len(
            selector.pending_promotions
        )
        return self.root_draft_state["submitted_count"] + len(
            [
                p
                for p in self.root_draft_state["pending"].values()
                if p["status"] == "reserved" and not p.get("recovery_already_submitted")
            ]
        ) + promotions < int(agent.acfg.steps)

    def attach_decision(self, node, context):
        mf = context.option.state_features["mf"]
        node.training_fraction = mf["fidelity"]
        mf["program_id"] = node.id
        self.root_draft_state["pending"][context.decision.decision_id]["option"][
            "state_features"
        ]["mf"] = copy.deepcopy(mf)
        return super().attach_decision(node, context)

    def _record_root_terminal(self, **kwargs):
        row = super()._record_root_terminal(**kwargs)
        node = kwargs.get("child_node")
        if node is not None:
            # MF workers return completed programs to the global legal pool.
            # The staged runner instead passes the locked continuation directly
            # to its owning worker. Pending reservations still prevent reuse.
            node.lock = False
            self._persist_root_state(self._agent)
        return row

    def _incumbent(self, agent):
        valid = [
            o
            for o in self.study7_selector.observations
            if o["valid"] and o.get("budget_eligible", True)
        ]
        full = [o["utility"] for o in valid if o["fidelity"] == 1.0]
        return (
            max(full, default=None)
            if full
            else max((o["utility"] for o in valid), default=None)
        )

    def _global_stagnation(self, agent):
        from engine.redesign_evidence import plateau_state

        return plateau_state(agent)["stale"]
