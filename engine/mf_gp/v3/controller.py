"""V3 native selection adapter: deferred fidelity, common scale and marginal costs."""
import copy
import time
import uuid
import math
import statistics
from dataclasses import asdict, replace
from engine.mf_gp.controller import MFSelector, PromotionRequested
from engine.mf_gp.model import acquisition, digest
from engine.mf_gp.v3.model import V3GaussianProcess
from engine.mf_gp.v3.context import BirthContext
from engine.mf_gp.v3.prepared import PreparedPrograms
from engine.parent_selectors.runtime import SearchExhausted, SelectionPending
from engine.parent_selectors.base import SelectionDecision, SelectorResult, CandidateScore, utc_timestamp


class PreparedRequested(Exception):
    def __init__(self,ticket,fidelity):
        self.ticket,self.fidelity=ticket,fidelity


class V3Selector(MFSelector):
    def __init__(self, config, records_provider, selection_logger, low_fraction=.2, policy_version=3, cost_prior=None):
        if low_fraction != .2 or policy_version != 3:
            raise ValueError("V3 requires 20% and policy 3")
        super().__init__(config,records_provider,selection_logger,low_fraction,2,cost_prior)
        self.policy_version=3
        self.gp=V3GaussianProcess()
        self.raw_contexts={}
        self.prepared=PreparedPrograms()
        self.probes={}

    def contract(self):
        c=super().contract()
        c.update(schema="mf_gp_v3", policy_version=3, cost_prior=self.cost_prior,
                 initialization="one_affordable_anchor_nonblocking", prepared_schema=1,
                 probe_contract=1)
        return c

    def observe(self, records):
        records=[r for r in records if r.transition_id not in self.seen]
        for rec in records:
            if rec.transition_id in self.seen: continue
            mf=rec.selection_features.get("mf",{})
            if rec.child_node_id:
                self.raw_contexts[rec.child_node_id]=copy.deepcopy(mf["raw_context"])
        super().observe(records)
        by_id={o.get("transition_id"):o for o in self.observations}
        for rec in records:
            o=by_id.get(rec.transition_id)
            e=self.prepared.entries.get(rec.decision_id)
            if o and e:
                o["probe_seconds"]=self.probes.get(rec.decision_id, {}).get("elapsed_seconds", 0.)
            if not o or o["kind"] != "expansion":
                if e and e["state"] == "generating":
                    self.prepared.finish(rec.decision_id, success=False, reason="generation_failure")
                continue
            node=((rec.metadata or {}).get("node_snapshot") or {}).get("node") or {}
            process=node.get("exec_time")
            o["execution_seconds"]=float(process or 0.)
            o["host_seconds"]=max(0.,float(rec.execution_wall_time_sec)-o["execution_seconds"])
            o["total_execution_seconds"]=float(rec.execution_wall_time_sec)
            e=self.prepared.entries.get(rec.decision_id)
            if e and e["state"] in ("generating","ready","executing"):
                self.prepared.finish(rec.decision_id, success=bool(o["valid"] and e["state"]=="executing"), reason=None if o["valid"] else "execution_or_generation_failure")

    def _fit(self,freeze=False):
        for p, program in self.programs.items():
            program["raw_context"]=self.raw_contexts[p]
        for o in self.observations:
            if o.get("program_id"):
                o["raw_context"]=self.raw_contexts[o["program_id"]]
        changed=self.gp.fit(self.observations)
        if changed and self.logger:
            self.logger._append("mf_gp_fits.jsonl",dict(model_version=self.gp.model_version,
                diagnostics=self.gp.fit_diagnostics,policy_version=3))

    def costs(self,program_id=None,operator=None):
        from engine.mf_gp.policy import estimate_costs
        cheap,full,gen,_=estimate_costs(self.observations,.2,self.cost_prior,program_id,operator)
        receipt=next((self.probes[t] for t,e in self.prepared.entries.items()
                      if e["program_id"]==program_id and t in self.probes),None)
        host=[o.get("host_seconds",0.) for o in self.observations if o.get("completed_runtime_observation")]
        overhead=statistics.median(host) if host else 0.
        if receipt and receipt.get("state")=="completed":
            m=receipt["metadata"]
            setup=receipt["elapsed_seconds"] # conservative includes warmup/probe overhead
            # Candidate declarations cannot replace measured evidence optimistically.
            cheap=max(cheap, setup+m["planned_low_steps"]*m["seconds_per_step"]+m["inference_seconds"])
            full=max(full, setup+m["planned_full_steps"]*m["seconds_per_step"]+m["inference_seconds"])
        cheap+=overhead;full+=overhead
        bounds=[o["execution_seconds"] for o in self.observations if program_id is not None
                and o.get("program_id")==program_id and o.get("fidelity")==1. and o.get("runtime_lower_bound")]
        return max(1.,cheap),max(1.,full),gen+120.,1.5*max([full]+bounds)

    def remaining_budget(self,agent):
        self.terminal=False
        return float(agent.acfg.time_limit)-(time.time()-agent.start_time)

    def pending_rows(self,agent):
        return self.prepared.pending_rows()+[p["row"] for p in self.pending_promotions.values()]

    def reserve(self,decision,option,commit_index):
        result=super().reserve(decision,option,commit_index)
        mf=option.state_features["mf"]
        raw=option.state_features.get("study7",{}).get("raw_context")
        if raw is None:
            raw=asdict(BirthContext(0.,0.,0.,0,0,.2,False,None))
        mf["raw_context"]=copy.deepcopy(raw)
        self.prepared.reserve(decision.decision_id,raw,option.operator)
        return result

    def generated(self,node,context):
        ticket=context.decision.decision_id
        self.prepared.generated(ticket,node.id,node.code)
        self.prepared.entries[ticket]["generation_seconds"] = node.generation_wall_time_sec
        self.raw_contexts[node.id]=copy.deepcopy(context.option.state_features["mf"]["raw_context"])

    def state_dict(self):
        return dict(super().state_dict(),raw_contexts=self.raw_contexts,
                    prepared=self.prepared.state_dict(),probes=self.probes)

    def load_state_dict(self,state):
        self.raw_contexts=copy.deepcopy(state["raw_contexts"])
        self.prepared.load_state_dict(state["prepared"])
        self.probes=copy.deepcopy(state["probes"])
        super().load_state_dict(state)

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
            chosen["fidelity"] if chosen else (getattr(parent, "training_fraction", None) or self.low_fraction)
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
            raw_context=asdict(BirthContext(utility, birth, sum(gains), len(gains), count, fidelity, ancestor is not None, rating if rating in (0,1,2) else None)),
            x=x,
            operator=operator,
            parent_id=parent.id,
            parent_utility=utility,
            parent_fidelity=fidelity,
            birth_commit_index=next(
                i for i, n in enumerate(agent.journal.nodes) if n.id == parent.id
            ),
        )

    def select(self, agent, options):
        self.observe(self.records_provider())
        self._fit(freeze=True)
        remaining = self.remaining_budget(agent)
        # One shared GPU: reserve outstanding execution work conservatively.
        outstanding = sum(max(0., p["guard"] - (time.time()-agent.start_time-p["started_active"]))
                          for p in self.pending_promotions.values())
        outstanding += sum(max(0., e.get("guard", 0.) - (time.time()-agent.start_time-e.get("started_active", 0.)))
                           for e in self.prepared.entries.values() if e["state"] == "executing")
        remaining -= outstanding
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
        if not agent.phase1_runtime._budget_available(agent):
            candidates = []  # executing prepared work is already charged
        valid_high = [
            o["utility"]
            for o in self.observations
            if o["valid"] and o["fidelity"] == 1.0 and o.get("budget_eligible", True)
        ]
        incumbent = max(valid_high, default=None)
        if self.calibration is None:
            initial = [p for p in candidates if self.programs[p]["initial"]]
            quality = {o["program_id"]:o["utility"] for o in self.observations if o["valid"]}
            affordable = [p for p in initial if self.costs(p)[3] <= remaining]
            if affordable:
                self.calibration = [max(affordable, key=lambda p:(quality[p],p))]
        actions, targets = [], {}
        feasibility = []
        evidence_key = digest([self.observations, self.probes, pending])
        for p in sorted(candidates):
            saved = self.programs[p]
            row = {k: saved[k] for k in ("program_id", "x", "operator", "raw_context")}
            row["fidelity"] = 1.0
            _, full, _, guard = self.costs(p)
            feasibility.append(dict(kind="promote", program_id=p, guard=guard, feasible=guard <= remaining))
            if guard <= remaining:
                actions.append(
                    dict(kind="promote", row=row, cost=full, guard=guard, program_id=p)
                )
                targets[p] = row
        ready = []
        for ticket, e in self.prepared.entries.items():
            if e["state"] == "deferred" and e.get("deferred_evidence") != evidence_key and self.costs(e["program_id"])[3] <= remaining:
                try: self.prepared.reconsider(ticket)
                except ValueError: pass
            if e["state"] != "ready" or ticket not in self.probes or self.probes[ticket]["state"] not in {"completed", "unsupported", "unknown", "timeout", "interrupted"}:
                continue
            row = dict(program_id=e["program_id"], raw_context=e["raw_context"],
                       x=BirthContext(**e["raw_context"]).features(self.gp.center,self.gp.scale),
                       operator=e["operator"], fidelity=1.)
            cheap, full, _, guard = self.costs(e["program_id"])
            initial = agent.phase1_runtime.root_draft_state["pending"][ticket]["initial"]
            added = False
            if not initial and guard <= remaining:
                actions.append(dict(kind="prepared", row=row, cost=full, guard=guard, ticket=ticket))
                added = True
            cheap_guard = 1.5*cheap + guard
            initial_left = .4*agent.acfg.time_limit - (time.time()-agent.start_time)
            if cheap_guard <= remaining and (not initial or 1.5*cheap <= initial_left):
                actions.append(dict(kind="prepared",row={**row,"fidelity":self.low_fraction},
                                    cost=cheap,guard=cheap_guard,ticket=ticket))
                added = True
            if added:
                targets[e["program_id"]] = row
                ready.append(ticket)
            else:
                self.prepared.defer(ticket,"no_completion_path")
                e["deferred_evidence"] = evidence_key
            feasibility.append(dict(kind="prepared", program_id=e["program_id"], full_guard=guard,
                low_then_full_guard=cheap_guard, feasible=added, initial=initial))
        if ready or sum(e["state"] in ("generating", "ready") for e in self.prepared.entries.values()) >= 3:
            options = []
        # Freeze one prospective identity per legal parent/operator, shared by
        # its hypothetical L/H alternatives, distinct from every past child.
        for index, option in enumerate(options):
            context = option.state_features["study7"]
            row = dict(
                program_id=f'fresh:{agent.phase1_runtime.root_draft_state["revision"]}:{index}',
                x=context["x"],
                raw_context=context["raw_context"],
                operator=option.operator,
                fidelity=1.0,
            )
            cheap, full, generation, guard = self.costs(operator=option.operator)
            feasibility.append(dict(kind="expand", program_id=row["program_id"],
                full_guard=generation*1.5+guard, low_then_full_guard=1.5*(cheap+generation)+guard))
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
        for p in (self.calibration or []):
            if p in completed_high or p in pending_high:
                continue
            selected = next(
                (a for a in actions if a["kind"] == "promote" and a["program_id"] == p),
                None,
            )
            if selected:
                break
        if not actions:
            self.logger._append("mf_gp_feasibility.jsonl", dict(remaining_seconds=remaining, alternatives=feasibility,
                pending=pending, reason="no_affordable_action"))
            if pending or any(e["state"] in ("generating", "ready") for e in self.prepared.entries.values()):
                raise SelectionPending(
                    agent.phase1_runtime.root_draft_state["revision"]
                )
            raise SearchExhausted("No MF action fits the remaining execution budget")
        self.terminal = all(a["row"]["fidelity"] == 1. for a in actions)
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
                pending=pending, targets=list(targets.values()),
                gp_state=self.gp.state_dict(), observations=copy.deepcopy(self.observations),
                pending_reserved_seconds=outstanding,
                cost_scope="future_incremental_wall_seconds",
                actions=actions, feasibility=feasibility,
                scores=scores,
                raw_acquisition=[v*max(1., a["cost"]) for v,a in zip(scores,actions)],
                fit_diagnostics=self.gp.fit_diagnostics,
                chosen=chosen,
                calibration=selected is not None,
                **(dict(policy_version=3, cost_prior=self.cost_prior,
                        completed_runtime_count=sum(o.get("completed_runtime_observation") is True for o in self.observations),
                        censored_runtime_count=sum(bool(o.get("runtime_lower_bound")) for o in self.observations))
                   if self.policy_version == 3 else {}),
                **details,
            ),
        )
        if action["kind"] == "prepared":
            self.prepared.start(action["ticket"], action["row"]["fidelity"])
            self.prepared.entries[action["ticket"]].update(guard=action["guard"],
                started_active=time.time()-agent.start_time, cost=action["cost"])
            agent.phase1_runtime._persist_root_state(agent)
            raise PreparedRequested(action["ticket"], action["row"]["fidelity"])
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
