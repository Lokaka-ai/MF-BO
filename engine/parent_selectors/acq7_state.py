"""Inherited initialization and frozen Acq7 performance-coordinate snapshots."""
import copy
import time

import numpy as np

from .base import utc_timestamp
from .branch7_state import Branch7State
from .study7_gp import Study7Error
from .study7_state import scientific_proposals


class Acq7State(Branch7State):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_diagnostics = None

    def freeze_reference(self, records):
        existing = self.reference is not None
        super().freeze_reference(records)
        if not existing:
            ids = set(self.reference["reference_node_ids"])
            values = np.asarray([r.child_utility for r in scientific_proposals(records)
                                 if r.child_node_id in ids], dtype=float)
            iqr = float(np.quantile(values, .75) - np.quantile(values, .25))
            absolute = .05 * float(np.median(np.abs(values)))
            source = "iqr" if iqr >= max(absolute, 1e-6) else "absolute_median_5pct" if absolute >= 1e-6 else "floor"
            self.freeze_diagnostics = {"timestamp": utc_timestamp(), "committed_record_count": len(records),
                "valid_root_count": len(values), "iqr": iqr, "fallback_source": source,
                "absolute_median_5pct": absolute, "reference_hash": self.reference["hash"],
                "branch_prior_std_utility": self.reference["s0"] * self.config.branch_prior_std}

    def features(self, agent, parent, operator, records):
        freezing = self.reference is None
        row = super().features(agent, parent, operator, records)
        if freezing:
            self.freeze_diagnostics.update(freeze_revision=agent.phase1_runtime.root_draft_state["revision"],
                active_seconds=max(0., time.time() - agent.start_time))
        row.update(schema=self.config.schema,
                   performance_v=(row["parent_utility"] - row["u_ref"]) / row["s0"],
                   performance_coordinate=self.config.performance_coordinate)
        if not np.isfinite(row["performance_v"]):
            raise Study7Error("Nonfinite frozen Acq7 performance coordinate")
        return row

    def state_dict(self):
        return {**super().state_dict(), "freeze_diagnostics": copy.deepcopy(self.freeze_diagnostics)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.freeze_diagnostics = copy.deepcopy(state.get("freeze_diagnostics"))
        if self.reference and (not self.freeze_diagnostics
                or self.freeze_diagnostics["reference_hash"] != self.reference["hash"]):
            raise Study7Error("Acq7 reference diagnostics missing or mismatched")
