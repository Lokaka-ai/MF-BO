"""Causal primary-graph metadata and seven-dimensional A/B inputs.

Only appended, completed transitions supply node scores. Debug replaces its
failed proposal's result rather than adding research depth. No text encoder,
private score, candidate code generation, or future journal score is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.linalg import solve

from .base import NORMAL_OPERATORS
from .gp_features import GP6_NAMES, gp_vector
from .transition_store import TransitionRecord

GP7_SCHEMA = "gp7_absolute_primary_ancestry_v1"
GP7_NAMES = (*GP6_NAMES, "absolute_parent")
GRAPH_SELECTOR_TYPES = frozenset({"gp7_ancestry_logei", "laplacian_anchor_logei", "gp7_long_horizon_logei"})


@dataclass(frozen=True)
class GraphConfig:
    ancestry_decay: float = 0.5
    graph_weight: float = 0.3
    use_ancestry_kernel: bool = True
    absolute_mode: str = "auto"
    laplacian_smoothing: float = 0.3
    covariance_ridge: float = 0.0625
    amplitude_cap: float = 5.0
    xi: float = 0.0

    @property
    def effective_graph_weight(self):
        return self.graph_weight if self.use_ancestry_kernel else 0.0

    def __post_init__(self):
        if type(self.use_ancestry_kernel) is not bool:
            raise ValueError("gp.graph.use_ancestry_kernel must be a boolean")
        if not math.isfinite(self.ancestry_decay) or not 0 < self.ancestry_decay < 1:
            raise ValueError("gp.graph.ancestry_decay must be in (0,1)")
        if not math.isfinite(self.graph_weight) or not 0 <= self.graph_weight <= 1:
            raise ValueError("gp.graph.graph_weight must be in [0,1]")
        for key in ("covariance_ridge", "amplitude_cap"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"gp.graph.{key} must be finite and positive")
        for key in ("laplacian_smoothing", "xi"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"gp.graph.{key} must be finite and nonnegative")
        if self.absolute_mode not in {"auto", "utility", "qwk", "log_loss"}:
            raise ValueError("absolute_mode must be auto/utility/qwk/log_loss")


def ancestry_matrix(chains: Sequence[Sequence[str]], gamma: float = 0.5) -> np.ndarray:
    """Normalized, dynamically extensible ID coordinates (not numerical IDs)."""
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("ancestry decay must be in (0,1)")
    if any(not c or len(set(c)) != len(c) or any(not isinstance(x, str) or not x for x in c)
           for c in chains):
        raise ValueError("Ordinary proposals need nonempty, acyclic primary ancestry")
    ids = {name: i for i, name in enumerate(sorted({x for c in chains for x in c}))}
    matrix = np.zeros((len(chains), len(ids)))
    for row, chain in enumerate(chains):
        for depth, name in enumerate(chain):
            matrix[row, ids[name]] = gamma ** depth
        matrix[row] /= np.linalg.norm(matrix[row])
    return matrix


def graph_matern32(x, ox, y, oy, graph_cross, *, length_scale=1.0,
                   amplitude=1.0, rho=0.25, graph_weight=0.3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    distance = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    q = math.sqrt(3.) * distance / length_scale
    state = (1. + q) * np.exp(-q)
    operation = rho + (1. - rho) * (np.asarray(ox)[:, None] == np.asarray(oy)[None, :])
    return amplitude ** 2 * operation * ((1. - graph_weight) * state + graph_weight * graph_cross)


def gp7_vector(features: dict, *, scale: float, draft_anchor: float,
               loss_scale: float, task_id: str, absolute_mode: str, operator: str) -> np.ndarray:
    # Initial draft/debug/aggregation are logged but NEVER trained by these arms.
    if operator not in NORMAL_OPERATORS:
        raise ValueError("GP7 models ordinary proposals only")
    parent = float(features["parent_utility"])
    mode = absolute_mode
    if mode == "auto":
        if "gp_metric_maximize" not in features:
            raise ValueError("Missing frozen metric direction for GP7")
        mode = ("qwk" if task_id == "aptos2019-blindness-detection" else "utility") \
            if features["gp_metric_maximize"] else "log_loss"
    if mode == "utility":
        absolute = parent
    elif mode == "qwk":
        absolute = (parent + 1.) / 2.
    else:
        if parent > 0 or not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("log_loss requires nonnegative raw loss and positive scale")
        absolute = -math.log1p(-parent / loss_scale)
    if not math.isfinite(absolute):
        raise ValueError("Non-finite absolute parent feature")
    return np.append(gp_vector(features, scale=scale, draft_anchor=draft_anchor, operator=operator), absolute)


class DecisionGraph:
    """One score-graph solve per feedback prefix, shared by all its candidates.

    The transition store supplies a completed prefix under the runtime lock.
    Neither node.lock nor newly generated/unreported node.metric changes scores.
    The graph revision is its source transition count, not a hash or extra file.
    """
    def __init__(self, smoothing: float):
        self.smoothing = float(smoothing)
        self._key = None
        self.parents: dict[str, str | None] = {}
        self.utilities: dict[str, float] = {}
        self.centers: dict[str, float] = {}
        self.roots: set[str] = set()
        self.revision = 0
        self.edge_count = 0

    def update(self, records: Sequence[TransitionRecord]) -> None:
        key = (len(records), records[-1].transition_id if records else None)
        if key == self._key:
            return
        parents, utilities, roots = {}, {}, set()
        for row in records:
            child, parent = row.child_node_id, row.parent_node_id
            if not child:
                continue
            if row.operator in {"draft", "aggregation"}:
                roots.add(parent)
                parents[child] = None
            elif row.operator == "debug":
                # All failed proposals, including intermediate failed repairs,
                # have topology even though they do not have a scored vertex.
                if parent not in parents:
                    raise ValueError("Debug feedback lacks its original proposal topology")
                parents[child] = parents[parent]
            else:
                parents[child] = parent
            if row.valid and row.child_utility is not None and math.isfinite(float(row.child_utility)):
                utilities[child] = float(row.child_utility)
        for root in roots:
            utilities.pop(root, None)
        ids = sorted(utilities)
        index = {name: i for i, name in enumerate(ids)}
        adjacency = np.zeros((len(ids), len(ids)))
        for child in ids:
            parent = parents.get(child)
            if parent in index and parent != child:
                adjacency[index[child], index[parent]] = adjacency[index[parent], index[child]] = 1.
        values = np.asarray([utilities[x] for x in ids])
        if len(ids) and self.smoothing:
            laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
            values = solve(np.eye(len(ids)) + self.smoothing * laplacian, values, assume_a="pos")
        self.parents, self.utilities, self.roots = parents, utilities, roots
        self.centers = dict(zip(ids, map(float, values)))
        self.revision, self.edge_count = len(records), int(adjacency.sum() // 2)
        self._key = key

    def ancestry(self, parent: str) -> list[str]:
        if parent not in self.utilities:
            raise ValueError(f"Candidate {parent!r} has no completed, valid feedback")
        chain, seen = [], set()
        cursor = parent
        while cursor is not None and cursor not in self.roots:
            if cursor in seen:
                raise ValueError("Cycle in primary proposal ancestry")
            seen.add(cursor)
            # Unscored intermediate failures are topology only, not depth.
            if cursor in self.utilities:
                chain.append(cursor)
            cursor = self.parents.get(cursor)
        return chain

    def features(self, parent: str, maximize: bool) -> dict:
        return {"gp_primary_ancestry": self.ancestry(parent),
                "gp_graph_center": self.centers[parent],
                "gp_graph_revision": self.revision,
                "gp_graph_nodes": len(self.utilities), "gp_graph_edges": self.edge_count,
                "gp_metric_maximize": bool(maximize)}
