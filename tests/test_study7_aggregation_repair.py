"""A repaired aggregation is a scientific root, never an ordinary GP target."""
import copy
import hashlib
from types import SimpleNamespace as NS

import pytest

from engine.parent_selectors.study7_state import scientific_proposals
from test_study7_math import causal_fixture, event


@pytest.mark.parametrize("sources", [(), ("a", "b")])
def test_successful_aggregation_repair_can_be_a_parent_without_new_gp_target(sources):
    state, records, agent, nodes, frozen = causal_fixture()
    records.append(event("ordinary", "b", op="improve", u=-1.35, pu=-1.4, frozen=frozen))
    state.sync(records)
    old_nodes = copy.deepcopy(state.nodes)
    reference = copy.deepcopy(state.reference)
    training = copy.deepcopy(state.training_data(records))
    records.append(event("aggregate", child="bad", op="aggregation", u=None,
                         is_closed=False, metadata={"fusion_source_ids": list(sources)}))
    records.append(event("first_debug", "bad", "bad2", "debug", None, is_closed=False))
    state.sync(records)
    assert state.nodes == old_nodes

    records.append(event("repair", "bad2", "repaired", "debug", -1.25,
                         metadata={"node_snapshot": {"node": {"code": "repaired code"}},
                                   "completed_active_seconds": 123.0}))
    raw = copy.deepcopy(records)
    parent = NS(id="repaired", agent_potential_level=1, agent_potential_assessed_at="repair")
    nodes.append(parent)
    features = state.features(agent, parent, "improve", records)
    scientific = state.nodes[parent.id]
    assert features["parent_utility"] == -1.25
    assert scientific["origin_proposal_id"] == "aggregate"
    assert scientific["outcome_transition_id"] == "repair"
    assert scientific["birth_commit_index"] == len(records) - 1
    assert scientific["scientific_parent_id"] is None
    assert scientific["birth_gain"] == 0
    assert scientific["fusion_sources"] == list(sources)
    assert scientific["fusion_provenance_status"] == (
        "explicit_sources" if sources else "unavailable_new_root")
    assert scientific["code_sha256"] == hashlib.sha256(b"repaired code").hexdigest()
    assert scientific["ancestor_weights"] == (
        {"repaired": .5, "a": .25, "b": .25} if sources else {"repaired": 1.0})
    assert {key: state.nodes[key] for key in old_nodes} == old_nodes
    assert state.reference == reference
    assert state.training_data(records) == training
    assert records == raw
    assert all(row.operator != "aggregation" for row in scientific_proposals(records))

    # A later repair must not replace the already committed successful outcome.
    committed = copy.deepcopy(state.nodes)
    records.append(event("late_debug", "bad", "later", "debug", -1.0))
    state.sync(records)
    assert state.nodes == committed


@pytest.mark.parametrize("metadata", [
    {"budget_eligible": False, "censored": True},
    {"failure_reason": "budget_censored"},
])
def test_censored_aggregation_repair_is_not_a_scientific_parent(metadata):
    state, records, _, _, _ = causal_fixture()
    old_nodes = copy.deepcopy(state.nodes)
    records += [event("aggregate", child="bad", op="aggregation", u=None, is_closed=False),
                event("repair", "bad", "repaired", "debug", -1.2, metadata=metadata)]
    state.sync(records)
    assert state.nodes == old_nodes
