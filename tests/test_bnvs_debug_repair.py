import json
from dataclasses import replace

import numpy as np
import pytest

from engine.parent_selectors.base import NORMAL_OPERATORS
from engine.parent_selectors.features import BNVS_ROOT_FEATURE_SCHEMA_VERSION
from engine.parent_selectors.selectors import BayesianNodeValueSelector
from engine.parent_selectors.surrogate import BayesianNodeValueSurrogate
from engine.parent_selectors.transition_store import TransitionStore
from engine.selection_logging import SelectionLogger
from test_bnvs_selector import _features, _option, _record


def _debug(index, parent, child):
    return replace(
        _record(index, parent=-10.0, incumbent=0.99, child=child),
        operator="debug",
        parent_node_id=parent.child_node_id,
        parent_metric_raw=None,
        parent_utility=None,
        improvement=None,
        timestamp=f"2026-09-05T00:00:{index:02d}Z",
    )


def _repaired(normal, debug):
    return replace(
        normal,
        child_node_id=debug.child_node_id,
        child_metric_raw=debug.child_metric_raw,
        child_utility=debug.child_utility,
        improvement=debug.child_utility - normal.parent_utility,
        valid=True,
    )


def _assert_same_fit(actual, expected):
    assert actual.training_transition_ids == expected.training_transition_ids
    assert actual.utility_scale == pytest.approx(expected.utility_scale)
    np.testing.assert_allclose(actual.posterior_mean, expected.posterior_mean, atol=1e-12)
    np.testing.assert_allclose(
        actual.posterior_precision_inverse, expected.posterior_precision_inverse, atol=1e-12
    )
    assert actual.posterior_shape == pytest.approx(expected.posterior_shape)
    assert actual.posterior_scale == pytest.approx(expected.posterior_scale)


def test_failed_normal_uses_debug_outcome_once_with_original_context():
    normal = replace(
        _record(0, parent=0.8, incumbent=0.869531, child=None),
        operator="evolution",
        selection_features=_features(0.8, 0.869531, "evolution"),
        reference_node_ids=("reference",),
        metadata={"original_context": True},
    )
    debug = _debug(1, normal, 0.884901717)
    model = BayesianNodeValueSurrogate()
    model.fit([normal, debug])
    expected = BayesianNodeValueSurrogate()
    expected.fit([_repaired(normal, debug)])

    _assert_same_fit(model, expected)
    assert model.training_transition_ids == [normal.transition_id]
    assert model.fitted_transition_count == 2
    effective = model._normal_records(model._effective_records([normal, debug]))[0]
    for field in (
        "transition_id", "decision_id", "parent_node_id", "operator",
        "reference_node_ids", "selection_features", "parent_utility", "timestamp",
    ):
        assert getattr(effective, field) == getattr(normal, field)
    assert model._target(effective) == pytest.approx(
        (0.884901717 - 0.869531) / model.utility_scale
    )
    assert effective.improvement == pytest.approx(0.884901717 - 0.8)
    assert normal.valid is False and normal.child_utility is None
    assert normal.metadata == {"original_context": True}


def test_direct_success_does_not_receive_later_debug_gains():
    normal = _record(0, parent=0.6, incumbent=0.8, child=0.7)
    debug = _debug(1, normal, 0.99)
    model, expected = BayesianNodeValueSurrogate(), BayesianNodeValueSurrogate()
    expected.fit([normal])
    model.fit([normal, debug])

    _assert_same_fit(model, expected)
    assert model._effective_records([normal, debug])[0] is normal
    assert model.fit_diagnostics["debug_repairs"] == []


def test_continuous_debug_chain_uses_first_valid_result_not_best_descendant():
    normal = _record(0, parent=0.6, incumbent=0.8, child=None)
    failed = _debug(1, normal, None)
    first = _debug(2, failed, 0.85)
    later = _debug(3, first, 0.99)
    sibling = _debug(4, failed, 0.95)
    model, expected = BayesianNodeValueSurrogate(), BayesianNodeValueSurrogate()
    model.fit([normal, failed, first, later, sibling])
    expected.fit([_repaired(normal, first)])

    _assert_same_fit(model, expected)
    assert model.fit_diagnostics["debug_repairs"][0]["debug_transition_id"] == first.transition_id


@pytest.mark.parametrize(("maximize", "parent", "incumbent", "child"), [
    (True, 0.2, 0.8, 0.6),
    (False, 0.9, 0.6, 0.7),
])
def test_successful_repair_below_original_incumbent_still_has_negative_h1(maximize, parent, incumbent, child):
    direction = 1 if maximize else -1
    normal = replace(
        _record(0, parent=parent * direction, incumbent=incumbent * direction, child=None),
        parent_metric_raw=parent,
    )
    debug = replace(_debug(1, normal, child), child_utility=child * direction)
    model = BayesianNodeValueSurrogate()
    model.fit([normal, debug])
    effective = model._effective_records([normal, debug])[0]

    assert effective.child_metric_raw == debug.child_metric_raw
    assert effective.child_utility == debug.child_utility
    assert effective.improvement == pytest.approx(debug.child_utility - normal.parent_utility)
    assert model._target(effective) == pytest.approx(np.clip(
        (debug.child_utility - normal.selection_features["global_best_utility"])
        / model.utility_scale, -1, 1
    ))
    assert model._target(effective) < 0


@pytest.mark.parametrize("changes", [
    {"valid": False},
    {"child_utility": None},
    {"child_utility": float("nan")},
    {"child_utility": float("inf")},
    {"child_metric_raw": None},
    {"child_metric_raw": float("nan")},
])
def test_no_valid_finite_debug_preserves_failure(changes):
    normal = _record(0, parent=0.6, incumbent=0.8, child=None)
    debug = replace(_debug(1, normal, 0.9), **changes)
    model, expected = BayesianNodeValueSurrogate(), BayesianNodeValueSurrogate()
    model.fit([normal, debug])
    expected.fit([normal])

    _assert_same_fit(model, expected)
    assert model._target(model._effective_records([normal, debug])[0]) == -1


@pytest.mark.parametrize("boundary_operator", ["draft", "aggregation", "improve", "evolution", "fusion"])
def test_stops_at_non_debug_boundary_and_ignores_fusion_references(boundary_operator):
    old = _record(0, parent=0.6, incumbent=0.8, child=None)
    boundary = replace(
        _record(1, parent=0.7, incumbent=0.8, child=None),
        operator=boundary_operator,
        parent_node_id="unrelated" if boundary_operator == "fusion" else old.child_node_id,
        reference_node_ids=(old.child_node_id,) if boundary_operator == "fusion" else (),
    )
    debug = _debug(2, boundary, 0.9)
    model = BayesianNodeValueSurrogate()
    effective = model._normal_records(model._effective_records([old, boundary, debug]))

    assert effective[0] is old and not effective[0].valid
    if boundary_operator in NORMAL_OPERATORS:
        assert [r.decision_id for r in effective] == [old.decision_id, boundary.decision_id]
        assert effective[1].valid and effective[1].child_utility == 0.9
    else:
        assert effective == [old]


def test_distinct_proposals_with_same_parent_remain_distinct():
    first = _record(0, parent=0.6, incumbent=0.8, child=None)
    second = replace(
        _record(1, parent=0.6, incumbent=0.8, child=None), parent_node_id=first.parent_node_id
    )
    debug = _debug(2, second, 0.9)
    unrelated = replace(_debug(3, first, 0.99), parent_node_id=first.parent_node_id)
    model, expected = BayesianNodeValueSurrogate(), BayesianNodeValueSurrogate()
    model.fit([first, second, debug, unrelated])
    expected.fit([first, _repaired(second, debug)])

    _assert_same_fit(model, expected)
    assert model.training_transition_ids == [first.transition_id, second.transition_id]


@pytest.mark.parametrize("refit_every", [1, 100])
@pytest.mark.parametrize("selection_rule", ["posterior_mean", "thompson"])
def test_visible_repair_refreshes_next_selection_and_logs_source(tmp_path, refit_every, selection_rule):
    store = TransitionStore(tmp_path / "transitions.jsonl")
    normal = _record(0, parent=0.8, incumbent=0.869531, child=None)
    store.append(normal)
    before_repair = store.records()
    model = BayesianNodeValueSurrogate(selection_logger=SelectionLogger(tmp_path))
    selector = BayesianNodeValueSelector(
        name="bnvs_ts", surrogate=model, records_provider=store.records,
        selection_rule=selection_rule, refit_every=refit_every,
        thompson_temperature=1.0, random_seed=42,
    )
    options = [_option(0.8, 0.869531), _option(0.7, 0.869531, "evolution")]
    before = selector.select(None, options)
    old_mean = model.posterior_mean.copy()
    assert before.decision.fallback_used is False
    failed_debug = _debug(1, normal, None)
    store.append(failed_debug)
    selector.select(None, options)
    if refit_every > 1:
        assert model.model_version == before.decision.model_version
    debug = _debug(2, failed_debug, 0.884901717)
    store.append(debug)
    after = selector.select(None, options)
    expected = BayesianNodeValueSurrogate()
    expected.fit([_repaired(normal, debug)])

    assert after.decision.fallback_used is False
    assert after.decision.model_version != before.decision.model_version
    assert not np.allclose(model.posterior_mean, old_mean)
    _assert_same_fit(model, expected)
    assert [score.posterior_mean_value for score in after.candidate_scores] == pytest.approx(
        expected.predict_values(options)[0]
    )
    replay = BayesianNodeValueSurrogate()
    replay.fit(before_repair)
    np.testing.assert_allclose(replay.posterior_mean, old_mean)
    assert replay._target(before_repair[0]) == -1
    fits = [json.loads(line) for line in (tmp_path / "model_fits.jsonl").read_text().splitlines()]
    assert fits[0]["fit_diagnostics"]["value"]["debug_repairs"] == []
    assert fits[-1]["training_transition_ids"] == [normal.transition_id]
    assert fits[-1]["fit_diagnostics"]["value"]["debug_repairs"] == [{
        "normal_transition_id": normal.transition_id,
        "normal_decision_id": normal.decision_id,
        "debug_transition_id": debug.transition_id,
        "available_at": debug.timestamp,
    }]
    if refit_every > 1:
        store.append(_debug(3, failed_debug, 0.99))
        assert selector.select(None, options).decision.model_version == after.decision.model_version


def test_repeated_reads_and_admitted_restart_do_not_duplicate_or_leak(tmp_path):
    path = tmp_path / "transitions.jsonl"
    store = TransitionStore(path)
    normal = _record(0, parent=0.6, incumbent=0.8, child=None)
    failed = _debug(1, normal, None)
    debug = _debug(2, failed, 0.85)
    for record in (normal, failed, debug, debug, normal, replace(normal, transition_id="duplicate")):
        store.append(record)
    raw_log = path.read_text()
    expected = BayesianNodeValueSurrogate()
    expected.fit([_repaired(normal, debug)])
    for admitted in (store, TransitionStore(path)):
        admitted.admit_child_ids({normal.child_node_id, failed.child_node_id, debug.child_node_id})
        model = BayesianNodeValueSurrogate()
        for _ in range(2):
            model.fit(admitted.records())
            _assert_same_fit(model, expected)
            assert model.fit_diagnostics["training_count"] == 1
        admitted.admit_child_ids({normal.child_node_id, failed.child_node_id})
        model.fit(admitted.records())
        assert model._target(model._effective_records(admitted.records())[0]) == -1
        assert model.fit_diagnostics["debug_repairs"] == []
    assert path.read_text() == raw_log


def test_root_mode_keeps_draft_outcome_and_original_normal_context():
    def root_record(record, target_available=True):
        return replace(record, selection_features={
            **record.selection_features,
            "incumbent_at_decision": record.selection_features["global_best_utility"],
        }, metadata={"feature_schema_version": BNVS_ROOT_FEATURE_SCHEMA_VERSION,
                     "target_available": target_available})

    draft = replace(root_record(_record(0, parent=0.0, incumbent=0.0, child=0.7), False), operator="draft")
    normal = root_record(_record(1, parent=0.7, incumbent=0.8, child=None))
    debug = root_record(_debug(2, normal, 0.85))
    model = BayesianNodeValueSurrogate(include_root_draft=True)
    expected = BayesianNodeValueSurrogate(include_root_draft=True)
    model.fit([draft, normal, debug])
    expected.fit([draft, _repaired(normal, debug)])

    _assert_same_fit(model, expected)
    assert model._effective_records([draft, normal, debug])[0] is draft
    assert model.debug_repairs_changed([draft, normal, debug]) is False
