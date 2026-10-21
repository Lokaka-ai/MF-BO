"""Independent numerical and causal checks for spec T01--T30/T36."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
from numpy.testing import assert_allclose
import pytest

from config.study7 import Component, Study7Config
from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.parent_selectors.study7_gp import State4GP, Study7Error, ancestor_kernel, matern32, rating_value
from engine.parent_selectors.study7_selector import winner
from engine.parent_selectors.study7_state import Study7State, closed, scientific_proposals
from engine.parent_selectors.transition_store import TransitionRecord
from engine.parent_selectors.features import utility_from_metric


def settings(ancestor=False, rating=False):
    return Study7Config(method_id={(False, False): "E02", (True, False): "E03",
                                 (False, True): "E04", (True, True): "E05"}[ancestor, rating],
                        ancestor=Component(ancestor), rating=Component(rating))


def row(z=(0, 0, 0, 0), op="improve", q=1, weights=None, parent=-1.4):
    return {"z": list(z), "operator": op, "rating_s": rating_value(q),
            "ancestor_weights": weights or {"a": 1.0}, "parent_utility": parent}


def event(key, parent="root", child=None, op="draft", u=-1.4, pu=None,
          frozen=None, is_closed=True, refs=(), metadata=None, cost=2.0):
    return TransitionRecord(key, key, parent, child or key, op, refs,
        -pu if pu is not None else None, -u if u is not None else None,
        pu, u, u-pu if u is not None and pu is not None else None,
        u is not None, cost/2, cost/2, cost, None, None, "study7_state4_ucb",
        {"study7": frozen} if frozen else {}, key,
        {"proposal_closed": is_closed, **(metadata or {})})


def causal_fixture(ancestor=True, rating=True):
    records = [event("a", u=-1.5), event("b", u=-1.4), event("c", u=-1.3)]
    nodes = [NS(id=key, agent_potential_level=2, agent_potential_assessed_at="initial") for key in "abc"]
    agent = NS(journal=NS(nodes=nodes))
    state = Study7State(settings(ancestor, rating))
    initial = state.features(agent, nodes[1], "improve", records)
    return state, records, agent, nodes, initial


def test_t01_matern_and_operation_closed_form():
    x = np.array([[0., 0.], [1., 2.]])
    d = np.sqrt(3) * np.sqrt(5) / .8
    assert_allclose(matern32(x, x, .8), [[1, (1+d)*np.exp(-d)], [(1+d)*np.exp(-d), 1]])
    gp = State4GP(settings(), {})
    k, _ = gp.covariance([row(), row(op="fusion")], [row(), row(op="fusion")], [1, 2, .25])
    assert_allclose(k, [[4, 1], [1, 4]])


def test_t02_state4_hand_calculation_sign_and_local_counts():
    state, records, agent, nodes, frozen = causal_fixture()
    assert state.reference["u_ref"] == -1.4
    assert_allclose(state.reference["s0"], .1)
    records += [event("gain", "b", op="improve", u=-1.3, pu=-1.4, frozen=frozen),
                event("loss", "b", op="improve", u=-1.45, pu=-1.4, frozen=frozen),
                event("failure", "b", op="improve", u=None, pu=-1.4),
                event("pending", "b", op="improve", u=None, pu=-1.4, is_closed=False),
                event("other", "a", op="evolution", u=-1.0, pu=-1.5)]
    got = state.features(agent, nodes[1], "improve", records)
    assert_allclose(got["z"], [0, 0, np.tanh(.05/.3), np.log(4)/(1+np.log(4))], atol=1e-14)
    assert got["local_closed_attempts"] == 3 and got["local_valid_count"] == 2
    assert utility_from_metric(1.4, False) == -1.4
    assert utility_from_metric(1.4, True) == 1.4


def test_t03_uuid_does_not_enter_state_kernel():
    gp = State4GP(settings(), {})
    x, y = row(), row()
    x["parent_id"], y["parent_id"] = "a", "unrelated"
    assert_allclose(gp.covariance([x, y], [x, y])[0], np.ones((2, 2)))


def test_t04_t11_tree_fusion_dedup_and_debug_do_not_add_generations():
    state, records, agent, nodes, frozen = causal_fixture()
    records += [event("p", "b", "bad", "improve", None, -1.4, frozen, False),
                event("d1", "bad", "bad2", "debug", None, is_closed=False),
                event("d2", "bad2", "good", "debug", -1.35),
                event("f", "good", "fused", "fusion", -1.3, -1.35, refs=("b", "b", "good"))]
    state.sync(records)
    assert "bad" not in state.nodes and "bad2" not in state.nodes
    assert_allclose(state.nodes["good"]["birth_gain"], .05)
    assert state.nodes["good"]["ancestor_weights"] == {"good": .5, "b": .5}
    assert state.nodes["fused"]["ancestor_weights"] == {"fused": .5, "good": .125, "b": .375}
    assert state.nodes["fused"]["fusion_sources"] == ["good", "b"]
    for node in state.nodes.values():
        assert_allclose(sum(node["ancestor_weights"].values()), 1.)
        assert min(node["ancestor_weights"].values()) >= 0


def test_t05_t06_t10_t11_content_aggregation_not_mean_embedding_or_identity():
    nodes = {"a": {"xi": [-1., 0.]}, "b": {"xi": [1., 0.]}, "clone": {"xi": [-1., 0.]}}
    left = [row(weights={"a": .5, "b": .5}), row(weights={"clone": 1.})]
    right = [row(weights={"b": 1.}), row(weights={"a": 1.})]
    k = ancestor_kernel(left, right, nodes, 1.)
    manual = np.array([[sum(w*v*matern32([nodes[i]["xi"]], [nodes[j]["xi"]], 1.)[0, 0]
                       for i, w in x["ancestor_weights"].items() for j, v in y["ancestor_weights"].items())
                       for y in right] for x in left])
    assert_allclose(k, manual)
    assert not np.isclose(k[0, 0], matern32([[0, 0]], [[1, 0]], 1.)[0, 0])
    assert k[1, 1] == 1.0


@pytest.mark.parametrize("ancestor,rating", [(False, False), (True, False), (False, True), (True, True)])
def test_t07_psd_random_dag(ancestor, rating):
    rng = np.random.default_rng(91)
    nodes, rows = {}, []
    for i in range(30):
        key = str(i)
        weights = {key: 1.0}
        if i:
            weights = {key: .5}
            for source in rng.choice(i, size=min(i, 3), replace=False):
                for support, weight in rows[source]["ancestor_weights"].items():
                    weights[support] = weights.get(support, 0) + .5*weight/min(i, 3)
        nodes[key] = {"xi": rng.uniform(-1, 1, 2)}
        rows.append(row(rng.uniform(-1, 1, 4), op=("improve", "evolution", "fusion")[i % 3],
                        q=i % 3, weights=weights))
    gp = State4GP(settings(ancestor, rating), nodes)
    k, _ = gp.covariance(rows, rows)
    assert_allclose(k, k.T, atol=1e-14)
    assert np.linalg.eigvalsh(k).min() > -1e-10
    ka = ancestor_kernel(rows, rows, nodes, 1.)
    assert np.linalg.eigvalsh(ka).min() > -1e-10
    assert np.diag(ka).max() <= 1 + 1e-12


def test_t08_t09_renaming_support_and_new_nodes_do_not_change_old_predictions():
    nodes = {"a": {"xi": [0., 0.]}, "b": {"xi": [.5, -.3]}}
    rows = [row(weights={"a": 1}), row(weights={"a": .5, "b": .5}, q=2)]
    model = State4GP(settings(True, True), nodes)
    model.fit(rows, [.1, -.2], training_ids=["1", "2"])
    before = model.predict(rows, .1)[:3]
    k = model.covariance(rows, rows)[0]
    nodes["future"] = {"xi": [1., 1.]}
    assert_allclose(model.covariance(rows, rows)[0], k, atol=0)
    renamed = {key + "new": val for key, val in reversed(list(nodes.items()))}
    relabelled = [{**r, "ancestor_weights": {key+"new": v for key, v in r["ancestor_weights"].items()}} for r in rows]
    other = State4GP(settings(True, True), renamed)
    other.fit(relabelled, [.1, -.2], training_ids=["1", "2"])
    assert_allclose(other.predict(relabelled, .1)[:3], before, atol=1e-12)


def test_t12_t30_all_component_toggles_and_prior_diagonals():
    nodes = {"a": {"xi": [0., 0.]}}
    rows = [row(q=0), row(q=2, op="fusion")]
    base = State4GP(settings(), nodes).covariance(rows, rows)[0]
    for ancestor, rating in [(False, False), (True, False), (False, True), (True, True)]:
        gp = State4GP(settings(ancestor, rating), nodes)
        k, components = gp.covariance(rows, rows)
        assert_allclose(k, base + ancestor*.5*base + rating*.0625*np.array([[1, -1], [-1, 1]]))
        assert_allclose(sum(components.values()), k)
        gp.fit(rows[:1], [.1], training_ids=["1"])
        _, _, _, detail = gp.predict(rows, .1)
        assert_allclose(detail[1]["diag_total"], sum(detail[1]["diag_"+key] for key in components))
        assert detail[1]["posterior_variance_residual"] < detail[1]["diag_total"]


def test_t13_t14_prior_prediction_and_strict_rating_neutrality():
    model = State4GP(settings(False, True), {})
    model.fit([], [], training_ids=[])
    rows = [row(q=q) for q in [0, 1, 2, None, True, "2", 2.0, float("nan")]]
    means, latent, predictive, _ = model.predict(rows, .2)
    assert_allclose(means, [-1.45, -1.4, -1.35, -1.4, -1.4, -1.4, -1.4, -1.4])
    assert_allclose(latent[[0, 2]], .2*np.sqrt(1.0625))
    assert_allclose(predictive**2 - latent**2, .2**2*.25**2)
    assert_allclose(means[2]-means[0], .5*.2)


def test_t15_t16_t17_joint_conditioning_theta_and_shadow():
    nodes = {"a": {"xi": [0., 0.]}}
    rows = [row([i/6, 0, 0, 0], q=(2 if i % 2 else 0), op=("fusion" if i % 3 else "improve")) for i in range(7)]
    scores = np.array([r["rating_s"] for r in rows])
    target = -4*scores
    model = State4GP(settings(True, True), nodes)
    model.fit(rows, target, training_ids=list(range(7)))
    tests = [row([.23, 0, 0, 0], q=2), row([.5, 0, 0, 0], q=0)]
    kt, _ = model.covariance(rows, rows)
    cross, _ = model.covariance(tests, rows)
    ktest, _ = model.covariance(tests, tests)
    c = kt + (.25**2 + 1e-8)*np.eye(len(rows))
    centered = target - .25*scores
    mean, latent, _, _ = model.predict(tests, .1)
    expected_mean = .25*np.array([1, -1]) + cross @ np.linalg.solve(c, centered)
    expected_var = ktest - cross @ np.linalg.solve(c, cross.T)
    assert_allclose(mean, -1.4+.1*expected_mean)
    assert_allclose(latent**2, .01*np.diag(expected_var))
    theta, variance = model.theta_posterior()
    assert_allclose(theta, .25+.0625*scores@np.linalg.solve(c, centered))
    assert_allclose(variance, .0625-.0625**2*scores@np.linalg.solve(c, scores))
    assert theta < 0
    shadow = model.predict(tests, .1, shadow=True)[0]
    assert_allclose(shadow, -1.4+.1*cross@np.linalg.solve(c, target))
    assert not np.allclose(shadow, mean-.1*.25*np.array([1, -1]))


def test_t18_t19_variance_shrinks_at_fixed_hyperparameters_and_noise_separate():
    gp = State4GP(settings(False, True), {})
    prior = gp.predict([row(q=2)], .1)[1][0]
    for n in range(1, 8):
        gp.fit([row(q=2)]*n, np.zeros(n), training_ids=list(range(n)))
        _, latent, pred, _ = gp.predict([row(q=2)], .1)
        assert latent[0] <= prior
        assert_allclose(pred**2-latent**2, .01*.25**2)
        prior = latent[0]


def test_t20_exact_argmax_does_not_use_old_tolerance():
    options = [ExpansionOption(key, "improve", (), {"study7": {"birth_commit_index": i}}, NS(id=key), OperatorPreview("improve"))
               for i, key in enumerate(["first", "second"])]
    assert winner(options, np.array([1., 1.+1e-12])) == 1
    assert winner(options, np.array([1., 1.])) == 0
    mu, latent = np.array([.8, .7]), np.array([.01, .2])
    assert winner(options, mu+np.sqrt(2.25)*latent) == winner(options, mu+1.5*latent) == 1


def test_t21_t22_t23_debug_label_failure_pending_and_negative_gain():
    state, records, agent, nodes, frozen = causal_fixture()
    records += [event("proposal", "b", "bad", "improve", None, -1.4, frozen, False, cost=3),
                event("debug1", "bad", "bad2", "debug", None, is_closed=False, cost=4),
                event("debug2", "bad2", "good", "debug", -1.45, cost=5),
                event("failure", "b", "failure", "improve", None, -1.4),
                event("pending", "b", "pending", "improve", None, -1.4, is_closed=False),
                event("late", "b", "late", "improve", -1.0, -1.4,
                      metadata={"budget_eligible": False, "censored": True})]
    state.sync(records)
    rows, targets, ids, exclusions = state.training_data(records)
    assert len(rows) == 1 and rows[0] == frozen
    assert_allclose(targets, [-.5])
    assert ids == [{"proposal_id": "proposal", "outcome_id": "debug2"}]
    folded = scientific_proposals(records)[3]
    assert folded.total_wall_time_sec == 12
    now = state.features(agent, nodes[1], "improve", records)
    assert now["local_closed_attempts"] == 2 and now["local_valid_count"] == 1
    assert {x["reason"] for x in exclusions} == {"not_ordinary_proposal", "closed_failure_no_quality_label", "pending_or_censored"}


def test_reset_gp_rows_require_full_parent_and_full_child():
    state, records, agent, nodes, cheap_features = causal_fixture()
    records.append(event("mixed", "b", "full", "improve", -1.35, -1.4, cheap_features))
    state.sync(records)
    full_node = NS(id="full", agent_potential_level=2, agent_potential_assessed_at="full")
    agent.journal.nodes.append(full_node)
    full_features = state.features(agent, full_node, "improve", records)
    records.append(event("full_edge", "full", "next", "improve", -1.3, -1.35, full_features))
    rows, targets, ids, exclusions = state.training_data(records, {"full", "next"})
    assert rows == [full_features]
    assert ids == [{"proposal_id": "full_edge", "outcome_id": "full_edge"}]
    assert any(item == {"proposal_id": "mixed", "reason": "not_full_full_transition"}
               for item in exclusions)


def test_t24_t25_t26_t27_frozen_history_rating_reference_and_replay():
    state, records, agent, nodes, frozen = causal_fixture()
    reference = copy.deepcopy(state.reference)
    records += [event("gain", "b", "future", "improve", -1.35, -1.4, frozen)]
    nodes[1].agent_potential_level = 0
    records += [event("future_draft", u=-.2), records[-1]]
    state.sync(records)
    rows, targets, _, _ = state.training_data(records)
    assert len(rows) == 1 and rows[0]["rating_s"] == 1
    assert rows[0]["local_closed_attempts"] == 0
    assert state.reference == reference
    other = Study7State(settings(True, True))
    other.load_state_dict(state.state_dict())
    other.sync(records)
    assert other.state_dict() == state.state_dict()
    assert other.features(agent, nodes[1], "improve", records) == state.features(agent, nodes[1], "improve", records)
    for node in state.nodes.values():
        assert len(node["xi"]) == 2
    assert len(rows[0]["z"]) == 4
    bad = state.state_dict()
    bad["reference"]["s0"] *= 2
    with pytest.raises(Study7Error, match="reference"):
        other.load_state_dict(bad)
    # Component switches never alter the four-state feature computation.
    for ancestor, rating in [(False, False), (True, False), (False, True), (True, True)]:
        fresh, _, _, _, initial = causal_fixture(ancestor, rating)
        assert_allclose(initial["z"], frozen["z"])


@pytest.mark.parametrize("sources", [("missing",), ("fusion",)])
def test_t36_illegal_sources_fail(sources):
    state, records, _, _, frozen = causal_fixture()
    records.append(event("fusion", "b", "fusion", "fusion", -1.3, -1.4, frozen, refs=sources))
    with pytest.raises(Study7Error, match="source|cyclic"):
        state.sync(records)


def test_numerical_bounds_map_gradient_and_resume(monkeypatch):
    nodes = {"a": {"xi": [0., 0.]}, "b": {"xi": [.5, -.4]}}
    rows = [row([i/10, np.sin(i), 0, 0], q=i % 3, weights={"a": .5, "b": .5}) for i in range(10)]
    gp = State4GP(settings(True, True), nodes)
    hp = np.array([.8, 1.2, .3])
    _, dell, damp = gp.covariance(rows, rows, hp, derivatives=True)
    for index, analytic in [(0, dell), (1, damp)]:
        left, right = hp.copy(), hp.copy()
        left[index] *= np.exp(1e-5)
        right[index] *= np.exp(-1e-5)
        numeric = (gp.covariance(rows, rows, left)[0]-gp.covariance(rows, rows, right)[0])/(2e-5)
        assert_allclose(analytic, numeric, atol=2e-9)
    target = np.sin(np.arange(10))*.3
    gp.fit(rows, target, training_ids=list(range(10)))
    assert gp.fit_diagnostics["objective_calls"] <= 150
    assert gp.fit_diagnostics["status"] in {"map_converged", "bounded_optimizer_limit"}
    restored = State4GP(settings(True, True), nodes)
    restored.load_state_dict(gp.state_dict())
    monkeypatch.setattr("engine.parent_selectors.study7_gp.minimize", lambda *a, **k: pytest.fail("Must restore fitted parameters"))
    restored.fit(rows, target, training_ids=list(range(10)))
    assert_allclose(restored.predict(rows, .1)[:3], gp.predict(rows, .1)[:3], atol=1e-14)
    assert restored.model_version == gp.model_version


def test_numerical_failure_limits_are_explicit():
    attempts = []
    with pytest.raises(Study7Error, match="three"):
        State4GP._factor(np.array([[1., 3.], [3., 1.]]), attempts)
    assert [x["jitter"] for x in attempts] == [1e-8, 1e-7, 1e-6]
    assert_allclose(State4GP.nonnegative_variance([-1e-12], [1.]), [0.])
    with pytest.raises(Study7Error, match="negative"):
        State4GP.nonnegative_variance([-1e-4], [1.])
    with pytest.raises(Study7Error, match="four"):
        State4GP(settings(), {}).covariance([row([0]*8)], [row()])


def test_late_debug_is_censored_not_failure():
    records = [event("p", "b", "bad", "improve", None, -1.4, is_closed=False),
               event("d", "bad", "late", "debug", None, is_closed=False,
                     metadata={"budget_eligible": False, "censored": True, "completed_active_seconds": 21601})]
    folded = scientific_proposals(records)[0]
    assert not closed(folded) and folded.metadata["censored"]
    assert folded.metadata["completed_active_seconds"] == 21601
