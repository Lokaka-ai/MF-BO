from engine.parent_selectors.acquisition import score_predictions
from engine.parent_selectors.base import ExpansionOption, OperatorPreview, PredictiveTransition
from conftest import add_valid_node, make_agent


def _option(node_id, utility, cost_operator="improve"):
    agent = make_agent()
    parent = add_valid_node(agent, node_id=node_id, metric=utility)
    features = {"parent_utility": utility, "global_best_utility": 0.9}
    return ExpansionOption(
        node_id,
        cost_operator,
        (),
        features,
        parent,
        OperatorPreview(cost_operator),
    )


def test_child_ucb_and_gain_ucb_answer_different_parent_questions():
    low_parent = _option("low", 0.2)
    high_parent = _option("high", 0.8)
    predictions = [
        PredictiveTransition("low", "improve", 0.30, 0.05, 5.0, 1.0),
        PredictiveTransition("high", "improve", 0.05, 0.05, 5.0, 1.0),
    ]
    child_scores = score_predictions(
        [low_parent, high_parent], predictions,
        acquisition_type="child_ucb", beta=1.0,
        use_validity_probability=False, cost_power=1.0,
    )
    gain_scores = score_predictions(
        [low_parent, high_parent], predictions,
        acquisition_type="gain_ucb", beta=1.0,
        use_validity_probability=False, cost_power=1.0,
    )
    assert max(child_scores, key=lambda item: item.acquisition_score).parent_node_id == "high"
    assert max(gain_scores, key=lambda item: item.acquisition_score).parent_node_id == "low"


def test_cost_aware_ei_prefers_equal_posterior_at_lower_cost():
    cheap = _option("cheap", 0.8)
    expensive = _option("expensive", 0.8)
    predictions = [
        PredictiveTransition("cheap", "improve", 0.2, 0.1, 1.0, 0.9),
        PredictiveTransition("expensive", "improve", 0.2, 0.1, 10.0, 0.9),
    ]
    scores = score_predictions(
        [cheap, expensive], predictions,
        acquisition_type="ei_per_cost", beta=1.5,
        use_validity_probability=True, cost_power=1.0,
    )
    assert scores[0].acquisition_score > scores[1].acquisition_score

