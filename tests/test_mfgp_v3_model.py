import copy
import json
from dataclasses import asdict
import numpy as np
import pytest
from engine.mf_gp.model import MultiFidelityGP
from engine.mf_gp.v3.model import V3GaussianProcess
from engine.mf_gp.v3.context import BirthContext


def row(p, f, y):
    return dict(program_id=p, operator="improve", fidelity=f, valid=True,
                budget_eligible=True, utility=y,
                raw_context=asdict(BirthContext(.5, .02, .04, 2, 4, .2, True, 1)))


def test_actual_gp_scale_refit_and_checkpoint_equivalence():
    rows = [row("a", .2, -.56), row("a", 1., -2.65), row("b", .2, -.42)]
    before = copy.deepcopy(rows)
    g = V3GaussianProcess()
    g.fit(rows[:1])
    singleton = g.scale
    g.fit(rows)
    assert g.scale > singleton
    query = [row("b", 1., 0)]
    expected = g.posterior(query)
    restored = V3GaussianProcess()
    restored.load_state_dict(json.loads(json.dumps(g.state_dict())))
    restored.fit(rows)
    for a, b in zip(expected, restored.posterior(query)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert rows == before


def test_no_cross_fidelity_standardization_or_invalid_admission():
    rows = [row("a", .2, .2), row("a", 1., .8)]
    g = V3GaussianProcess()
    g.fit(rows)
    assert g.center == pytest.approx(.5)
    assert g.scale == pytest.approx(.3)
    h = row("bad", 1., 100)
    h["budget_eligible"] = False
    assert not g.fit(rows + [h])
    assert len(g.rows) == 2


def test_same_code_pending_information_reduces_uncertainty():
    g = V3GaussianProcess()
    g.fit([row("a", .2, .5)])
    target = row("b", 1., 0)
    _, before = g.posterior([target])
    _, after = g.posterior([target], [row("b", .2, 0)])
    assert 0 <= after[0, 0] < before[0, 0]


def test_legacy_checkpoint_and_missing_context_rejected():
    g = V3GaussianProcess()
    with pytest.raises(ValueError, match="contract"):
        g.load_state_dict(MultiFidelityGP(.2).state_dict())
    bad = row("a", .2, .4)
    del bad["raw_context"]
    with pytest.raises(ValueError, match="raw birth context"):
        g.fit([bad])
