import pytest
from engine.mf_gp.v3.mes import information, score_actions
from engine.mf_gp.v3.model import V3GaussianProcess
from dataclasses import asdict
from engine.mf_gp.v3.context import BirthContext


def test_independent_fidelity_has_zero_information():
    assert information(0,1,0,1,0,[0,1,2]) == 0


def test_information_handles_negative_transfer_and_noise():
    a = information(0,1,0,1,.8,[0,1,2])
    b = information(0,1,0,1,-.8,[0,1,2])
    noisy = information(0,1,0,4,.8,[0,1,2])
    assert a == pytest.approx(b, abs=1e-6)
    assert a > noisy > 0


def test_real_gp_challenger_uses_same_pool_and_marginal_cost():
    gp = V3GaussianProcess()
    raw = asdict(BirthContext(.5,0,0,0,0,.2,False,None))
    target = dict(program_id='p',operator='draft',raw_context=raw,fidelity=1.)
    actions = [dict(row=dict(target,fidelity=.2),cost=10),dict(row=target,cost=100)]
    a, details = score_actions(gp,[target],actions,samples=4)
    b, repeated = score_actions(gp,[target],actions,samples=4)
    assert a == b and details == repeated
    assert all(v >= 0 for v in a)
    actions[0]['cost'] = 20
    c, _ = score_actions(gp,[target],actions,samples=4)
    assert c[0] == pytest.approx(a[0]/2)
    assert c[1] == a[1]
