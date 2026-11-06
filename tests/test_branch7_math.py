"""Independent finite-Gaussian checks of the actual Branch7 production kernel."""
import copy
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import cholesky

from config.branch7 import Branch7ModelConfig
from config.study7 import Component, Study7Config
from engine.parent_selectors.branch7_gp import BranchGP, branch_kernel
from engine.parent_selectors.gp_surrogate import GaussianProposalSurrogate
from engine.parent_selectors.study7_gp import State4GP, Study7Error
from tests_gp6.test_core import record, features, option


def setting(dim=4, rating=True, ancestor=False, tau=1.5):
    return Branch7ModelConfig(method_id="BR06" if dim == 6 else "BR05",
        base_dimension=dim, rating=Component(rating), ancestor=Component(ancestor), branch_prior_std=tau)


def rows(n=5, dim=4):
    rng = np.random.default_rng(10)
    return [{"z": rng.normal(size=dim).tolist(), "operator": ("improve", "evolution", "fusion")[i % 3],
        "parent_utility": -1.6 + .02 * i, "parent_id": str(i), "branch_effect_load": 1,
        "root_lineage_id": f"branch{i % 2}", "rating_s": [-1, 0, 1][i % 3],
        "ancestor_weights": {str(i): 1.}, "reference_hash": "frozen"} for i in range(n)]


@pytest.mark.parametrize("dim", [4, 6])
@pytest.mark.parametrize("tau", [1.5, 2.])
def test_k01_k07_k08_k11_k12_full_joint_conditioning(dim, tau):
    train, candidate = rows(5, dim), rows(3, dim)
    candidate[2]["root_lineage_id"] = "unobserved"
    gp = BranchGP(setting(dim, tau=tau), {})
    targets = np.array([.2, -.9, .3, -.7, .1])
    gp.fit(train, targets, training_ids=list(range(5)))
    scale = .08
    mean, sd, pred, _ = gp.predict(candidate, scale)
    all_rows = train + candidate
    x = np.array([r["z"] for r in all_rows])
    distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    q = np.sqrt(3) * distance
    ops = np.array([r["operator"] for r in all_rows])
    kbase = (1 + q) * np.exp(-q) * (.25 + .75 * (ops[:, None] == ops[None, :]))
    names = list(dict.fromkeys(r["root_lineage_id"] for r in all_rows))
    z = np.array([[r["root_lineage_id"] == name for name in names] for r in all_rows], dtype=float)
    s = np.array([r["rating_s"] for r in all_rows])
    latent = kbase + tau ** 2 * z @ z.T + .0625 * np.outer(s, s)
    c = latent[:5, :5] + (.25 ** 2 + 1e-8) * np.eye(5)
    cross = latent[5:, :5]
    alpha = np.linalg.solve(c, targets - .25 * s[:5])
    expected = np.array([r["parent_utility"] for r in candidate]) + scale * (.25 * s[5:] + cross @ alpha)
    covariance = latent[5:, 5:] - cross @ np.linalg.solve(c, cross.T)
    assert_allclose(mean, expected, atol=1e-12)
    assert_allclose(sd ** 2, scale ** 2 * np.diag(covariance), atol=1e-12)
    assert_allclose(pred ** 2 - sd ** 2, (scale * .25) ** 2, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > -1e-12
    posterior = gp.branch_posterior(names, scale)
    assert_allclose(posterior["mean"], tau ** 2 * z[:5].T @ alpha, atol=1e-12)
    assert_allclose(posterior["covariance"], tau ** 2 * np.eye(len(names)) - tau ** 4 * z[:5].T @ np.linalg.solve(c, z[:5]), atol=1e-12)
    assert posterior["mean"][-1] == 0 and posterior["sd"][-1] == tau
    assert max(posterior["sd"][:-1]) < tau
    th, var = gp.theta_posterior()
    assert_allclose(th, .25 + .0625 * s[:5] @ alpha)
    assert_allclose(var, .0625 - .0625 ** 2 * s[:5] @ np.linalg.solve(c, s[:5]))


def test_k02_k03_k04_k05_mask_shared_across_operations_and_tau_units():
    x = rows(4)
    x[0]["root_lineage_id"] = x[1]["root_lineage_id"] = "same"
    for row in x[2:]:
        row.update(branch_effect_load=0, root_lineage_id=None, rating_s=0)
    expected = np.zeros((4, 4)); expected[:2, :2] = 2.25
    assert_allclose(branch_kernel(x, x, 1.5), expected)
    assert_allclose(branch_kernel(x, x, 2.), expected * 4 / 2.25)
    gp = BranchGP(setting(), {})
    for hp in ([1, 1, .25], [.3, 4, .5]):
        assert_allclose(gp.covariance(x, x, hp)[1]["branch"], expected)
    assert_allclose(branch_kernel([], x, 1.5), np.empty((0, 4)))


@pytest.mark.parametrize("dim", [4, 6])
def test_k06_k07_k14_permutation_growth_and_tau_difference(dim):
    x = rows(6, dim)
    gp = BranchGP(setting(dim), {})
    kernel, _ = gp.covariance(x, x)
    assert_allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-12
    renamed = copy.deepcopy(x)
    for row in renamed:
        row["root_lineage_id"] = {"branch0": "z", "branch1": "a"}[row["root_lineage_id"]]
    assert_allclose(gp.covariance(renamed, renamed)[0], kernel)
    extra = copy.deepcopy(x[0]); extra["root_lineage_id"] = "new"
    assert_allclose(gp.covariance([*x, extra], [*x, extra])[0][:6, :6], kernel)
    stronger = BranchGP(setting(dim, tau=2.), {})
    assert_allclose(stronger.mean(x), gp.mean(x))
    assert_allclose(stronger.covariance(x, x)[0] - kernel, 1.75 * branch_kernel(x, x, 1.))


@pytest.mark.parametrize("dim", [4, 6])
def test_k17_finite_difference_full_map_gradient(dim):
    x = rows(9, dim)
    gp = BranchGP(setting(dim), {})
    y = np.linspace(-1.7, .7, len(x))
    eta = np.log([.8, 1.4, .3])
    value, actual = gp.objective(x, y, eta)
    expected = []
    for i in range(3):
        plus, minus = eta.copy(), eta.copy()
        plus[i] += 1e-5; minus[i] -= 1e-5
        expected.append((gp.objective(x, y, plus)[0] - gp.objective(x, y, minus)[0]) / 2e-5)
    assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
    _, _, amplitude = gp.covariance(x, x, np.exp(eta), derivatives=True)
    assert_allclose(amplitude, 2 * gp.covariance(x, x, np.exp(eta))[1]["state"])


def test_k10_scalar_random_effect_posterior():
    x = rows(4)
    for row in x: row["root_lineage_id"] = "a"
    gp = BranchGP(setting(rating=False), {})
    gp.rows = x
    c = 2.25 * np.ones((4, 4)) + .0625 * np.eye(4)
    gp.factor = cholesky(c, lower=True)
    y = np.array([.1, .2, .3, .4])
    gp.alpha = np.linalg.solve(c, y)
    posterior = gp.branch_posterior(["a", "new"], .07)
    variance = 1 / (1 / 2.25 + 4 / .0625)
    assert_allclose(posterior["covariance"][0][0], variance, atol=1e-12)
    assert_allclose(posterior["mean"][0], variance * y.sum() / .0625)
    assert_allclose(posterior["sd_utility"], .07 * np.array(posterior["sd"]))


@pytest.mark.parametrize("dim", [4, 6])
def test_k09_k16_k18_draft_mask_zero_prior_predicts_and_theta_can_negative(dim):
    gp = BranchGP(setting(dim), {})
    x = rows(3, dim)
    gp.fit([], [], training_ids=[])
    mean, sd, _, detail = gp.predict(x, .1)
    expected = np.array([r["parent_utility"] + .025 * r["rating_s"] for r in x])
    assert_allclose(mean, expected)
    assert_allclose(sd ** 2, .01 * (1 + 2.25 + .0625 * np.array([r["rating_s"] for r in x]) ** 2))
    for row in x: row.update(branch_effect_load=0, root_lineage_id=None, rating_s=0)
    gp.fit(x, [-1., 0., 1.], training_ids=list(range(3)))
    assert gp.branch_posterior(["unobserved"], .1)["sd"] == [1.5]
    assert gp.theta_posterior() == (.25, .0625)
    x = rows(6, dim)
    for row in x: row.update(rating_s=1, branch_effect_load=1)
    gp.fit(x, [-100.] * 6, training_ids=list(range(6)))
    assert gp.theta_posterior()[0] < 0


@pytest.mark.parametrize("dim", [4, 6])
def test_d08_cache_invalidation_and_restored_fit(dim):
    x = rows(9, dim); y = np.linspace(-.2, .3, 9)
    gp = BranchGP(setting(dim), {})
    assert gp.fit(x, y, training_ids=list(range(9)))
    saved = gp.state_dict()
    assert not gp.fit(x, y, training_ids=list(range(9)))
    other = BranchGP(setting(dim), {}); other.load_state_dict(saved)
    assert other.fit(x, y, training_ids=list(range(9)))
    assert other.fit_diagnostics["status"] == "restored_frozen_fit"
    assert_allclose(other.predict(x, .1)[:3], gp.predict(x, .1)[:3])
    for field, value in [("root_lineage_id", "different"), ("reference_hash", "different")]:
        changed = copy.deepcopy(x); changed[0][field] = value
        assert gp.fit(changed, y, training_ids=list(range(9)))
    gp.config = replace(gp.config, branch_prior_std=2.)
    assert gp.fit(x, y, training_ids=list(range(9)))
    x[0].update(branch_effect_load=0, rating_s=0)
    assert gp.fit(x, y, training_ids=list(range(9)))


@pytest.mark.parametrize("dim", [4, 6])
def test_shadow_reconditions_full_model_without_mutating_it(dim):
    x = rows(5, dim); y = np.linspace(-1., .3, 5)
    gp = BranchGP(setting(dim), {}); gp.fit(x, y, training_ids=list(range(5)))
    before = copy.deepcopy(gp.state_dict()); alpha = gp.alpha.copy()
    shadow = gp.without_branch()
    k, _ = shadow.covariance(x, x)
    expected = np.linalg.solve(k + (.25 ** 2 + shadow.jitter) * np.eye(5), y - shadow.mean(x))
    assert_allclose(shadow.alpha, expected)
    gp.predict(x, .1, shadow=True)
    assert gp.state_dict() == before
    assert_allclose(gp.alpha, alpha)
    assert gp.config.branch_prior_std == 1.5 and shadow.config.branch_prior_std == 0


def test_d01_gp6_off_components_matches_native_admission_map_and_prediction():
    records = [record('draft', value=-1.7)]
    for i in range(10):
        records.append(record(str(i), parent='draftchild', operator='improve', parent_value=-1.7,
            value=-1.7+.03*np.sin(i), state=features(-1.7, -1.6, gp_trials=i, gp_last_gain=.02*np.cos(i))))
    native = GaussianProposalSurrogate(); native.fit(records)
    gp = BranchGP(setting(6, rating=False, tau=0.), {})
    train = []
    targets = []
    for r in records:
        baseline = native.baseline(r.selection_features, r.operator)
        train.append({'z':native._design(r.selection_features,r.operator).tolist(), 'operator':r.operator,
            'parent_utility':baseline, 'branch_effect_load':int(r.operator!='draft'),
            'root_lineage_id':'draft' if r.operator!='draft' else None, 'rating_s':0})
        targets.append((r.child_utility-baseline)/native.utility_scale)
    gp.fit(train,targets,training_ids=list(range(len(train))))
    assert_allclose(gp.hyperparameters,native.hyperparameters,rtol=1e-7,atol=1e-8)
    candidate = option('a',features(-1.5,-1.4))
    row={'z':native._design(candidate.state_features,candidate.operator).tolist(), 'operator':candidate.operator,
         'parent_utility':candidate.parent_utility,'branch_effect_load':1,'root_lineage_id':'draft','rating_s':0}
    assert_allclose(gp.predict([row],native.utility_scale)[:3],native.predict_quality([candidate]),rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize('ancestor,rating',[(False,False),(True,False),(False,True),(True,True)])
def test_d02_k13_k15_gp4_off_branch_matches_all_historical_components(ancestor,rating):
    x=rows(5); nodes={str(i):{'xi':[.1*i,.05*i]} for i in range(5)}
    x[-1]['ancestor_weights']={'0':.25,'1':.25,'4':.5}
    old=State4GP(Study7Config(ancestor=Component(ancestor),rating=Component(rating)),nodes)
    new=BranchGP(setting(4,rating,ancestor,tau=0.),nodes)
    y=np.linspace(-.5,.5,5)
    old.fit(x,y,training_ids=list(range(5)));new.fit(x,y,training_ids=list(range(5)))
    assert_allclose(old.covariance(x,x)[0],new.covariance(x,x)[0])
    assert_allclose(old.predict(x,.08)[:3],new.predict(x,.08)[:3])


@pytest.mark.parametrize('change',[{'root_lineage_id':None},{'branch_effect_load':2},
    {'branch_effect_load':True},{'branch_effect_load':0,'rating_s':1}])
def test_missing_or_illegal_branch_inputs_fail_closed(change):
    x=rows(1);x[0].update(change)
    with pytest.raises(Study7Error):BranchGP(setting(),{}).fit(x,[.1],training_ids=['a'])
