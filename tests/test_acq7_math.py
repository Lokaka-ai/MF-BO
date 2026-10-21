"""Independent scalar Gaussian/kernel oracles for the Acq7 scientific contract."""
import copy
from dataclasses import replace
import math

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from config.acq7 import ACQ7_ARMS, Acq7ModelConfig
from config.study7 import Component
from engine.parent_selectors.acq7_gp import AcqGP, branch_covariance, psd_covariance
from engine.parent_selectors.acq7_acquisition import gaussian_envelope, joint_thompson, knowledge_gradient
from engine.parent_selectors.branch7_gp import BranchGP
from engine.parent_selectors.study7_gp import Study7Error


def config(method="AQ01"):
    kind, tau, rated, acq, slope = ACQ7_ARMS[method]
    return Acq7ModelConfig(method_id=method, schema="acq7_gp4_v1", branch_kind=kind,
        branch_prior_std=tau, slope_prior_std=slope, rating=Component(rated), acquisition_type=acq)


def rows():
    return [{"parent_id": str(i), "operator": "improve" if i % 2 else "fusion",
        "z": [math.tanh(v), .1*i, -.05*i, .1], "performance_v": v,
        "parent_utility": -1.5 + .2*v, "s0": .2, "u_ref": -1.5,
        "branch_effect_load": 1, "root_lineage_id": "b" if i == 2 else "a",
        "rating_s": [-1, 1, 0, 1][i]} for i, v in enumerate([-3., 0., 1., 4.])]


def oracle_kernel(left, right, cfg, hp):
    out = np.zeros((len(left), len(right)))
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            distance = np.sqrt(sum((x-y)**2 for x,y in zip(a["z"], b["z"])))
            q = np.sqrt(3.) * distance / hp[0]
            value = hp[1]**2 * (.25 + .75*(a["operator"] == b["operator"])) * (1+q)*np.exp(-q)
            if cfg.rating.enabled:
                value += .0625*a["rating_s"]*b["rating_s"]
            if a["branch_effect_load"] and b["branch_effect_load"] and a["root_lineage_id"] == b["root_lineage_id"]:
                extra = cfg.branch_prior_std**2
                if cfg.branch_kind == "level_matern32":
                    t = np.sqrt(3.)*abs(a["performance_v"]-b["performance_v"])/cfg.branch_lengthscale
                    extra *= (1+t)*np.exp(-t)
                if cfg.branch_kind == "affine":
                    extra += cfg.slope_prior_std**2*a["performance_v"]*b["performance_v"]
                value += extra
            out[i,j] = value
    return out


@pytest.mark.parametrize("method", ACQ7_ARMS)
def test_kernel_joint_conditional_and_map_derivatives(method):
    cfg, data = config(method), rows()
    gp = AcqGP(cfg, {})
    hp = np.array([.8, 1.3, .4])
    kernel, dell, damp = gp.covariance(data, data, hp, derivatives=True)
    assert_allclose(kernel, oracle_kernel(data, data, cfg, hp), atol=2e-14)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10
    for column, analytic in [(0,dell),(1,damp)]:
        plus, minus = hp.copy(), hp.copy()
        plus[column] *= np.exp(1e-5); minus[column] *= np.exp(-1e-5)
        numerical = (oracle_kernel(data,data,cfg,plus)-oracle_kernel(data,data,cfg,minus))/2e-5
        assert_allclose(analytic, numerical, atol=1e-9, rtol=1e-8)
    target = np.array([.2,-.1,.3])
    gp.fit(data[:3],target,training_ids=["one","two","three"])
    result = gp.predict_joint(data,.2)
    hp = gp.hyperparameters
    train = oracle_kernel(data[:3],data[:3],cfg,hp) + (hp[2]**2+gp.jitter)*np.eye(3)
    cross = oracle_kernel(data,data[:3],cfg,hp)
    m = np.array([.25*r["rating_s"] if cfg.rating.enabled else 0. for r in data])
    expected_mean = np.array([r["parent_utility"] for r in data]) + .2*(m+cross@np.linalg.solve(train,target-m[:3]))
    expected_cov = .04*(oracle_kernel(data,data,cfg,hp)-cross@np.linalg.solve(train,cross.T))
    assert_allclose(result["mean"],expected_mean,atol=1e-13)
    assert_allclose(result["latent_covariance"],expected_cov,atol=1e-13)
    marginal = gp.predict(data,.2)
    assert_allclose(result["mean"],marginal[0],atol=1e-13)
    assert_allclose(result["latent_sd"],marginal[1],atol=1e-13)
    assert_allclose(result["predictive_sd"]**2-result["latent_sd"]**2,.04*hp[2]**2,atol=1e-15)
    short = gp.predict_joint(data[:2],.2)
    assert_allclose(short["latent_covariance"],result["latent_covariance"][:2,:2],atol=1e-13)
    eta = np.log([.8,1.3,.4]);value,gradient=gp.objective(data[:3],target,eta)
    numerical=[]
    for index in range(3):
        direction=np.eye(3)[index]*1e-5
        numerical.append((gp.objective(data[:3],target,eta+direction)[0]-gp.objective(data[:3],target,eta-direction)[0])/2e-5)
    assert_allclose(gradient,numerical,atol=1e-7,rtol=1e-7)


def test_constant_parity_tau_h_loading_and_relabeling():
    data=rows();cfg=config();gp=AcqGP(cfg,{})
    assert_allclose(gp.covariance(data,data)[0],BranchGP(cfg,{}).covariance(data,data)[0])
    one=branch_covariance(data,data,cfg);two=branch_covariance(data,data,config("AQ02"))
    assert_allclose(two-one,1.75*np.array([[x["root_lineage_id"]==y["root_lineage_id"] for y in data] for x in data]))
    zero=copy.deepcopy(data);zero[0].update(branch_effect_load=0,root_lineage_id=None,rating_s=0)
    assert_allclose(branch_covariance(zero,zero,cfg)[0],0)
    zero[1].update(branch_effect_load=0,root_lineage_id=None,rating_s=0)
    assert branch_covariance(zero,zero,cfg)[0,1]==0
    renamed=copy.deepcopy(data)
    for row in renamed:row["root_lineage_id"]="other-"+row["root_lineage_id"]
    assert_allclose(branch_covariance(renamed,renamed,cfg),one)
    data[0]["root_lineage_id"]=None
    with pytest.raises(Study7Error):gp.covariance(data,data)


def test_level_unsaturated_coordinate_and_fixed_lengthscale():
    data=rows();cfg=config("AQ03")
    data[0]["z"][0]=data[3]["z"][0]=1.
    matrix=branch_covariance(data,data,cfg)
    assert matrix[0,3] < .001
    assert_allclose(np.diag(matrix),2.25)
    assert matrix[0,2]==0
    assert_allclose(branch_covariance(data,data,replace(cfg,branch_lengthscale=1e12)),
                    branch_covariance(data,data,config()),atol=1e-10)
    gp=AcqGP(cfg,{})
    assert_allclose(gp.covariance(data,data,[.25,1,.25])[1]["branch"],
                    gp.covariance(data,data,[4,1,.25])[1]["branch"])


def test_affine_feature_map_negative_cross_scale_and_no_double_intercept():
    data=rows();cfg=config("AQ06");matrix=branch_covariance(data,data,cfg)
    design=np.zeros((4,4))
    for i,row in enumerate(data):
        j=0 if row["root_lineage_id"]=="a" else 2
        design[i,j:j+2]=1.5,row["performance_v"]
    assert_allclose(matrix,design@design.T)
    assert matrix[0,3]<0 and np.linalg.eigvalsh(matrix).min()>-1e-10
    assert_allclose(branch_covariance(data,data,replace(cfg,slope_prior_std=0)),branch_covariance(data,data,config()))
    for scale in [.001,10.]:
        transformed=copy.deepcopy(data)
        for row in transformed:row["performance_v"]=(row["parent_utility"]-row["u_ref"])/scale
        difference=branch_covariance(transformed,transformed,cfg)-branch_covariance(transformed,transformed,replace(cfg,slope_prior_std=0))
        expected=np.array([[(a["parent_utility"]-a["u_ref"])*(b["parent_utility"]-b["u_ref"])*(a["root_lineage_id"]==b["root_lineage_id"]) for b in data] for a in data])
        assert_allclose(scale**2*difference,expected,atol=1e-12)


def test_aq04_aq05_aq07_identical_posteriors_and_cache_restore():
    hashes=[]
    for method in ["AQ04","AQ05","AQ07"]:
        gp=AcqGP(config(method),{});data=rows()
        assert gp.fit(data,[.1,.2,-.3,.2],training_ids=list(range(4)))
        hashes.append(gp.predict_joint(data,.2)["posterior_hash"])
        assert not gp.fit(data,[.1,.2,-.3,.2],training_ids=list(range(4)))
        data=copy.deepcopy(data);data[0]["performance_v"]+=.1
        assert gp.fit(data,[.1,.2,-.3,.2],training_ids=list(range(4)))
        saved=gp.state_dict();restored=AcqGP(config(method),{});restored.load_state_dict(saved)
        restored.fit(data,[.1,.2,-.3,.2],training_ids=list(range(4)))
        assert_allclose(restored.predict_joint(data,.2)["latent_covariance"],gp.predict_joint(data,.2)["latent_covariance"])
    assert len(set(hashes))==1


@pytest.mark.parametrize("method",["AQ01","AQ03","AQ06"])
def test_component_conditioning_and_unobserved_branch(method):
    data=rows();gp=AcqGP(config(method),{});gp.fit(data[:2],[.2,-.1],training_ids=[0,1])
    probe=gp.branch_function_posterior("new",[-2.,0.,2.])
    assert_allclose(probe["mean"],0)
    expected=branch_covariance([dict(data[0],root_lineage_id="new",performance_v=v) for v in [-2,0,2]],
        [dict(data[0],root_lineage_id="new",performance_v=v) for v in [-2,0,2]],gp.config)
    assert_allclose(probe["covariance"],expected,atol=1e-13)
    joint=gp.predict_joint(data,.2)
    _,components=gp.covariance(data,data[:2])
    mean=sum((matrix@gp.alpha for matrix in components.values()))
    assert_allclose(joint["mean"],np.array([r["parent_utility"] for r in data])+.2*mean)
    if method=="AQ06":
        coefficients=gp.affine_coefficients(["a","new"])
        assert_allclose(np.asarray(coefficients["covariance"])[2:,2:],np.diag([2.25,1.]))


def test_psd_roundoff_only_and_duplicate_joint_not_independent():
    sigma=np.array([[1.,1.],[1.,1.]])
    matrix,root,diag=psd_covariance(sigma)
    rng=np.random.default_rng(42)
    values,_=joint_thompson([0,0],sigma,rng,root)
    assert_allclose(values[0],values[1],atol=1e-15)
    assert_allclose(root@root.T,matrix,atol=1e-14)
    psd_covariance(np.diag([1.,-1e-12]))
    with pytest.raises(Study7Error):psd_covariance(np.diag([1.,-1e-3]))
    before=copy.deepcopy(rng.bit_generator.state)
    _,info=joint_thompson([3.],np.array([[1.]]),rng)
    assert not info["draw"] and rng.bit_generator.state==before


def test_joint_ts_empirical_mean_covariance_and_two_option_win_probability():
    mu=np.array([.1,-.2]);sigma=np.array([[.7,.35],[.35,.5]])
    _,root,_=psd_covariance(sigma);rng=np.random.default_rng(7301)
    samples=np.array([joint_thompson(mu,sigma,rng,root)[0] for _ in range(25000)])
    assert_allclose(samples.mean(0),mu,atol=.015)
    assert_allclose(np.cov(samples.T),sigma,atol=.02)
    expected=ndtr((mu[0]-mu[1])/np.sqrt(sigma[0,0]+sigma[1,1]-2*sigma[0,1]))
    assert abs(np.mean(samples[:,0]>samples[:,1])-expected)<.015


@pytest.mark.parametrize("seed",range(8))
def test_envelope_independent_quadrature(seed):
    rng=np.random.default_rng(seed);a=rng.normal(size=5);b=rng.normal(size=5)
    crossings=sorted({float((a[i]-a[j])/(b[j]-b[i])) for i in range(5) for j in range(i) if b[i]!=b[j]})
    bounds=[-np.inf,*crossings,np.inf]
    oracle=sum(quad(lambda z: float(np.max(a+b*z))*np.exp(-z*z/2)/np.sqrt(2*np.pi),lo,hi,epsabs=1e-11)[0] for lo,hi in zip(bounds[:-1],bounds[1:]))
    actual,_=gaussian_envelope(a,b,max(a),include_segments=True)
    assert_allclose(actual,oracle-max(a),atol=1e-10)


@pytest.mark.parametrize("a,b",[([0,0],[1,1]),([0,-10],[0,1]),([0,-40],[0,1]),([0,10],[0,1]),([1e12,1e12-1],[0,1]),([0,0,0],[-1,0,1])])
def test_envelope_tails_equal_slopes_and_wide_centering(a,b):
    value,details=gaussian_envelope(a,b,max(a),include_segments=True)
    assert np.isfinite(value) and value>=-1e-12 and details["segment_count"]>=1
    if len(a)==2 and b==[0,1]:
        gap=a[1]-a[0]
        expected=gap*ndtr(gap)+np.exp(-gap*gap/2)/np.sqrt(2*np.pi)-max(0,gap)
        assert_allclose(value,expected,atol=1e-14)


def test_kg_zero_common_shift_noise_and_cross_covariance():
    for mu,sigma in [([3.],[[2.]]),([1.,2.],np.zeros((2,2))),([1.,2.],np.ones((2,2)))]:
        values,_=knowledge_gradient(mu,sigma,.2)
        assert_allclose(values,0,atol=1e-14)
    mu=np.array([.3,0]);sigma=np.array([[1.,.2],[.2,.6]])
    low,_=knowledge_gradient(mu,sigma,.01);high,_=knowledge_gradient(mu,sigma,10.)
    assert np.all(low>high)
    independent,_=knowledge_gradient(mu,np.diag(np.diag(sigma)),.01)
    assert not np.allclose(independent,low)


def test_rokg_same_z_ei_bound_not_ei_plus_kg_and_affine_equivariance():
    mu=np.array([.2,-.4]);sigma=np.array([[.5,-.15],[-.15,1.]])
    incumbent=.6;noise=.3
    kg,_=knowledge_gradient(mu,sigma,noise)
    ro,_=knowledge_gradient(mu,sigma,noise,incumbent=incumbent)
    direct=[]
    for j in range(2):
        d=np.sqrt(sigma[j,j]+noise);gap=mu[j]-incumbent
        ei=gap*ndtr(gap/d)+d*np.exp(-.5*(gap/d)**2)/np.sqrt(2*np.pi)
        direct.append(ei);assert ro[j]>=ei-1e-12
        restricted,_=gaussian_envelope([incumbent,mu[j]],[0,d],incumbent)
        assert_allclose(restricted,ei,atol=1e-13)
        oracle=quad(lambda z: (max(incumbent,mu[j]+d*z,float(np.max(mu+sigma[:,j]/d*z)))-incumbent)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-9,limit=300)[0]
        assert_allclose(ro[j],oracle,atol=1e-8)
    assert not np.allclose(ro,np.array(direct)+kg)
    for retained in [None,incumbent]:
        base,_=knowledge_gradient(mu,sigma,noise,incumbent=retained)
        changed,_=knowledge_gradient(3*mu+7,9*sigma,9*noise,incumbent=None if retained is None else 3*retained+7)
        assert_allclose(changed,3*base,atol=1e-13)


def test_acquisition_errors_do_not_fallback():
    with pytest.raises(Study7Error):knowledge_gradient([0,1],np.eye(2),.1,max_seconds=-1)
    with pytest.raises(Study7Error):knowledge_gradient([0,np.nan],np.eye(2),.1)
    with pytest.raises(Study7Error):gaussian_envelope([np.inf],[0.])
    from engine.parent_selectors.acq7_acquisition import _nonnegative
    with pytest.raises(Study7Error):_nonnegative(-.01,1.)


@pytest.mark.parametrize("method",ACQ7_ARMS)
def test_real_gp4_bounded_map_and_consistent_utility_reexpression(method):
    cfg=config(method);data=[]
    for i in range(12):
        row=copy.deepcopy(rows()[i%4]);row["parent_id"]=str(i)
        row["z"][1]=.04*i;row["performance_v"]+=.03*i
        data.append(row)
    targets=np.sin(np.arange(12))*.2
    gp=AcqGP(cfg,{});gp.fit(data,targets,training_ids=list(range(12)))
    assert gp.fit_diagnostics["status"] in {"map_converged","bounded_optimizer_limit"}
    assert gp.fit_diagnostics["objective_calls"]<=150
    before=gp.predict_joint(data,.2)
    transformed=copy.deepcopy(data)
    for row in transformed:
        row["parent_utility"]=2.5*row["parent_utility"]+4.
        row["u_ref"]=2.5*row["u_ref"]+4.;row["s0"]*=2.5
    other=AcqGP(cfg,{})
    other.fit(transformed,targets,training_ids=list(range(12)))
    after=other.predict_joint(transformed,.5)
    assert_allclose(after["mean"],2.5*before["mean"]+4,atol=1e-12)
    assert_allclose(after["latent_covariance"],2.5**2*before["latent_covariance"],atol=1e-12)
