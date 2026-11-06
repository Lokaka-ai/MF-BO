"""Offline finite-pool MF-MES challenger; never a live selector default.

Uses the marginal truncation approximation H_p <= sampled max(H_pool),
not an exact joint conditional distribution given the global maximum.
Candidate observation noise is included; the target maximum is latent quality.
"""
import math
import numpy as np
from scipy.special import log_ndtr
from scipy.integrate import quad


def information(mu_h, var_h, mu_y, var_y, cross, maxima):
    if var_h <= 1e-14 or var_y <= 1e-14 or abs(cross) < 1e-14:
        return 0.
    rho = cross / math.sqrt(var_h * var_y)
    if abs(rho) > 1 + 1e-8:
        raise ValueError("Invalid joint covariance")
    rho = max(-1 + 1e-12, min(1 - 1e-12, rho))
    residual = math.sqrt(1-rho*rho)
    entropy_normal = .5 * math.log(2*math.pi*math.e)
    values = []
    for maximum in maxima:
        alpha = (maximum-mu_h)/math.sqrt(var_h)
        log_z = float(log_ndtr(alpha))
        def integrand(z):
            log_density = -.5*z*z-.5*math.log(2*math.pi) + float(log_ndtr((alpha-rho*z)/residual))-log_z
            return -math.exp(log_density)*log_density if log_density > -740 else 0.
        # Conditional tails can move far from zero for unlikely truncations.
        width = max(12., abs(alpha*rho)+12.)
        entropy, error = quad(integrand, -width, width, epsabs=1e-7, limit=200,
                              points=[max(-width+.001,min(width-.001,alpha/rho))])
        value = entropy_normal-entropy
        if not math.isfinite(value) or value < -1e-5 or error > 1e-4:
            raise ValueError("Unreliable MF-MES quadrature")
        values.append(max(0., value))
    if not values:
        raise ValueError("MF-MES requires maximum samples")
    return float(np.mean(values))


def score_actions(gp, targets, actions, *, pending=(), seed=42, samples=32):
    if not targets or samples < 1 or any(t["fidelity"] != 1. for t in targets):
        raise ValueError("Nonempty full-fidelity target pool required")
    # Every concrete evaluated program has its H counterpart in the target pool.
    index = {t["program_id"]: i for i,t in enumerate(targets)}
    if len(index) != len(targets):
        raise ValueError("Duplicate target identities")
    if any(a["row"]["program_id"] not in index for a in actions):
        raise ValueError("Action outside frozen target pool")
    mean, cov = gp.posterior(list(targets)+[a["row"] for a in actions], pending)
    n = len(targets)
    vals, vecs = np.linalg.eigh(cov[:n,:n])
    if min(vals) < -1e-7:
        raise ValueError("Invalid target covariance")
    rng = np.random.default_rng(seed)
    draws = mean[:n]+rng.standard_normal((samples,n)) @ (vecs*np.sqrt(np.maximum(vals,0))).T
    maxima = draws.max(axis=1)
    noise = gp.noise([a["row"] for a in actions])*gp.scale**2
    raw = []
    for j,a in enumerate(actions):
        if not math.isfinite(a["cost"]) or a["cost"] <= 0:
            raise ValueError("Positive incremental cost required")
        i = index[a["row"]["program_id"]]
        raw.append(information(mean[i],cov[i,i],mean[n+j],cov[n+j,n+j]+noise[j],cov[i,n+j],maxima))
    return [v/a["cost"] for v,a in zip(raw,actions)], dict(
        raw_information=raw, maxima=maxima.tolist(), seed=seed,
        approximation="marginal_high_below_sampled_finite_pool_maximum", offline_only=True)
