"""Versioned scheduling policy. No model changes or benchmark-label access."""
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def validate_policy(version, prior, fraction):
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError("Unsupported MF scheduling policy")
    if version == 2:
        if not prior or prior.get("training_fraction") != fraction:
            raise ValueError("Policy 2 requires a measured cost profile for this fraction")
        for key in ("low_seconds", "full_seconds"):
            if not math.isfinite(float(prior[key])) or float(prior[key]) <= 0:
                raise ValueError("Invalid measured runtime")
    return version, prior


def load_cost_prior(path, protocol_dir, fraction):
    if not path:
        return None
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    protocol = json.loads((Path(protocol_dir) / "protocol.json").read_text())
    if (receipt.get("schema") != 1 or receipt.get("protocol_id") != protocol["protocol_id"]
            or receipt.get("training_fraction") != fraction
            or receipt.get("measurement") != "candidate_process_wall_seconds"
            or receipt.get("completed") is not True
            or len(receipt.get("code_sha256", "")) != 64):
        raise ValueError("Cost profile is incomplete or belongs to another protocol/fraction")
    prior = {key: receipt[key] for key in ("low_seconds", "full_seconds", "training_fraction")}
    prior.update(receipt_sha256=hashlib.sha256(raw).hexdigest(),
                 code_sha256=receipt["code_sha256"], protocol_id=receipt["protocol_id"],
                 source="measured_same_code_pair")
    validate_policy(2, prior, fraction)
    return prior


def estimate_costs(observations, fraction, prior, program_id=None, operator=None):
    # Failed/censored seconds remain in the original ledger, not successful-runtime samples.
    measured = [o for o in observations if o.get("completed_runtime_observation") is True
                and o.get("execution_seconds", 0) > 0]
    low = [o["execution_seconds"] for o in measured if o["fidelity"] == fraction]
    high = [o["execution_seconds"] for o in measured if o["fidelity"] == 1.0]
    pairs = {}
    for o in measured:
        pairs.setdefault(o["program_id"], {}).setdefault(o["fidelity"], o["execution_seconds"])
    ratios = [p[1.0] / p[fraction] for p in pairs.values() if fraction in p and 1.0 in p]
    ratio = statistics.median(ratios) if ratios else prior["full_seconds"] / prior["low_seconds"]
    own = pairs.get(program_id, {}).get(fraction)
    cheap = own or (statistics.median(low) if low else prior["low_seconds"])
    full = own * ratio if own else (statistics.median(high) if high else cheap * ratio)
    generation = [o["generation_seconds"] for o in observations
                  if o.get("generation_seconds", 0) > 0 and (operator is None or o.get("operator") == operator)]
    if not generation:
        generation = [o["generation_seconds"] for o in observations if o.get("generation_seconds", 0) > 0]
    # Upper completed costs plus comparable censored lower bounds constrain feasibility.
    upper = sorted(high)[min(len(high)-1, math.ceil(.9*len(high))-1)] if high else full
    bounds = [o["execution_seconds"] for o in observations
              if program_id is not None and o.get("program_id") == program_id
              and o.get("fidelity") == 1.0 and o.get("runtime_lower_bound")]
    guard = 1.5 * max([full, upper, own * max(ratios) if own and ratios else 0.] + bounds)
    return max(1., cheap), max(1., full), statistics.median(generation) if generation else 60., max(1., guard)


def fill_calibration(previous, candidates, programs, observations, seed):
    # Preserve admitted/pending slots. No replacement or endless repair barrier.
    selected = list(previous)
    eligible = sorted(p for p in candidates if programs[p]["initial"] and p not in selected)
    if not selected and eligible:
        quality = {o["program_id"]: o["utility"] for o in observations
                   if o.get("valid") and o.get("budget_eligible", True)}
        best = max(eligible, key=lambda p: (quality[p], p))
        selected.append(best)
        eligible.remove(best)
    if len(selected) < 2 and eligible:
        selected.append(random.Random(seed).choice(eligible))
    return selected


def exact_ceiling_reached(observations, metric):
    # Exact equality, no tolerance/plateau test. Signed utility is higher-is-better.
    bounds = {("roc_auc", True): 1., ("accuracy", True): 1.,
              ("quadratic_kappa", True): 1., ("rmse", False): 0.,
              ("rmsle", False): 0., ("log_loss", False): 0.}
    ceiling = bounds.get((metric.get("name"), metric.get("maximize")))
    return ceiling is not None and any(o.get("valid") and o.get("budget_eligible", True)
        and o.get("fidelity") == 1. and o.get("utility") == ceiling for o in observations)
