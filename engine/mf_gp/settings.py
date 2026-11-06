"""Fidelity identity, separate from the numeric GP hyperparameters."""


def validate_low_fraction(value):
    if isinstance(value, bool) or value not in (0.1, 0.2, 0.5):
        raise ValueError("MF-GP low fraction must be 0.1, 0.2, or legacy 0.5")
    return float(value)


def low_fraction(cfg):
    value = validate_low_fraction(getattr(cfg.evaluation, "mf_gp_low_fraction", 0.5))
    if getattr(cfg.evaluation, "multifidelity_mode", None) != "mf_gp" and value != 0.5:
        raise ValueError("Configurable low fraction applies only to MF-GP")
    return value
