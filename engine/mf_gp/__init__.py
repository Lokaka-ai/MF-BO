"""Opt-in contextual two-fidelity BO. Staged modes do not import this package."""

MODE = "mf_gp"


def enabled(cfg):
    return getattr(getattr(cfg, "evaluation", None), "multifidelity_mode", None) == MODE
