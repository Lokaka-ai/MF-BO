"""V3 GP: raw birth context is immutable; normalized geometry is not.

This subclass deliberately has a different checkpoint contract from v1/v2.
It retains the existing covariance, noise, MAP thresholds and acquisition API.
"""
from engine.mf_gp.model import MultiFidelityGP
from engine.mf_gp.v3.context import BirthContext, common_scale


class V3GaussianProcess(MultiFidelityGP):
    def __init__(self):
        super().__init__(.2)
        self.contract.update(version=3, normalization="all_admitted_common_mean_std",
                             context="immutable_raw_birth_context_v1")

    def transformed(self, row):
        raw = row.get("raw_context")
        if raw is None:
            raise ValueError("V3 requires raw birth context; v2 rows cannot be migrated implicitly")
        context = BirthContext(**raw)
        return dict(row, x=context.features(self.center, self.scale))

    def fit(self, observations, reference=None):
        if reference is not None:
            raise ValueError("V3 normalization is based on all admitted observations")
        valid = [o for o in observations if o.get("valid") and o.get("budget_eligible", True)]
        self.center, self.scale = common_scale(o["utility"] for o in valid)
        self.reference_frozen = bool(valid)  # legacy field gates MAP, not frozen scaling
        # Validate only quality-admitted contexts; generation failures have no row.
        return super().fit([self.transformed(o) for o in valid])

    def posterior(self, rows, pending=()):
        return super().posterior([self.transformed(r) for r in rows],
                                 [self.transformed(r) for r in pending])
