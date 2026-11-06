"""Common normalization with immutable birth-time context, independent of NumPy."""
from dataclasses import dataclass
import math
import statistics


def common_scale(utilities):
    y = list(utilities)
    if any(not math.isfinite(v) for v in y):
        raise ValueError("Nonfinite utility")
    if not y:
        return 0., 1.
    center = statistics.fmean(y)
    return center, max(statistics.pstdev(y), .05 * abs(center), 1e-6)


@dataclass(frozen=True)
class BirthContext:
    utility: float
    birth_gap: float
    gain_sum: float
    gain_count: int
    expansion_count: int
    parent_fidelity: float
    ancestor_present: bool
    rating: int | None

    def __post_init__(self):
        if any(not math.isfinite(v) for v in (self.utility, self.birth_gap, self.gain_sum)):
            raise ValueError("Nonfinite context")
        for n in (self.gain_count, self.expansion_count):
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("Invalid context count")
        if self.parent_fidelity not in (.2, 1.) or self.rating not in (None, 0, 1, 2):
            raise ValueError("Invalid context category")

    def features(self, center, scale):
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid normalization")
        count = math.log1p(self.expansion_count)
        return [math.tanh((self.utility-center)/scale),
                math.tanh(self.birth_gap/scale),
                math.tanh(self.gain_sum/((self.gain_count+1)*scale)),
                count/(1+count), float(self.parent_fidelity == 1.),
                float(self.ancestor_present),
                self.rating-1. if self.rating is not None else 0.,
                float(self.rating is not None)]
