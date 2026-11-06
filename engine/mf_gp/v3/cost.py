"""Comparable measured costs and remaining-work estimates for policy v3.

All times are wall seconds, not allocated GPU seconds. Components are mutually
exclusive. Host review never silently becomes candidate-process training time.
"""
from dataclasses import dataclass, asdict
import math


def seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative seconds")
    return float(value)


@dataclass(frozen=True)
class CostRecord:
    program_id: str
    fidelity: float
    state: str
    generation: float = 0.
    probe: float = 0.
    process: float = 0.
    host: float = 0.

    def __post_init__(self):
        if not self.program_id or self.fidelity not in (.2, 1.):
            raise ValueError("Invalid v3 program/fidelity")
        if self.state not in ("completed", "failed", "timeout", "interrupted"):
            raise ValueError("Invalid execution state")
        for key in ("generation", "probe", "process", "host"):
            seconds(getattr(self, key))

    @property
    def spent(self):
        return self.generation + self.probe + self.process + self.host

    @property
    def completed_process_sample(self):
        return self.process if self.state == "completed" else None

    @property
    def completion_lower_bound(self):
        return self.process if self.state == "timeout" else None


@dataclass(frozen=True)
class Workload:
    setup: float
    steps: int
    seconds_per_step: float
    inference: float
    host: float

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("Invalid planned step count")
        for key in ("setup", "seconds_per_step", "inference", "host"):
            seconds(getattr(self, key))

    @property
    def remaining(self):
        return self.setup + self.steps * self.seconds_per_step + self.inference + self.host


def action_cost(workload, *, generation_remaining=0., probe_remaining=0.):
    """Ready programs pass zero sunk generation/probe costs here."""
    return workload.remaining + seconds(generation_remaining) + seconds(probe_remaining)


def completion_guard(cost, comparable_timeout_bounds=(), margin=1.5):
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("Invalid safety margin")
    return margin * max([seconds(cost)] + [seconds(v) for v in comparable_timeout_bounds])


def feasible(remaining, execution_guard, *, required_full_guard=0.):
    """No persistent global terminal latch; caller supplies action-local bounds."""
    return seconds(execution_guard) + seconds(required_full_guard) <= seconds(remaining)
