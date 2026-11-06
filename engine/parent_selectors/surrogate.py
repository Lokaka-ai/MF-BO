"""Small-data Bayesian gain, cost and validity models for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
import time
import uuid
from typing import Sequence

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from engine.parent_selectors.base import (
    NORMAL_OPERATORS,
    ExpansionOption,
    PredictiveTransition,
)
from engine.parent_selectors.features import (
    BNVS_FEATURE_NAMES,
    BNVS_FEATURE_SCHEMA_VERSION,
    BNVS_ROOT_FEATURE_NAMES,
    BNVS_ROOT_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    REWARD_FEATURE_NAMES,
    REWARD_ONLY_FEATURE_SCHEMA_VERSION,
    reward_scale_from_records,
    vectorize_bnvs_features,
    vectorize_bnvs_root_features,
    vectorize_features,
    vectorize_reward_features,
)
from engine.parent_selectors.transition_store import TransitionRecord
from engine.selection_logging import SelectionLogger


def cost_to_log(cost: float) -> float:
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and non-negative")
    return math.log1p(cost)


def cost_from_log(log_cost: float) -> float:
    if not math.isfinite(log_cost):
        raise ValueError("log cost must be finite")
    return max(math.expm1(log_cost), 0.0)


class SurrogateNotFitted(RuntimeError):
    pass


class BayesianGainModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = BayesianRidge(compute_score=True)
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.training_transition_ids: list[str] = []
        self.model_version: str | None = None
        self.fit_diagnostics: dict = {}

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        usable = [
            record for record in records
            if record.valid and record.improvement is not None
        ]
        if not usable:
            raise ValueError("gain model has no valid continuous-gain transitions")
        x = np.asarray(
            [vectorize_features(record.selection_features) for record in usable],
            dtype=float,
        )
        y = np.asarray([record.improvement for record in usable], dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("gain model input contains non-finite values")

        x_scaled = self.scaler.fit_transform(x)
        self.target_mean = float(np.mean(y))
        self.target_scale = float(np.std(y))
        if self.target_scale < 1e-12:
            self.target_scale = 1.0
        y_scaled = (y - self.target_mean) / self.target_scale
        self.model.fit(x_scaled, y_scaled)
        # sklearn <=1.6 omits prior variance in unobserved SVD directions when
        # n_samples < n_features. Reconstruct the exact posterior covariance;
        # the fitted mean is unchanged. Also correct on newer sklearn releases.
        centered = x_scaled - x_scaled.mean(axis=0)
        precision = self.model.lambda_ * np.eye(centered.shape[1]) + self.model.alpha_ * centered.T.dot(centered)
        self.model.sigma_ = np.linalg.solve(precision, np.eye(precision.shape[0]))

        train_pred = self.model.predict(x_scaled) * self.target_scale + self.target_mean
        self.training_transition_ids = [record.transition_id for record in usable]
        self.model_version = f"gain-{len(usable)}-{uuid.uuid4().hex[:12]}"
        self.fit_diagnostics = {
            "training_count": len(usable),
            "training_rmse": float(np.sqrt(np.mean((train_pred - y) ** 2))),
            "target_mean": self.target_mean,
            "target_std": float(np.std(y)),
            "alpha": float(self.model.alpha_),
            "lambda": float(self.model.lambda_),
        }

    def predict(self, options: Sequence[ExpansionOption]) -> tuple[np.ndarray, np.ndarray]:
        if self.model_version is None:
            raise SurrogateNotFitted("gain model has not been fitted")
        x = np.asarray(
            [vectorize_features(dict(option.state_features)) for option in options],
            dtype=float,
        )
        if not np.isfinite(x).all():
            raise ValueError("gain prediction input contains non-finite values")
        mean_scaled, std_scaled = self.model.predict(
            self.scaler.transform(x), return_std=True
        )
        means = mean_scaled * self.target_scale + self.target_mean
        stds = np.maximum(std_scaled * self.target_scale, 1e-9)
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("gain model returned non-finite predictions")
        return means, stds


class BayesianCostModel:
    def __init__(self, default_cost: float = 600.0):
        self.default_cost = max(float(default_cost), 1e-6)
        self.scaler = StandardScaler()
        self.model = BayesianRidge(compute_score=True)
        self.is_fitted = False
        self.operator_costs: dict[str, list[float]] = {}
        self.global_costs: list[float] = []
        self.training_count = 0

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        usable = [
            record for record in records
            if math.isfinite(record.total_wall_time_sec)
            and record.total_wall_time_sec >= 0
        ]
        self.operator_costs = {}
        self.global_costs = []
        for record in usable:
            cost = max(float(record.total_wall_time_sec), 1e-6)
            self.operator_costs.setdefault(record.operator, []).append(cost)
            self.global_costs.append(cost)
        self.training_count = len(usable)

        if len(usable) < 2:
            self.is_fitted = False
            return
        x = np.asarray(
            [vectorize_features(record.selection_features) for record in usable],
            dtype=float,
        )
        y = np.asarray([cost_to_log(max(record.total_wall_time_sec, 0.0)) for record in usable])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("cost model input contains non-finite values")
        self.model.fit(self.scaler.fit_transform(x), y)
        self.is_fitted = True

    def fallback_cost(self, operator: str) -> float:
        operator_costs = self.operator_costs.get(operator, ())
        if operator_costs:
            return max(float(median(operator_costs)), 1e-6)
        if self.global_costs:
            return max(float(median(self.global_costs)), 1e-6)
        return self.default_cost

    def predict(self, options: Sequence[ExpansionOption]) -> np.ndarray:
        if not self.is_fitted:
            return np.asarray([self.fallback_cost(option.operator) for option in options])
        x = np.asarray(
            [vectorize_features(dict(option.state_features)) for option in options],
            dtype=float,
        )
        log_cost = self.model.predict(self.scaler.transform(x))
        costs = np.asarray([max(cost_from_log(float(value)), 1e-6) for value in log_cost])
        if not np.isfinite(costs).all():
            raise ValueError("cost model returned non-finite predictions")
        return costs


class BetaBernoulliValidityModel:
    def __init__(self):
        self.counts: dict[str, tuple[int, int]] = {}

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        mutable: dict[str, list[int]] = {}
        for record in records:
            success, failure = mutable.setdefault(record.operator, [0, 0])
            if record.valid:
                success += 1
            else:
                failure += 1
            mutable[record.operator] = [success, failure]
        self.counts = {
            operator: (counts[0], counts[1]) for operator, counts in mutable.items()
        }

    def predict(self, operator: str) -> float:
        success, failure = self.counts.get(operator, (0, 0))
        return (success + 1.0) / (success + failure + 2.0)


@dataclass
class BayesianTransitionSurrogate:
    default_cost: float = 600.0
    selection_logger: SelectionLogger | None = None

    def __post_init__(self):
        self.gain_model = BayesianGainModel()
        self.cost_model = BayesianCostModel(default_cost=self.default_cost)
        self.validity_model = BetaBernoulliValidityModel()
        self.model_version: str | None = None
        self.fitted_transition_count = -1
        self.last_fit_wall_time_sec = 0.0

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        started = time.perf_counter()
        self.gain_model.fit(records)
        self.cost_model.fit(records)
        self.validity_model.fit(records)
        self.model_version = self.gain_model.model_version
        self.fitted_transition_count = len(records)
        self.last_fit_wall_time_sec = max(time.perf_counter() - started, 0.0)
        if self.selection_logger is not None:
            self.selection_logger.append_model_fit({
                "model_version": self.model_version,
                "training_transition_ids": self.gain_model.training_transition_ids,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "fit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fit_wall_time_sec": self.last_fit_wall_time_sec,
                "fit_diagnostics": {
                    "gain": self.gain_model.fit_diagnostics,
                    "cost_training_count": self.cost_model.training_count,
                    "cost_model_fitted": self.cost_model.is_fitted,
                    "validity_counts": self.validity_model.counts,
                },
            })

    def predict(self, options: Sequence[ExpansionOption]) -> list[PredictiveTransition]:
        means, stds = self.gain_model.predict(options)
        costs = self.cost_model.predict(options)
        predictions = []
        for option, mean_gain, std_gain, cost in zip(options, means, stds, costs):
            predictions.append(PredictiveTransition(
                parent_node_id=option.parent_node_id,
                operator=option.operator,
                mean_gain=float(mean_gain),
                std_gain=float(max(std_gain, 1e-9)),
                predicted_total_cost=float(max(cost, 1e-6)),
                p_valid=float(self.validity_model.predict(option.operator)),
            ))
        return predictions


@dataclass
class RewardOnlyBayesianLinearSurrogate:
    """Run-local conjugate Bayesian linear model for valid transition gains."""

    selection_logger: SelectionLogger | None = None
    prior_precision: float = 4.0
    intercept_precision: float = 0.25
    prior_shape: float = 2.0
    prior_scale: float = 1.0
    uncertainty_multiplier: float = 2.0
    uncertainty_floor: float = 0.10

    def __post_init__(self):
        self.posterior_mean: np.ndarray | None = None
        self.posterior_precision_inverse: np.ndarray | None = None
        self.posterior_shape = 0.0
        self.posterior_scale = 0.0
        self.reward_scale = 1.0
        self.model_version: str | None = None
        self.fitted_transition_count = -1
        self.last_fit_wall_time_sec = 0.0
        self.training_transition_ids: list[str] = []
        self.fit_diagnostics: dict = {}

    @staticmethod
    def _usable(records: Sequence[TransitionRecord]) -> list[TransitionRecord]:
        return [
            record for record in records
            if record.valid
            and record.improvement is not None
            and math.isfinite(float(record.improvement))
        ]

    def _design(self, features: dict[str, float]) -> np.ndarray:
        compact = vectorize_reward_features(
            features,
            utility_scale=self.reward_scale,
        )
        return np.asarray([1.0, *compact], dtype=float)

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        started = time.perf_counter()
        usable = self._usable(records)
        if not usable:
            raise ValueError("reward-only model has no valid continuous-gain transitions")

        self.reward_scale = reward_scale_from_records(usable)
        x = np.asarray(
            [self._design(record.selection_features) for record in usable],
            dtype=float,
        )
        y = np.asarray(
            [float(record.improvement) / self.reward_scale for record in usable],
            dtype=float,
        )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("reward-only model input contains non-finite values")

        dimension = x.shape[1]
        prior_precision = np.eye(dimension, dtype=float) * max(
            float(self.prior_precision), 1e-9
        )
        prior_precision[0, 0] = max(float(self.intercept_precision), 1e-9)
        posterior_precision = prior_precision + x.T @ x
        posterior_precision_inverse = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_precision_inverse @ x.T @ y
        posterior_shape = max(float(self.prior_shape), 1e-9) + len(usable) / 2.0
        residual_term = float(y @ y - posterior_mean @ posterior_precision @ posterior_mean)
        posterior_scale = max(
            float(self.prior_scale) + 0.5 * max(residual_term, 0.0),
            1e-12,
        )

        self.posterior_mean = posterior_mean
        self.posterior_precision_inverse = (
            posterior_precision_inverse + posterior_precision_inverse.T
        ) / 2.0
        self.posterior_shape = posterior_shape
        self.posterior_scale = posterior_scale
        self.training_transition_ids = [record.transition_id for record in usable]
        self.model_version = f"reward-only-{len(usable)}-{uuid.uuid4().hex[:12]}"
        self.fitted_transition_count = len(records)
        self.last_fit_wall_time_sec = max(time.perf_counter() - started, 0.0)

        train_mean = x @ posterior_mean * self.reward_scale
        train_target = y * self.reward_scale
        self.fit_diagnostics = {
            "training_count": len(usable),
            "training_rmse": float(
                np.sqrt(np.mean((train_mean - train_target) ** 2))
            ),
            "reward_scale": self.reward_scale,
            "posterior_shape": self.posterior_shape,
            "posterior_scale": self.posterior_scale,
            "prior_precision": float(self.prior_precision),
            "intercept_precision": float(self.intercept_precision),
            "uncertainty_multiplier": float(self.uncertainty_multiplier),
            "uncertainty_floor": float(self.uncertainty_floor),
        }
        if self.selection_logger is not None:
            self.selection_logger.append_model_fit({
                "model_version": self.model_version,
                "model_type": "conjugate_bayesian_linear_reward_only",
                "training_transition_ids": self.training_transition_ids,
                "feature_schema_version": REWARD_ONLY_FEATURE_SCHEMA_VERSION,
                "feature_names": list(REWARD_FEATURE_NAMES),
                "fit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fit_wall_time_sec": self.last_fit_wall_time_sec,
                "fit_diagnostics": {"reward": self.fit_diagnostics},
            })

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.posterior_mean is None or self.posterior_precision_inverse is None:
            raise SurrogateNotFitted("reward-only model has not been fitted")
        return self.posterior_mean, self.posterior_precision_inverse

    def _design_matrix(self, options: Sequence[ExpansionOption]) -> np.ndarray:
        x = np.asarray(
            [self._design(dict(option.state_features)) for option in options],
            dtype=float,
        )
        if not np.isfinite(x).all():
            raise ValueError("reward-only prediction input contains non-finite values")
        return x

    def predict(self, options: Sequence[ExpansionOption]) -> list[PredictiveTransition]:
        posterior_mean, precision_inverse = self._require_fitted()
        x = self._design_matrix(options)
        means = x @ posterior_mean * self.reward_scale
        expected_noise_variance = self.posterior_scale / max(
            self.posterior_shape - 1.0,
            1e-9,
        )
        epistemic_variance = np.einsum("ij,jk,ik->i", x, precision_inverse, x)
        standardized_stds = np.sqrt(
            np.maximum((1.0 + epistemic_variance) * expected_noise_variance, 0.0)
        ) * max(float(self.uncertainty_multiplier), 0.0)
        standardized_stds = np.maximum(
            standardized_stds,
            max(float(self.uncertainty_floor), 0.0),
        )
        stds = standardized_stds * self.reward_scale
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("reward-only model returned non-finite predictions")
        return [
            PredictiveTransition(
                parent_node_id=option.parent_node_id,
                operator=option.operator,
                mean_gain=float(mean_gain),
                std_gain=float(std_gain),
                predicted_total_cost=None,
                p_valid=None,
            )
            for option, mean_gain, std_gain in zip(options, means, stds)
        ]

    def latent_std_gains(self, options: Sequence[ExpansionOption]) -> np.ndarray:
        """Return uncalibrated epistemic std of the latent mean gain.

        Unlike ``predict()``, this excludes observation noise, the uncertainty
        multiplier, and the predictive uncertainty floor. It is suitable for
        experimental latent-UCB acquisition bonuses.
        """
        _, precision_inverse = self._require_fitted()
        x = self._design_matrix(options)
        expected_noise_variance = self.posterior_scale / max(
            self.posterior_shape - 1.0,
            1e-9,
        )
        epistemic_variance = np.einsum("ij,jk,ik->i", x, precision_inverse, x)
        stds = np.sqrt(
            np.maximum(epistemic_variance * expected_noise_variance, 0.0)
        ) * self.reward_scale
        if not np.isfinite(stds).all():
            raise ValueError("reward-only latent uncertainty is non-finite")
        return stds

    def sample_gains(
        self,
        options: Sequence[ExpansionOption],
        *,
        seed: int,
        temperature: float,
    ) -> np.ndarray:
        """Jointly sample latent gains so candidate correlations are preserved."""
        posterior_mean, precision_inverse = self._require_fitted()
        x = self._design_matrix(options)
        rng = np.random.default_rng(int(seed))
        precision_sample = rng.gamma(
            shape=self.posterior_shape,
            scale=1.0 / self.posterior_scale,
        )
        noise_variance = 1.0 / max(float(precision_sample), 1e-12)
        covariance = noise_variance * precision_inverse
        covariance = (covariance + covariance.T) / 2.0
        cholesky = np.linalg.cholesky(
            covariance + np.eye(covariance.shape[0], dtype=float) * 1e-12
        )
        raw_draw = posterior_mean + cholesky @ rng.standard_normal(len(posterior_mean))
        multiplier = max(float(self.uncertainty_multiplier), 0.0)
        calibrated_draw = posterior_mean + multiplier * (raw_draw - posterior_mean)
        effective_temperature = min(max(float(temperature), 0.0), 1.0)
        tempered_draw = posterior_mean + effective_temperature * (
            calibrated_draw - posterior_mean
        )
        gains = x @ tempered_draw * self.reward_scale
        if not np.isfinite(gains).all():
            raise ValueError("reward-only Thompson sample contains non-finite values")
        return gains


@dataclass
class BayesianNodeValueSurrogate:
    """Run-local Bayesian linear model for exact-10 BNVS node values."""

    target_type: str = "incumbent_h1"
    selection_logger: SelectionLogger | None = None
    prior_precision: float = 4.0
    intercept_precision: float = 0.25
    prior_shape: float = 2.0
    prior_scale: float = 1.0
    include_root_draft: bool = False

    def __post_init__(self):
        if self.target_type not in {"incumbent_h1", "local_gain"}:
            raise ValueError(f"unsupported BNVS target type: {self.target_type}")
        if self.include_root_draft and self.target_type != "incumbent_h1":
            raise ValueError("root draft requires incumbent_h1")
        self.feature_names = BNVS_ROOT_FEATURE_NAMES if self.include_root_draft else BNVS_FEATURE_NAMES
        self.feature_schema_version = (
            BNVS_ROOT_FEATURE_SCHEMA_VERSION if self.include_root_draft else BNVS_FEATURE_SCHEMA_VERSION
        )
        self.posterior_mean: np.ndarray | None = None
        self.posterior_precision_inverse: np.ndarray | None = None
        self.posterior_shape = 0.0
        self.posterior_scale = 0.0
        self.utility_scale = 1.0
        self.model_version: str | None = None
        self.fitted_transition_count = -1
        self.last_fit_wall_time_sec = 0.0
        self.training_transition_ids: list[str] = []
        self.fit_diagnostics: dict = {}

    @staticmethod
    def _normal_records(
        records: Sequence[TransitionRecord],
    ) -> list[TransitionRecord]:
        return [record for record in records if record.operator in NORMAL_OPERATORS]

    @staticmethod
    def _effective_records(
        records: Sequence[TransitionRecord],
    ) -> list[TransitionRecord]:
        """Replace failed normal outcomes using only this admitted history prefix."""
        effective: list[TransitionRecord] = []
        normal_indices: dict[str, int] = {}
        by_child: dict[str, TransitionRecord] = {}
        for record in records:
            if record.operator in NORMAL_OPERATORS:
                if record.decision_id in normal_indices:
                    continue
                normal_indices[record.decision_id] = len(effective)
            effective.append(record)
            if (
                record.operator == "debug"
                and record.valid
                and record.child_metric_raw is not None
                and math.isfinite(float(record.child_metric_raw))
                and record.child_utility is not None
                and math.isfinite(float(record.child_utility))
            ):
                origin = by_child.get(record.parent_node_id)
                while origin is not None and origin.operator == "debug":
                    origin = by_child.get(origin.parent_node_id)
                if origin is not None and origin.operator in NORMAL_OPERATORS:
                    index = normal_indices[origin.decision_id]
                    normal = effective[index]
                    # The first valid outcome wins, including direct success.
                    if not normal.valid:
                        improvement = (
                            record.child_utility - normal.parent_utility
                            if normal.parent_utility is not None
                            and math.isfinite(float(normal.parent_utility)) else None
                        )
                        effective[index] = replace(
                            normal,
                            child_node_id=record.child_node_id,
                            child_metric_raw=record.child_metric_raw,
                            child_utility=record.child_utility,
                            improvement=improvement,
                            valid=True,
                            metadata={
                                **(normal.metadata or {}),
                                "debug_repair": {
                                    "normal_transition_id": normal.transition_id,
                                    "normal_decision_id": normal.decision_id,
                                    "debug_transition_id": record.transition_id,
                                    "available_at": record.timestamp,
                                },
                            },
                        )
            if record.child_node_id is not None:
                by_child[record.child_node_id] = record
        return effective

    @staticmethod
    def _repair_sources(records: Sequence[TransitionRecord]) -> list[dict]:
        return [record.metadata["debug_repair"] for record in records
                if record.operator in NORMAL_OPERATORS
                and (record.metadata or {}).get("debug_repair")]

    def debug_repairs_changed(self, records: Sequence[TransitionRecord]) -> bool:
        return self._repair_sources(self._effective_records(records)) != self.fit_diagnostics.get(
            "debug_repairs", []
        )

    @staticmethod
    def _scale(records: Sequence[TransitionRecord]) -> float:
        has_valid_value = any(
            record.valid
            and record.child_utility is not None
            and math.isfinite(float(record.child_utility))
            for record in records
        )
        return reward_scale_from_records(records) if has_valid_value else 1.0

    def _design(self, features: dict[str, float]) -> np.ndarray:
        vectorizer = vectorize_bnvs_root_features if self.include_root_draft else vectorize_bnvs_features
        values = vectorizer(
            features,
            utility_scale=self.utility_scale,
        )
        return np.asarray([1.0, *values], dtype=float)

    def _target(self, record: TransitionRecord) -> float:
        if self.include_root_draft:
            incumbent = record.selection_features.get("incumbent_at_decision")
            if incumbent is None or not math.isfinite(float(incumbent)):
                raise ValueError("H1 target unavailable: no pre-action incumbent")
            if not record.valid or record.child_utility is None:
                return -1.0
            return float(np.clip((record.child_utility - incumbent) / self.utility_scale, -1.0, 1.0))
        if not record.valid or record.child_utility is None:
            return -1.0
        if self.target_type == "local_gain":
            if record.improvement is None or not math.isfinite(float(record.improvement)):
                return -1.0
            raw = float(record.improvement) / self.utility_scale
        else:
            pre_action_best = record.selection_features.get("global_best_utility")
            if pre_action_best is None:
                pre_action_best = record.parent_utility
            if pre_action_best is None or not math.isfinite(float(pre_action_best)):
                return -1.0
            raw = (
                float(record.child_utility) - float(pre_action_best)
            ) / self.utility_scale
        return float(min(max(raw, -1.0), 1.0))

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        """Fit on all observed normal outcomes; an empty history yields the prior."""
        started = time.perf_counter()
        effective_records = self._effective_records(records)
        observed = self._normal_records(effective_records)
        scale_records = observed
        if self.include_root_draft:
            # This filter is deliberately local to BNVS; draft remains a bypass
            # operator for every legacy selector.
            scale_records = [r for r in effective_records if r.operator in (*NORMAL_OPERATORS, "draft")]
            if any((r.metadata or {}).get("feature_schema_version") != self.feature_schema_version
                   for r in scale_records):
                raise ValueError("incompatible root-draft observation schema")
            observed = [r for r in scale_records if (r.metadata or {}).get("target_available", False)]
        self.utility_scale = self._scale(scale_records)
        design = np.asarray(
            [self._design(record.selection_features) for record in observed],
            dtype=float,
        ).reshape(len(observed), len(self.feature_names) + 1)
        targets = np.asarray([self._target(record) for record in observed], dtype=float)
        if not np.isfinite(design).all() or not np.isfinite(targets).all():
            raise ValueError("BNVS model input contains a non-finite value")

        dimension = len(self.feature_names) + 1
        prior_precision = np.eye(dimension, dtype=float) * max(
            float(self.prior_precision), 1e-9
        )
        prior_precision[0, 0] = max(float(self.intercept_precision), 1e-9)
        posterior_precision = prior_precision + design.T @ design
        posterior_precision_inverse = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_precision_inverse @ design.T @ targets
        posterior_shape = max(float(self.prior_shape), 1e-9) + len(observed) / 2.0
        residual = float(
            targets @ targets
            - posterior_mean @ posterior_precision @ posterior_mean
        )
        posterior_scale = max(
            float(self.prior_scale) + 0.5 * max(residual, 0.0),
            1e-12,
        )

        self.posterior_mean = posterior_mean
        self.posterior_precision_inverse = (
            posterior_precision_inverse + posterior_precision_inverse.T
        ) / 2.0
        self.posterior_shape = posterior_shape
        self.posterior_scale = posterior_scale
        self.training_transition_ids = [record.transition_id for record in observed]
        label = "h1" if self.target_type == "incumbent_h1" else "local"
        self.model_version = f"bnvs-{label}-{len(observed)}-{uuid.uuid4().hex[:12]}"
        self.fitted_transition_count = len(records)
        self.last_fit_wall_time_sec = max(time.perf_counter() - started, 0.0)
        train_mean = design @ posterior_mean
        train_rmse = (
            float(np.sqrt(np.mean((train_mean - targets) ** 2)))
            if len(observed) else None
        )
        self.fit_diagnostics = {
            "training_count": len(observed),
            "training_rmse": train_rmse,
            "utility_scale": self.utility_scale,
            "target_type": self.target_type,
            "debug_repairs": self._repair_sources(effective_records),
            "posterior_shape": self.posterior_shape,
            "posterior_scale": self.posterior_scale,
            "prior_precision": float(self.prior_precision),
            "intercept_precision": float(self.intercept_precision),
        }
        if self.selection_logger is not None:
            self.selection_logger.append_model_fit({
                "model_version": self.model_version,
                "model_type": "conjugate_bayesian_linear_bnvs",
                "target_type": self.target_type,
                "training_transition_ids": self.training_transition_ids,
                "feature_schema_version": self.feature_schema_version,
                "feature_names": list(self.feature_names),
                "fit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fit_wall_time_sec": self.last_fit_wall_time_sec,
                "fit_diagnostics": {"value": self.fit_diagnostics},
            })

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.posterior_mean is None or self.posterior_precision_inverse is None:
            raise SurrogateNotFitted("BNVS model has not been fitted")
        return self.posterior_mean, self.posterior_precision_inverse

    def _design_matrix(self, options: Sequence[ExpansionOption]) -> np.ndarray:
        matrix = np.asarray(
            [self._design(dict(option.state_features)) for option in options],
            dtype=float,
        )
        if not np.isfinite(matrix).all():
            raise ValueError("BNVS prediction input contains a non-finite value")
        return matrix

    def predict_values(
        self,
        options: Sequence[ExpansionOption],
    ) -> tuple[np.ndarray, np.ndarray]:
        posterior_mean, precision_inverse = self._require_fitted()
        design = self._design_matrix(options)
        means = design @ posterior_mean
        noise_variance = self.posterior_scale / max(
            self.posterior_shape - 1.0,
            1e-9,
        )
        epistemic = np.einsum(
            "ij,jk,ik->i", design, precision_inverse, design
        )
        stds = np.sqrt(np.maximum((1.0 + epistemic) * noise_variance, 0.0))
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("BNVS prediction is non-finite")
        return means, stds

    def sample_values(
        self,
        options: Sequence[ExpansionOption],
        *,
        seed: int,
        temperature: float = 0.35,
    ) -> np.ndarray:
        posterior_mean, precision_inverse = self._require_fitted()
        design = self._design_matrix(options)
        rng = np.random.default_rng(int(seed))
        precision_sample = rng.gamma(
            shape=self.posterior_shape,
            scale=1.0 / self.posterior_scale,
        )
        covariance = precision_inverse / max(float(precision_sample), 1e-12)
        covariance = (covariance + covariance.T) / 2.0
        cholesky = np.linalg.cholesky(
            covariance + np.eye(covariance.shape[0], dtype=float) * 1e-12
        )
        sampled = posterior_mean + cholesky @ rng.standard_normal(len(posterior_mean))
        effective_temperature = min(max(float(temperature), 0.0), 1.0)
        tempered = posterior_mean + effective_temperature * (sampled - posterior_mean)
        values = design @ tempered
        if not np.isfinite(values).all():
            raise ValueError("BNVS Thompson sample is non-finite")
        return values
