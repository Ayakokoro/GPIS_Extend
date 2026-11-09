"""Validated CPU/batch inference, with no numerical-integration fallback.

Full-rank multivariate cases use the learned approximation. Deterministic,
one-dimensional, duplicate, and rank-one cases have explicit analytic handling.
Other singular Gaussians and inputs outside the recorded training domain raise
DomainError unless the caller explicitly opts into domain extrapolation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.special import log_ndtr
import torch

try:
    from .model import OrthantNet
except ImportError:
    from model import OrthantNet


class DomainError(ValueError):
    """A mathematically valid input is outside this experiment's supported domain."""


@dataclass(frozen=True)
class PreparedGaussian:
    z: np.ndarray
    corr: np.ndarray
    dimension: int
    log_probability_if_deterministic: Optional[float] = None
    method: str = "neural"


def _log_difference(log_a: float, log_b: float) -> float:
    if log_b == -np.inf:
        return float(log_a)
    if log_a <= log_b:
        return -np.inf
    return float(log_a + np.log(-np.expm1(log_b - log_a)))


def _normal_interval_log_probability(lower: float, upper: float) -> float:
    if lower >= upper:
        return -np.inf
    if lower >= 0:
        return _log_difference(float(log_ndtr(-lower)), float(log_ndtr(-upper)))
    return _log_difference(float(log_ndtr(upper)), float(log_ndtr(lower)))


def prepare_gaussian(mean, covariance, max_dim: int = 5,
                     singular_tolerance: float = 1e-12) -> PreparedGaussian:
    """Validate PSD, standardize, remove constants and merge duplicate noises.

Zero covariance means a deterministic strict-sign event. Tiny positive physical
variances are not treated as zero; this preserves invariance to unit changes.
"""
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 1 or not 1 <= len(mean) <= max_dim:
        raise ValueError(f"mean must be a vector with 1..{max_dim} entries")
    if covariance.shape != (len(mean), len(mean)):
        raise ValueError("covariance shape does not match mean")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Gaussian parameters must be finite")
    magnitude = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-11 * magnitude):
        raise ValueError("covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance / magnitude)
    if eigenvalues[0] < -1e-10:
        raise ValueError("covariance is not positive semidefinite")
    variance = covariance.diagonal()
    if (variance < 0).any():
        raise ValueError("negative marginal variance")
    deterministic = variance == 0
    if deterministic.any() and np.any(covariance[deterministic] != 0):
        raise ValueError("zero-variance coordinates must have zero covariance")
    if np.any(mean[deterministic] <= 0):
        return PreparedGaussian(np.empty(0), np.empty((0, 0)), 0, -np.inf, "deterministic-zero")
    active = ~deterministic
    if not active.any():
        return PreparedGaussian(np.empty(0), np.empty((0, 0)), 0, 0.0, "deterministic-one")
    sd = np.sqrt(variance[active])
    z = mean[active] / sd
    if not np.isfinite(z).all():
        raise DomainError("standardized mean exceeds float64 range; no integration fallback")
    # Divide separately so a product of very large standard deviations does
    # not overflow merely because the physical field uses different units.
    corr = covariance[np.ix_(active, active)] / sd[:, None] / sd[None, :]
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    if np.max(np.abs(corr)) > 1 + 1e-10:
        raise ValueError("standardized covariance has invalid correlation")
    corr = np.clip(corr, -1, 1)
    # For identical standardized noise, the smallest z is the strictest sign.
    retained = []
    for index in np.argsort(z, kind="stable"):
        if not any(corr[index, old] >= 1 - 2e-14 for old in retained):
            retained.append(int(index))
    z, corr = z[retained], corr[np.ix_(retained, retained)]
    n = len(z)
    if n == 1:
        return PreparedGaussian(z, corr, n, float(log_ndtr(z[0])), "analytic-one-dimensional")
    values, vectors = np.linalg.eigh(corr)
    if values[0] < -1e-9:
        raise ValueError("standardized covariance is not positive semidefinite")
    if values[-2] <= min(singular_tolerance, 64 * np.finfo(float).eps * n):
        coefficients = vectors[:, -1] * np.sqrt(max(0.0, values[-1]))
        lower, upper = -np.inf, np.inf
        for location, coefficient in zip(z, coefficients):
            if coefficient > 0:
                lower = max(lower, -location / coefficient)
            elif coefficient < 0:
                upper = min(upper, -location / coefficient)
            elif location <= 0:
                return PreparedGaussian(z, corr, n, -np.inf, "analytic-rank-one-zero")
        return PreparedGaussian(z, corr, n,
                                _normal_interval_log_probability(lower, upper), "analytic-rank-one")
    if values[0] <= singular_tolerance:
        raise DomainError("rank-deficient or unresolved near-singular Gaussian; no integration fallback")
    return PreparedGaussian(z, corr, n)


class OrthantPredictor:
    """One shared network for original and equality-conditioned Gaussians."""
    approximate = True
    numerical_fallback = False

    def __init__(self, checkpoint, device: str = "cpu", allow_out_of_domain: bool = False):
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("unsupported orthant checkpoint format")
        self.model = OrthantNet(**state["model_config"]).to(self.device).eval()
        self.model.load_state_dict(state["model_state"])
        self.domain = state.get("domain", {})
        self.allow_out_of_domain = allow_out_of_domain

    def _validate_domain(self, prepared: PreparedGaussian):
        if self.allow_out_of_domain or prepared.log_probability_if_deterministic is not None:
            return
        low = self.domain.get("minimum_z", -np.inf)
        high = self.domain.get("maximum_z", np.inf)
        if np.min(prepared.z) < low - 1e-8 or np.max(prepared.z) > high + 1e-8:
            raise DomainError(f"standardized mean outside trained [{low:g}, {high:g}] domain; no fallback")
        minimum = self.domain.get("minimum_correlation_eigenvalue", 0)
        if np.linalg.eigvalsh(prepared.corr)[0] < 0.9 * minimum:
            raise DomainError("correlation is more singular than the training domain; no fallback")

    def predict_standardized_batch(self, z, corr, dim, batch_size: int = 8192,
                                   assume_validated: bool = False) -> np.ndarray:
        """Return log probabilities for padded standardized parameters.

        ``assume_validated`` is an explicit fast path for the experiment's
        generated/evaluated dataset, bypassing PSD and training-domain checks.
        Ordinary application inputs should use predict()/predict_batch().
        """
        z, corr, dim = np.asarray(z), np.asarray(corr), np.asarray(dim, dtype=np.int64)
        if z.ndim != 2 or z.shape[1] != self.model.max_dim or corr.shape != (len(z), z.shape[1], z.shape[1]):
            raise ValueError("invalid padded standardized batch shape")
        if dim.shape != (len(z),) or np.any((dim < 1) | (dim > self.model.max_dim)):
            raise ValueError("dimensions must be 1..max_dim")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not assume_validated:
            return self.predict_log_probability_batch([row[:n] for row, n in zip(z, dim)],
                                                       [matrix[:n, :n] for matrix, n in zip(corr, dim)])
        predictions = []
        with torch.inference_mode():
            for first in range(0, len(z), batch_size):
                last = first + batch_size
                result = self.model(torch.as_tensor(z[first:last], dtype=torch.float32, device=self.device),
                                    torch.as_tensor(corr[first:last], dtype=torch.float32, device=self.device),
                                    torch.as_tensor(dim[first:last], dtype=torch.long, device=self.device))
                predictions.append(result.cpu().numpy().astype(np.float64))
        return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float64)

    def predict_log_probability_batch(self, means: Sequence, covariances: Sequence) -> np.ndarray:
        if len(means) != len(covariances):
            raise ValueError("mean and covariance batch lengths differ")
        prepared = [prepare_gaussian(m, c, self.model.max_dim) for m, c in zip(means, covariances)]
        result = np.empty(len(prepared), dtype=np.float64)
        indices = []
        for index, item in enumerate(prepared):
            self._validate_domain(item)
            if item.log_probability_if_deterministic is not None:
                result[index] = item.log_probability_if_deterministic
            else:
                indices.append(index)
        if indices:
            z = np.zeros((len(indices), self.model.max_dim), dtype=np.float32)
            corr = np.repeat(np.eye(self.model.max_dim, dtype=np.float32)[None], len(indices), axis=0)
            dim = np.empty(len(indices), dtype=np.int64)
            for row, index in enumerate(indices):
                item = prepared[index]
                dim[row] = item.dimension
                z[row, :item.dimension] = item.z
                corr[row, :item.dimension, :item.dimension] = item.corr
            with torch.inference_mode():
                values = self.model(torch.as_tensor(z, device=self.device),
                                    torch.as_tensor(corr, device=self.device),
                                    torch.as_tensor(dim, device=self.device)).cpu().numpy()
            for index, value in zip(indices, values):
                if not np.isfinite(value):
                    raise ArithmeticError("network produced a non-finite log probability")
                # Enforce the mathematical marginal upper bound in float64,
                # independently of the float32 bound used inside the network.
                result[index] = min(float(value), float(np.min(log_ndtr(prepared[index].z))))
        return result

    def predict_log_probability(self, mean, covariance) -> float:
        return float(self.predict_log_probability_batch([mean], [covariance])[0])

    @staticmethod
    def _exponentiate(values):
        finite = np.isfinite(values)
        if np.any(values[finite] < np.log(np.nextafter(0.0, 1.0))):
            raise ArithmeticError("probability underflows float64; use predict_log_probability")
        return np.exp(values)

    def predict_batch(self, means: Sequence, covariances: Sequence) -> np.ndarray:
        return self._exponentiate(self.predict_log_probability_batch(means, covariances))

    def predict(self, mean, covariance) -> float:
        return float(self.predict_batch([mean], [covariance])[0])

    def predict_conditioned(self, mean, covariance) -> float:
        """Input already conditioned on equalities, e.g. F0=0 and known G0."""
        return self.predict(mean, covariance)

    def predict_given_positive_log(self, mean, covariance, origin_index: int = 0) -> float:
        """P(all coordinates positive | one origin coordinate positive).

        The joint, including the origin, must have at most five coordinates.
        This does not substitute F0=0 for the camera's positive-sign event.
        """
        mean, covariance = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
        # Full validation precedes accessing the conditioning marginal.
        prepare_gaussian(mean, covariance, self.model.max_dim)
        if not 0 <= origin_index < len(mean):
            raise ValueError("invalid origin index")
        variance = covariance[origin_index, origin_index]
        if variance == 0:
            if mean[origin_index] <= 0:
                raise ValueError("cannot condition on a zero-probability positive origin")
            denominator = 0.0
        else:
            denominator = float(log_ndtr(mean[origin_index] / np.sqrt(variance)))
        return min(0.0, self.predict_log_probability(mean, covariance) - denominator)

    def predict_given_positive(self, mean, covariance, origin_index: int = 0) -> float:
        value = self.predict_given_positive_log(mean, covariance, origin_index)
        return float(self._exponentiate(np.asarray([value]))[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("input", help="JSON containing mean and covariance, already equality-conditioned if desired")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-origin", type=int)
    parser.add_argument("--allow-out-of-domain", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    predictor = OrthantPredictor(args.checkpoint, args.device, args.allow_out_of_domain)
    if args.positive_origin is None:
        log_probability = predictor.predict_log_probability(payload["mean"], payload["covariance"])
    else:
        log_probability = predictor.predict_given_positive_log(payload["mean"], payload["covariance"], args.positive_origin)
    print(json.dumps({"log_probability": log_probability, "approximate": True,
                      "numerical_integration_fallback": False}))


if __name__ == "__main__":
    main()
