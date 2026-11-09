"""Gaussian statistics and independently scrambled Genz orthant labels.

All coordinates are physical distances along one ray. The SE amplitude is
``sigma`` and its correlation length is ``length_scale``. The learned target
only sees standardized means and correlations, so both birth models share
one mathematical probability function.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri_exp
from scipy.stats import qmc


@dataclass(frozen=True)
class LabelEstimate:
    probability: float
    log_probability: float
    error: float
    relative_error: float
    samples: int
    status: str


def _positions(positions, length_scale, sigma):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be a finite one-dimensional array")
    if not np.isfinite(length_scale) or length_scale <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma and length_scale must be finite and positive")
    return positions / length_scale


def se_covariance(positions, length_scale=1.0, sigma=1.0):
    q = _positions(positions, length_scale, sigma)
    return sigma ** 2 * np.exp(-0.5 * (q[:, None] - q[None, :]) ** 2)


def se_features(positions, length_scale=1.0, sigma=1.0, conditioned=False):
    """SE power-series feature rows, optionally removing birth value/slope.

    This Gram construction avoids subtracting nearly equal covariances near
    the birth point. The series is extended far beyond its Poisson peak;
    discarded diagonal mass is checked rather than repaired with jitter.
    ``conditioned=True`` retains terms k>=2, exactly the F0,G0 Schur model
    on a collinear ray (transverse birth derivatives have zero covariance).
    """
    q = _positions(positions, length_scale, sigma)
    maximum = float(np.max(np.abs(q), initial=0.0))
    if maximum > 20.0:
        raise ValueError("feature reference supports |distance/ell| <= 20")
    degree = max(40, int(math.ceil(maximum ** 2 + 14 * maximum + 48)))
    features = np.empty((len(q), degree + 1), dtype=np.float64)
    features[:, 0] = np.exp(-0.5 * q * q)
    for k in range(1, degree + 1):
        features[:, k] = features[:, k - 1] * q / math.sqrt(k)
    if np.any(np.abs(np.sum(features * features, axis=1) - 1.0) > 5e-14):
        raise ArithmeticError("SE feature truncation/roundoff exceeds reference tolerance")
    return sigma * features[:, 2 if conditioned else 0:]


def surface_conditioned_statistics(positions, mean_values, origin_mean,
                                   origin_directional_derivative, outward_derivative,
                                   length_scale=1.0, sigma=1.0):
    q = _positions(positions, length_scale, sigma)
    means = np.asarray(mean_values, dtype=np.float64)
    if means.shape != q.shape or not np.all(np.isfinite(means)):
        raise ValueError("mean_values must match positions")
    if not all(np.isfinite(x) for x in (origin_mean, origin_directional_derivative, outward_derivative)):
        raise ValueError("birth statistics must be finite")
    c = np.exp(-0.5 * q * q)
    mean = means - c * origin_mean + length_scale * q * c * (
        outward_derivative - origin_directional_derivative)
    features = se_features(positions, length_scale, sigma, conditioned=True)
    return mean, features @ features.T


def standardized_statistics(mean, covariance):
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("Gaussian dimensions do not agree")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Gaussian statistics must be finite")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-13 * max(1.0, np.max(np.abs(covariance)))):
        raise ValueError("covariance must be symmetric")
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError("standardization requires strictly positive variances")
    sd = np.sqrt(variances)
    correlation = covariance / sd[:, None] / sd[None, :]
    return mean / sd, correlation


def feature_factor(features):
    """Lower factor from feature QR; no covariance perturbation or jitter."""
    features = np.asarray(features, dtype=np.float64)
    factor = np.linalg.qr(features.T, mode="reduced")[1].T
    signs = np.where(np.diag(factor) < 0.0, -1.0, 1.0)
    return factor * signs[None, :]


def _from_log(logp, relative_error, samples, status="ok"):
    probability = float(np.exp(logp))
    return LabelEstimate(probability, float(logp), probability * relative_error,
                         float(relative_error), int(samples), status)


def _analytic(z, correlation):
    n = len(z)
    if n == 1 or np.array_equal(correlation, np.eye(n)):
        return _from_log(float(np.sum(log_ndtr(z))), 4e-15, 0)
    if np.array_equal(correlation, np.ones((n, n))):
        return _from_log(float(log_ndtr(np.min(z))), 4e-15, 0)
    if n == 2 and z[0] == 0.0 and z[1] == 0.0:
        p = 0.25 + math.asin(float(correlation[0, 1])) / (2 * math.pi)
        return _from_log(math.log(p) if p > 0 else -math.inf, 4e-15, 0)
    return None


def _genz_replicates(z, factors, power, replicates, seed):
    """Return [replicate,batch] log estimates, with stable Gaussian tails."""
    batch, n = z.shape
    samples = 1 << power
    result = np.empty((replicates, batch))
    for replicate in range(replicates):
        engine = qmc.Sobol(max(1, n - 1), scramble=True,
                           seed=(int(seed) + 104729 * replicate) % (2 ** 32))
        uniforms = engine.random_base2(power)
        logu = np.log(np.clip(uniforms, np.finfo(float).tiny, 1 - np.finfo(float).eps))
        latent = np.zeros((batch, samples, max(1, n - 1)))
        log_weight = np.zeros((batch, samples))
        for i in range(n):
            conditional = np.broadcast_to(z[:, i, None], (batch, samples)).copy()
            for j in range(i):
                conditional += factors[:, i, j, None] * latent[:, :, j]
            sd = factors[:, i, i, None]
            nonzero = sd[:, 0] > 0
            log_survival = np.empty_like(conditional)
            log_survival[nonzero] = log_ndtr(conditional[nonzero] / sd[nonzero])
            log_survival[~nonzero] = np.where(conditional[~nonzero] > 0, 0.0, -np.inf)
            log_weight += log_survival
            if i < n - 1:
                draw = -ndtri_exp(logu[None, :, i] + log_survival)
                latent[:, :, i] = np.where(np.isfinite(draw), draw, 0.0)
        result[replicate] = logsumexp(log_weight, axis=1) - math.log(samples)
    return result


def _genz_replicates_cuda(z, factors, power, replicates, seed):
    """The same Genz transform in CUDA float64, including logarithmic tails."""
    import torch
    batch, n = z.shape
    samples = 1 << power
    uniforms = np.stack([qmc.Sobol(max(1, n - 1), scramble=True,
                                  seed=(int(seed) + 104729 * r) % (2 ** 32)).random_base2(power)
                         for r in range(replicates)])
    uniforms = np.clip(uniforms, np.finfo(float).tiny, 1 - np.finfo(float).eps)
    logu = torch.as_tensor(np.log(uniforms), dtype=torch.float64, device='cuda')
    means = torch.as_tensor(z, dtype=torch.float64, device='cuda')
    lower = torch.as_tensor(factors, dtype=torch.float64, device='cuda')
    log_weight = torch.zeros((replicates, batch, samples), dtype=torch.float64, device='cuda')
    latent = []
    for i in range(n):
        conditional = means[None, :, i, None].expand(replicates, batch, samples).clone()
        for j in range(i):
            conditional += lower[None, :, i, j, None] * latent[j]
        sd = lower[None, :, i, i, None]
        log_survival = torch.special.log_ndtr(conditional / torch.where(sd > 0, sd, 1.0))
        log_survival = torch.where(sd > 0, log_survival,
                                   torch.where(conditional > 0, 0.0, -torch.inf))
        log_weight += log_survival
        if i < n - 1:
            logp = logu[:, None, :, i] + log_survival
            # ndtri(exp(logp)) is safe throughout the representable range.
            # Beyond it, solve logPhi(-y)=logp directly, never erfinv(2p-1).
            draw = -torch.special.ndtri(torch.exp(logp.clamp_min(-700.0)))
            tail = (logp < -700.0) & torch.isfinite(logp)
            if bool(torch.any(tail)):
                target = logp[tail]
                magnitude = torch.sqrt(-2 * target - torch.log(-2 * target) - math.log(2 * math.pi))
                for _ in range(3):
                    magnitude += (torch.special.log_ndtr(-magnitude) - target) / (magnitude + 1 / magnitude)
                draw[tail] = magnitude
            latent.append(torch.where(torch.isfinite(logp), draw, 0.0))
    return (torch.logsumexp(log_weight, dim=2) - math.log(samples)).cpu().numpy()


def orthant_qmc_batch(z, correlations, *, factors=None, seed=12345,
                      relative_tolerance=5e-4, initial_power=9, max_power=14,
                      replicates=8, batch_size=128, backend='cpu'):
    """Positive-orthant labels for standardized Gaussian rows of one size.

    Error estimates are three standard errors across independent Sobol
    scramblings, not rigorous bounds. Failing the relative target returns an
    explicit status; no probability floor, covariance jitter, or tail clamp
    is used. Optional factors preserve near-birth feature precision that a
    covariance Cholesky decomposition can lose.
    """
    z = np.asarray(z, dtype=np.float64)
    correlations = np.asarray(correlations, dtype=np.float64)
    if z.ndim != 2 or not 1 <= z.shape[1] <= 5 or correlations.shape != (len(z), z.shape[1], z.shape[1]):
        raise ValueError("expected z[B,n] and correlation[B,n,n], 1<=n<=5")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(correlations)):
        raise ValueError("finite Gaussian parameters required")
    if relative_tolerance <= 0 or not 1 <= initial_power <= max_power <= 22 or replicates < 2:
        raise ValueError("invalid QMC precision settings")
    if backend not in ('cpu', 'cuda'):
        raise ValueError("backend must be cpu or cuda")
    if factors is not None:
        factors = np.asarray(factors, dtype=np.float64)
        if factors.shape != correlations.shape:
            raise ValueError("factor dimensions do not agree")
    n = z.shape[1]
    labels = [None] * len(z)
    prepared_z, prepared_l, indices = [], [], []
    for index, (mean, corr) in enumerate(zip(z, correlations)):
        if not np.allclose(corr, corr.T, atol=2e-13, rtol=0) or not np.allclose(np.diag(corr), 1.0, atol=2e-13, rtol=0):
            raise ValueError("correlation must be symmetric with unit diagonal")
        if np.linalg.eigvalsh(corr)[0] < -2e-13:
            raise ValueError("correlation is not positive semidefinite")
        analytic = _analytic(mean, corr)
        if analytic is not None:
            labels[index] = analytic
            continue
        order = np.argsort(mean, kind="stable")
        try:
            if factors is None:
                lower = np.linalg.cholesky(corr[np.ix_(order, order)])
            else:
                factor = factors[index]
                if not np.allclose(factor @ factor.T, corr, rtol=2e-12, atol=2e-13):
                    raise ValueError("supplied factor does not reproduce correlation")
                lower = feature_factor(factor[order])
        except np.linalg.LinAlgError:
            labels[index] = LabelEstimate(math.nan, math.nan, math.inf, math.inf, 0, "matrix_factorization_failed")
            continue
        prepared_z.append(mean[order]); prepared_l.append(lower); indices.append(index)
    if not indices:
        return labels
    prepared_z, prepared_l = np.asarray(prepared_z), np.asarray(prepared_l)
    pending = np.arange(len(indices))
    for power in range(initial_power, max_power + 1):
        next_pending = []
        for begin in range(0, len(pending), batch_size):
            selection = pending[begin:begin + batch_size]
            estimator = _genz_replicates_cuda if backend == 'cuda' else _genz_replicates
            estimates = estimator(prepared_z[selection], prepared_l[selection], power, replicates, seed)
            reference = np.max(estimates, axis=0)
            finite = np.isfinite(reference)
            scaled = np.zeros_like(estimates)
            scaled[:, finite] = np.exp(estimates[:, finite] - reference[finite])
            average = np.mean(scaled, axis=0)
            relative_error = np.full(len(selection), np.inf)
            relative_error[finite] = np.maximum(8e-15, 3 * np.std(scaled[:, finite], axis=0, ddof=1)
                                                / math.sqrt(replicates) / average[finite])
            for local, prepared_index in enumerate(selection):
                if not finite[local]:
                    label = LabelEstimate(0.0, -math.inf, math.inf, math.inf,
                                          replicates * (1 << power), "zero_or_unresolved_probability")
                else:
                    logp = reference[local] + math.log(average[local])
                    success = relative_error[local] <= relative_tolerance
                    label = _from_log(logp, relative_error[local], replicates * (1 << power),
                                      "ok" if success else "relative_error_target_not_met")
                labels[indices[prepared_index]] = label
                if label.status != "ok" and power < max_power:
                    next_pending.append(prepared_index)
        pending = np.asarray(next_pending, dtype=np.int64)
        if len(pending) == 0:
            break
    return labels


def orthant_qmc(z, correlation, **options):
    factor = options.pop("factor", None)
    return orthant_qmc_batch(np.asarray(z)[None, :], np.asarray(correlation)[None, :, :],
                             factors=None if factor is None else np.asarray(factor)[None, :, :],
                             **options)[0]


def positive_origin_probability(z, correlation, **options):
    """P(Y[1:]>0 | Y[0]>0), retaining the exterior birth as a sign event."""
    z = np.asarray(z, dtype=np.float64)
    joint = orthant_qmc(z, correlation, **options)
    if joint.status != "ok":
        return joint
    return _from_log(joint.log_probability - float(log_ndtr(z[0])),
                     joint.relative_error, joint.samples)


def family_split(group_id, seed=12345, train_fraction=6 / 7, validation_fraction=1 / 14):
    """Stable, order-independent family split; prior/surface pairs share IDs."""
    if train_fraction < 0 or validation_fraction < 0 or train_fraction + validation_fraction > 1:
        raise ValueError("invalid split fractions")
    # SplitMix64 integer mixing, with no dependence on sample row order.
    mask = (1 << 64) - 1
    x = (int(group_id) + int(seed) + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    value = x / float(1 << 64)
    return 0 if value < train_fraction else 1 if value < train_fraction + validation_fraction else 2
