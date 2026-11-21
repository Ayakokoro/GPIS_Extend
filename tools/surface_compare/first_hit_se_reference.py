"""Independent SE first-downcross reference on an evenly sampled ray SDF profile.

The fine and coarse results use the same Gaussian realizations. A miss has
distance +inf. The finite mesh can miss crossings between its sample points.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfftfreq
from scipy.special import log_ndtr, ndtri_exp


def _spectral_grid(t, length_scale):
    """Real Fourier series of a periodized unit-variance SE covariance."""
    dt = t[1]
    n = next_fast_len(len(t) + int(np.ceil(10 * length_scale / dt)), real=True)
    # Gaussian replicas are at least 10 length scales outside the ray.
    # The omitted spectrum beyond angular frequency 10/ell is negligible.
    omega = 2 * np.pi * rfftfreq(n, dt)
    modes = np.searchsorted(omega * length_scale, 10.)
    omega = omega[:modes]
    variance = np.exp(-0.5 * (omega * length_scale) ** 2)
    variance[1:] *= 2
    variance /= variance.sum()
    return n, omega, variance


def _responses(t, n, omega, variance, length_scale):
    """Covariance ratios for conditioning on the value and slope at zero."""
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    spectrum[0] = variance[0]
    spectrum[1:len(omega)] = 0.5 * variance[1:]
    value = irfft(spectrum, n=n, norm='forward')[:len(t)]
    slope_var = np.dot(variance, omega**2)
    spectrum[:len(omega)] *= -1j * omega
    slope = irfft(spectrum, n=n, norm='forward')[:len(t)] / slope_var
    # Validate the actual representation on this grid, including the slope law.
    target = np.exp(-0.5 * (t / length_scale) ** 2)
    error = max(np.max(np.abs(value - target)),
                np.max(np.abs(slope - t * target)) / length_scale,
                abs(slope_var * length_scale**2 - 1))
    if error > 5e-12:
        raise ArithmeticError(f'Spectral SE covariance check failed: {error:g}')
    return value, slope, float(error)


def _first_downcross(t, field, outgoing_slope):
    down = (field[:, :-1] > 0) & (field[:, 1:] <= 0)
    if outgoing_slope is not None:
        down[:, 0] = field[:, 1] <= 0
    rows = np.flatnonzero(down.any(axis=1))
    idx = down[rows].argmax(axis=1)
    left = field[rows, idx]
    right = field[rows, idx + 1]
    distance = np.full(len(field), np.inf)
    from_surface = (idx == 0) if outgoing_slope is not None else np.zeros(len(idx), bool)
    linear = ~from_surface
    fraction = np.empty(len(rows))
    fraction[linear] = left[linear] / (left[linear] - right[linear])
    if np.any(from_surface):
        # Interpolate the returning zero using F(0)=0, F'(0)>0, F(dt)<=0.
        rise = outgoing_slope * t[1]
        fraction[from_surface] = rise / (rise - right[from_surface])
    distance[rows] = t[idx] + fraction * (t[idx + 1] - t[idx])
    return distance


def sample_reference(t, mean, sigma, length_scale, samples, seed, *,
                     start_noise=None, start_noise_slope=None, exterior=False,
                     mean_start_slope=None, checks=None):
    """Return fine/coarse first-downcross arrays for an SE field plus mean.

    t starts at zero and has an even number of intervals. start_noise and
    start_noise_slope specify observations of the centered process, in world
    units. exterior=True conditions the initial total field to be positive.
    For a prescribed outgoing zero, mean_start_slope is the SDF's directional
    derivative there. In its absence it is estimated from the supplied mean.
    """
    t, mean = np.asarray(t, dtype=float), np.asarray(mean, dtype=float)
    if (t.ndim != 1 or mean.shape != t.shape or len(t) < 3 or len(t) % 2 != 1
            or t[0] != 0 or t[1] <= 0 or not np.all(np.isfinite(mean))
            or not np.allclose(np.diff(t), t[1], rtol=1e-9, atol=0)):
        raise ValueError('Expected uniform grid starting at zero with even interval count')
    if not (sigma > 0 and length_scale > 0 and t[1] <= length_scale / 4 and samples > 0):
        raise ValueError('Require positive parameters and grid spacing <= length_scale/4')
    if exterior and start_noise is not None:
        raise ValueError('Use either a fixed initial value or exterior truncation')
    outgoing_slope = None
    if start_noise is not None and mean[0] + start_noise == 0:
        if mean_start_slope is None:
            mean_start_slope = (-3 * mean[0] + 4 * mean[1] - mean[2]) / (2 * t[1])
        if start_noise_slope is None or mean_start_slope + start_noise_slope <= 0:
            raise ValueError('A ray launched at zero requires a fixed positive outgoing slope')
        outgoing_slope = mean_start_slope + start_noise_slope

    n, omega, variance = _spectral_grid(t, length_scale)
    value_response, slope_response, error = _responses(t, n, omega, variance, length_scale)
    if checks is not None:
        checks.update(mesh_step=float(t[1]), coarse_step=float(2*t[1]),
                      grid_points=len(t), period=n*t[1], fourier_frequencies=len(omega),
                      max_normalized_value_slope_covariance_error=error)
    scale = sigma * np.sqrt(variance)
    rng = np.random.default_rng(seed)
    fine = np.empty(samples)
    coarse = np.empty(samples)
    # Bound temporary spectral and spatial arrays for long rays.
    batch_size = max(1, min(256, 2_000_000 // n))
    for begin in range(0, samples, batch_size):
        size = min(batch_size, samples - begin)
        cosine = rng.standard_normal((size, len(omega))) * scale
        sine = rng.standard_normal((size, len(omega) - 1)) * scale[1:]
        spectrum = np.zeros((size, n // 2 + 1), dtype=complex)
        spectrum[:, 0] = cosine[:, 0]
        spectrum[:, 1:len(omega)] = 0.5 * (cosine[:, 1:] - 1j * sine)
        field = irfft(spectrum, n=n, axis=1, norm='forward')[:, :len(t)]

        if start_noise is not None:
            field += (start_noise - field[:, :1]) * value_response
        elif exterior:
            rows = np.flatnonzero(field[:, 0] + mean[0] <= 0)
            if len(rows):
                # Exact upper-tail inversion, including very rare exterior starts.
                log_tail = log_ndtr(mean[0] / sigma) + np.log1p(-rng.random(len(rows)))
                new_value = -sigma * ndtri_exp(log_tail)
                field[rows] += (new_value - field[rows, 0])[:, None] * value_response
        if start_noise_slope is not None:
            old_slope = np.sum(sine * omega[1:], axis=1)
            field += (start_noise_slope - old_slope)[:, None] * slope_response
        field += mean
        if start_noise is not None:
            field[:, 0] = mean[0] + start_noise
        fine[begin:begin+size] = _first_downcross(t, field, outgoing_slope)
        coarse[begin:begin+size] = _first_downcross(t[::2], field[:, ::2], outgoing_slope)
    return fine, coarse


def _nested_check(fine, coarse):
    fine_hit, coarse_hit = np.isfinite(fine), np.isfinite(coarse)
    both = fine_hit & coarse_hit
    change = np.abs(fine[both] - coarse[both])
    finite_fine = np.sort(fine[fine_hit])
    finite_coarse = np.sort(coarse[coarse_hit])
    thresholds = np.union1d(finite_fine, finite_coarse)
    fine_counts = np.searchsorted(finite_fine, thresholds, side='right')
    coarse_counts = np.searchsorted(finite_coarse, thresholds, side='right')
    return dict(fine_hits=int(np.sum(fine_hit)), coarse_hits=int(np.sum(coarse_hit)),
                fine_only_hits=int(np.sum(fine_hit & ~coarse_hit)),
                coarse_only_hits=int(np.sum(coarse_hit & ~fine_hit)),
                max_empirical_cdf_change=float(np.max(np.abs(fine_counts - coarse_counts),
                                                      initial=0) / len(fine)),
                median_abs_change_both_hit=float(np.median(change)) if len(change) else None,
                p99_abs_change_both_hit=float(np.quantile(change, .99)) if len(change) else None,
                max_abs_change_both_hit=float(np.max(change)) if len(change) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rays', type=Path, help='Probe ray and sample specification')
    parser.add_argument('mean', type=Path, help='Probe uniform SDF profile CSV')
    parser.add_argument('output', type=Path, help='Directory for reference.csv and reference_checks.json')
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--length-scale', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    spec = json.loads(args.rays.read_text(encoding='utf-8-sig'))
    profiles = {}
    with args.mean.open(newline='', encoding='utf-8-sig') as source:
        for row in csv.DictReader(source):
            profiles.setdefault(row['ray'], []).append(row)
    samples = spec['samples']
    summary = dict(master_seed=args.seed, sigma=args.sigma, length_scale=args.length_scale,
                   samples_per_ray=samples, rng='numpy PCG64', rays={})
    args.output.mkdir(exist_ok=True, parents=True)
    with (args.output / 'reference.csv').open('w', newline='', encoding='utf-8') as dest:
        writer = csv.writer(dest)
        writer.writerow(('ray', 'realization', 'distance', 'coarse_distance'))
        for i, ray in enumerate(spec['rays']):
            name = ray['name']
            rows = profiles[name]
            t = np.array([float(row['t']) for row in rows])
            if abs(t[-1] - ray['length']) > 1e-6 * float(ray['length']):
                raise ValueError(f'Incomplete mean profile for {name}')
            mean = np.array([float(row['mean']) for row in rows])
            if rows[0].get('mean_slope'):
                mean_slope = float(rows[0]['mean_slope'])
                slope_source = 'exported SDF directional derivative'
            else:
                # Exact for the cubic mean when these samples share a voxel.
                mean_slope = float(np.dot([-11, 18, -9, 2], mean[:4]) / (6*t[1]))
                slope_source = 'four-point forward difference'
            conditions = dict(exterior=True)
            if 'start_derivative' in ray:
                conditions = dict(start_noise=-mean[0], mean_start_slope=mean_slope,
                                  start_noise_slope=ray['start_derivative'] - mean_slope)
            seed = int(np.random.SeedSequence([args.seed, i]).generate_state(1)[0])
            checks = dict(seed=seed, start_mean=float(mean[0]), start_mean_slope=mean_slope,
                          mean_slope_source=slope_source, conditions=conditions)
            fine, coarse = sample_reference(t, mean, args.sigma, args.length_scale, samples,
                                            seed, **conditions, checks=checks)
            if np.any((fine <= 0) | (coarse <= 0)) or np.any(np.isnan(fine) | np.isnan(coarse)):
                raise ArithmeticError(f'Invalid first downcross for {name}')
            checks.update(_nested_check(fine, coarse))
            summary['rays'][name] = checks
            writer.writerows((name, j, format(f, '.12g') if np.isfinite(f) else '',
                              format(c, '.12g') if np.isfinite(c) else '')
                             for j, (f, c) in enumerate(zip(fine, coarse)))
            print(f'{name}: {checks["fine_hits"]}/{samples} fine hits, '
                  f'{checks["fine_only_hits"]} extra over coarse mesh', flush=True)
    (args.output / 'reference_checks.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
