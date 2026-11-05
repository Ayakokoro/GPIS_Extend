"""Reproducible, family-split orthant data; no renderer or neural inference.

Example: python -m tools.gaussian_orthant_nn.generate --groups 1000
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

from .gaussian import (family_split, feature_factor, orthant_qmc_batch,
                       se_covariance, se_features, standardized_statistics,
                       surface_conditioned_statistics)


def _profile(rng, positions):
    origin = rng.uniform(-2.5, 2.5)
    linear = rng.normal(0.0, 0.45)
    amplitudes = rng.normal(0.0, 0.65, 3)
    frequencies = rng.uniform(0.3, 2.5, 3)
    phases = rng.uniform(-np.pi, np.pi, 3)
    values = origin + linear * positions + np.sum(amplitudes[:, None] * (
        np.sin(frequencies[:, None] * positions + phases[:, None]) - np.sin(phases[:, None])), axis=0)
    slope = linear + np.sum(amplitudes * frequencies * np.cos(phases))
    return values, origin, float(slope)


def _record(group_id, split, mode, positions, mean, covariance, features,
            origin_mean=math.nan, origin_slope=math.nan, birth_slope=math.nan):
    z, correlation = standardized_statistics(mean, covariance)
    sd = np.sqrt(np.diag(covariance))
    factor = feature_factor(features / sd[:, None])
    return dict(group_id=group_id, split=split, mode=mode, positions=positions,
                z=z, corr=correlation, factor=factor, dim=len(z), mean=mean,
                minimum_eigenvalue=float(np.linalg.eigvalsh(correlation)[0]),
                origin_mean=origin_mean, origin_slope=origin_slope, birth_slope=birth_slope)


def make_families(groups, seed, generic_fraction=0.2):
    rng = np.random.default_rng(seed)
    records = []
    for group_id in range(groups):
        n = int(rng.integers(1, 6))
        split = family_split(group_id, seed)
        near_birth = rng.uniform() < 0.1
        extent = np.exp(rng.uniform(np.log(0.003), np.log(0.2))) if near_birth else rng.uniform(0.4, 8.0)
        positions = np.sort(rng.uniform(0.08, 1.0, n)) * extent
        # A subset has close neighbours; equality is never manufactured by a
        # covariance jitter or correlation clamp.
        if n > 1 and rng.uniform() < 0.12:
            positions[-1] = positions[-2] + extent * np.exp(rng.uniform(np.log(1e-4), np.log(0.03)))
        mean, origin, origin_slope = _profile(rng, positions)
        birth_slope = extent * rng.uniform(0.1, 2.0) if near_birth else np.exp(rng.uniform(np.log(0.05), np.log(2.0)))
        records.append(_record(group_id, split, 0, positions, mean, se_covariance(positions),
                               se_features(positions), origin, origin_slope, birth_slope))
        conditional_mean, conditional_covariance = surface_conditioned_statistics(
            positions, mean, origin, origin_slope, birth_slope)
        records.append(_record(group_id, split, 1, positions, conditional_mean, conditional_covariance,
                               se_features(positions, conditioned=True), origin, origin_slope, birth_slope))
        if rng.uniform() < generic_fraction:
            # A Gram matrix of random signed loadings is SPD with probability
            # one and supplies correlations of both signs.
            loadings = rng.normal(size=(n, n + 2))
            loadings /= np.linalg.norm(loadings, axis=1)[:, None]
            covariance = loadings @ loadings.T
            generic_mean = rng.uniform(-4.0, 4.0, n)
            if rng.uniform() < 0.1:
                generic_mean -= rng.uniform(1.0, 3.0)
            records.append(_record(group_id, split, 2, positions, generic_mean, covariance, loadings))
    return records


def generate(args):
    begin = time.perf_counter()
    if args.backend == 'auto':
        try:
            import torch
            args.backend = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.backend = 'cpu'
    records = make_families(args.groups, args.seed, args.generic_fraction)
    rejected = Counter({name: 0 for name in (
        'input_mean_outside_training_range', 'probability_below_training_range',
        'relative_error_target_not_met', 'matrix_factorization_failed',
        'zero_or_unresolved_probability', 'nonfinite_log_probability')})
    rejection_breakdown = Counter()
    def reject(record, reason):
        rejected[reason] += 1
        rejection_breakdown[(record['mode'], record['dim'], reason)] += 1
    selected = []
    for record in records:
        if np.max(np.abs(record['z'])) > args.maximum_standardized_mean:
            reject(record, 'input_mean_outside_training_range')
        else:
            selected.append(record)
    print(f"Built {len(records)} candidate rows in {time.perf_counter()-begin:.2f}s; "
          f"labeling {len(selected)} rows", flush=True)
    accepted = []
    label_options = dict(relative_tolerance=args.relative_tolerance,
                         initial_power=args.initial_power, max_power=args.max_power,
                         replicates=args.replicates, batch_size=args.batch_size, backend=args.backend)
    for n in range(1, 6):
        for split in range(3):
            rows = [r for r in selected if r['dim'] == n and r['split'] == split]
            if not rows:
                continue
            started = time.perf_counter()
            labels = orthant_qmc_batch(np.array([r['z'] for r in rows]),
                                       np.array([r['corr'] for r in rows]),
                                       factors=np.array([r['factor'] for r in rows]),
                                       seed=args.seed + 1009 * n + 100003 * split,
                                       **label_options)
            kept = 0
            for record, label in zip(rows, labels):
                if label.status != 'ok':
                    reject(record, label.status)
                elif not np.isfinite(label.log_probability):
                    reject(record, 'nonfinite_log_probability')
                elif label.log_probability < args.minimum_log_probability:
                    reject(record, 'probability_below_training_range')
                elif not np.isfinite(label.relative_error) or label.relative_error > args.relative_tolerance:
                    reject(record, 'relative_error_target_not_met')
                else:
                    record['label'] = label
                    accepted.append(record)
                    kept += 1
            print(f"n={n} split={split}: {kept}/{len(rows)} labels retained, "
                  f"{time.perf_counter()-started:.2f}s", flush=True)
    if not accepted:
        raise RuntimeError("No labels passed the requested quality gates")
    # Group and mode order make output independent of batch processing order.
    accepted.sort(key=lambda r: (r['group_id'], r['mode']))
    size = len(accepted)
    z = np.zeros((size, 5), np.float64)
    correlation = np.broadcast_to(np.eye(5), (size, 5, 5)).copy()
    factor = correlation.copy()
    positions = np.zeros((size, 5), np.float64)
    means = np.zeros((size, 5), np.float64)
    for index, record in enumerate(accepted):
        n = record['dim']
        z[index, :n] = record['z']; correlation[index, :n, :n] = record['corr']
        factor[index, :n, :n] = record['factor']; positions[index, :n] = record['positions']
        means[index, :n] = record['mean']
    data = dict(z=z, corr=correlation, factor=factor, positions=positions, mean=means)
    for name, dtype in [('group_id', np.int64), ('split', np.int8), ('mode', np.int8), ('dim', np.int8),
                        ('minimum_eigenvalue', np.float64), ('origin_mean', np.float64),
                        ('origin_slope', np.float64), ('birth_slope', np.float64)]:
        data[name] = np.asarray([r[name] for r in accepted], dtype=dtype)
    for name, attribute in [('logp', 'log_probability'), ('probability', 'probability'),
                            ('label_error', 'error'), ('label_relative_error', 'relative_error'),
                            ('label_samples', 'samples')]:
        data[name] = np.asarray([getattr(r['label'], attribute) for r in accepted])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    counts = {str(split): int(np.count_nonzero(data['split'] == split)) for split in range(3)}
    manifest = dict(version=1, seed=args.seed, requested_groups=args.groups,
                    candidate_rows=len(records), accepted_rows=size, split_counts=counts,
                    mode_counts={str(i): int(np.count_nonzero(data['mode'] == i)) for i in range(3)},
                    dimension_counts={str(i): int(np.count_nonzero(data['dim'] == i)) for i in range(1, 6)},
                    rejected=dict(rejected), label_options=label_options,
                    rejection_breakdown=[dict(mode=mode, dim=dim, reason=reason, count=count)
                                         for (mode, dim, reason), count in sorted(rejection_breakdown.items())],
                    candidate_mode_dimension_counts={str(mode): {str(n): sum(r['mode'] == mode and r['dim'] == n for r in records)
                                                                  for n in range(1, 6)} for mode in range(3)},
                    accepted_mode_dimension_counts={str(mode): {str(n): int(np.count_nonzero((data['mode'] == mode) & (data['dim'] == n)))
                                                                 for n in range(1, 6)} for mode in range(3)},
                    minimum_log_probability=args.minimum_log_probability,
                    maximum_standardized_mean=args.maximum_standardized_mean,
                    label_log_probability_range=[float(data['logp'].min()), float(data['logp'].max())],
                    minimum_correlation_eigenvalue=float(data['minimum_eigenvalue'].min()),
                    near_singular_rows={str(threshold): int(np.count_nonzero(data['minimum_eigenvalue'] <= threshold))
                                        for threshold in (1e-12, 1e-8, 1e-4)},
                    max_label_relative_error=float(data['label_relative_error'].max()),
                    total_seconds=time.perf_counter()-begin,
                    mode_names={'0': 'unconditioned_se', '1': 'surface_value_gradient_conditioned_se',
                                '2': 'generic_spd'},
                    splitting='SplitMix64 family IDs; prior, surface, generic family members never cross splits',
                    probability_policy='No probability floor, no covariance jitter; failed labels are rejected',
                    error_policy='Three standard errors across independently scrambled Sobol Genz replicates; estimates, not certificates',
                    se_parameters={'sigma': 1.0, 'length_scale': 1.0},
                    storage_dtype='float64 statistics; factor preserves near-birth feature precision',
                    output=str(args.output))
    manifest_path = args.manifest or args.output.parent / 'generation.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gaussian-orthant-nn/dataset.npz'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--groups', type=int, default=32000)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--generic-fraction', type=float, default=0.2)
    parser.add_argument('--relative-tolerance', type=float, default=5e-4)
    parser.add_argument('--initial-power', type=int, default=9)
    parser.add_argument('--max-power', type=int, default=14)
    parser.add_argument('--replicates', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--backend', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--minimum-log-probability', type=float, default=-60.0)
    parser.add_argument('--maximum-standardized-mean', type=float, default=12.0)
    args = parser.parse_args()
    if args.groups <= 0 or not 0 <= args.generic_fraction <= 1:
        parser.error('positive --groups and --generic-fraction within [0,1] required')
    generate(args)


if __name__ == '__main__':
    main()
