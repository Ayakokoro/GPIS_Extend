"""Compare exported production laws with independent dense SE path samples.

Distances and speeds are normalized: q=t/ell, w=ell*V/sigma.
No covariance jitter or learned/LUT probabilities are used by the reference.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr, ndtri, erf
from scipy.integrate import trapezoid


def features(q, surface=False, degree=170):
    q = np.asarray(q, dtype=float)
    f = np.empty((len(q), degree + 1))
    f[:, 0] = np.exp(-.5*q*q)
    for k in range(1, degree + 1):
        f[:, k] = f[:, k-1]*q/np.sqrt(k)
    v = q[:, None]*f
    v[:, 1:] -= np.sqrt(np.arange(1, degree+1))*f[:, :-1]
    assert np.max(np.abs(np.sum(f*f, axis=1)-1)) < 2e-13
    assert np.max(np.abs(np.sum(v*v, axis=1)-1)) < 2e-12
    return f[:, 2 if surface else 0:], v[:, 2 if surface else 0:]


def profile_mean(case, q):
    knots = np.asarray(case['mean_knots'])
    result = np.interp(q, knots[:, 0], knots[:, 1])
    if case['start_kind'] == 'surface':
        c = np.exp(-.5*q*q)
        result = result-c*case['start_value']+q*c*(case['start_gradient']-case['start_slope'])
    return result


def partition(case, subdivisions, end=6.):
    knots = np.asarray(case['mean_knots'])[:, 0]
    values = np.r_[np.arange(int(np.floor(end*subdivisions))+1)/subdivisions,
                   knots[(knots >= 0) & (knots <= end)], end]
    # The C++ grid/physical conversion differs by a few ulps at shared faces.
    return np.unique(np.round(values, 12))


def model_transmittance(model, q, subdivisions):
    data = next(x for x in model['integration'] if x['subdivisions_per_length'] == subdivisions)
    segments = np.array([row[:4] for row in data['segments']], dtype=float)
    assert np.all(segments[:, 1] > segments[:, 0]) and np.all(segments[:, 3] >= 0)
    assert np.max(np.abs(segments[1:, 0]-segments[:-1, 1])) < 1e-10
    x = np.r_[segments[0, 0], segments[:, 1]]
    h = np.r_[0., np.cumsum((segments[:, 1]-segments[:, 0])*segments[:, 3])]
    return np.exp(-np.interp(q, x, h))


def transmittance_reference(case, samples, seed, batch_size=4096):
    q = partition(case, 128)
    coarse = partition(case, 64)
    coarse_indices = np.searchsorted(q, coarse)
    assert np.max(np.abs(q[coarse_indices]-coarse)) < 1e-11
    surface = case['start_kind'] == 'surface'
    f, _ = features(q, surface)
    mu = profile_mean(case, q)
    if surface:
        assert abs(mu[0]) < 1e-12
    counts = np.zeros(len(q), dtype=np.int64)
    coarse_counts = np.zeros(len(coarse), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for offset in range(0, samples, batch_size):
        size = min(batch_size, samples-offset)
        z = rng.standard_normal((size, f.shape[1]))
        if not surface:
            lower = ndtr(-case['start_value'])
            z[:, 0] = ndtri(lower+(1-lower)*rng.random(size))
        y = z @ f.T + mu
        positive = y > 0
        positive[:, 0] = True  # exterior was truncated; surface is an equality.
        counts += np.logical_and.accumulate(positive, axis=1).sum(axis=0)
        coarse_counts += np.logical_and.accumulate(positive[:, coarse_indices], axis=1).sum(axis=0)
    fine = counts/samples
    coarse_t = coarse_counts/samples
    # Compare only identical query endpoints; interpolation is for plotting.
    delta = float(np.max(np.abs(fine[coarse_indices]-coarse_t)))
    # Symmetric envelope of the Wilson interval, including at empirical 0/1.
    z95 = 1.959963984540054
    center = (fine+z95*z95/(2*samples))/(1+z95*z95/samples)
    radius = z95*np.sqrt(fine*(1-fine)/samples+z95*z95/(4*samples*samples))/(1+z95*z95/samples)
    width = radius+np.abs(center-fine)
    width[0] = 0  # survival at the conditioned origin is exactly one.
    return dict(q=q.tolist(), reference=fine.tolist(),
        reference_coarse=np.interp(q, coarse, coarse_t).tolist(),
        ci95=width.tolist(),
        reference_grid_max_delta=delta, samples=samples)


def weighted_cdf(v, weight, grid):
    order = np.argsort(v)
    values = v[order]
    cumulative = np.r_[0., np.cumsum(weight[order])]
    return cumulative[np.searchsorted(values, grid, side='right')]


def normal_cdf_sums(v, weight, grid, variance):
    # G_perp is a centered independent 2D Gaussian in every test profile.
    out = np.zeros(len(grid))
    for j, c in enumerate(grid[1:-1], 1):
        out[j] = np.dot(weight, np.exp(-v*v*(1-c*c)/(2*variance*c*c)))
    out[-1] = np.sum(weight)
    return out


def normal_cdf_from_speed(grid, cdf, normal_grid, variance):
    # Integrate exp(-a*w^2) exactly for the piecewise-uniform PDF defined by
    # the production CDF samples. The final tail is checked before use.
    mass = np.maximum(0., np.diff(cdf))
    lo, hi = grid[:-1], grid[1:]
    result = np.zeros(len(normal_grid))
    for j, c in enumerate(normal_grid[1:-1], 1):
        a = (1-c*c)/(2*variance*c*c)
        average = np.sqrt(np.pi)/(2*np.sqrt(a))*(erf(np.sqrt(a)*hi)-erf(np.sqrt(a)*lo))/(hi-lo)
        result[j] = np.dot(mass, average)
    result[-1] = 1.
    return result


def quantiles(grid, cdf):
    return {name: float(np.interp(p, cdf, grid)) for name, p in [('p10', .1), ('p50', .5), ('p90', .9)]}


def gradient_reference(case, probe, samples, seed, selections=None, batch_size=4096):
    batch_size = min(batch_size, max(1, samples//8))
    t = probe['q']
    surface = case['start_kind'] == 'surface'
    q = partition(case, 128, t)
    q = q[q < t-1e-11]
    if surface:
        q = q[q > 0]
    fine_q = q.copy()
    selections = selections or {'baseline': None, 'selected': None}
    for selected in selections.values():
        if selected is not None:
            assert 0 < selected < t
            q = np.unique(np.r_[q, selected])
    fi = np.searchsorted(q, fine_q)
    coarse_q = partition(case, 64, t)
    coarse_q = coarse_q[coarse_q < t-1e-11]
    if surface:
        coarse_q = coarse_q[coarse_q > 0]
    ci = np.searchsorted(q, coarse_q)
    assert np.max(np.abs(q[ci]-coarse_q), initial=0) < 1e-11
    f, _ = features(q, surface)
    ft, vt = features(np.array([t]), surface)
    ft, vt = ft[0], vt[0]
    var = ft @ ft
    mean_t = profile_mean(case, np.array([t]))[0]
    mean = profile_mean(case, q) - (f @ ft)*mean_t/var
    residual = f - (f @ ft)[:, None]*ft/var
    slope = probe['mean_slope']
    if surface:
        slope += np.exp(-.5*t*t)*(t*case['start_value']+(1-t*t)*(case['start_gradient']-case['start_slope']))
    mean_v = -slope - (vt @ ft)*mean_t/var
    residual_v = vt - (vt @ ft)*ft/var
    speed_grid = np.array(probe['speed_cdf'])[:, 0]
    normal_grid = np.linspace(0, 1, 101)
    transverse_variance = -np.expm1(-t*t) if surface else 1.
    rng = np.random.default_rng(seed)
    sums = np.zeros(len(speed_grid))
    normal_sums = np.zeros(len(normal_grid))
    coarse_sums = np.zeros(len(speed_grid))
    total = total_squared = moment = coarse_total = 0.
    # Eight independent groups estimate ratio-CDF Monte Carlo uncertainty.
    groups = np.zeros((8, len(speed_grid)+1))
    normal_groups = np.zeros((8, len(normal_grid)))
    checks = {name:dict(cdf=np.zeros(len(speed_grid)), total=0., square=0.)
              for name in selections}
    for offset in range(0, samples, batch_size):
        size = min(batch_size, samples-offset)
        z = rng.standard_normal((size, residual.shape[1]))
        y = z @ residual.T + mean
        v = z @ residual_v + mean_v
        positive = y > 0
        # Extra coordinates only validate each production history selection;
        # they must not change the dense reference's prescribed discretization.
        valid = np.all(positive[:, fi], axis=1) & (v > 0)
        coarse_valid = np.all(positive[:, ci], axis=1) & (v > 0)
        velocity = v[valid]
        weight = velocity  # Rice crossing-speed weighting is essential.
        batch_total = weight.sum()
        batch_cdf = weighted_cdf(velocity, weight, speed_grid)
        sums += batch_cdf
        batch_normal = normal_cdf_sums(velocity, weight, normal_grid, transverse_variance)
        normal_sums += batch_normal
        total += batch_total
        total_squared += np.dot(weight, weight)
        moment += np.dot(weight, velocity)
        coarse_v = v[coarse_valid]
        coarse_sums += weighted_cdf(coarse_v, coarse_v, speed_grid)
        coarse_total += coarse_v.sum()
        group = min(7, offset*8//samples)
        groups[group, :-1] += batch_cdf
        groups[group, -1] += batch_total
        normal_groups[group] += batch_normal
        base_valid = (v > 0)
        if not surface:
            base_valid &= positive[:, 0]
        for name in checks:
            check_valid = base_valid.copy()
            selected = selections[name]
            if selected is not None:
                check_valid &= positive[:, np.searchsorted(q, selected)]
            check_v = v[check_valid]
            checks[name]['cdf'] += weighted_cdf(check_v, check_v, speed_grid)
            checks[name]['total'] += check_v.sum()
            checks[name]['square'] += check_v @ check_v
    assert total > 0 and coarse_total > 0 and np.all(groups[:, -1] > 0), case['id']
    cdf = sums/total
    t95 = 2.364624251  # Student t with seven degrees of freedom.
    error = t95*np.std(groups[:, :-1]/groups[:, -1, None], axis=0, ddof=1)/np.sqrt(8)
    normal_error = t95*np.std(normal_groups/groups[:, -1, None], axis=0, ddof=1)/np.sqrt(8)
    return dict(q=t, selected=None, samples=samples, ess=float(total*total/total_squared),
        speed_grid=speed_grid.tolist(), reference_cdf=cdf.tolist(),
        reference_cdf_ci95=error.tolist(), normal_grid=normal_grid.tolist(),
        reference_normal_cdf=(normal_sums/total).tolist(),
        reference_normal_cdf_ci95=normal_error.tolist(),
        reference_mean_speed=float(moment/total), reference_speed_quantiles=quantiles(speed_grid, cdf),
        reference_grid_cdf_max_delta=float(np.max(np.abs(cdf-coarse_sums/coarse_total))),
        reference_flux_mass=float(total/samples), transverse_variance=float(transverse_variance),
        model_mc_validation={name:dict(cdf=(x['cdf']/x['total']).tolist(),
            ess=float(x['total']**2/x['square'])) for name,x in checks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=262144)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()
    if args.samples < 8:
        parser.error('--samples must be at least 8 for the uncertainty groups')
    output = args.output or args.model.with_name('results.json')
    if output.exists():
        raise SystemExit(f'Preserving existing results: {output}')
    model = json.loads(args.model.read_text(encoding='utf-8'))
    result = dict(settings=dict(**model.get('settings', {})), cases=[])
    for key in ('experiment', 'model_settings', 'integration_levels'):
        if key in model:
            result[key] = model[key]
    result['settings'].update(dict(ell=model['ell'], sigma=model['sigma'], samples=args.samples,
        seed=args.seed, reference_dq=1/128, coarse_reference_dq=1/64,
        feature_degree=170, speed_unit='w=ell*V/sigma',
        description='Independent Gaussian-feature Monte Carlo, shared coarse/fine histories; fixed-position Rice-weighted gradients.'))
    started = time.perf_counter()
    for index, case in enumerate(model['cases']):
        print(f"{case['id']}: transmittance", flush=True)
        # A width sweep uses paired latent GP paths within each birth kind.
        # This reduces noise in changes between widths without coupling models.
        seed_offset = (10000 if case['start_kind'] == 'surface' else 20000) if 'width_over_length' in case else 100*index
        case_seed = args.seed+seed_offset
        trans = transmittance_reference(case, args.samples, case_seed)
        trans['models'] = {}
        q, ref = np.array(trans['q']), np.array(trans['reference'])
        for name, data in case['models'].items():
            levels = sorted(x['subdivisions_per_length'] for x in data['integration'])
            assert len(levels) >= 2
            fine = model_transmittance(data, q, levels[-1])
            coarse = model_transmittance(data, q, levels[-2])
            error = fine-ref
            trans['models'][name] = dict(values=fine.tolist(), max_abs_error=float(np.max(np.abs(error))),
                rmse=float(np.sqrt(trapezoid(error*error, q)/(q[-1]-q[0]))),
                endpoint_error=float(error[-1]), integration_max_delta=float(np.max(np.abs(fine-coarse))),
                integration_coarse_divisions=levels[-2], integration_fine_divisions=levels[-1])
        row = {k: v for k, v in case.items() if k != 'models'}
        row.update(transmittance=trans, probes=[])
        for j, probe in enumerate(case['models']['baseline']['probes']):
            print(f"{case['id']}: gradient q={probe['q']}", flush=True)
            selections = {name: law['probes'][j]['selected'] for name, law in case['models'].items()}
            data = gradient_reference(case, probe, args.samples, case_seed+j+1, selections)
            if data['ess'] < 1500:
                print(f"  low ESS={data['ess']:.0f}; increasing reference to {4*args.samples} samples", flush=True)
                data = gradient_reference(case, probe, 4*args.samples, case_seed+j+1, selections)
            knots = np.asarray(case['mean_knots'])
            data['selection_details'] = {name: dict(q=u, mean=None if u is None else float(np.interp(u, knots[:, 0], knots[:, 1])))
                                         for name, u in selections.items()}
            data['models'] = {}
            v = np.array(data['speed_grid'])
            nc = np.array(data['normal_grid'])
            for name, law in case['models'].items():
                p = law['probes'][j]
                assert p['q'] == probe['q']
                cdf = np.array(p['speed_cdf'])[:, 1]
                assert np.all(np.isfinite(cdf)) and np.min(np.diff(cdf)) > -1e-8
                assert cdf[-1] > 1-1e-6, 'Extend speed export to cover the remaining mass'
                normal = normal_cdf_from_speed(v, cdf, nc, data['transverse_variance'])
                coarse_normal = normal_cdf_from_speed(v[::2], cdf[::2], nc, data['transverse_variance'])
                data['models'][name] = dict(cdf=cdf.tolist(), normal_cdf=normal.tolist(),
                    ks=float(np.max(np.abs(cdf-data['reference_cdf']))),
                    normal_ks=float(np.max(np.abs(normal-data['reference_normal_cdf']))),
                    normal_quadrature_max_delta=float(np.max(np.abs(normal-coarse_normal))),
                    mean_speed=float(trapezoid(1-cdf, v)), quantiles=quantiles(v, cdf))
                data['model_mc_validation'][name]['max_cdf_difference'] = float(np.max(
                    np.abs(cdf-data['model_mc_validation'][name]['cdf'])))
                if name == 'selected':
                    data['selected'] = p['selected']
            row['probes'].append(data)
        result['cases'].append(row)
        result['elapsed_seconds'] = time.perf_counter()-started
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(f"{case['id']}: done, T errors="+str({k:v['max_abs_error'] for k,v in trans['models'].items()}), flush=True)
    print(f'Wrote {output}; elapsed={result["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
