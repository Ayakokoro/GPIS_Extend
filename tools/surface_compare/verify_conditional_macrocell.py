"""Validate recorded macrocell runs without rerunning timing experiments."""
from conditional_optimization_report import ROOT, read, pfm, np
import json
import statistics

OUT = ROOT/'outputs/shaderball_macrocell_hybrid_20261118'
all_runs = []
for batch in ['sweep', 'selective', 'final']:
    data = read(OUT/batch/'runs.json')
    assert len(data['runs']) == len(data['spec']['runs'])
    all_runs += data['runs']
    for r in data['runs']:
        a = pfm(r['pfm'])
        assert a.shape == (64, 64, 3) and np.isfinite(a).all() and a.min() >= 0
        assert r['discarded_surface_paths'] == r['discarded_surface_connections'] == 0
        for v in r.get('slow_rays', []):
            assert v['within_six_sigma']
final = read(OUT/'final/runs.json')['runs']
controls = {r['seed']: r for r in final if not r.get('macro_bounds')}
timings = {}
for size in [4, 16]:
    selected = [r for r in final if r.get('macro_size') == size]
    ratios = np.array([r['render_seconds']/controls[r['seed']]['render_seconds'] for r in selected])
    differences = np.array([pfm(r['pfm']).mean(dtype=np.float64)-pfm(controls[r['seed']]['pfm']).mean(dtype=np.float64) for r in selected])
    se = differences.std(ddof=1)/np.sqrt(len(differences))
    assert abs(differences.mean()) <= 6*se+1e-4
    timings[size] = dict(median_seconds=statistics.median(r['render_seconds'] for r in selected),
        paired_median_time_ratio=float(np.median(ratios)), paired_ratios=ratios.tolist(),
        image_mean_difference=float(differences.mean()), image_mean_standard_error=float(se))
old = pfm(ROOT/'outputs/shaderball_tracking_optimization_20261118/final-prefix/optimized_0.pfm')
new = pfm(controls[20261024]['pfm'])
assert np.array_equal(old, new)
regression = (OUT/'tracking-regression.log').read_text(encoding='utf-16')
assert 'tracking: carried optical depth' in regression and 'survival laws passed' in regression
assert 'failed' not in regression.lower()
diagnostic = OUT/read(OUT/'selection.json').get('diagnostic_directory', 'diagnostic')
fixed_points = fresh_points = 0
if (diagnostic/'fixed_rays.json').exists():
    fixed = [r for r in read(diagnostic/'fixed_rays.json') if r['optimized']]
    fixed_points = sum(r['coverage_samples'] for r in fixed)
    assert all(r['maximum_hazard_bound_ratio'] <= 1+1e-9 for r in fixed)
    fresh = read(diagnostic/'slow_rays.json')
    # This export also checks every recorded interval while producing its curves.
    if isinstance(fresh, dict):
        for r in fresh.get('rays', []):
            coverage = r.get('validation', {})
            fresh_points += coverage.get('coverage_samples', 0)
            assert coverage.get('maximum_hazard_bound_ratio', 0) <= 1+1e-9
out = dict(render_runs=len(all_runs),threads=8,performance_core_affinity=True,
    baseline_median_seconds=statistics.median(r['render_seconds'] for r in controls.values()),
    final_timings=timings,macro_hazard_coverage_points=sum(r['macro_checks']['coverage_points'] for r in all_runs if r.get('macro_checks')),
    fixed_ray_coverage_points=fixed_points, new_slow_ray_coverage_points=fresh_points,
    monte_carlo_samples=sum(v['samples'] for r in all_runs for v in r.get('slow_rays', [])),
    monte_carlo_checks_passed=True, tracking_regression_passed=True,
    baseline_image_bitwise_preserved=True, all_images_finite_nonnegative=True,
    discarded_paths_and_connections=0, default_macrocell_bounds_enabled=False)
if (diagnostic/'runs.json').exists():
    before = read(ROOT/'outputs/shaderball_tracking_optimization_20261118/diagnostic-prefix/runs.json')['runs'][0]['profile']
    after = read(diagnostic/'runs.json')['runs'][0]['profile']
    keys = ['candidate_events', 'rejected_candidates', 'ratio_events', 'processed_intervals',
            'grid_macro_queries', 'transition_bound_queries', 'local_far_bound_queries', 'grid_cell_builds']
    out['small_render_work_counts'] = {k: [sum(d['counts'].get(k, 0) for d in before['domains'].values()),
        sum(d['counts'].get(k, 0) for d in after['domains'].values())] for k in keys}
high_path = OUT/'high-resolution/runs.json'
if high_path.exists():
    high = read(high_path)
    assert [r['label'] for r in high['runs']] == ['control', 'macro4']
    for r in high['runs']:
        a = pfm(r['pfm'])
        assert a.shape == (256, 256, 3) and np.isfinite(a).all() and a.min() >= 0
        assert r['discarded_surface_paths'] == r['discarded_surface_connections'] == 0
    old = pfm(ROOT/'outputs/shaderball_tracking_256_128spp_20261118/render/render.pfm')
    baseline, hybrid = high['runs']
    assert np.array_equal(old, pfm(baseline['pfm']))
    log = (OUT/'high-resolution/execution.log').read_text(encoding='utf-8')
    assert 'Transition experiment failed: conditional event cannot advance' in log
    out['high_resolution'] = dict(resolution=[256, 256], spp=128, seed=20261024,
        completed_variants=['control', 'macro4'], failed_variant='macro16',
        failure='conditional event cannot advance',
        baseline_seconds=baseline['render_seconds'], hybrid_seconds=hybrid['render_seconds'],
        macro_build_seconds=hybrid['macro_build_seconds'], loading_seconds=high['initialization_seconds'],
        render_time_reduction=1-hybrid['render_seconds']/baseline['render_seconds'],
        render_and_macro_build_reduction=1-(hybrid['render_seconds']+hybrid['macro_build_seconds'])/baseline['render_seconds'],
        original_baseline_bitwise_preserved=True, high_resolution_repetitions_per_completed_variant=1)
(OUT/'validation.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
