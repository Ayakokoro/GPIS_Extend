"""Check completed artifacts before publishing the shaderball comparison."""
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/shaderball_tracking_optimization_20261118'
sys.path.insert(0, str(ROOT/'build/fp-gradient-validation-deps'))
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def pfm(path):
    with Path(path).open('rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split()); scale = float(f.readline())
        return np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)[::-1]


selection = read(OUT/'selection.json')
final_directory = selection.get('final_directory', 'final')
diagnostic_directory = selection.get('diagnostic_directory', 'diagnostic')
standalone_directory = selection.get('standalone_directory', 'standalone')
final = read(OUT/final_directory/'runs.json')
assert final['spec']['size'] == 64 and final['spec']['spp'] == 8 and final['spec']['threads'] == 8
optimized = [r for r in final['runs'] if r['mode'] == 'coupled']
controls = [r for r in final['runs'] if r['mode'] == 'integral_three']
assert len(optimized) >= 6 and len(controls) == len(optimized)
times = [r['render_seconds'] for r in optimized]
assert max(times) <= 1, times
for run in final['runs']:
    a = pfm(run['pfm'])
    assert a.shape == (64, 64, 3) and np.isfinite(a).all() and a.min() >= 0 and a.max() > 0
    assert run['discarded_surface_paths'] == run['discarded_surface_connections'] == 0
    for ray in run.get('slow_rays', []):
        assert ray['within_six_sigma'] and ray['reference']['refinement_difference'] < 1e-7
assert len(optimized[0]['slow_rays']) == len(controls[0]['slow_rays']) == 4
assert all(r['samples'] == 12000 for r in optimized[0]['slow_rays'])
assert all(r['samples'] == 4000 for r in controls[0]['slow_rays'])
image_checks = {}
comparisons = {
    'legacy_integral_bitwise_preserved': (controls[0]['pfm'], OUT/'baseline/integral_three_0.pfm'),
    'profile_and_production_identical': (optimized[0]['pfm'], OUT/diagnostic_directory/'optimized.pfm'),
    'standalone_tungsten_identical': (optimized[0]['pfm'], OUT/standalone_directory/'render.pfm'),
}
for label, (a, b) in comparisons.items():
    aa, bb = pfm(a), pfm(b)
    image_checks[label] = dict(equal=bool(np.array_equal(aa, bb)), max_absolute_difference=float(np.max(np.abs(aa-bb))))
    assert image_checks[label]['equal'], (label, image_checks[label])
fixed = read(OUT/diagnostic_directory/'fixed_rays.json')
fresh = read(OUT/diagnostic_directory/'slow_rays.json')
assert len(fixed) == 8
for ray in fixed:
    assert ray['maximum_hazard_bound_ratio'] <= 1+1e-9 and ray['coverage_samples'] > 0
    assert all(e['hazard'] <= e['majorant']*(1+1e-10) for e in ray['events'])
for ray in fresh['rays']:
    assert ray['validation']['replay_exact'] and ray['validation']['maximum_hazard_bound_ratio'] <= 1+1e-9
log = (OUT/'tracking-regression-prefix.log').read_bytes()
text = log.decode('utf-16') if log[:2] in (b'\xff\xfe', b'\xfe\xff') else log.decode('utf-8-sig')
assert 'coupled transition: 43776 coverage checks' in text
assert 'medium/shadow routing and fine/adaptive/cheap survival laws passed' in text
old = read(ROOT/'outputs/shaderball_transition_integral_20261118/scene.json')
new = read(OUT/'scene.json')
for key in ['primitives', 'camera', 'integrator', 'bsdfs']:
    assert old[key] == new[key], key
strip_tracking = lambda medium: {k: v for k, v in medium.items() if not k.startswith('tracking_')}
assert strip_tracking(old['media'][0]) == strip_tracking(new['media'][0])
assert new['renderer']['spp'] == 8 and not new['renderer']['adaptive_sampling']
differences = np.array([a['image']['mean_rgb']-b['image']['mean_rgb'] for a, b in zip(optimized, controls)])
se = float(differences.std(ddof=1)/np.sqrt(len(differences)))
assert abs(differences.mean()) <= 6*se+1e-4
no_split = read(OUT/'no-split-optimized/runs.json')
no_split_log = (OUT/'no-split-optimized/execution.log').read_text(encoding='utf-8')
assert 'conditional event cannot advance' in no_split_log and len(no_split['runs']) == 1
standalone_log = (OUT/standalone_directory/'execution.log').read_text(encoding='utf-8')
standalone_seconds = float(re.search(r'Finished render\. Render time ([0-9.]+)s', standalone_log)[1])
assert 0 < standalone_seconds <= 1
report = dict(
    target_met=True, render_only=True, loading_seconds=final['initialization_seconds'],
    threads=8, performance_core_affinity=read(OUT/final_directory/'host.json')['performance_core_affinity'],
    render_repetitions=len(times), render_seconds=times, median_seconds=float(np.median(times)),
    maximum_seconds=max(times), control_median_seconds=float(np.median([r['render_seconds'] for r in controls])),
    standalone_renderer_seconds=standalone_seconds,
    tracking_regression_passed=True, coupled_coverage_checks=43776,
    fixed_ray_coverage_samples=sum(r['coverage_samples'] for r in fixed),
    new_slow_ray_coverage_samples=sum(r['validation']['coverage_samples'] for r in fresh['rays']),
    fixed_ray_monte_carlo_samples=dict(original=16000, optimized=48000),
    fixed_ray_monte_carlo_passed=True, discarded_paths_and_connections=0,
    scene_field_camera_materials_and_integrator_preserved=True, image_checks=image_checks,
    paired_image_mean_difference=float(differences.mean()), paired_image_standard_error=se,
    disabled_transition_subdivision=dict(recommended=False, first_render_seconds=no_split['runs'][0]['render_seconds'],
        next_seed_status='numeric_error: conditional event cannot advance', selected_threshold=4),
)
(OUT/'validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ['target_met', 'median_seconds', 'maximum_seconds', 'control_median_seconds',
    'fixed_ray_coverage_samples', 'new_slow_ray_coverage_samples', 'paired_image_mean_difference', 'paired_image_standard_error']}, indent=2))
