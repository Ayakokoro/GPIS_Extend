"""Include the two completed Renewal+ images explicitly accepted by the user.

Reads original render outputs, preserves their warnings, and completes native
FLIP reporting. No renderer is invoked and no original render file is changed.
"""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path
import re
from types import SimpleNamespace

import gpis_shaderball_parameter_comparison as s

OUT = s.DEFAULT_OUT
HISTORY = OUT/'history/before_accepting_nee_warnings'
TARGETS = ((.2, .01), (.4, .01))
PATTERN = re.compile(r'^1D NEE sanity check failed!\s+([-+\deE.]+)\s+([-+\deE.]+)\s*$')


def main():
    args = SimpleNamespace(output=OUT, threads=32, flip=s.DEFAULT_FLIP)
    assert not (OUT/'runner.lock').exists(), 'A render queue is active'
    original = s.load(HISTORY/'manifest.json')
    old_runner = HISTORY/Path(s.__file__).name
    assert s.sha(old_runner) == original['runner_sha256']
    # Changes to the report must not silently change render behavior.
    old_ast = ast.parse(old_runner.read_text(encoding='utf-8'))
    new_ast = ast.parse(Path(s.__file__).read_text(encoding='utf-8'))
    for name in ('scene_for', 'render_one', 'verified', 'validate_scenes'):
        lookup = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(lookup(old_ast)) == ast.dump(lookup(new_ast)), name
    for name, expected in original['runtime_sha256'].items():
        assert s.sha(OUT/'renderer'/name) == expected, name
    assert s.sha(args.flip) == original['flip']['sha256']
    planned, warnings = [], []
    for ell, sigma in TARGETS:
        mode = 'gpis_renewal_plus'
        folder = s.folder(args, ell, sigma, mode)
        scene = s.scene_for(args, ell, sigma, mode)
        job = s.load(folder/'job.json')
        expected = dict(scene=scene, seed=s.SEED, renderer_sha256=original['runtime_sha256']['tungsten.exe'])
        assert s.load(folder/'scene.json') == scene and job['identity'] == expected
        medium = scene['media'][0]
        assert medium['1D_sampling'] and medium['1D_sampling_scheme'] == 'mis'
        assert medium['step_size'] == .0025 and medium['1D_gradient_correlationXY']
        text = (folder/'render.log').read_text(encoding='utf-8', errors='replace')
        assert 'Finished render.' in text
        progress = re.findall(r'Completed\s+(\d+)/(\d+)\s+spp,\s*([\d.]+)/[\d.]+\s+sec', text)
        assert progress and progress[-1][:2] == ('128', '128')
        error_lines = [(i, line) for i, line in enumerate(text.splitlines(), 1) if s.ERRORS.search(line)]
        assert len(error_lines) == 1, error_lines
        line_number, message = error_lines[0]
        match = PATTERN.fullmatch(message)
        assert match, message
        reconstructed, constraint = map(float, match.groups())
        rgb, a, b, variance = [s.pfm(folder/name) for name in
            ('render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm')]
        assert rgb.shape == variance.shape == (256, 256, 3) and rgb.max() > 0
        assert s.np.allclose((a+b)/2, rgb, rtol=4e-6, atol=4e-7)
        with s.Image.open(folder/'render.png') as image:
            assert image.size == (256, 256)
        warning = dict(kind='1d_nee_gradient_consistency', count=1, message=message,
            log_line=line_number, reconstructed_gradient=reconstructed, constraint_gradient=constraint,
            relative_mismatch=abs(reconstructed/constraint-1), check_threshold=.01,
            accepted_for_comparison=True)
        row = dict(identity=expected, seconds=float(progress[-1][2]), timing_source='original_render_log',
            finished_at=dt.datetime.fromtimestamp((folder/'render.log').stat().st_mtime, dt.timezone.utc).isoformat(),
            command=job['command'], resolution=[256,256], spp=128, seed=s.SEED,
            threads=int(job['command'][job['command'].index('-t')+1]),
            mean_luminance=float((rgb@s.Y).mean()), sha256={name:s.sha(folder/name) for name in s.FILES},
            warnings=[warning], inclusion=dict(policy='include_completed_with_known_warning',
                user_request='仍然作为渲染图放上去进行比较', recorded_at=s.now(),
                importer_sha256=s.sha(Path(__file__))))
        if (folder/'result.json').exists():
            existing = s.load(folder/'result.json')
            assert existing['sha256'] == row['sha256'] and existing['warnings'] == row['warnings']
            row = existing
        planned.append((folder, row))
        warnings.append(dict(group=s.key(ell, sigma), ell=ell, sigma=sigma, mode=mode,
            folder=folder.relative_to(OUT).as_posix(), warnings=[warning]))
    # Register only the two explicitly accepted results, retaining original files.
    for folder, row in planned:
        s.save(folder/'result.json', row)
    s.save(OUT/'warnings.json', dict(policy='include_completed_with_known_warning',
        accepted_render_count=2, sources=warnings,
        historical_failures='history/before_accepting_nee_warnings/failures.json',
        original_render_files_unchanged=True))
    manifest = s.load(OUT/'manifest.json')
    assert manifest['runner_sha256'] == original['runner_sha256']
    manifest.update(report_runner_sha256=s.sha(Path(s.__file__)),
        original_runner=old_runner.relative_to(OUT).as_posix(),
        warning_acceptance=dict(record='warnings.json', sha256=s.sha(OUT/'warnings.json'),
            renders=2, importer=str(Path(__file__).relative_to(s.ROOT)), importer_sha256=s.sha(Path(__file__))))
    s.save(OUT/'manifest.json', manifest)
    s.prepare(args)
    s.status(args)
    s.report(args)
    old_metrics = s.load(HISTORY/'metrics.json')
    metrics = s.load(OUT/'metrics.json')
    lookup = {(r['group'], r['test'], r['reference']):r for r in metrics['comparisons']}
    for row in old_metrics['comparisons']:
        assert lookup[(row['group'], row['test'], row['reference'])] == row
    assert len(metrics['sources']) == 40 and len(metrics['comparisons']) == 60 and metrics['complete']
    assert len(metrics['included_with_warnings']) == 2
    for ell, sigma in s.GROUPS:
        for mode in s.MODES:
            s.verified(args, ell, sigma, mode)
    flip_records = list((OUT/'flip').glob('*/*/*/result.json'))
    assert len(flip_records) == 120
    for path in flip_records:
        record = s.load(path)
        for name, expected in record['sha256'].items():
            assert s.sha(path.parent/name) == expected
        assert record['identity']['executable_sha256'] == original['flip']['sha256']
        for kind in ('reference', 'test'):
            command = record['identity']['command']
            source = Path(command[command.index('-r' if kind == 'reference' else '-t')+1])
            assert s.sha(source) == record['identity'][kind+'_sha256']
    s.save(OUT/'failures.json', [])
    s.status(args, complete=True)
    verification = s.load(OUT/'verification.json')
    verification.update(prior_pairs_preserved=len(old_metrics['comparisons']),
        native_flip_records=120, original_render_files_unchanged=True,
        all_render_and_flip_hashes_verified=True, report_runner_sha256=s.sha(Path(s.__file__)),
        warning_acceptance_sha256=s.sha(OUT/'warnings.json'),
        figure_sha256={p.relative_to(OUT).as_posix():s.sha(p) for p in (OUT/'figures').glob('*/*.png')})
    s.save(OUT/'verification.json', verification)
    print('PASS: 40 renders, 60 pairs, 120 native FLIP records; 2 accepted warnings; 54 prior pairs unchanged.')


if __name__ == '__main__':
    main()
