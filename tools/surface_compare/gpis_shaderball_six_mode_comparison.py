"""Seven matched shaderball modes and all fourteen requested native-FLIP pairs.

Run with the project's plot Python. This reuses verified render outputs; it
does not change renderer code, scene parameters, or the archived experiments.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from figures import Image
from run import load, save, sha, pfm
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links
from gpis_shaderball_renewal import ROOT, BASE, REPORT, GROUPS, SEED, original, verified
from gpis_shaderball_renewal_3d import destination as renewal_destination
from gpis_shaderball_conditional_surface_policies import destination as conditional_destination
from gpis_shaderball_shadow_ablation_report import grayscale_flip, image_array

OUT = REPORT/'six_mode_surface_comparison'
BOUNDS = (44, 164, 213, 216)
PPD = 67.0
METHODS = ('classic_local', 'classic_global', 'start_conditioning', 'neural')
REFERENCES = ('gpis_global', 'gpis_renewal_plus', 'gpis_renewal_plus_surface_conditioned')
MODES = METHODS + REFERENCES
LABELS = dict(zip(MODES, (
    'classic_local', 'classic_global', 'start_conditioning', 'neural', 'GPIS global',
    'GPIS renewal+', 'GPIS renewal+（非 GPIS 表面保留实际 F 和梯度）')))
PAIRS = tuple((test, ref) for test in METHODS for ref in REFERENCES) + (
    ('gpis_renewal_plus', 'gpis_global'),
    ('gpis_renewal_plus_surface_conditioned', 'gpis_global'))


def relative(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def crop(pixels):
    x0, y0, x1, y1 = BOUNDS
    return pixels[y0:y1, x0:x1]


def normalized_scene(scene):
    result = copy.deepcopy(scene)
    result['renderer'].pop('output_directory')
    return result


def provenance_records():
    experiment_file = REPORT/'shadow_condition_ablation/experiment.json'
    experiment = load(experiment_file)
    assert experiment['status'] == 'complete_archived_variant'
    archive = experiment_file.parent
    assert sha(archive/'experiment.patch') == experiment['patch_sha256']
    for row in experiment['sources']:
        for stage in ('before', 'after'):
            assert sha(archive/f'source_{stage}'/row['path']) == row[f'{stage}_sha256']
    assert sha(Path(experiment['before_executable'])) == experiment['before_renderer_sha256']
    assert sha(Path(experiment['experimental_executable'])) == experiment['experimental_renderer_sha256']
    conditional = load(REPORT/'conditional_surface_policies/compatibility.json')
    bridge = load(REPORT/'renderer_compatibility.json')
    assert conditional['passed'] and bridge['passed']
    assert all(row['passed'] for row in bridge['checks'])
    records = []
    for path in (experiment_file, REPORT/'conditional_surface_policies/compatibility.json',
                 REPORT/'conditional_surface_policies/implementation.json', REPORT/'renderer_compatibility.json'):
        records.append(dict(path=relative(path), sha256=sha(path)))
    return experiment, conditional, records


def read_group(key, experiment, conditional):
    folders = {
        'classic_local': original(key, 'classic_local'),
        'classic_global': original(key, 'classic_global'),
        'start_conditioning': conditional_destination(key, 'retain_endpoint'),
        'neural': original(key, 'neural'),
        'gpis_global': original(key, 'sparse_3d_global'),
        'gpis_renewal_plus': renewal_destination(key, 'renewal_3d'),
        'gpis_renewal_plus_surface_conditioned': renewal_destination(key, 'renewal_3d_shadow_conditioned'),
    }
    data = {mode: verified(folder) for mode, folder in folders.items()}
    reference = data['classic_global']['scene']
    gp = reference['media'][0]['gaussian_process']
    sigma = .02 if key.endswith('0p02') else .04
    assert gp['covariance'] == dict(type='squared_exponential', lengthScale=.2, sigma=sigma)
    rows = []
    for mode, d in data.items():
        folder = folders[mode]
        result, scene = d['source']['result'], d['scene']
        assert result['resolution'] == [256, 256] and result['spp'] == 128
        assert folder.name == f'seed{SEED}'
        assert result.get('seed', SEED) == SEED
        assert np.isfinite(d['rgb']).all() and (d['rgb'] >= 0).all()
        for section in ('primitives', 'bsdfs', 'camera', 'integrator', 'renderer'):
            actual, expected = copy.deepcopy(scene[section]), copy.deepcopy(reference[section])
            if section == 'renderer':
                actual.pop('output_directory'); expected.pop('output_directory')
            assert actual == expected, (key, mode, section)
        medium = scene['media'][0]
        material = medium['material'] if mode in METHODS else medium['phase_function']['bsdf']
        assert material == reference['media'][0]['material']
        if mode in ('classic_local', 'neural'):
            assert medium['type'] == ('neural_causal_renewal' if mode == 'neural' else mode)
            assert medium['covariance'] == gp['covariance']
            field = Path(medium['field']['file'])
            metadata = load(Path(str(field)+'.json'))
            assert sha(field) == result['provenance']['field_sha256']
            assert metadata['source_mean'] == gp['mean'] and metadata['sigma'] == sigma
            assert metadata['dx'] == .005
            d['field'] = dict(path=str(field), sha256=sha(field),
                              metadata_sha256=sha(Path(str(field)+'.json')), dx=.005)
        else:
            assert medium['gaussian_process'] == gp, (key, mode, 'gaussian_process')
        if mode == 'classic_global':
            assert medium['type'] == mode
        if mode == 'neural':
            model = Path(medium['model'])
            assert medium['backend'] == 'scalar'
            assert sha(model) == result['provenance']['model_sha256']
            model_info = load(model)
            assert model_info['format'] == 'macrofacet.causal_renewal' and model_info['version'] == 1
            assert model_info['mean_convention'] == 'b=m/sigma'
            assert model_info['distance_convention'] == 'q=distance_from_birth/ell'
            run_file = folder.parent.parent/'run_neural.json'
            run = load(run_file)
            expected = dict(ell=.2, sigma=sigma, stage='final', size=256, spp=128, seed=SEED)
            assert all(run['parameters'][name] == value for name,value in expected.items())
            matching = [item for item in run['sources'] if Path(item['folder']) == folder]
            assert len(matching) == 1 and matching[0]['result'] == result
            d['model'] = dict(path=str(model), sha256=sha(model), backend=medium['backend'],
                format=model_info['format'], version=model_info['version'],
                mean_convention=model_info['mean_convention'],
                distance_convention=model_info['distance_convention'],
                run_record=str(run_file), run_record_sha256=sha(run_file))
        if mode == 'start_conditioning':
            assert medium['type'] == 'global_conditional'
            assert medium['non_gp_surface_policy'] == 'retain_endpoint'
            assert result['provenance']['renderer_sha256'] == conditional['current_renderer_sha256']
            assert result['discarded_surface_births'] == dict(paths=0, shadow_connections=0)
            archived = original(key, 'global_conditional')
            verified(archived)
            for name in ('render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm'):
                assert np.array_equal(pfm(folder/name), pfm(archived/name))
            d['matches_archived_conditional'] = True
        if mode in REFERENCES:
            assert medium['type'] == 'sparse_conv_noise'
            assert not medium['1D_sampling'] and not medium['single_realization']
            assert medium['1D_sampling_scheme'] == 'uni' and not medium['1D_gradient_correlationXY']
            assert medium['step_size'] == .01 and medium['impulse_density'] == 10
            assert medium['correlation_context'] == ('global' if mode == 'gpis_global' else 'renewal+')
        if mode == 'gpis_renewal_plus':
            assert result['provenance']['renderer_sha256'] == experiment['before_renderer_sha256']
        if mode == 'gpis_renewal_plus_surface_conditioned':
            assert result['provenance']['renderer_sha256'] == experiment['experimental_renderer_sha256']
        d['png'] = folder/'render.png'
        assert image_array(d['png']).shape == (256, 256, 3)
        rows.append(dict(group='A' if sigma == .02 else 'B', mode=mode, label=LABELS[mode],
            folder=str(folder), internal_mode=medium['type'], result=result,
            result_sha256=sha(folder/'result.json'), render_log_sha256=sha(folder/'render.log'),
            field=d.get('field'), model=d.get('model'),
            matches_archived_conditional=d.get('matches_archived_conditional')))
    sparse_global = normalized_scene(data['gpis_global']['scene'])
    sparse_global['media'][0]['correlation_context'] = 'renewal+'
    assert sparse_global == normalized_scene(data['gpis_renewal_plus']['scene'])
    assert sparse_global == normalized_scene(data['gpis_renewal_plus_surface_conditioned']['scene'])
    return dict(key=key, label='A' if sigma == .02 else 'B', sigma=sigma,
                data=data, images={mode:d['png'] for mode,d in data.items()}, flip={}), rows


def linear_statistics(test, reference, region):
    select = crop if region == 'bottom' else lambda pixels: pixels
    a, b = select(test['y']), select(reference['y'])
    delta = a-b
    half_delta = [select(x-y) for x,y in zip(test['halves'], reference['halves'])]
    assert np.allclose(delta, .5*(half_delta[0]+half_delta[1]), rtol=1e-4, atol=3e-7)
    noise = .5*(half_delta[0]-half_delta[1])
    return dict(pixels=delta.size, test_mean_y=float(a.mean()), reference_mean_y=float(b.mean()),
        mean_delta_y=float(delta.mean()), relative_mean_delta=float(a.mean()/b.mean()-1),
        luminance_rmse=float(np.sqrt(np.mean(delta*delta))),
        approximate_paired_mean_se=float(np.sqrt(np.sum(noise*noise))/delta.size))


def known_flip_results():
    known = {}
    reports = [REPORT/report/'metrics.json'
               for report in ('shadow_condition_ablation', 'conditional_surface_policies')]
    prior = OUT/'history/before_neural/metrics.json'
    if prior.exists():
        reports.append(prior)
    for report in reports:
        for row in load(report)['flip_records']:
            result = row['result']
            identity = result['identity']
            key = (identity['reference_sha256'], identity['test_sha256'], identity['executable_sha256'])
            if key in known:
                assert known[key] == result['statistics']
            known[key] = result['statistics']
    return known


def calculate_flip(group, known):
    comparisons, records, historical_checks = [], [], 0
    for mode, d in group['data'].items():
        path = OUT/'inputs'/group['label']/mode/'bottom.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop(image_array(d['png']))).save(path)
        d['bottom_png'] = path
    for test, reference in PAIRS:
        pair = f'{test}_vs_{reference}'
        folder = OUT/'flip'/group['label']/pair
        a, b = group['data'][test], group['data'][reference]
        full = evaluate(DEFAULT_FLIP, b['png'], a['png'], folder/'full', PPD)
        bottom = evaluate(DEFAULT_FLIP, b['bottom_png'], a['bottom_png'], folder/'bottom_input', PPD)
        gray, errors = grayscale_flip(b['png'], a['png'], folder/'full_gray', full)
        for kind, result in (('full', full), ('bottom_input', bottom), ('full_gray', gray)):
            record = folder/kind/'result.json'
            records.append(dict(group=group['label'], pair=pair, kind=kind, path=relative(record),
                                sha256=sha(record), result=result))
            identity = result['identity']
            old_key = (identity['reference_sha256'], identity['test_sha256'], identity['executable_sha256'])
            if old_key in known:
                assert result['statistics'] == known[old_key], (group['label'], pair, kind)
                historical_checks += 1
        group['flip'][pair] = dict(test=test, reference=reference,
            error_map=folder/'full'/full['error_map'],
            bottom_error_map=folder/'bottom_input'/bottom['error_map'],
            full=full['statistics'], bottom=bottom['statistics'])
        comparisons.append(dict(group=group['label'], ell=.2, sigma=group['sigma'], pair=pair,
            test=test, reference=reference, test_label=LABELS[test], reference_label=LABELS[reference],
            full=full['statistics'], bottom=bottom['statistics'],
            full_map=relative(group['flip'][pair]['error_map']),
            bottom_input_map=relative(group['flip'][pair]['bottom_error_map']),
            bottom_full_context_quantized_mean=float(crop(errors).mean()),
            quantization_absolute_bound=1/255,
            linear={region:linear_statistics(a,b,region) for region in ('full','bottom')}))
        print(group['label'], test, 'vs', reference,
              'FLIP', full['statistics']['mean'], 'bottom', bottom['statistics']['mean'], flush=True)
    assert len(group['flip']) == len(PAIRS)
    return comparisons, records, historical_checks


def write_csv(comparisons):
    fields = ('group','ell','sigma','test','reference','region','mean_flip','weighted_median',
              'maximum','mean_delta_y','relative_mean_delta','luminance_rmse')
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for region in ('full','bottom'):
                stats = comparison[region]
                row = {name:comparison[name] for name in ('group','ell','sigma')}
                row.update(test=comparison['test_label'], reference=comparison['reference_label'],
                           region=region, mean_flip=stats['mean'], weighted_median=stats['weighted_median'],
                           maximum=stats['maximum'])
                row.update({name:comparison['linear'][region][name]
                            for name in ('mean_delta_y','relative_mean_delta','luminance_rmse')})
                writer.writerow(row)


def link_existing_reports():
    """Add a checked navigation link while preserving each historical report."""
    updates = []
    for name in ('', 'shadow_condition_ablation', 'conditional_surface_policies'):
        folder = REPORT/name
        page, record = folder/'comparison.html', folder/'verification.json'
        verification = load(record)
        assert sha(page) == verification['generated_sha256']['comparison.html']
        backup = OUT/'history'/((name or 'main')+'_comparison_before_six_mode.html')
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(page.read_bytes())
        body = page.read_text(encoding='utf-8')
        marker = 'six-mode-surface-comparison'
        body = re.sub(r'\n?<p class="note" id="'+marker+r'">.*?</p>', '', body)
        target = Path(os.path.relpath(OUT/'comparison.html',folder)).as_posix()
        note = (f'\n<p class="note" id="{marker}"><b>七模式完整比较（含 neural）：</b> '
                f'<a href="{html.escape(target,quote=True)}">classic_local、classic_global、start_conditioning、neural '
                '与三个 GPIS 参考的全部 FLIP，以及两种 Renewal+ 对 Global 的 FLIP</a>。'
                '包含非 GPIS 表面保留实际 F 和梯度的 Renewal+ 变体，A/B 两组共 28 个配对。</p>')
        assert '</h1>' in body
        page.write_text(body.replace('</h1>', '</h1>'+note, 1), encoding='utf-8')
        verification['generated_sha256']['comparison.html'] = sha(page)
        verification['local_links'] = verify_links(page)
        verification['six_mode_surface_comparison'] = dict(report=target,
            prior_html_backup=str(backup), prior_html_sha256=sha(backup), historical_figures_preserved=True)
        save(record,verification)
        updates.append(dict(page=relative(page), sha256=sha(page), backup=relative(backup),
                            backup_sha256=sha(backup)))
    return updates


def main(args):
    OUT.mkdir(parents=True,exist_ok=True)
    # Preserve the previous 22 pairs so an additive update cannot silently
    # change their inputs or numerical results. The public report URL is stable.
    prior_file = OUT/'history/before_neural/metrics.json'
    if not prior_file.exists() and (OUT/'metrics.json').exists():
        previous = load(OUT/'metrics.json')
        if 'neural' not in previous['modes']:
            verification = load(OUT/'verification.json')
            assert sha(OUT/'metrics.json') == verification['generated_sha256']['metrics.json']
            prior_file.parent.mkdir(parents=True, exist_ok=True)
            prior_file.write_bytes((OUT/'metrics.json').read_bytes())
            (prior_file.parent/'verification.json').write_bytes((OUT/'verification.json').read_bytes())
    previous = load(prior_file) if prior_file.exists() else None
    experiment, conditional, provenance = provenance_records()
    groups, sources = [], []
    for key in GROUPS:
        group, rows = read_group(key, experiment, conditional)
        groups.append(group); sources.extend(rows)
    assert len(sources) == len(GROUPS)*len(MODES)
    help_result = subprocess.run([str(DEFAULT_FLIP),'--help'], capture_output=True,
        text=True, encoding='utf-8', errors='replace', check=True)
    help_text = help_result.stdout+help_result.stderr
    assert 'FLIP v1.7.' in help_text
    (OUT/'flip_tool_help.txt').write_text(help_text,encoding='utf-8')
    comparisons, records, checked = [], [], 0
    known = known_flip_results()
    for group in groups:
        rows, provenance_rows, count = calculate_flip(group,known)
        comparisons.extend(rows); records.extend(provenance_rows); checked += count
    assert len(comparisons) == len(GROUPS)*len(PAIRS) and len(records) == 3*len(comparisons)
    retained_pairs = 0
    if previous:
        current = {(row['group'],row['pair']):row for row in comparisons}
        for row in previous['comparisons']:
            assert current[row['group'],row['pair']] == row, 'Existing comparison changed'
            retained_pairs += 1
    metrics = dict(complete=True, groups=list(GROUPS), modes=list(MODES), labels=LABELS,
        display_alias=dict(start_conditioning=dict(internal_mode='global_conditional',
            non_gp_surface_policy='retain_endpoint')),
        settings=dict(resolution=[256,256],spp=128,seed=SEED,ell=.2,sigma=[.02,.04]),
        pair_count_per_group=len(PAIRS),bottom_bounds=list(BOUNDS),sources=sources,
        comparisons=comparisons,flip_records=records,provenance=provenance,
        prior_pairs_preserved=retained_pairs,
        previous_metrics=(dict(path=relative(prior_file),sha256=sha(prior_file)) if previous else None),
        historical_flip_statistics_matches=checked,
        tool=dict(path=str(DEFAULT_FLIP),sha256=sha(DEFAULT_FLIP),version='1.7',ppd=PPD),
        definitions=dict(input='Original filmic PNG, no exposure fitting or denoising.',
            bottom_maps='Crops of full-image FLIP maps for consistent filter context.',
            bottom_statistics='Native FLIP computed on lossless cropped input PNGs; filter context differs.',
            scope='Matched common scene settings; algorithms, field representations and historical renderer binaries differ.',
            retained_variant='Archived sparse 3D Renewal+ retains actual endpoint F and gradient on same-medium non-GP surface direct-light connections with valid prior state; main continuation and GP collision rules unchanged.',
            reference='The 128 spp GPIS references contain sampling and numerical approximation error.'))
    save(OUT/'metrics.json',metrics)
    write_csv(comparisons)
    if args.metrics_only:
        print('Metrics complete:',OUT/'metrics.json',flush=True)
        return
    from gpis_shaderball_six_mode_view import write_report
    generated = write_report(OUT,groups,sources)
    assert 'comparison.html' in generated
    files = list(dict.fromkeys(['metrics.json','metrics.csv','flip_tool_help.txt',*generated]))
    scripts = [Path(__file__),Path(__file__).with_name('gpis_shaderball_six_mode_view.py'),
               Path(__file__).with_name('gpis_shaderball_flip.py')]
    verification = dict(passed=True,render_records=len(sources),comparison_pairs=len(comparisons),
        flip_records=len(records),historical_flip_statistics_matches=checked,
        prior_pairs_preserved=retained_pairs,
        generated_sha256={name:sha(OUT/name) for name in files},
        scripts={str(path.relative_to(ROOT)):sha(path) for path in scripts},
        input_sha256_verified=True,common_scene_parameters_matched=True,
        fresh_retain_bit_identical_to_archived=True)
    save(OUT/'verification.json',verification)
    verification['local_links'] = verify_links(OUT/'comparison.html')
    verification['linked_reports'] = link_existing_reports()
    save(OUT/'verification.json',verification)
    print('Verified:',len(sources),'renders,',len(comparisons),'pairs,',len(records),'native FLIP records',flush=True)
    print(OUT/'comparison.html',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-only',action='store_true')
    main(parser.parse_args())
