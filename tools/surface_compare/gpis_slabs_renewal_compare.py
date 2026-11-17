"""Add matched Renewal+ renders to an existing slab CDF comparison.

python -B tools/surface_compare/gpis_slabs_renewal_compare.py
python -B tools/surface_compare/gpis_slabs_renewal_compare.py --output outputs/gpis_three_slabs_compare_20261116/smoke
"""
from __future__ import annotations

import argparse
import copy
import html
from pathlib import Path
import re

import numpy as np
from gpis_two_slabs_cdf5 import BASE, METHODS, masks, render
from run import ROOT, load, pfm, save, sha

RENEWAL = 'gpis_renewal_plus'
LABELS = {'numeric': 'Single history / numeric',
          'gauss_legendre5': 'Single history / five-point CDF',
          RENEWAL: 'Renewal+ / 1D MIS'}
BEGIN = '<!-- BEGIN SLABS RENEWAL COMPARISON -->'
END = '<!-- END SLABS RENEWAL COMPARISON -->'


def common_scene(scene):
    result = copy.deepcopy(scene)
    medium = result.pop('media')[0]
    result['renderer'].pop('output_directory')
    result['shared_medium'] = dict(name=medium['name'], max_bounces=medium['max_bounces'],
                                  gaussian_process=medium['gaussian_process'],
                                  material=medium.get('material', medium.get('phase_function', {}).get('bsdf')))
    return result


def verify_record(args, record):
    directory = args.output / record['method'] / f'seed{record["seed"]}'
    assert record == load(directory / 'result.json')
    assert record['scene'] == load(directory / 'scene.json')
    assert record['renderer_sha256'] == sha(args.renderer)
    job = load(directory / 'job.json')['command']
    assert job[job.index('-t') + 1] == str(args.threads)
    assert job[job.index('-s') + 1] == str(record['seed'])
    for name, digest in record['sha256'].items():
        assert sha(directory / name) == digest, str(directory / name)
    log = (directory / 'render.log').read_text(encoding='utf-8', errors='replace')
    assert 'Finished render.' in log and f'Completed {args.spp}/{args.spp} spp' in log
    assert not re.search(r'internal error|unrecoverable|sanity check failed|Unable to load|'
                         r'Gradient invalid|Failed to sample gradient|Conditioning .* error', log, re.I)
    pixels, variance = [pfm(directory / name) for name in ('render.pfm', 'samplesVariance.pfm')]
    assert pixels.shape == variance.shape == (args.size, args.size, 3)
    assert pixels.max() <= 1.00001
    luminance = pixels @ np.array([.2126, .7152, .0722])
    assert record['roi'] == {name: float(luminance[mask].mean())
                             for name, mask in masks(args.size, args.slabs).items()}
    if record['method'] == RENEWAL:
        # Absorbing visibility paths contribute zero or one.
        assert np.max(np.abs(pixels * args.spp - np.round(pixels * args.spp))) < 2e-3
        assert np.allclose(variance * (args.spp - 1), pixels * (1 - pixels), rtol=2e-4, atol=2e-6)
    return pixels


def write_report(args, records, comparisons, averages):
    left = 'left_double' if args.slabs == 3 else 'left_single'
    summary = []
    for method, row in averages.items():
        summary.append(f'<tr><td>{LABELS[method]}</td><td>{row["overlap"]:.6f}</td>'
                       f'<td>{row[left]:.6f}</td><td>{row["right_single"]:.6f}</td>'
                       f'<td>{row["render_seconds"]:.2f}</td></tr>')
    details = []
    for seed in args.seeds:
        rows, pictures = [], []
        for record in (r for r in records if r['seed'] == seed):
            relative = f'{record["method"]}/seed{seed}'
            rows.append(f'<tr><td>{LABELS[record["method"]]}</td>'
                        f'<td>{record["roi"]["overlap"]:.6f}</td><td>{record["render_seconds"]:.2f}</td>'
                        f'<td><a href="{relative}/render.pfm">PFM</a> / '
                        f'<a href="{relative}/scene.json">scene</a> / '
                        f'<a href="{relative}/render.log">log</a></td></tr>')
            pictures.append(f'<figure><a href="{relative}/render.png"><img src="{relative}/render.png" '
                            f'alt="{html.escape(LABELS[record["method"]])}; seed {seed}"></a>'
                            f'<figcaption>{LABELS[record["method"]]}</figcaption></figure>')
        deltas = ''.join(f'<tr><td>{LABELS[r["method"]]}</td>'
                         f'<td>{r["overlap_delta"]:+.6f}</td><td>{r["image_rmse"]:.6f}</td></tr>'
                         for r in comparisons if r['seed'] == seed)
        details.append(f'<h3>Seed {seed}</h3><div class="images">' + ''.join(pictures) + '</div>'
                       '<table><tr><th>Method</th><th>Overlap transmittance</th><th>Seconds</th><th>Files</th></tr>'
                       + ''.join(rows) + '</table><table><tr><th>Compared with Renewal+</th>'
                       '<th>Overlap delta (method - Renewal+)</th><th>Linear RGB RMSE</th></tr>'
                       + deltas + '</table>')
    section = (BEGIN + '<section id="renewal-comparison"><h2>Renewal+ comparison</h2>'
               f'<p>{args.size}&times;{args.size}, {args.spp} spp, {args.threads} threads, '
               f'{len(args.seeds)} seed(s). Same renderer, mean field, covariance, material, camera and medium boundary.</p>'
               '<p>Renewal+: 1D sparse convolution, MIS, correlated transverse gradients; '
               'step 0.0025, impulse density 10. Single history uses numeric or fixed five-point CDF.</p>'
               '<h3>Mean across seeds</h3><table><tr><th>Method</th>'
               f'<th>{args.slabs}-layer overlap</th><th>Left: {args.slabs - 1} layer(s)</th>'
               '<th>Right: 1 layer</th><th>Seconds</th></tr>' + ''.join(summary) + '</table>'
               + ''.join(details) + '<p>Finite-spp comparison; Renewal+ is a sampled reference, not exact ground truth. '
               'Differences include field approximation, marching step and sampling noise. '
               'Absorbing transmission probes survival along one segment; it does not test history after a reflection.</p>'
               '<p><a href="renewal_metrics.json">Combined metrics</a> / '
               '<a href="renewal_manifest.json">Renewal+ configuration</a> / '
               '<a href="renewal_verification.json">Validation</a></p></section>' + END)
    path = args.output / 'comparison.html'
    page = path.read_text(encoding='utf-8')
    if BEGIN in page:
        assert page.count(BEGIN) == page.count(END) == 1
        start, end = page.index(BEGIN), page.index(END) + len(END)
        page = page[:start] + section + page[end:]
    else:
        end = page.index('</h1>') + len('</h1>')
        page = page[:end] + section + page[end:]
    title = f'{"Three" if args.slabs == 3 else "Two"} slabs: single history / Renewal+'
    page = re.sub(r'<title>.*?</title>', f'<title>{title}</title>', page, count=1)
    page = re.sub(r'<h1>.*?</h1>', f'<h1>{title}</h1>', page, count=1)
    path.write_text(page, encoding='utf-8')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/gpis_three_slabs_compare_20261116')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    baseline = load(args.output / 'manifest.json')
    metrics = load(args.output / 'metrics.json')
    assert metrics['complete'] and len(metrics['records']) == 2 * len(baseline['seeds'])
    args.renderer = Path(baseline['renderer'])
    args.size, args.spp, args.threads, args.seeds = [baseline[k] for k in ('size', 'spp', 'threads', 'seeds')]
    args.slabs = baseline.get('geometry', {}).get('slabs', 2)
    assert sha(args.renderer) == baseline['renderer_sha256'], 'Renderer changed since CDF comparison'
    for record in metrics['records']:
        verify_record(args, record)
    source = BASE / 'final/gpis_renewal_plus/seed20261024/scene.json'
    source_scene = load(source)
    template = copy.deepcopy(metrics['records'][0]['scene'])
    old_medium = template['media'][0]
    template['media'] = copy.deepcopy(source_scene['media'])
    template['media'][0]['gaussian_process'] = copy.deepcopy(old_medium['gaussian_process'])
    assert template['media'][0]['phase_function']['bsdf'] == old_medium['material']
    assert common_scene(template) == common_scene(metrics['records'][0]['scene'])
    manifest = dict(source=str(source), source_sha256=sha(source),
                    baseline_manifest_sha256=sha(args.output / 'manifest.json'),
                    baseline_metrics_sha256=sha(args.output / 'metrics.json'),
                    renderer=str(args.renderer), renderer_sha256=sha(args.renderer),
                    size=args.size, spp=args.spp, threads=args.threads, seeds=args.seeds,
                    methods=[*METHODS, RENEWAL], medium=template['media'][0],
                    notes=['Renewal+ medium copied from the original two-slab experiment; only its mean field changes for three slabs.',
                           'All non-medium scene settings and the GP, material and medium name match the CDF comparison.',
                           'Existing CDF renders are verified and reused.',
                           'Renewal+ uses sampled sparse convolution; the CDF setting applies only to single history.'])
    manifest_path = args.output / 'renewal_manifest.json'
    if manifest_path.exists():
        assert load(manifest_path) == manifest
    else:
        save(manifest_path, manifest)
    renewal = [render(args, template, RENEWAL, seed) for seed in args.seeds]
    records = [r for seed in args.seeds for r in [*metrics['records'], *renewal] if r['seed'] == seed]
    images, comparisons = {}, []
    for record in records:
        images[record['method'], record['seed']] = verify_record(args, record)
        assert common_scene(record['scene']) == common_scene(template)
    for seed in args.seeds:
        ref = next(r for r in renewal if r['seed'] == seed)
        for row in (r for r in records if r['seed'] == seed and r['method'] in METHODS):
            comparisons.append(dict(seed=seed, method=row['method'], reference=RENEWAL,
                                    overlap_delta=row['roi']['overlap'] - ref['roi']['overlap'],
                                    image_rmse=float(np.sqrt(np.mean((images[row['method'], seed] - images[RENEWAL, seed]) ** 2)))))
    averages = {}
    for method in [*METHODS, RENEWAL]:
        rows = [r for r in records if r['method'] == method]
        averages[method] = {name: float(np.mean([r['roi'][name] for r in rows])) for name in masks(args.size, args.slabs)}
        averages[method]['render_seconds'] = float(np.mean([r['render_seconds'] for r in rows]))
    save(args.output / 'renewal_metrics.json', dict(complete=True, records=records,
                                                  comparisons=comparisons, averages=averages))
    page = write_report(args, records, comparisons, averages)
    links = re.findall(r'(?:src|href)="([^"]+)"', page)
    for link in links:
        if link != 'renewal_verification.json':
            assert (args.output / html.unescape(link)).is_file(), link
    save(args.output / 'renewal_verification.json',
         dict(passed=True, complete=True, render_records=len(records), new_render_records=len(renewal),
              all_render_hashes_verified=True, matched_common_scene=True, matched_renderer=True,
              matched_threads_and_seeds=True, renewal_bernoulli_variance_verified=True,
              local_links=len(links), baseline_verification_sha256=sha(args.output / 'verification.json')
              if (args.output / 'verification.json').exists() else None,
              report_sha256=sha(args.output / 'comparison.html'),
              source_sha256={p: sha(ROOT / p) for p in ('tools/surface_compare/gpis_slabs_renewal_compare.py',
                                                       'tools/surface_compare/gpis_two_slabs_cdf5.py')}))
    print(averages, flush=True)
    print(args.output / 'comparison.html', flush=True)


if __name__ == '__main__':
    main()
