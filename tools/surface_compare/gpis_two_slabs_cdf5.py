"""Compare numeric and fixed five-point CDFs on shared-field slab scenes.

python tools/surface_compare/gpis_two_slabs_cdf5.py --size 64 --spp 256
python tools/surface_compare/gpis_two_slabs_cdf5.py --slabs 3
"""
from __future__ import annotations

import argparse
import copy
import html
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from run import ROOT, load, pfm, save, sha

BASE = ROOT / 'outputs/gpis_two_slabs_compare_20261113'
METHODS = ('numeric', 'gauss_legendre5')


def masks(size, slabs=2):
    v = (np.arange(size) + .5) / size * 2 - 1
    x, z = np.meshgrid(v * 2 * np.tan(np.deg2rad(22)), v * 2 * np.tan(np.deg2rad(22)))
    # The camera looks down -y with +z up, so screen x maps to world -x.
    result = dict(overlap=(np.abs(x) < .30) & (np.abs(z) < .40))
    result['left_double' if slabs == 3 else 'left_single'] = (x > -.69) & (x < -.51) & (np.abs(z) < .40)
    result['right_single'] = (x > .51) & (x < .69) & (np.abs(z) < .40)
    return result


def render(args, template, method, seed):
    directory = args.output / method / f'seed{seed}'
    scene = copy.deepcopy(template)
    if method in METHODS:
        scene['media'][0]['single_history']['probability_method'] = method
    else:
        assert method == 'gpis_renewal_plus'
        assert scene['media'][0]['type'] == 'sparse_conv_noise'
        assert scene['media'][0]['correlation_context'] == 'renewal+'
    scene['camera']['resolution'] = [args.size, args.size]
    scene['renderer'].update(output_directory=str(directory), spp=args.spp,
                             spp_step=min(8, args.spp), enable_resume_render=False,
                             checkpoint_interval='0')
    scene_path = directory / 'scene.json'
    record_path = directory / 'result.json'
    if record_path.exists():
        record = load(record_path)
        assert record['scene'] == scene and record['renderer_sha256'] == sha(args.renderer)
        for name, digest in record['sha256'].items():
            assert sha(directory / name) == digest
        image = pfm(directory / 'render.pfm')
        brightness = image @ np.array([.2126, .7152, .0722])
        roi = {name: float(brightness[mask].mean()) for name, mask in masks(args.size, args.slabs).items()}
        if record['roi'] != roi:
            record['roi'] = roi
            save(record_path, record)
        return record
    if scene_path.exists():
        raise RuntimeError(f'Unverified existing attempt retained: {directory}')
    save(scene_path, scene)
    command = [str(args.renderer), '-t', str(args.threads), '-s', str(seed),
               '-d', str(directory), str(scene_path)]
    save(directory / 'job.json', dict(command=command))
    print('START', method, seed, flush=True)
    start = time.perf_counter()
    with (directory / 'render.log').open('wb') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.timeout,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    wall = time.perf_counter() - start
    text = (directory / 'render.log').read_text(encoding='utf-8', errors='replace')
    if (result.returncode or 'Finished render.' not in text or
            f'Completed {args.spp}/{args.spp} spp' not in text or
            re.search(r'internal error|unrecoverable error|sanity check failed|Unable to load', text, re.I)):
        raise RuntimeError(f'Render failed: {directory / "render.log"}')
    image = pfm(directory / 'render.pfm')
    assert image.shape == (args.size, args.size, 3) and image.max() <= 1.00001
    brightness = image @ np.array([.2126, .7152, .0722])
    timing = re.findall(r'Completed \d+/\d+ spp,\s*([\d.]+)/[\d.]+ sec', text)
    files = ('scene.json', 'render.log', 'render.png', 'render.pfm', 'samplesVariance.pfm')
    record = dict(method=method, seed=seed, scene=scene, renderer_sha256=sha(args.renderer),
                  render_seconds=float(timing[-1]), wall_seconds=wall,
                  roi={name: float(brightness[mask].mean()) for name, mask in masks(args.size, args.slabs).items()},
                  sha256={name: sha(directory / name) for name in files})
    save(record_path, record)
    print('DONE', method, seed, record['render_seconds'], record['roi']['overlap'], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path,
                        default=ROOT / 'build/Release/tungsten.exe')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--slabs', type=int, choices=(2, 3), default=2)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--spp', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261024, 20261113])
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / 'outputs/gpis_three_slabs_compare_20261116' if args.slabs == 3 else BASE / 'cdf5_comparison_20261116'
    args.output, args.renderer = args.output.resolve(), args.renderer.resolve()
    if min(args.size, args.spp, args.threads) <= 0 or args.spp % min(8, args.spp):
        parser.error('size/threads must be positive and spp must be divisible by its spp_step')
    source = BASE / 'single_history_analytic_1024_comparison/final/single_history_gpis/seed20261024/scene.json'
    template = load(source)
    if args.slabs == 3:
        template['media'][0]['gaussian_process']['mean']['func'] = 'three_slabs'
    manifest = dict(source=str(source), source_sha256=sha(source), renderer=str(args.renderer),
                    renderer_sha256=sha(args.renderer), size=args.size, spp=args.spp,
                    threads=args.threads, seeds=args.seeds, methods=METHODS,
                    notes=['Same executable, scene, sampling quality, thread count and seeds for both methods.',
                           'Five-point bivariate CDF and fixed five-point recursive higher-dimensional CDF; no high-precision CDF fallback.',
                           'CDF approximation does not alter the existing speed/flux integration algorithms.',
                           'Image differences include finite-spp noise; they do not certify a converged bias.'])
    if args.slabs == 3:
        manifest['geometry'] = dict(slabs=3, centers=[[.18, .02, 0], [-.18, -.02, 0], [.18, -.06, 0]],
                                    size=[1.2, .02, 1.2], center_spacing_y=.04, face_gap=.02,
                                    layout='A/B/A: the third slab aligns with the first slab in x',
                                    shared_process=True)
        manifest['notes'].append('Only the GP mean changes from two_slabs to three_slabs; camera, bounds, material, covariance and tracking settings are preserved.')
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert load(manifest_path) == dict(manifest, methods=list(METHODS))
    else:
        save(manifest_path, manifest)
    records, comparisons = [], []
    for seed in args.seeds:
        rows = {method: render(args, template, method, seed) for method in METHODS}
        records.extend(rows.values())
        images = {method: pfm(args.output / method / f'seed{seed}/render.pfm') for method in METHODS}
        comparisons.append(dict(seed=seed,
                                speedup=rows['numeric']['render_seconds'] / rows['gauss_legendre5']['render_seconds'],
                                image_rmse=float(np.sqrt(np.mean((images['numeric'] - images['gauss_legendre5']) ** 2))),
                                overlap_delta=rows['gauss_legendre5']['roi']['overlap'] - rows['numeric']['roi']['overlap']))
        save(args.output / 'metrics.json', dict(complete=len(comparisons) == len(args.seeds),
                                               records=records, comparisons=comparisons))
    reference_section = ''
    if args.slabs == 3:
        reference_path = BASE / 'cdf5_comparison_20261116/metrics.json'
        if reference_path.exists():
            reference = load(reference_path)
            reference_rows, reference_pictures, layer_comparisons = [], [], []
            for record in records:
                old = next((r for r in reference['records']
                            if r['method'] == record['method'] and r['seed'] == record['seed']
                            and r['scene']['camera']['resolution'] == [args.size, args.size]
                            and r['scene']['renderer']['spp'] == args.spp), None)
                if old is None:
                    continue
                old_scene, new_scene = copy.deepcopy(old['scene']), copy.deepcopy(record['scene'])
                for scene in (old_scene, new_scene):
                    scene['media'][0]['gaussian_process']['mean'].pop('func')
                    scene['renderer'].pop('output_directory')
                assert old_scene == new_scene, 'Layer comparison changed another scene parameter'
                old_directory = reference_path.parent / old['method'] / f'seed{old["seed"]}'
                for name in ('render.png', 'render.pfm'):
                    assert sha(old_directory / name) == old['sha256'][name]
                delta = record['roi']['overlap'] - old['roi']['overlap']
                layer_comparisons.append(dict(method=record['method'], seed=record['seed'],
                                              two_layers=old['roi']['overlap'],
                                              three_layers=record['roi']['overlap'], delta=delta))
                reference_rows.append(f'<tr><td>{record["method"]}</td><td>{record["seed"]}</td>'
                                      f'<td>{old["roi"]["overlap"]:.6f}</td><td>{record["roi"]["overlap"]:.6f}</td>'
                                      f'<td>{delta:+.6f}</td></tr>')
                image_path = os.path.relpath(old_directory / 'render.png', args.output).replace('\\', '/')
                reference_pictures.append(f'<figure><img src="{html.escape(image_path)}">'
                                          f'<figcaption>Two slabs / {record["method"]} / {record["seed"]}</figcaption></figure>')
            if layer_comparisons:
                metrics = load(args.output / 'metrics.json')
                metrics['layer_comparisons'] = layer_comparisons
                save(args.output / 'metrics.json', metrics)
                reference_section = ('<h2>Two / three layers</h2>'
                                     '<p>Previous two-slab images use the same scene parameters, quality and seeds.</p>'
                                     '<table><tr><th>Method</th><th>Seed</th><th>Two layers</th><th>Three layers</th><th>Delta</th></tr>'
                                     + ''.join(reference_rows) + '</table><div class="images">'
                                     + ''.join(reference_pictures) + '</div>')
    rows, pictures = [], []
    left_region = 'left_double' if args.slabs == 3 else 'left_single'
    for record in records:
        relative = f'{record["method"]}/seed{record["seed"]}'
        rows.append(f'<tr><td>{record["method"]}</td><td>{record["seed"]}</td>'
                    f'<td>{record["render_seconds"]:.2f}</td><td>{record["roi"]["overlap"]:.6f}</td>'
                    f'<td>{record["roi"][left_region]:.6f}</td><td>{record["roi"]["right_single"]:.6f}</td>'
                    f'<td><a href="{relative}/scene.json">scene</a> / <a href="{relative}/render.log">log</a></td></tr>')
        pictures.append(f'<figure><img src="{relative}/render.png"><figcaption>'
                        f'{html.escape(record["method"])} / {record["seed"]}</figcaption></figure>')
    label = 'Three' if args.slabs == 3 else 'Two'
    page = (f'<!doctype html><meta charset="utf-8"><title>{label} slabs: five-point CDF trial</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:20px}'
            'td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}'
            '.images{display:flex;flex-wrap:wrap}figure{margin:12px}img{width:256px;image-rendering:pixelated}</style>'
            f'<h1>{label} slabs: numeric / fixed five-point CDF</h1>'
            f'<p>{args.size}×{args.size}, {args.spp} spp, {args.threads} threads per render. '
            'Fixed five-point CDF without high-precision CDF fallback. Same executable and scene for both modes.</p>'
            f'<p>{"A/B/A layout; one continuous GP field. Slab centers: (0.18,0.02,0), (-0.18,-0.02,0), (0.18,-0.06,0). Thickness 0.02; face gaps 0.02." if args.slabs == 3 else "Two slabs share one continuous GP field."}</p>'
            f'<table><tr><th>Method</th><th>Seed</th><th>Seconds</th><th>{args.slabs}-layer overlap</th>'
            f'<th>Left: {2 if args.slabs == 3 else 1} layer(s)</th><th>Right: 1 layer</th><th>Files</th></tr>'
            + ''.join(rows) + '</table><div class="images">' + ''.join(pictures) + '</div>'
            + reference_section +
            '<p>Finite-spp trial; image differences include sampling noise. '
            '<a href="metrics.json">Metrics</a> / <a href="manifest.json">Configuration</a></p>')
    (args.output / 'comparison.html').write_text(page, encoding='utf-8')
    print(args.output / 'comparison.html', flush=True)


if __name__ == '__main__':
    main()
