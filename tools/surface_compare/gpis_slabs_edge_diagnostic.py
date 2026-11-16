"""Diagnose selection-group rings without changing the production scene.

python -B tools/surface_compare/gpis_slabs_edge_diagnostic.py
"""
from __future__ import annotations

import copy
import html
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gpis_two_slabs_cdf5 import render
from run import ROOT, load, pfm, save, sha


def main():
    base = ROOT / 'outputs/gpis_three_slabs_compare_20261116'
    out = base / 'edge_diagnostic_20261116'
    manifest = load(base / 'manifest.json')
    seed = 20261024
    template = load(base / f'gauss_legendre5/seed{seed}/scene.json')
    assert sha(Path(manifest['renderer'])) == manifest['renderer_sha256']
    for method in ('numeric', 'gauss_legendre5', 'gpis_renewal_plus'):
        d = base / method / f'seed{seed}'
        record = load(d / 'result.json')
        assert all(sha(d / name) == digest for name, digest in record['sha256'].items())
    spec = dict(seed=seed, baseline_scene=template, renderer_sha256=manifest['renderer_sha256'],
                variants={'half_selection_step': {'selection_step_over_length': .25},
                          'entry_plus_0p01': {'boundary_position_y': .005, 'boundary_scale_y': .61}},
                notes=['Only the stated parameter changes; geometry, kernel, camera, quality and seed match.',
                       'The boundary variant raises the entry plane from 0.30 to 0.31 and keeps the exit at -0.30.',
                       'No production source or original rendered image is modified.'])
    if (out / 'manifest.json').exists():
        assert load(out / 'manifest.json') == spec
    else:
        save(out / 'manifest.json', spec)
    variants = {}
    for variant in spec['variants']:
        scene = copy.deepcopy(template)
        if variant == 'half_selection_step':
            scene['media'][0]['single_history']['selection_step_over_length'] = .25
        else:
            boundary = next(p for p in scene['primitives'] if p['name'] == 'shared_process_bounds')
            boundary['transform']['position'][1] = .005
            boundary['transform']['scale'][1] = .61
        args = SimpleNamespace(output=out / variant, renderer=Path(manifest['renderer']),
                               size=manifest['size'], spp=manifest['spp'], threads=manifest['threads'],
                               slabs=3, timeout=600)
        variants[variant] = render(args, scene, 'gauss_legendre5', seed)
    paths = {'baseline': base / f'gauss_legendre5/seed{seed}/render.png',
             **{name: out / name / f'gauss_legendre5/seed{seed}/render.png' for name in variants},
             'renewal': base / f'gpis_renewal_plus/seed{seed}/render.png'}
    images = {name: pfm(path.with_suffix('.pfm'))[:, :, 0] for name, path in paths.items()}
    size = manifest['size']
    extent = 2 * np.tan(np.deg2rad(22))
    v = ((np.arange(size) + .5) / size * 2 - 1) * extent
    x, z = np.meshgrid(v, v)
    radius = np.hypot(x, z)
    strip = (x > -.73) & (x < -.51) & (abs(z) < .48)
    profiles = []
    for lo in np.arange(.52, .90, .02):
        mask = strip & (radius >= lo) & (radius < lo + .02)
        if mask.any():
            profiles.append(dict(radius=float(lo + .01), pixels=int(mask.sum()),
                                 **{name: float(image[mask].mean()) for name, image in images.items()}))
    circles = dict(baseline=float(2 * np.sqrt((.3 / .28) ** 2 - 1)),
                   entry_plus_0p01=float(2 * np.sqrt((.3 / .29) ** 2 - 1)))
    metrics = dict(complete=True, records=variants, ring_radius_at_y0=circles,
                   radial_profiles=profiles, profile_region='-0.73 < screen x < -0.51; abs(z) < 0.48')
    save(out / 'metrics.json', metrics)
    labels = {'baseline': 'Original: selection interval 0.10',
              'half_selection_step': 'Selection interval 0.05',
              'entry_plus_0p01': 'Entry plane y=0.31 (was 0.30)',
              'renewal': 'Renewal+ reference'}
    cards = []
    for name, path in paths.items():
        link = html.escape(os.path.relpath(path, out).replace('\\', '/'))
        circle = ''
        if name in circles:
            circle = f'<circle cx="32" cy="32" r="{circles[name] / (2 * extent) * 64}" fill="none" stroke="#ff4a38" stroke-width="0.25"/>'
        cards.append(f'<figure><img src="{link}" alt="{labels[name]}"><figcaption>{labels[name]}</figcaption>'
                     f'<svg viewBox="0 0 64 64"><image href="{link}" width="64" height="64"/>{circle}</svg></figure>')
    colors = dict(baseline='#b72434', half_selection_step='#2876c5', entry_plus_0p01='#b87814', renewal='#23885b')
    curves = []
    for name in paths:
        points = ' '.join(f'{50 + (r["radius"] - .5) / .4 * 540:.2f},{260 - r[name] / .4 * 220:.2f}' for r in profiles)
        curves.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
    plot = ('<svg viewBox="0 0 640 310" class="plot"><path d="M50 30V260H600" fill="none" stroke="#333"/>'
            '<text x="15" y="20">Transmittance</text><text x="270" y="305">Radius at y=0</text>'
            + ''.join(f'<text x="{50 + (r - .5) / .4 * 540}" y="282">{r:.1f}</text>' for r in [.5, .6, .7, .8, .9])
            + ''.join(f'<text x="12" y="{264 - t / .4 * 220}">{t:.1f}</text>' for t in [0, .1, .2, .3, .4])
            + ''.join(curves) + '</svg>')
    legend = ''.join(f'<span style="color:{colors[name]};margin-right:16px">{labels[name]}</span>' for name in paths)
    page = ('<!doctype html><meta charset="utf-8"><title>Single-history edge diagnosis</title>'
            '<style>body{font:16px/1.5 system-ui;margin:30px auto;max-width:1240px;padding:20px}'
            '.images{display:flex;flex-wrap:wrap}figure{width:280px;margin:10px}figure img,figure svg{width:256px;image-rendering:pixelated}'
            '.plot{width:640px;max-width:100%}</style><h1>Single-history selection-group ring</h1>'
            f'<p>{size}&times;{size}, {manifest["spp"]} spp, seed {seed}. '
            'Red circles mark where the first slab minimum reaches distance 0.30 from the medium entry.</p>'
            '<p>Original radius: 0.76931; raised-entry radius: 0.52973. '
            'The original geometry and render outputs are retained.</p><div class="images">'
            + ''.join(cards) + '</div><h2>Left-side radial profile</h2><p>' + legend + '</p>' + plot
            + '<p>Profile uses -0.73 &lt; screen x &lt; -0.51 and |z| &lt; 0.48. '
            'The original and raised-entry panels show each image twice, with the analytical circle overlaid below.</p>'
            '<p><a href="metrics.json">Metrics</a> / <a href="manifest.json">Configuration</a> / '
            '<a href="../comparison.html">Original comparison</a></p>')
    (out / 'comparison.html').write_text(page, encoding='utf-8')
    print(circles, flush=True)
    print(out / 'comparison.html', flush=True)


if __name__ == '__main__':
    main()
