"""Exact production path replay and an offline viewer for shaderball grooves.

Build: cmake --build build --config Release --target surface_path_trace --parallel 1
Run with NumPy + Pillow: python path_trace_report.py run
The scene and renderer transport settings are not modified.
"""
from __future__ import annotations
import argparse
import csv
import base64
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/gpis_shaderball_compare/ell0p2_sigma0p02'
OUT = BASE / 'groove_paths'
MODES = ['sparse_3d_global', 'classic_local', 'classic_global', 'global_conditional']
PIXELS = [dict(label=n, xy=p) for n, p in [
    ('inner_left', [85, 101]), ('inner_bottom', [124, 113]),
    ('inner_right', [165, 83]), ('far_rim', [149, 48]),
    ('center_control', [126, 82]), ('lip_control', [126, 132]),
    ('base_groove', [177, 166])]]
WEIGHTS = np.array([.2126, .7152, .0722])


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pfm(p):
    with p.open('rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split()); scale = float(f.readline())
        a = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)
    return np.flipud(a).astype(float) * abs(scale)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    exe = ROOT / 'build/Release/surface_path_trace.exe'
    for mode in MODES:
        spec = dict(scene=str(BASE / 'final' / mode / 'seed20261024/scene.json'),
                    seed=20261024, samples=args.samples, verify_samples=args.samples,
                    pixels=PIXELS, export_mean_mesh=mode == MODES[0], mesh_grid=48)
        sp = OUT / (mode + '.spec.json'); op = OUT / (mode + '.json')
        save(sp, spec)
        print('Replay', mode, args.samples, flush=True)
        with (OUT / (mode + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run([str(exe), str(sp), str(op)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        save(OUT / (mode + '.provenance.json'), dict(executable_sha256=sha(exe),
             scene_sha256=sha(Path(spec['scene'])), spec_sha256=sha(sp), output_sha256=sha(op)))
    report()


def report():
    data = {m: load(OUT / (m + '.json')) for m in MODES}
    provenance = {m: load(OUT / (m + '.provenance.json')) for m in MODES}
    for mode, d in data.items():
        assert provenance[mode]['output_sha256'] == sha(OUT / (mode + '.json'))
        assert provenance[mode]['scene_sha256'] == sha(Path(d['spec']['scene']))
        assert d['observer_mismatches'] == 0
    rows = []; checks = []; mesh = data[MODES[0]].pop('geometry')
    for mode, d in data.items():
        reference = pfm(BASE / 'final' / mode / 'seed20261024/render.pfm')
        for pixel in PIXELS:
            paths = [p for p in d['paths'] if p['pixel_label'] == pixel['label']]
            counts = np.array([p['medium_scatters'] for p in paths])
            radiance = np.array([p['radiance'] for p in paths])
            y = radiance @ WEIGHTS
            hist = Counter(map(int, counts))
            termination = Counter(p['termination'] for p in paths)
            contributions = Counter(); tiny = 0; short = 0
            for p in paths:
                vertices = [e for e in p['events'] if e['type'] == 'medium_scatter']
                assert len(vertices) == p['medium_scatters']
                for v in vertices:
                    assert abs(np.linalg.norm(v['incoming']) - 1) < 2e-5
                    assert abs(np.linalg.norm(v['outgoing']) - 1) < 2e-5
                    assert np.allclose(np.array(v['throughput_before']) * v['scatter_weight'],
                                       v['throughput_after'], rtol=2e-6, atol=1e-12)
                steps = [np.linalg.norm(np.subtract(b['position'], a['position'])) for a, b in zip(vertices, vertices[1:])]
                tiny += any(t < 1e-5 for t in steps)
                short += any(t < .001 for t in steps)
                for e in p['events']:
                    if e['type'] == 'direct_estimate':
                        contributions[e['medium_scatter_order']] += float(np.array(e['emission_added']) @ WEIGHTS)
            first128 = np.mean(radiance[:128], axis=0)
            x0, y0 = pixel['xy']; original = reference[y0, x0]
            error = float(np.max(np.abs(first128 - original)))
            checks.append(dict(mode=mode, pixel=pixel['xy'], samples=128,
                replay_rgb=first128.tolist(), original_rgb=original.tolist(), max_abs_error=error,
                matches=bool(np.allclose(first128, original, rtol=3e-5, atol=2e-7))))
            if mode == MODES[0] and len(paths) >= 512:
                original512 = pfm(BASE / 'highspp_512' / mode / 'seed20261024/render.pfm')[y0, x0]
                mean512 = np.mean(radiance[:512], axis=0)
                checks.append(dict(mode=mode, pixel=pixel['xy'], samples=512,
                    replay_rgb=mean512.tolist(), original_rgb=original512.tolist(),
                    max_abs_error=float(np.max(np.abs(mean512-original512))),
                    matches=bool(np.allclose(mean512, original512, rtol=3e-5, atol=2e-7))))
            rows.append(dict(mode=mode, label=pixel['label'], pixel=pixel['xy'], samples=len(paths),
                mean_medium_scatters=float(counts.mean()), median_medium_scatters=float(np.median(counts)),
                max_medium_scatters=int(counts.max()), single_fraction=float(np.mean(counts == 1)),
                multiple_fraction=float(np.mean(counts >= 2)), histogram=dict(sorted(hist.items())),
                termination=dict(termination), mean_luminance=float(y.mean()),
                luminance_standard_error=float(y.std(ddof=1) / np.sqrt(len(y))),
                zero_radiance_fraction=float(np.mean(y == 0)),
                tiny_step_path_fraction=tiny / len(paths), short_step_path_fraction=short / len(paths),
                direct_luminance_by_order={str(k): v / len(paths) for k, v in sorted(contributions.items())}))
    verification = dict(all_original_pixels_match=all(c['matches'] for c in checks),
                        pixel_checks=checks, observer_verified_samples=sum(d['verified_samples'] for d in data.values()),
                        observer_mismatches=sum(d['observer_mismatches'] for d in data.values()))
    save(OUT / 'statistics.json', rows); save(OUT / 'verification.json', verification)
    with (OUT / 'statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['mode', 'label', 'samples', 'mean_medium_scatters', 'single_fraction', 'multiple_fraction',
                  'mean_luminance', 'luminance_standard_error', 'zero_radiance_fraction', 'tiny_step_path_fraction']
        writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    # A static, directly shareable pixel map; numbered locations are identical in all modes.
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    canvas = Image.new('RGB', (1024, 310), '#18222c'); draw = ImageDraw.Draw(canvas)
    for i, mode in enumerate(MODES):
        source = BASE / ('highspp_512' if mode == MODES[0] else 'final') / mode / 'seed20261024/render.png'
        canvas.paste(Image.open(source).convert('RGB'), (256*i, 34))
        draw.text((256*i+5, 7), 'GPIS global' if i == 0 else mode, font=font, fill='white')
        for number, pixel in enumerate(PIXELS, 1):
            x, y = pixel['xy']; x += 256*i; y += 34
            draw.ellipse((x-3, y-3, x+3, y+3), outline='#ffde3b', width=1)
            draw.text((x+4, y-12), str(number), font=font, fill='#ffde3b', stroke_width=1, stroke_fill='black')
    canvas.save(OUT / 'pixels.png')
    template = (Path(__file__).with_name('path_trace_viewer.html')).read_text(encoding='utf-8')
    # Keep all samples in the viewer, but omit unused duplicate collision records.
    # Full precision and every event remain in the per-mode JSON files.
    viewer_data = {}
    event_keys = {'type','position','origin','outgoing','incoming','gradient','reason','bounce',
                  'medium_scatter_order','emission_added','scatter_weight','throughput_after',
                  'strategy','status','transmittance','endpoint'}
    for mode, d in data.items():
        viewer_data[mode] = dict(paths=[])
        for path in d['paths']:
            q = {k: path[k] for k in ['sample','pixel_label','medium_scatters','surface_reflections','radiance','termination']}
            q['events'] = [{k: v for k, v in e.items() if k in event_keys} for e in path['events']
                           if e['type'] not in ['medium_collision','medium_exit']]
            viewer_data[mode]['paths'].append(q)
    bundle = dict(modes=MODES, pixels=PIXELS, data=viewer_data, statistics=rows, verification=verification, mesh=mesh)
    raw = json.dumps(bundle, separators=(',', ':'), allow_nan=False).encode('utf-8')
    payload = json.dumps(base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii'))
    (OUT / 'comparison.html').write_text(template.replace('__PATH_DATA__', payload), encoding='utf-8')
    save(OUT / 'manifest.json', dict(scene='shaderball-stationary-270', sigma=.02, length_scale=.2,
        seed=20261024, resolution=[256, 256], modes=MODES, pixels=PIXELS, verification=verification,
        provenance=provenance,
        notes=['CPU production PathTracer, including NEE/MIS RNG draws.',
               'Counts are successful continuation medium scatters; mirror micro-material makes these reflections.',
               'Transparent medium boundaries and direct-light branches are not continuation reflections.',
               'Three classic modes sample statistical collision events; GPIS samples an explicit field.',
               'Viewer mesh is the mean surface, not a realized GPIS field.',
               'Statistics cover selected pixels, not the entire groove.']))
    print('Original pixel matches:', verification['all_original_pixels_match'],
          'verified paths:', verification['observer_verified_samples'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'report'])
    parser.add_argument('--samples', type=int, default=512)
    args = parser.parse_args()
    if args.action == 'run':
        assert args.samples >= 128
        run(args)
    else:
        report()
