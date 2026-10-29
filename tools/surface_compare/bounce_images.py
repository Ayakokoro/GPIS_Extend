"""Production-path images split by scattering order, without changing max_bounces.

Build: cmake --build build --config Release --target surface_bounce_images --parallel 1
Run with NumPy + Pillow: python bounce_images.py run --modes sparse_3d_global
Then: python bounce_images.py report
All display panels use the same filmic curve and exposure. Raw data are linear PFM.
"""
from __future__ import annotations
import argparse
import base64
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/gpis_shaderball_compare/ell0p2_sigma0p02'
OUT = BASE / 'bounce_orders'
EXE = ROOT / 'build/Release/surface_bounce_images.exe'
MODES = ['classic_local', 'classic_global', 'global_conditional', 'sparse_3d_global']
LABELS = ['classic_local', 'classic_global', 'global_conditional', 'GPIS global']
LAYERS = ['order00', 'order01', 'order02', 'order03', 'order04', 'order05plus', 'beauty']
PIXELS = [('inner_left', [85, 101]), ('inner_bottom', [124, 113]),
          ('inner_right', [165, 83]), ('far_rim', [149, 48]),
          ('center_control', [126, 82]), ('lip_control', [126, 132]),
          ('base_groove', [177, 166])]
WEIGHTS = np.array([.2126, .7152, .0722])
# Small neighborhoods around previously selected pixels, fixed before new images.
REGIONS = {'inner_left': [81, 97, 90, 106], 'inner_right': [161, 79, 170, 88],
           'far_rim': [145, 44, 154, 53], 'base_groove': [173, 162, 182, 171],
           'center_control': [122, 78, 131, 87]}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pfm(path):
    with path.open('rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        rgb = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)
    return np.flipud(rgb).astype(float) * abs(scale)


def filmic(rgb, gain=1):
    x = np.maximum(0, rgb * gain - .004)
    return np.uint8(np.clip(x * (6.2*x + .5) / (x * (6.2*x + 1.7) + .06), 0, 1) * 255 + .5)


def validate(folder):
    result = load(folder / 'result.json')
    spec = result['spec']
    data = np.array([pfm(folder / (key + '.pfm')) for key in LAYERS])
    assert np.isfinite(data).all() and data.min() >= -1e-7
    assert result['observer_mismatches'] == 0
    assert result['verified_samples'] > 0
    error = np.abs(data[:6].sum(axis=0) - data[6])
    assert np.allclose(data[:6].sum(axis=0), data[6], rtol=2e-6, atol=2e-7)
    x0, y0, x1, y1 = result['bounds']
    reference_path = Path(spec['scene']).with_name('render.pfm')
    reference = pfm(reference_path)
    reference_result = load(reference_path.with_name('result.json'))
    check = {'reconstruction_max_abs_error': float(error.max()),
             'verified_paths': result['verified_samples'], 'observer_mismatches': 0}
    if reference_result['spp'] == spec['samples']:
        a = data[6, y0:y1, x0:x1]
        b = reference[y0:y1, x0:x1]
        check['original_image_max_abs_error'] = float(np.abs(a-b).max())
        check['matches_original_image'] = bool(np.allclose(a, b, rtol=3e-5, atol=2e-7))
        assert check['matches_original_image'], check
    return data, check


def check_recorded_paths(mode, images, samples):
    # Independent check of the bin labels against the previously saved event
    # replay. The sum check alone would not catch an off-by-one order label.
    path = BASE / 'groove_paths' / (mode + '.json')
    recorded = load(path)
    checks = []
    for label, (x, y) in PIXELS:
        paths = [p for p in recorded['paths'] if p['pixel_label'] == label and p['sample'] < samples]
        assert len(paths) == samples
        bins = np.zeros((6, 3))
        for p in paths:
            if p['renderer_clamped']:
                continue
            for e in p['events']:
                if 'emission_added' not in e:
                    continue
                value = np.array(e['emission_added'])
                if not np.any(value):
                    continue
                if e['type'] == 'direct_estimate':
                    order = e['bounce'] + 1
                elif e['type'] == 'surface':
                    assert e['primitive'] in ('floor', 'light')
                    order = e['bounce'] + int(e['primitive'] == 'floor')
                else:
                    assert e['type'] in ('environment', 'medium_collision', 'medium_exit')
                    order = e['bounce']
                bins[min(order, 5)] += value / samples
        observed = images[:6, y, x]
        assert np.allclose(bins, observed, rtol=2e-6, atol=2e-7), (mode, label, bins, observed)
        checks.append(dict(pixel=[x, y], label=label, samples=samples,
                           max_abs_error=float(np.abs(bins-observed).max())))
    return dict(recorded_paths_sha256=sha(path), per_pixel_checks=checks)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        folder = OUT / mode
        spec = dict(scene=str(BASE / 'final' / mode / 'seed20261024/scene.json'),
                    seed=20261024, samples=args.samples, threads=args.threads,
                    verify_pixels=[p for _, p in PIXELS], verify_samples=16)
        spec_path = OUT / (mode + '.spec.json')
        if (folder / 'provenance.json').exists():
            previous = load(folder / 'provenance.json')
            # Independent sampling is keyed by pixel/sample, not worker count.
            physical = lambda s: {k: v for k, v in s.items() if k != 'threads'}
            if previous['executable_sha256'] == sha(EXE) and physical(load(spec_path)) == physical(spec):
                assert previous['scene_sha256'] == sha(Path(spec['scene']))
                for name, digest in previous['outputs'].items():
                    assert sha(folder / name) == digest, name
                validate(folder)
                print('Already complete:', mode, flush=True)
                continue
            raise RuntimeError('Completed output uses different inputs; choose a new output directory')
        save(spec_path, spec)
        executable_hash = sha(EXE)
        scene_hash = sha(Path(spec['scene']))
        print('Render', mode, args.samples, 'spp', args.threads, 'threads', flush=True)
        with (OUT / (mode + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run([str(EXE), str(spec_path), str(folder)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        assert sha(EXE) == executable_hash and sha(Path(spec['scene'])) == scene_hash
        _, checks = validate(folder)
        files = list(folder.glob('*.pfm')) + [folder / 'result.json']
        save(folder / 'provenance.json', dict(executable_sha256=executable_hash,
             scene_sha256=scene_hash, spec_sha256=sha(spec_path),
             completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             outputs={p.name: sha(p) for p in files}, validation=checks))
        print('Complete', mode, json.dumps(checks), flush=True)


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def sheet(data, cumulative, crop=False):
    cell = 256 if not crop else 320
    cell_height = cell if not crop else round(cell*107/140)
    left, top, gap, title = 175, 114, 18, 30
    im = Image.new('RGB', (left + 4*(cell+gap)+16, top + 4*(cell_height+title+gap)+70), '#f5f7fa')
    d = ImageDraw.Draw(im)
    heading = '累计到 1、2、3、4 次反射' if cumulative else '恰好 1、2、3、4 次反射的贡献'
    d.text((24, 16), heading + (' · 凹槽放大' if crop else ''), font=font(26), fill='#172338')
    d.text((24, 56), '原 GPIS shaderball · ℓ=0.2 · σ=0.02 · 256×256 · 128 spp · 统一 filmic / 曝光', font=font(17), fill='#52617b')
    for m, (mode, label) in enumerate(zip(MODES, LABELS)):
        y = top + m*(cell_height+title+gap)
        d.text((14, y+title+cell_height//2), label, font=font(16), fill='#172338')
        for order in range(1, 5):
            x = left + (order-1)*(cell+gap)
            rgb = data[mode][:order+1].sum(axis=0) if cumulative else data[mode][order]
            if crop:
                rgb = rgb[38:145, 55:195]
            panel = Image.fromarray(filmic(rgb)).resize((cell, cell_height), Image.Resampling.NEAREST)
            d.text((x, y), ('累计 ≤' if cumulative else '恰好 ') + str(order), font=font(18), fill='#172338')
            im.paste(panel, (x, y+title))
    d.text((24, im.height-48), '累计图含 0 次贡献；地面反射计数，透明边界不计数。保留完整追踪至上限 20；未降噪。', font=font(17), fill='#52617b')
    return im


def report():
    data, checks, provenance, statistics, results = {}, {}, {}, [], {}
    for mode in MODES:
        folder = OUT / mode
        provenance[mode] = load(folder / 'provenance.json')
        for name, digest in provenance[mode]['outputs'].items():
            assert sha(folder / name) == digest, (mode, name)
        result = results[mode] = load(folder / 'result.json')
        assert result['bounds'] == [0, 0, 256, 256] and result['samples'] == 128
        assert sha(Path(result['spec']['scene'])) == provenance[mode]['scene_sha256']
        data[mode], checks[mode] = validate(folder)
        checks[mode]['bin_labels_match_recorded_paths'] = check_recorded_paths(mode, data[mode], result['samples'])
        for k, key in enumerate(LAYERS):
            Image.fromarray(filmic(data[mode][k])).save(folder / (key + '.png'))
        for k in range(1, 5):
            Image.fromarray(filmic(data[mode][:k+1].sum(axis=0))).save(folder / f'cumulative{k:02}.png')
        for region, (x0, y0, x1, y1) in REGIONS.items():
            values = data[mode][:, y0:y1, x0:x1] @ WEIGHTS
            statistics.append(dict(mode=mode, region=region, bounds=[x0, y0, x1, y1],
                                   luminance=values.mean(axis=(1, 2)).tolist()))
    for cumulative in (False, True):
        for crop in (False, True):
            name = ('cumulative' if cumulative else 'exact') + ('_groove' if crop else '') + '.png'
            sheet(data, cumulative, crop).save(OUT / name)
    save(OUT / 'validation.json', checks)
    save(OUT / 'statistics.json', dict(order=LAYERS, regions=REGIONS, rows=statistics))
    payload = dict(modes=MODES, labels=LABELS, layers=LAYERS, width=256, height=256,
                   pixels=[dict(label=n, xy=p) for n, p in PIXELS], statistics=statistics,
                   images={mode: base64.b64encode(np.ascontiguousarray(data[mode].astype('<f4')).tobytes()).decode('ascii') for mode in MODES})
    template = (Path(__file__).with_name('bounce_images_viewer.html')).read_text(encoding='utf-8')
    (OUT / 'comparison.html').write_text(template.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':'))), encoding='utf-8')
    save(OUT / 'manifest.json', dict(scene='gpis-scenes/shaderball/shaderball-stationary-270.json',
         sigma=.02, length_scale=.2, resolution=[256, 256], spp=128, seed=20261024, max_bounces=20,
         definition=results[MODES[0]]['definition'],
         cumulative='Includes order 0 plus orders 1 through k; exact panels show only order k.',
         reference='Full production trace with unchanged NEE, MIS, Russian roulette, and gradient rejection.',
         sampling_note='128 spp diagnostic; not claimed to be converged.',
         termination_fraction_channels=['outward_crossing_rejected on the continuation path (not shadow rays)',
                                        'integrator bounce_limit', 'renderer-clamped sample'],
         source_provenance=provenance, validation=checks,
         outputs={p.name: sha(p) for p in OUT.glob('*.png')}))
    print('Report:', str(OUT / 'comparison.html'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'report'])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args()
    run(args) if args.action == 'run' else report()


if __name__ == '__main__':
    main()
