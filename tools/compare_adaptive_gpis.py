"""Render matched adaptive / uniform-online / legacy GPIS scenes and an HTML report.

Uses only Python's standard library. The uniform-online mode retains the same
history and gradient law as adaptive; legacy uses the older batch-32 renderer.
Neither a low-spp image nor the legacy batch approximation is ground truth.
"""
import argparse
import array
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def pfm(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'PF':
            raise RuntimeError(f'Expected RGB PFM: {path}')
        w, h = map(int, stream.readline().split())
        scale = float(stream.readline())
        data = array.array('f')
        data.frombytes(stream.read())
    if (scale < 0) != (sys.byteorder == 'little'):
        data.byteswap()
    if len(data) != w*h*3 or not all(math.isfinite(x) and x >= 0 for x in data):
        raise RuntimeError(f'Invalid pixels: {path}')
    if max(data) <= 0:
        raise RuntimeError(f'Black render: {path}')
    return w, h, [x*abs(scale) for x in data]


def configured_scene(source, mode, size, spp):
    scene = copy.deepcopy(source)
    scene['camera']['resolution'] = [size]*2
    scene['renderer'].update(spp=spp, spp_step=1)
    medium = scene['media'][0]
    if mode == 'uniform':
        medium['adaptive']['enabled'] = False
    elif mode == 'refined':
        medium['adaptive']['minimum_step'] *= .5
        medium['adaptive']['root_tolerance'] *= .5
        medium['adaptive']['rank_tolerance'] *= .01
    elif mode == 'legacy':
        options = medium.pop('adaptive')
        medium.update(type='function_space_gaussian_process', sample_points=32,
                      step_size=options['minimum_step'], skip_space=0)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT/'build/Release/tungsten.exe')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/adaptive_gpis_double_20261114/comparison')
    parser.add_argument('--scenes', nargs='+', choices=['sphere', 'plane', 'shaderball'], default=['sphere', 'plane'])
    parser.add_argument('--modes', nargs='+', choices=['adaptive', 'uniform', 'legacy', 'refined'], default=['adaptive', 'uniform', 'legacy'])
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--spp', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()
    if min(args.size, args.spp, args.threads) <= 0:
        parser.error('size, spp and threads must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()
    binary_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    rows = []
    images = {}
    for name in args.scenes:
        source = json.loads((ROOT/f'data/example-scenes/adaptive-gpis/{name}.json').read_text())
        for mode in args.modes:
            folder = output/name/mode
            folder.mkdir(parents=True, exist_ok=True)
            scene = configured_scene(source, mode, args.size, args.spp)
            identity = dict(scene=scene, binary_sha256=binary_hash, seed=args.seed, threads=args.threads)
            record_file = folder/'result.json'
            if record_file.exists():
                record = json.loads(record_file.read_text())
                if record['identity'] != identity:
                    raise RuntimeError(f'Inputs changed; select a new output directory: {folder}')
            else:
                (folder/'scene.json').write_text(json.dumps(scene, indent=2)+'\n')
                command = [str(executable), '-t', str(args.threads), '-s', str(args.seed),
                           '-d', str(folder), str(folder/'scene.json')]
                print('Rendering', name, mode, flush=True)
                begin = time.perf_counter()
                with (folder/'render.log').open('wb') as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               cwd=ROOT, timeout=900)
                wall = time.perf_counter()-begin
                log = (folder/'render.log').read_text(encoding='utf-8', errors='replace')
                if completed.returncode or re.search(
                    r'Caught an internal error|Unable to load|Gradient invalid|Failed to sample gradient|'
                    r'Matrix square root failed|Wrong normal direction|Conditioning .* failed', log):
                    raise RuntimeError(f'Render failed; inspect {folder / "render.log"}')
                if not re.search(rf'Completed {args.spp}/{args.spp} spp', log):
                    raise RuntimeError(f'Incomplete render: {folder}')
                _, _, pixels = pfm(folder/'render.pfm')
                records = re.findall(r'^Adaptive GPIS statistics: (.+)$', log, re.M)
                record = dict(identity=identity, scene=name, mode=mode,
                              wall_seconds=wall, render_seconds=float(re.findall(
                                  r'Completed \d+/\d+ spp, ([\d.]+)/', log)[-1]),
                              mean_rgb=sum(pixels)/len(pixels),
                              statistics=json.loads(records[-1]) if records else None)
                record_file.write_text(json.dumps(record, indent=2)+'\n')
            _, _, images[name, mode] = pfm(folder/'render.pfm')
            rows.append({k: v for k, v in record.items() if k != 'identity'})
    # Additional invocations can add a scene/mode without rerunning completed
    # work. Include only records from this binary and these sampling settings.
    for record_file in sorted(output.glob('*/*/result.json')):
        record = json.loads(record_file.read_text())
        key = record['scene'], record['mode']
        identity = record['identity']
        if key in images or identity['binary_sha256'] != binary_hash:
            continue
        source = json.loads((ROOT/f'data/example-scenes/adaptive-gpis/{key[0]}.json').read_text())
        if (identity['seed'] != args.seed or identity['threads'] != args.threads or
                identity['scene'] != configured_scene(source, key[1], args.size, args.spp)):
            continue
        _, _, images[key] = pfm(record_file.parent/'render.pfm')
        rows.append({k: v for k, v in record.items() if k != 'identity'})
    rows.sort(key=lambda row: (row['scene'], row['mode']))
    for row in rows:
        reference = images.get((row['scene'], 'uniform'))
        if reference is not None:
            pixels = images[row['scene'], row['mode']]
            row['linear_rgb_rmse_vs_uniform'] = math.sqrt(sum((a-b)**2 for a, b in zip(pixels, reference))/len(pixels))
    report = dict(resolution=[args.size]*2, spp=args.spp, seed=args.seed,
                  binary_sha256=binary_hash, runs=rows,
                  note='Low-spp smoke/comparison, not equal-quality or equal-time reproduction. '
                       'Uniform uses the same online sampler; legacy has different batch-history and gradient approximations.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    cards = []
    for row in rows:
        stats = row['statistics']
        text = f'{row["render_seconds"]:.3f} s render; {row["wall_seconds"]:.3f} s wall'
        if stats:
            text += f'; {stats["queries"]/max(1,stats["rays"]):.1f} queries/ray'
            text += f'; invalid gradients: {stats["invalid_gradients"]}'
            text += f'; opaque origins: {stats["invalid_births"]}'
            if 'arithmetic' in stats:
                text += f'; arithmetic: {stats["arithmetic"]}'
            elif 'high_precision_rays' in stats:
                text += f'; high-precision rays: {stats["high_precision_rays"]}/{stats["rays"]}'
        if 'linear_rgb_rmse_vs_uniform' in row:
            text += f'; RGB RMSE vs uniform: {row["linear_rgb_rmse_vs_uniform"]:.5g}'
        path = f'{row["scene"]}/{row["mode"]}'
        cards.append(f'<article><h2>{html.escape(row["scene"])} / {html.escape(row["mode"])}</h2>'
                     f'<img src="{path}/render.png"><p>{html.escape(text)}</p>'
                     f'<a href="{path}/render.log">Log</a> · <a href="{path}/scene.json">Scene</a></article>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Adaptive GPIS comparison</title>'
                '<style>body{font:16px system-ui;background:#181b20;color:#e5e7eb;margin:32px}'
                'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}'
                'article{background:#252b34;padding:20px;border-radius:10px}img{width:100%;image-rendering:pixelated}'
                'h2{font-size:18px}a{color:#8ed8fa}p{line-height:1.5}</style>'
                f'<h1>Adaptive function-space GPIS</h1><p>{args.size} × {args.size}, {args.spp} spp, seed {args.seed}.</p>'
                '<p>Uniform-online shares the adaptive sampler and gradient law. Legacy uses batch-32 function space. '
                'These low-spp images are not ground truth or evidence of equal-time speedup. '
                'The shaderball fixture uses Lambert reflection to validate small-light NEE at low spp. '
                'It is separate from the existing mirror shaderball with sparse-convolution 1D MIS.</p>'
                '<p><a href="report.json">Machine-readable report</a></p><main>'+''.join(cards)+'</main></html>')
    (output/'comparison.html').write_text(document, encoding='utf-8')
    print(output/'comparison.html')


if __name__ == '__main__':
    main()
