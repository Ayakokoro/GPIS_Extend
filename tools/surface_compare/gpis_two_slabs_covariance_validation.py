"""Matched renderer comparison for stable analytic LocalLaw covariances."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time

import numpy as np
from run import ROOT, pfm, sha


def main():
    base = ROOT / 'outputs/gpis_two_slabs_compare_20261113'
    out = base / 'local_law_analytic_optimization'
    checker = ROOT / 'build/Release/sparse_history_gpis_test.exe'
    for mode, name in [('covariance', 'law_validation.json'),
                       ('covariance-benchmark', 'construction_benchmark.json')]:
        result = subprocess.run([str(checker), mode], cwd=ROOT, capture_output=True,
                                text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        parsed = json.loads(result.stdout)
        (out / name).write_text(json.dumps(parsed, indent=2), encoding='utf-8')
        print(f'{mode}: {result.stdout.strip()}', flush=True)
    after = out / 'after/renderer'
    after.mkdir(parents=True, exist_ok=True)
    for path in [ROOT / 'build/Release/tungsten.exe', *sorted((ROOT / 'build/Release').glob('*.dll'))]:
        target = after / path.name
        if target.exists():
            assert sha(target) == sha(path), 'Frozen optimized renderer changed'
        else:
            shutil.copy2(path, target)
    original = json.loads((base / 'single_history_cost_probe/scene.json').read_text())
    spp = 256
    records = {}
    comparisons = []
    for seed, order in [(20261024, ('before', 'after')), (20261113, ('after', 'before'))]:
        for version in order:
            folder = out / version / f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            exe = out / version / 'renderer/tungsten.exe'
            scene = json.loads(json.dumps(original))
            scene['renderer'].update(spp=spp, output_directory=str(folder), enable_resume_render=False)
            scene_path = folder / 'scene.json'
            scene_path.write_text(json.dumps(scene, indent=2), encoding='utf-8')
            result_path = folder / 'result.json'
            if result_path.exists():
                record = json.loads(result_path.read_text())
                assert record['renderer_sha256'] == sha(exe)
                assert record['scene_sha256'] == sha(scene_path)
                assert record['image_sha256'] == sha(folder / 'render.pfm')
            else:
                print(f'START {version} seed={seed} 64x64 {spp}spp threads=16', flush=True)
                command = [str(exe), '-t', '16', '-s', str(seed), '-d', str(folder), str(scene_path)]
                started = time.monotonic()
                with (folder / 'render.log').open('wb') as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                elapsed = time.monotonic() - started
                text = (folder / 'render.log').read_text(errors='replace')
                assert process.returncode == 0 and 'Finished render.' in text, str(folder)
                assert 'sanity check failed' not in text.lower(), str(folder)
                times = re.findall(rf'Completed\s+{spp}/{spp}\s+spp,\s*([\d.]+)/[\d.]+\s+sec', text)
                profiles = re.findall(r'GPIS single-history profile:\s*(\{.*\})', text)
                assert times and profiles, text[-2000:]
                assert np.isfinite(pfm(folder / 'render.pfm')).all()
                record = dict(version=version, seed=seed, resolution=[64, 64], spp=spp, threads=16,
                              render_seconds=float(times[-1]), wall_seconds=elapsed,
                              renderer_sha256=sha(exe), scene_sha256=sha(scene_path),
                              image_sha256=sha(folder / 'render.pfm'), profile=json.loads(profiles[-1]))
                result_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
                print(f'DONE {version} seed={seed} {record["render_seconds"]:.2f}s', flush=True)
            records[f'{version}_{seed}'] = record
        old, new = records[f'before_{seed}'], records[f'after_{seed}']
        delta = pfm(out / 'before' / f'seed{seed}/render.pfm') - pfm(out / 'after' / f'seed{seed}/render.pfm')
        before_counts = {k: v for k, v in old['profile']['exterior_path'].items() if not k.endswith('_seconds')}
        after_counts = {k: v for k, v in new['profile']['exterior_path'].items() if not k.endswith('_seconds')}
        comparisons.append(dict(seed=seed, image_identical=old['image_sha256'] == new['image_sha256'],
                                counts_identical=before_counts == after_counts,
                                max_absolute_pixel_difference=float(np.abs(delta).max()),
                                before_seconds=old['render_seconds'], after_seconds=new['render_seconds']))
    old_mean = float(np.mean([x['before_seconds'] for x in comparisons]))
    new_mean = float(np.mean([x['after_seconds'] for x in comparisons]))
    report = dict(complete=True, comparison=comparisons, average_before_seconds=old_mean,
                  average_after_seconds=new_mean, speedup=old_mean/new_mean,
                  reduction_fraction=1-new_mean/old_mean,
                  note='Sequential matched runs with reversed ordering for seed 2. Detail profiling enabled in both versions.')
    (out / 'render_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    assert all(x['image_identical'] and x['counts_identical'] for x in comparisons), 'Inspect numerical/path differences'


if __name__ == '__main__':
    main()
