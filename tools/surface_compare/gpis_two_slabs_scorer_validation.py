"""Matched old/new renderer timings and image checks for scalar history scoring."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time

from run import ROOT, pfm, sha
import numpy as np


def main():
    base = ROOT / 'outputs/gpis_two_slabs_compare_20261113'
    out = base / 'scorer_scalar_optimization'
    source = base / 'neural_single_history_continued'
    before = out / 'before/renderer/tungsten.exe'
    assert before.exists(), 'Freeze the original renderer before rebuilding'
    checker = ROOT / 'build/Release/sparse_history_gpis_test.exe'
    for mode, name in [('scorer', 'score_validation.json'), ('scorer-benchmark', 'score_benchmark.json')]:
        checked = subprocess.run([str(checker), mode], cwd=ROOT, capture_output=True, text=True, check=True,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        result = json.loads(checked.stdout)
        (out / name).write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'{mode}: {checked.stdout.strip()}', flush=True)
    after_dir = out / 'after/renderer'
    after_dir.mkdir(parents=True, exist_ok=True)
    for path in [ROOT / 'build/Release/tungsten.exe', *sorted((ROOT / 'build/Release').glob('*.dll'))]:
        destination = after_dir / path.name
        if destination.exists():
            assert sha(path) == sha(destination), 'Frozen optimized renderer changed'
        else:
            shutil.copy2(path, destination)
    records = {}
    comparison = []
    for seed, order in [(20261024, ('before', 'after')), (20261113, ('after', 'before'))]:
        original = json.loads((source / f'final/global_conditional_neural/seed{seed}/scene.json').read_text())
        for version in order:
            folder = out / version / f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            exe = out / version / 'renderer/tungsten.exe'
            result_path = folder / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                assert result['renderer_sha256'] == sha(exe)
                assert result['model_sha256'] == sha(source / 'model/model.json')
                for name, digest in result['images_sha256'].items():
                    assert sha(folder / name) == digest
                records[f'{version}_{seed}'] = result
                continue
            scene = json.loads(json.dumps(original))
            scene['renderer']['output_directory'] = str(folder)
            scene['renderer']['enable_resume_render'] = False
            (folder / 'scene.json').write_text(json.dumps(scene, indent=2), encoding='utf-8')
            command = [str(exe), '-t', '16', '-s', str(seed), '-d', str(folder), str(folder / 'scene.json')]
            print(f'START {version} seed={seed} 64x64 1024spp threads=16', flush=True)
            start = time.monotonic()
            with (folder / 'render.log').open('wb') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            elapsed = time.monotonic() - start
            text = (folder / 'render.log').read_text(errors='replace')
            assert process.returncode == 0 and 'Finished render.' in text, str(folder)
            assert 'sanity check failed' not in text.lower(), str(folder)
            times = re.findall(r'Completed\s+1024/1024\s+spp,\s*([\d.]+)/[\d.]+\s+sec', text)
            assert times, text[-1000:]
            statistics = re.findall(r'GPIS single-point neural statistics:\s*(\{.*\})', text)
            assert statistics
            names = ['render.png', 'render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm']
            for name in names[1:]:
                assert np.isfinite(pfm(folder / name)).all()
            result = dict(version=version, seed=seed, resolution=[64, 64], spp=1024, threads=16,
                          renderer_sha256=sha(exe),
                          model_sha256=sha(source / 'model/model.json'),
                          render_seconds=float(times[-1]), wall_seconds=elapsed,
                          diagnostics=json.loads(statistics[-1]),
                          images_sha256={name: sha(folder / name) for name in names})
            result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            records[f'{version}_{seed}'] = result
            print(f'DONE {version} seed={seed} {result["render_seconds"]:.2f}s', flush=True)
        old, new = records[f'before_{seed}'], records[f'after_{seed}']
        delta = pfm(out / 'before' / f'seed{seed}/render.pfm') - pfm(out / 'after' / f'seed{seed}/render.pfm')
        comparison.append(dict(seed=seed,
            images_identical=old['images_sha256'] == new['images_sha256'],
            diagnostics_identical=old['diagnostics'] == new['diagnostics'],
            maximum_absolute_pixel_difference=float(np.max(np.abs(delta))),
            before_seconds=old['render_seconds'], after_seconds=new['render_seconds'],
            speedup=old['render_seconds']/new['render_seconds']))
    old_mean = float(np.mean([v['before_seconds'] for v in comparison]))
    new_mean = float(np.mean([v['after_seconds'] for v in comparison]))
    report = dict(complete=True, comparison=comparison,
                  average_before_seconds=old_mean, average_after_seconds=new_mean,
                  speedup=old_mean/new_mean, reduction_fraction=1-new_mean/old_mean,
                  note='Serial runs, opposite ordering for the two seeds; unchanged model and rendering settings. Timings include rendering, not training.')
    (out / 'render_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    assert all(v['images_identical'] and v['diagnostics_identical'] for v in comparison), 'Investigate old/new mismatch'


if __name__ == '__main__':
    main()
