"""Add same-order GPIS FLIP maps below the existing bounce viewer's renders.

<plot-python> -B tools/surface_compare/bounce_images_flip.py

Uses the installed FLIP executable, native full-resolution inputs, and the
viewer's exact filmic/quantization for every gain and cumulative preset.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import subprocess
from pathlib import Path

from gpis_shaderball_flip import DEFAULT_FLIP, REFERENCE, evaluate, rgb, verify_links
from figures import Image, plt
import numpy as np
from bounce_images import OUT, LAYERS, MODES, filmic, load, pfm, save, sha

GAINS = (1, 4, 16)
VARIANTS = tuple(LAYERS) + tuple(f'cumulative{k:02}' for k in range(1, 6))
PAYLOAD = re.compile(r'(<script id="payload" type="application/json">)(.*?)(</script>)', re.S)


def load_sources(report):
    original = (report / 'comparison.html').read_text(encoding='utf-8')
    match = PAYLOAD.search(original)
    assert match, 'Original bounce viewer payload is missing'
    payload = json.loads(match[2])
    assert payload['modes'] == MODES and payload['layers'] == LAYERS
    manifest = load(report / 'manifest.json')
    assert manifest['resolution'] == [payload['width'], payload['height']]
    sources, hashes = {}, {}
    for mode in MODES:
        folder = report / mode
        provenance = load(folder / 'provenance.json')
        assert provenance == manifest['source_provenance'][mode]
        paths = [folder / (layer + '.pfm') for layer in LAYERS] + [folder / 'result.json']
        for path in paths:
            digest = sha(path)
            assert digest == provenance['outputs'][path.name], str(path)
            hashes[path.relative_to(report).as_posix()] = digest
        hashes[f'{mode}/provenance.json'] = sha(folder / 'provenance.json')
        data = np.array([pfm(folder / (layer + '.pfm')) for layer in LAYERS])
        assert np.isfinite(data).all() and data.min() >= 0
        # These are the float32 values actually displayed by the JS viewer.
        packed = np.ascontiguousarray(data.astype('<f4'))
        assert packed.tobytes() == base64.b64decode(payload['images'][mode]), mode
        sources[mode] = packed.astype(np.float64)
    return payload, manifest, sources, hashes


def linear_variant(data, variant):
    if variant.startswith('cumulative'):
        # JS sums each channel sequentially using doubles, before tone mapping.
        return np.cumsum(data[:int(variant[-2:]) + 1], axis=0)[-1]
    return data[LAYERS.index(variant)]


def build(report, exe, ppd):
    payload, original_manifest, sources, source_hashes = load_sources(report)
    dest = report / 'flip'
    dest.mkdir(parents=True, exist_ok=True)
    help_result = subprocess.run([str(exe), '--help'], capture_output=True,
                                 text=True, encoding='utf-8', errors='replace', check=True)
    help_text = help_result.stdout + help_result.stderr
    assert '1.7' in help_text, 'Expected the locally installed FLIP 1.7'
    (dest / 'tool_help.txt').write_text(help_text, encoding='utf-8')
    exe_hash = sha(exe)
    rows, evaluations, entries, input_hashes = [], [], {}, {}
    source_png_matches, self_checks = 0, 0
    relative = lambda path: path.relative_to(report).as_posix()
    for gain in GAINS:
        entries[str(gain)] = {}
        for variant in VARIANTS:
            inputs = {}
            for mode in MODES:
                image = filmic(linear_variant(sources[mode], variant), gain)
                path = dest / 'inputs' / f'gain{gain}' / variant / (mode + '.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
                input_hashes[relative(path)] = sha(path)
                inputs[mode] = path
                existing = report / mode / (variant + '.png')
                if gain == 1 and existing.exists():
                    assert np.array_equal(image, rgb(existing)), str(existing)
                    source_png_matches += 1
            group = entries[str(gain)][variant] = {}
            for mode in MODES:
                folder = dest / 'evaluations' / f'gain{gain}' / variant / mode
                result = evaluate(exe, inputs[REFERENCE], inputs[mode], folder, ppd)
                group[mode] = dict(map=relative(folder / result['error_map']),
                                   pooled=relative(folder / result['pooled_values']),
                                   mean=result['statistics']['mean'],
                                   input=relative(inputs[mode]), reference=relative(inputs[REFERENCE]))
                if mode == REFERENCE:
                    assert all(value == 0 for value in result['statistics'].values())
                    self_checks += 1
                row = dict(gain=gain, variant=variant, mode=mode,
                           reference_mode=REFERENCE, self_comparison=mode == REFERENCE,
                           ppd=ppd, **result['statistics'])
                row.update(group[mode])
                rows.append(row)
                evaluations.append(dict(gain=gain, variant=variant, mode=mode,
                                        record=relative(folder / 'result.json'),
                                        sha256=sha(folder / 'result.json')))
            print(f'FLIP complete: gain={gain}, {variant}', flush=True)

    with (dest / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gradient = np.round(plt.get_cmap('magma')(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    Image.fromarray(np.repeat(gradient[None, :, :], 12, axis=0)).save(dest / 'colorbar.png')
    viewer = dict(version='v1.7', ppd=ppd, reference=REFERENCE, entries=entries,
                  colorbar='flip/colorbar.png', gains=list(GAINS))
    manifest = dict(tool=dict(path=str(exe), version='1.7', sha256=exe_hash,
                             method='LDR-FLIP', ppd=ppd, colormap='magma', color_range=[0, 1]),
                    resolution=original_manifest['resolution'], spp=original_manifest['spp'],
                    sigma=original_manifest['sigma'], length_scale=original_manifest['length_scale'],
                    reference=REFERENCE, gains=list(GAINS), variants=list(VARIANTS),
                    display='Same filmic transform and 8-bit rounding as the existing viewer; no per-image normalization.',
                    cumulative='Sum orders 0 through k in linear double precision before filmic; cumulative05 includes the 5+ bin.',
                    zoom='Crop the full-resolution FLIP map; no FLIP recomputation on cropped/resized images. Means are full-frame.',
                    sampling_note='Same 128 spp GPIS global reference; sampling noise is included, not a converged truth.',
                    source_sha256=source_hashes, input_sha256=input_hashes,
                    evaluations=evaluations, viewer=viewer)
    save(dest / 'manifest.json', manifest)
    payload['flip'] = viewer
    template = Path(__file__).with_name('bounce_images_viewer.html').read_text(encoding='utf-8')
    page = report / 'comparison.html'
    page.write_text(template.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':'))), encoding='utf-8')
    original_manifest['flip'] = dict(manifest='flip/manifest.json', sha256=sha(dest / 'manifest.json'),
                                     metrics='flip/metrics.csv', reference=REFERENCE, ppd=ppd)
    save(report / 'manifest.json', original_manifest)
    # Verify inputs and original data after the calculations and page update.
    for name, digest in {**source_hashes, **input_hashes}.items():
        assert sha(report / name) == digest, name
    assert sha(exe) == exe_hash
    for entry in rows:
        assert (report / entry['map']).is_file() and (report / entry['pooled']).is_file()
    verification = dict(original_sources_unchanged=True, viewer_float32_matches_sources=True,
                        input_images=len(input_hashes), original_gain1_png_pixel_matches=source_png_matches,
                        comparisons=len(rows) - self_checks, reference_self_checks=self_checks,
                        self_check_statistics_all_zero=True, link_check=verify_links(page),
                        page_sha256=sha(page), manifest_sha256=sha(dest / 'manifest.json'))
    save(dest / 'verification.json', verification)
    print(json.dumps(verification), flush=True)
    print('Report:', page, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=OUT)
    parser.add_argument('--flip-exe', type=Path, default=DEFAULT_FLIP)
    parser.add_argument('--ppd', type=float, default=67)
    args = parser.parse_args()
    assert args.ppd > 0 and np.isfinite(args.ppd)
    build(args.report.resolve(), args.flip_exe.resolve(), args.ppd)


if __name__ == '__main__':
    main()
