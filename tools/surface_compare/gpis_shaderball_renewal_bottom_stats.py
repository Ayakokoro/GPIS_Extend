"""Describe the existing 3D Renewal+/global difference in fixed base regions.

No new paths are rendered. Luminance statistics use linear PFMs, whereas the
two appearance panels show untouched filmic PNGs. The split-half noise estimate
is descriptive and assumes approximately independent pixel sample streams.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from run import save, sha
from figures import plt, Image
from gpis_shaderball_renewal_3d import (
    REPORT, GROUPS, GLOBAL, MODE, WEIGHTS, original, destination, verified,
)

OUT = REPORT/'bottom_audit'
REGIONS = {
    'bottom': [44, 164, 213, 216],
    'left': [44, 174, 98, 216],
    'right': [158, 174, 213, 216],
    'center_control': [112, 68, 144, 100],
}


def crop(a, bounds):
    x0, y0, x1, y1 = bounds
    return a[y0:y1, x0:x1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources, images = [], [], []
    for group in GROUPS:
        data = {GLOBAL: verified(original(group, GLOBAL)),
                MODE: verified(destination(group, 'renewal_3d'))}
        renewal, global_ = data[MODE], data[GLOBAL]
        delta = renewal['y']-global_['y']
        half_delta = [a-b for a, b in zip(renewal['halves'], global_['halves'])]
        assert np.allclose(delta, .5*(half_delta[0]+half_delta[1]),
                           rtol=1e-4, atol=3e-7)
        noise = .5*(half_delta[0]-half_delta[1])
        label = 'A' if group.endswith('0p02') else 'B'
        for name, bounds in REGIONS.items():
            d = crop(delta, bounds)
            n = crop(noise, bounds)
            h = [crop(a, bounds) for a in half_delta]
            global_mean = float(crop(global_['y'], bounds).mean())
            renewal_mean = float(crop(renewal['y'], bounds).mean())
            mean = float(d.mean())
            approximate_se = float(np.sqrt(np.sum(n*n))/d.size)
            rms = float(np.sqrt(np.mean(d*d)))
            noise_rms = float(np.sqrt(np.mean(n*n)))
            rows.append(dict(group=label, key=group, region=name, bounds=bounds,
                pixels=d.size, global_mean_y=global_mean, renewal_mean_y=renewal_mean,
                renewal_minus_global_mean_y=mean, relative_mean_difference=mean/global_mean,
                split_half_difference_means=[float(a.mean()) for a in h],
                same_mean_sign_in_both_halves=bool(h[0].mean()*h[1].mean()>0),
                approximate_paired_mean_se=approximate_se,
                descriptive_mean_over_se=mean/approximate_se if approximate_se else None,
                pixel_difference_rms=rms, pixel_split_noise_rms=noise_rms,
                rms_over_split_noise=rms/noise_rms if noise_rms else None,
                noise_subtracted_rms_estimate=float(np.sqrt(max(0, rms*rms-noise_rms*noise_rms))),
                half_difference_map_cosine=float(np.sum(h[0]*h[1]) /
                    max(np.sqrt(np.sum(h[0]**2)*np.sum(h[1]**2)), 1e-30))))
        sources.extend(dict(group=label, mode=mode, folder=d['source']['folder'],
                            result_sha256=sha(Path(d['source']['folder'])/'result.json'),
                            recorded_hashes=d['source']['result']['sha256'])
                       for mode, d in data.items())
        images.append((label, data, delta, noise))

    fig, axes = plt.subplots(2, 4, figsize=(16, 5.8), squeeze=False)
    scale = .12
    for i, (label, data, delta, noise) in enumerate(images):
        for j, mode in enumerate((GLOBAL, MODE)):
            image = np.asarray(Image.open(Path(data[mode]['source']['folder'])/'render.png').convert('RGB'))
            axes[i, j].imshow(crop(image, REGIONS['bottom']), interpolation='nearest')
            axes[i, j].set_title(('GPIS global' if mode==GLOBAL else 'GPIS 3D Renewal+')+' · filmic PNG')
        for j, value in enumerate((delta, noise), 2):
            handle = axes[i, j].imshow(crop(value, REGIONS['bottom']), cmap='RdBu_r',
                                       vmin=-scale, vmax=scale, interpolation='nearest')
            axes[i, j].set_title('Linear ΔY = Renewal+ − global' if j==2 else 'Split noise = (ΔY_A − ΔY_B)/2')
        axes[i, 0].set_ylabel(label+' group')
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Fixed bottom crop: x=[44,213), y=[164,216) · 256², 128 spp\n'
                 'Signed linear-light difference and paired split-half noise; no new render', fontsize=15)
    fig.subplots_adjust(left=.035, right=.94, top=.78, bottom=.16, wspace=.06, hspace=.36)
    fig.colorbar(handle, cax=fig.add_axes([.955, .18, .012, .56]), label='Linear luminance')
    fig.text(.035, .035, 'Red: Renewal+ brighter. Blue: Renewal+ darker. '
             'Shared ±0.12 scale; image-only statistics cannot separate memory-model and discretization effects.', fontsize=10)
    fig.savefig(OUT/'bottom_difference.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    summary = dict(complete=True, scope='Existing 128-spp images; current 3D Renewal+ versus global.',
        regions=REGIONS, selection='Fixed geometric regions supplied before calculating these statistics; not selected by sorting errors.',
        estimate=dict(delta='renewal_Y - global_Y', half_delta='renewal_half_Y - global_half_Y',
            noise='(half_delta_A-half_delta_B)/2',
            approximate_mean_se='sqrt(sum(noise_pixel**2))/pixel_count',
            assumptions='Odd/even halves have comparable independent sampling errors. Pixel sample streams are approximately independent. Pairing retains shared-randomness cancellation between the two models.',
            limitations='Single seed and one split. SE is descriptive, not a convergence proof or an independent-seed confidence interval. It excludes discretization and sparse-convolution approximation bias, and cannot attribute a coherent difference to a specific implementation cause.'),
        validation='Both render records, recorded file hashes, finite nonnegative PFMs, and half reconstruction verified.',
        sources=sources, statistics=rows,
        runner_sha256=sha(Path(__file__)),
        figure=dict(path='bottom_difference.png', sha256=sha(OUT/'bottom_difference.png')))
    save(OUT/'image_statistics.json', summary)
    for row in rows:
        print(row['group'], row['region'],
              'delta', round(row['renewal_minus_global_mean_y'], 7),
              'relative', round(row['relative_mean_difference']*100, 3), '%',
              'SE', round(row['approximate_paired_mean_se'], 7),
              'half means', [round(x, 7) for x in row['split_half_difference_means']],
              'RMS/noise', round(row['rms_over_split_noise'], 3))


if __name__=='__main__':
    main()
