"""Publish explicitly provisional checkpoint previews without changing render jobs."""
import os
import re
import time

import test_conditional_groove as g
import render_conditional_groove_final as p


def main():
    stage = g.OUT/'res256_spp256'
    preview = stage/'renewal_plus_progress'
    preview.mkdir(exist_ok=True)
    records, rows = [], []
    for group in g.GROUPS:
        folder = stage/group/'view45/total/gpis_renewal_plus/seed202610281'
        text = (folder/'render.log').read_text(encoding='utf-8',errors='replace')
        steps = re.findall(r'Completed (\d+)/256 spp[^\n]*\nSaving checkpoint[^\n]*\nSaving checkpoint took',text)
        complete = (folder/'result.json').exists() and g.load(folder/'result.json')['status']=='complete'
        spp = 256 if complete else int(steps[-1]) if steps else 0
        cells = [f'<th>{group}</th>']
        for mode in ('classic_global','global_conditional'):
            image = f'{group}/view45/total/{mode}/seed202610281/exposure025.png'
            cells.append(f'<td><img src="{image}" width="256" height="256"><br>256 spp · 已完成</td>')
        if spp:
            source = folder/('render.pfm' if complete else 'render_checkpoint.pfm')
            # Native checkpoint files are replaced atomically after each segment.
            image = g.read_image(source)
            target = preview/(group+f'_{spp}spp.png')
            g.write_png(target,g.filmic(image*p.EXPOSURE))
            scene = g.load(folder/'scene.json')
            roi = float(image[g.roi_mask(scene['camera'])].mean())
            cells.append(f'<td><img src="renewal_plus_progress/{target.name}" width="256" height="256"><br><b>{spp}/256 spp'+
                         (' · 已完成' if complete else ' · 进行中')+f'</b><br>ROI {roi:.6g}</td>')
            records.append(dict(group=group,spp=spp,complete=complete,roi_mean=roi))
        else:
            cells.append('<td>首个 32 spp 分段计算中</td>')
            records.append(dict(group=group,spp=0,complete=False))
        local = stage/group/'view45/total/classic_local/seed202610281'
        local_record = local/'result.json'
        if local_record.exists() and g.load(local_record)['status']=='complete':
            local_image = (local/'exposure025.png').relative_to(stage).as_posix()
            local_roi = g.load(local_record)['roi_mean']
            cells.append(f'<td><img src="{local_image}" width="256" height="256"><br>256 spp · 已完成<br>ROI {local_roi:.6g}</td>')
        else:
            cells.append('<td>classic_local 渲染中</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>U 槽 · 四模式渲染对比与进度</title><style>body{font:15px system-ui;margin:28px;background:#f5f7fa;color:#243047}p{max-width:1080px;line-height:1.7}table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #ccd4df;text-align:center}img{display:block}a{color:#175ab0}</style>
<h1>U 槽：四模式渲染对比与进度</h1><p><a href="comparison.html">正式对比页、ROI 与差分</a> · <a href="renewal_plus_manifest.json">GPIS 配置</a></p>
<p>原生 256×256；GPIS 列明确标注当前已保存的 spp，未完成的图只供预览。classic 与 conditional 是已完成的 256 spp 结果。
全部使用曝光 0.25 和同一 filmic 映射，无降噪。GPIS 为 3D sparse-convolution renewal+；classic_local 列仅显示已完成的 256 spp 结果。页面每分钟刷新。</p>
<table><tr><th>组别</th><th>classic_global</th><th>global_conditional</th><th>GPIS renewal+</th><th>classic_local</th></tr>'''
    target = stage/'renewal_plus_progress.html'
    temporary = target.with_suffix('.html.tmp')
    temporary.write_text(page+''.join(rows)+'</table></html>',encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temporary,target)
            break
        except PermissionError:
            if attempt==9:
                print('Preview page is temporarily locked; retain the previous page and retry later.')
                return
            time.sleep(.5)
    p.atomic_save(preview/'status.json',records)
    print('preview '+', '.join(f'{x["group"]}={x["spp"]}' for x in records))


if __name__=='__main__': main()
