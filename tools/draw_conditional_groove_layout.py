"""Draw the actual view45 U-groove layout as a standalone SVG and HTML page."""
from pathlib import Path
import html
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/conditional_ugroove'
SCENE = OUT / 'res256_spp256/A_strong/view45/total/classic_global/seed202610281/scene.json'
scene = json.loads(SCENE.read_text(encoding='utf-8'))
meta = json.loads((OUT / 'fields/main/u_groove.nvdb.json').read_text(encoding='utf-8'))
geometry = meta['geometry']
cam = scene['camera']['transform']
lamp = next(p for p in scene['primitives'] if p.get('name') == 'right_area')
cp, target = cam['position'], cam['look_at']
lp, ls = lamp['transform']['position'], lamp['transform']['scale']
w, depth, length = (geometry[k] for k in ('width', 'depth', 'length'))
base, wall = geometry['base_thickness'], geometry['wall_thickness']
distance = math.dist(cp, target)
angle = math.degrees(math.atan2(abs(target[0]-cp[0]), cp[1]-target[1]))

BLUE, GOLD, RED = '#2475ba', '#bb8117', '#d04e45'
INK, MUTED, LINE = '#24384b', '#617384', '#d9e2e8'
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1080" viewBox="0 0 1600 1080" role="img" aria-labelledby="title desc">
<title id="title">U 槽正式场景：相机、面积光与观察区</title>
<desc id="desc">Y 轴朝上，槽宽沿 x、槽长沿 z。相机在左上方以 45 度看右内壁；面积光在右上方朝负 y 发光。附槽口放大剖面与俯视投影。</desc>
<defs>
  <pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse"><rect width="10" height="10" fill="#e7edf1"/><path d="M-2 2L2-2M0 10L10 0M8 12L12 8" stroke="#cbd5de" stroke-width="1"/></pattern>
</defs>
<style>text{font-family:"Microsoft YaHei UI","Microsoft YaHei",sans-serif;fill:#24384b} .small{fill:#617384}</style>
<rect width="1600" height="1080" fill="#f2f5f7"/>''']


def element(tag, **attrs):
    encoded = ' '.join(f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"' for key, value in attrs.items())
    parts.append(f'<{tag} {encoded}/>')


def text(x, y, value, size=20, color=INK, anchor='start', weight=400, **attrs):
    extra = ' '.join(f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"' for key, value in attrs.items())
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" text-anchor="{anchor}" font-weight="{weight}" {extra}>{html.escape(value)}</text>')


def line(a, b, color=INK, width=2, dash=None):
    attrs = dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1], stroke=color, stroke_width=width, stroke_linecap='round')
    if dash:
        attrs['stroke_dasharray'] = dash
    element('line', **attrs)


def arrow(a, b, color=INK, width=2, head=10, dash=None):
    line(a, b, color, width, dash)
    theta = math.atan2(b[1]-a[1], b[0]-a[0])
    points = [b, (b[0]-head*math.cos(theta-.40), b[1]-head*math.sin(theta-.40)),
              (b[0]-head*math.cos(theta+.40), b[1]-head*math.sin(theta+.40))]
    element('polygon', points=' '.join(f'{x:.2f},{y:.2f}' for x, y in points), fill=color)


def panel(x, y, width, height):
    element('rect', x=x, y=y, width=width, height=height, rx=18, fill='white', stroke=LINE)


def circle(p, r, fill, stroke='none', width=2):
    element('circle', cx=p[0], cy=p[1], r=r, fill=fill, stroke=stroke, stroke_width=width)


def transform(ox, oy, scale):
    return lambda x, y: (ox+scale*x, oy-scale*y)


def groove(project):
    points = [project(x, y) for x, y in meta['polygon_xy']]
    element('polygon', points=' '.join(f'{x:.2f},{y:.2f}' for x, y in points), fill='url(#hatch)', stroke='#667f92', stroke_width=2)
    line(project(w/2, -.15*w), project(w/2, -.65*w), RED, 7)
    p = project(target[0], target[1])
    circle(p, 6, 'white', BLUE, 2)
    line((p[0]-10, p[1]), (p[0]+10, p[1]), BLUE, 1.5)
    line((p[0], p[1]-10), (p[0], p[1]+10), BLUE, 1.5)


def dimension(a, b, color=MUTED):
    line(a, b, color, 1.5)
    dx, dy = b[0]-a[0], b[1]-a[1]
    norm = math.hypot(dx, dy)
    px, py = -dy/norm*5, dx/norm*5
    for p in (a, b):
        line((p[0]-px, p[1]-py), (p[0]+px, p[1]+py), color, 1.5)


text(44, 57, 'U 槽：相机与光源的实际方位', 32, weight=700)
text(46, 91, '正式版 view45   /   Y-up   /   x：槽宽    y：高度    z：槽长', 20, MUTED)
panel(32, 116, 998, 860)
panel(1054, 116, 514, 408)
panel(1054, 546, 514, 430)

# Cross section, with equal x/y scale and scene-derived coordinates.
text(64, 158, '01 / 整体剖面', 23, weight=600)
text(64, 190, 'x–y 平面，z = 0；相机和灯在槽口两侧', 19, MUTED)
project = transform(570, 690, 800)
c, t = project(*cp[:2]), project(*target[:2])
line(project(-.2, 0), project(.18, 0), LINE, 1.5, '5 6')
groove(project)
arrow(c, t, BLUE, 3, 13)
parts.append(f'<g transform="translate({c[0]:.3f} {c[1]:.3f}) rotate({angle:.3f})">')
element('rect', x=-42, y=-18, width=34, height=36, rx=5, fill=BLUE)
element('polygon', points='-8,-11 8,-20 8,20 -8,11', fill=BLUE)
circle((0, 0), 4, 'white')
parts.append('</g>')
text(250, 253, '相机：左上方 → 右内壁', 23, BLUE, weight=600)
text(250, 283, f'位置 ({cp[0]:.4f}, {cp[1]:.4f}, 0)', 18, MUTED)
text(250, 310, f'FOV {scene["camera"]["fov"]}°  ·  注视距离 {distance:.1f}', 18, MUTED)
line(c, (c[0], c[1]+150), MUTED, 1.5, '5 5')
radius = 90
angle_end = (c[0]+radius*math.sin(math.radians(angle)), c[1]+radius*math.cos(math.radians(angle)))
element('path', d=f'M {c[0]} {c[1]+radius} A {radius} {radius} 0 0 0 {angle_end[0]} {angle_end[1]}', fill='none', stroke=BLUE, stroke_width=2)
text(c[0]+24, c[1]+116, f'{angle:.0f}°', 23, BLUE, weight=600)
text(c[0]-15, c[1]+160, '−y', 18, MUTED, 'end')

light_left = project(lp[0]-ls[0]/2, lp[1])
light_right = project(lp[0]+ls[0]/2, lp[1])
light_center = project(*lp[:2])
text(light_center[0], 345, '面积光：右上方', 23, GOLD, 'middle', 600)
text(light_center[0], 377, f'中心 ({lp[0]:.3f}, {lp[1]:.3f}, 0)', 18, MUTED, 'middle')
text(light_center[0], 406, f'尺寸 {ls[0]:.1f} × {ls[2]:.1f}  ·  发射值 {lamp["emission"]}', 18, MUTED, 'middle')
element('rect', x=light_left[0], y=light_left[1]-6, width=light_right[0]-light_left[0], height=12, rx=3, fill='#f1ca75', stroke=GOLD, stroke_width=2)
for offset in (-.10, 0, .10):
    p = project(lp[0]+offset, lp[1])
    arrow((p[0], p[1]+10), (p[0], p[1]+64), GOLD, 2.5, 10)
text(light_center[0], 543, '发光面法线：−y', 20, GOLD, 'middle')
text(light_center[0], 571, '灯面平行 x–z 平面', 18, MUTED, 'middle')

dimension(project(-w/2, .025), project(w/2, .025))
text(570, 651, f'w = {w:g}', 19, MUTED, 'middle')
line((t[0]+12, t[1]), (724, t[1]), BLUE, 1.5)
text(737, 704, '注视点在右内壁', 21, BLUE, weight=600)
text(737, 734, f'({target[0]:.3f}, {target[1]:.3f}, 0)', 18, MUTED)
text(490, 845, '左壁', 19, MUTED, 'middle')
text(650, 845, '右壁', 19, MUTED, 'middle')
arrow((136, 758), (236, 758), INK, 2, 10)
arrow((136, 758), (136, 658), INK, 2, 10)
circle((136, 758), 7, 'white', INK, 1.5)
circle((136, 758), 2, INK)
text(247, 765, '+x（槽宽）', 18)
text(136, 642, '+y（槽口朝上）', 18, anchor='middle')
text(64, 811, '⊙ +z：沿槽长，垂直本页向外', 18, MUTED)
for x, color, label in [(64, BLUE, '相机视线'), (330, GOLD, '面积光与朝向'), (654, RED, '右内壁 ROI')]:
    line((x, 894), (x+32, 894), color, 4)
    text(x+44, 901, label, 19, color)
text(64, 949, '位置按场景坐标同比例绘制；图标大小和线宽仅作标记。', 18, MUTED)

# Detail view of the mean U-shaped surface.
text(1082, 157, '02 / 槽口剖面放大', 23, weight=600)
text(1082, 188, '均值几何；尺寸使用场景单位', 18, MUTED)
zoom = transform(1276, 274, 1100)
groove(zoom)
dimension(zoom(-w/2, .032), zoom(w/2, .032))
text(1276, 224, f'槽宽 {w:g}', 18, MUTED, 'middle')
dimension(zoom(-w/2-wall-.02, 0), zoom(-w/2-wall-.02, -depth))
text(1077, 329, f'深 {depth:g}', 17, MUTED, 'middle', transform='rotate(-90 1077 329)')
text(1453, 280, 'y = 0', 17, MUTED)
text(1170, 337, '左壁', 18, MUTED, 'middle')
text(1390, 337, '右壁', 18, MUTED, 'middle')
text(1276, 419, f'底板 {base:g}', 19, MUTED, 'middle')
text(1082, 477, 'ROI：x=0.05，y∈[−0.065, −0.015]', 18, RED)
text(1082, 506, '|z| < 0.1；蓝色准星为相机注视点。', 17, MUTED)

# Top view: orthographic projections of the camera, solid, and lamp.
text(1082, 587, '03 / 俯视投影', 23, weight=600)
text(1082, 618, '沿 −y 看向槽口；坐标平面为 x–z', 18, MUTED)
top = transform(1312, 763, 400)
for xmin, xmax, fill in [(-w/2-wall, -w/2, 'url(#hatch)'), (-w/2, w/2, '#f6f8fa'), (w/2, w/2+wall, 'url(#hatch)')]:
    x, y = top(xmin, length/2)
    element('rect', x=x, y=y, width=(xmax-xmin)*400, height=length*400, fill=fill, stroke='#667f92', stroke_width=1.5)
line(top(w/2, -w), top(w/2, w), RED, 6)
x, y = top(lp[0]-ls[0]/2, lp[2]+ls[2]/2)
element('rect', x=x, y=y, width=ls[0]*400, height=ls[2]*400, fill='#fceaca', stroke=GOLD, stroke_width=2)
text(top(lp[0], 0)[0], 663, '面积光', 18, GOLD, 'middle')
text(top(lp[0], 0)[0], 756, 'y = 0.3', 17, GOLD, 'middle')
text(top(lp[0], 0)[0], 783, '朝下发光', 17, GOLD, 'middle')
ct, tt = top(cp[0], cp[2]), top(target[0], target[2])
arrow(ct, tt, BLUE, 2, 10, '6 5')
circle(ct, 7, BLUE)
circle(tt, 5, 'white', BLUE, 2)
text(ct[0], ct[1]-22, '相机', 18, BLUE, 'middle')
dimension(top(-.21, length/2), top(-.21, -length/2))
text(1204, 763, f'槽长 {length:g}', 17, MUTED, 'middle', transform='rotate(-90 1204 763)')
arrow((1523, 801), (1523, 707), INK, 2, 10)
text(1523, 692, '+z', 18, anchor='middle')
arrow((1412, 888), (1523, 888), INK, 2, 10)
text(1535, 895, '+x', 18)
arrow((1130, 790), (1130, 839), BLUE, 2, 9)
text(1082, 874, '画面向上 = −z', 19, BLUE)
text(1082, 909, '蓝色虚线为视线的水平投影。', 17, MUTED)
text(1082, 947, '红色观察区沿槽长取中央 |z| < 0.1。', 17, MUTED)

text(46, 1015, '对应正式配置：256 × 256 / 256 spp / view45；各模式共用这套相机、光源与均值几何。', 19)
text(46, 1050, '面积光向 −y 半球发射，金色箭头只表示灯面法线；环境发射值 0.002。', 18, MUTED)
parts.append('</svg>')
svg = '\n'.join(parts)
(OUT / 'layout.svg').write_text(svg, encoding='utf-8')
relative_scene = SCENE.relative_to(OUT).as_posix()
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>U 槽相机与光源方位</title><style>body{margin:0;background:#f2f5f7;color:#24384b;font:16px "Microsoft YaHei UI",sans-serif}main{max-width:1600px;margin:auto}svg{display:block;width:100%;height:auto}nav{padding:16px 3%;line-height:1.8}a{color:#2475ba;margin-right:24px}</style><main>'''
page += svg + f'<nav><a href="layout.svg">下载 SVG</a><a href="layout.png">下载 PNG</a><a href="{relative_scene}">场景 JSON</a><a href="res256_spp256/comparison.html">正式渲染对比</a></nav></main></html>'
(OUT / 'layout.html').write_text(page, encoding='utf-8')
print(OUT / 'layout.svg')
print(OUT / 'layout.html')
