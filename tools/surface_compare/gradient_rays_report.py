"""Build the local, reproducible shaderball gradient-ray report after refinement."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import math
from pathlib import Path
import re
import shutil
import sys

import gradient_rays as g
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtri
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SELECTION = [
    ("P1",".",656,"improved","明显改善：球体下方的入射光线"),
    ("P2",".",600,"improved","明显改善：球体下沿的入射光线"),
    ("P3",".",1348,"improved","多次反射：conditional 已经改善，bridge 进一步改善"),
    ("C1","dense",15386,"classic","classic 更好：差距主要来自起点条件化"),
    ("C2","ultradense",15293,"classic","classic 更好：bridge 仅有很小的增益"),
    ("C3","expanded",16184,"classic","classic 更好：较小但稳定的差距"),
    ("F1","dense",914,"control","修正失效：两个 anchor 漏掉中间的生存约束"),
    ("N1",".",194,"control","无明显变化：三种分布几乎重合"),
]
COLORS = dict(classic="#737d8c",conditional="#e39127",bridge="#287ae1",reference="#111827")
LABELS = dict(classic="classic_global",conditional="global_conditional",bridge="conditional + bridge",reference="GP prefix reference")
STYLES = dict(classic=":",conditional="--",bridge="-",reference="-")
MARKERS = dict(classic="o",conditional="s",bridge="^",reference="x")

def inline_svg(path,key):
    svg=path.read_text(encoding="utf-8")
    svg=svg[svg.index("<svg"):]
    # Matplotlib reuses glyph IDs between figures. Namespace every definition
    # and reference so eight inline figures can be toggled independently.
    svg=re.sub(r'\bid="([^"]+)"',lambda m:f'id="{key}_{m[1]}"',svg)
    svg=re.sub(r'((?:xlink:)?href)="#([^"]+)"',lambda m:f'{m[1]}="#{key}_{m[2]}"',svg)
    return re.sub(r'url\(#([^)]+)\)',lambda m:f'url(#{key}_{m[1]})',svg)

def plot_differences(folder,key,curves,scale,limits):
    fig,axes=plt.subplots(1,2,figsize=(13,3.2),layout="constrained")
    pairs=[("classic","conditional"),("conditional","bridge")]
    result={}
    for ax,(left,right) in zip(axes,pairs):
        delta=curves[left+"_cdf"]-curves[right+"_cdf"]
        maximum=float(np.max(abs(delta)))
        result[left+"_vs_"+right]=maximum
        ax.axhline(0,color="#8a94a4",lw=.8)
        ax.plot(curves["v"]/scale,delta,color=COLORS[left],lw=1.6)
        ax.set(xlim=limits,xlabel="axial gradient v / (sigma gamma)",ylabel="CDF difference",
            title=f"{left} - {right}: max |difference| = {maximum:.3g}")
        if maximum<1e-12:
            ax.set_ylim(-1e-12,1e-12)
            ax.text(.5,.8,"Numerically coincident (below 1e-12)",ha="center",transform=ax.transAxes,fontsize=9)
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ax.grid(alpha=.15)
    fig.savefig(folder/f"{key}-differences.png",dpi=150)
    fig.savefig(folder/f"{key}-differences.svg")
    plt.close(fig)
    return result

def covariance_checks(folder,details):
    checks=[]
    for r in details["records"]:
        law=g.models(r)["bridge"]; ray=g.Ray(r)
        queries=np.asarray(r["query_derivatives"])
        pdf_error=float(np.max(abs(law.pdf(queries)-r["production_pdf"])))
        cdf_error=float(np.max(abs(law.cdf(queries)-r["production_cdf"])))
        f,_=ray.features(r["anchor_positions"])
        mu=np.asarray(r["anchor_means"])+f@ray.coefficient_mean
        beta=(f@ray.free3)@ray.unit_gradient
        residual=f@ray.free4; sd=np.linalg.norm(residual,axis=1)
        p,q=mu/sd,beta/sd; rho=(residual@residual.T)/np.outer(sd,sd)
        production=r["production"]
        anchor_error=float(max(np.max(abs(p-production["anchor_p"])/(1+abs(p))),
            np.max(abs(q-production["anchor_q"])/(1+abs(q))),abs(rho[0,1]-production["anchor_rho"])))
        row=dict(id=r["id"],**ray.checks(),pdf_max_error=pdf_error,cdf_max_error=cdf_error,
                 anchor_max_relative_error=anchor_error)
        row["pass"]=bool(cdf_error<2e-7 and pdf_error<2e-5 and anchor_error<2e-6
                         and row["endpoint_error"]<1e-9 and row["kernel_relative_error"]<1e-12)
        checks.append(row)
    controls=[]
    # Check the reference integration against exact one-anchor probit laws,
    # including a tail-dominated endpoint distribution.
    normals=ndtri(qmc.Sobol(1,scramble=True,seed=731,bits=52).random_base2(20).ravel())
    for mu,sd,p,q in [(-.2,.1,-.4,-1.2),(.4,.1,-1.,-.5),(-.8,.1,1.,-3.)]:
        law=g.Density(mu,sd,p,q,True)
        ref=g.Prefix((p+normals)/(-q),mu,sd)
        grid=g.grid_for({"model":law},ref)
        error=g.ks(law,ref,grid)
        controls.append(dict(mu=mu,sd=sd,p=p,q=q,cdf_max_error=error,pass_=error<1e-5))
    validation=dict(production_checks=checks,reference_one_anchor_controls=controls,
        all_pass=all(r["pass"] for r in checks) and all(r["pass_"] for r in controls))
    g.write(folder/"validation.json",validation)
    if not validation["all_pass"]:
        raise AssertionError("Diagnostic validation failed; inspect validation.json")
    return validation

def angle_statistics(row,curves):
    r=row["record"]; w=np.asarray(r["wi"])
    axis=np.eye(3)[np.argmin(abs(w))]; b1=np.cross(w,axis);b1/=np.linalg.norm(b1)
    basis=np.array([b1,np.cross(w,b1)])
    scale=r["sigma"]*r["gamma"]; qt=r["t"]*r["gamma"]
    prior=basis@np.asarray(r["end_mean_gradient"])
    conditional=prior+math.exp(-qt*qt/2)*(basis@(np.asarray(r["birth_gradient"])-r["birth_mean_gradient"]))
    sd=scale*math.sqrt(-math.expm1(-qt*qt))
    inverse={}
    for name in COLORS:
        F=np.maximum.accumulate(curves[name+"_cdf"])
        unique,index=np.unique(F,return_index=True)
        inverse[name]=PchipInterpolator(unique,curves["v"][index])
    values={name:[] for name in COLORS}
    for repeat in range(4):
        sample=qmc.Sobol(3,scramble=True,seed=784+repeat,bits=52).random_base2(15)
        transverse=ndtri(sample[:,1:])
        for name in COLORS:
            radius=np.linalg.norm((prior if name=="classic" else conditional)
                +(scale if name=="classic" else sd)*transverse,axis=1)
            v=inverse[name](sample[:,0])
            values[name].append(float(np.mean(np.degrees(np.arctan2(radius,-v)))))
    return {name:dict(mean_degrees=float(np.mean(v)),qmc_standard_error=float(np.std(v,ddof=1)/2)) for name,v in values.items()}

def anchor_ablation(row,curves):
    r=row["record"]; p=r["production"]
    normal=ndtri(qmc.Sobol(2,scramble=True,seed=1419,bits=52).random_base2(19))
    rho=p["anchor_rho"]
    normal[:,1]=rho*normal[:,0]+math.sqrt(max(0,1-rho*rho))*normal[:,1]
    bounds=np.min((np.asarray(p["anchor_p"])+normal)/(-np.asarray(p["anchor_q"])),axis=1)
    exact=g.Prefix(bounds,p["mu"],p["sd"])
    cdf=exact.cdf(curves["v"])
    curves["two_anchor_cdf"]=cdf
    return dict(exact_two_anchor_vs_reference_ks=float(np.max(abs(cdf-curves["reference_cdf"]))),
                current_probit_vs_exact_two_anchor_ks=float(np.max(abs(curves["bridge_cdf"]-cdf))))

def projected(points,scene):
    camera=scene["camera"]; t=camera["transform"]
    origin=np.asarray(t["position"]); f=np.asarray(t["look_at"])-origin;f/=np.linalg.norm(f)
    right=np.cross(f,t["up"]);right/=np.linalg.norm(right);up=np.cross(right,f)
    p=np.asarray(points)-origin
    depth=p@f; scale=np.tan(np.deg2rad(camera["fov"])/2)
    return np.column_stack([(p@right/depth/scale+1)*128,(1-p@up/depth/scale)*128])

def plot_case(folder,key,row,curves,background,scene):
    r=row["record"]; scale=r["sigma"]*r["gamma"]
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    ax=axs[0,0];ax.imshow(background,extent=(-.5,255.5,255.5,-.5))
    path=projected(r["vertices"][1:],scene)
    ax.plot(path[:,0],path[:,1],color="white",linewidth=1.1,marker=".",markersize=4)
    segment=projected([r["birth_position"],r["position"]],scene)
    ax.plot(segment[:,0],segment[:,1],color="#ee5265",linewidth=2.2,marker="o",markersize=4)
    ax.plot(*r["pixel"],marker="o",markersize=11,markerfacecolor="none",markeredgecolor="#ffeb63",markeredgewidth=1.7)
    ax.set(xlim=(-.5,255.5),ylim=(255.5,-.5),xlabel="camera pixel x",ylabel="camera pixel y",
           title=f"{key}: pixel {tuple(r['pixel'])}, sample {r['sample']}, hit #{r['bounce']+1}")
    ax=axs[0,1]
    distance=np.linspace(0,r["t"],len(r["mean_profile"]))*r["gamma"]
    ax.plot(distance,np.asarray(r["mean_profile"])/r["sigma"],color="#858b96",lw=1.6,label="actual shaderball SDF / sigma")
    x=curves["s"]*r["gamma"];mean=curves["bridge_mean"]/r["sigma"];sd=curves["bridge_sd"]/r["sigma"]
    ax.plot(x,mean,color="#337c99",label="GP mean | birth jet, F(t)=0")
    ax.fill_between(x,mean-2*sd,mean+2*sd,color="#337c99",alpha=.18,label="+/- 2 conditional std. dev.")
    for j,s in enumerate([r["t"]/3,max(2*r["t"]/3,r["t"]-1/r["gamma"])]):
        ax.axvline(s*r["gamma"],color="#9a5de2",ls=":",lw=1.5,label="production anchors" if j==0 else None)
    ax.axhline(0,color="#121b2a",lw=.9)
    ax.set(xlabel="distance / ell",ylabel="field value / sigma",ylim=(-8,12),title="Prefix profile (surface band; before positivity test)")
    ax.legend(fontsize=8,loc="upper right")
    for col,suffix in [(0,"pdf"),(1,"cdf")]:
        ax=axs[1,col]
        handles={}
        if suffix=="cdf":
            handles["two_anchor"],=ax.plot(curves["v"]/scale,curves["two_anchor_cdf"],color="#9a5de2",ls=(0,(3,2,1,2)),lw=1,
                marker="d",markevery=(.1,.23),markersize=4,markerfacecolor="white",label="exact 2-anchor (QMC)",gid="curve-cdf-two_anchor")
        # Sparse, staggered markers reveal coincident curves in static exports.
        # The HTML also permits hiding each curve; no values are offset.
        for i,name in enumerate(["reference","bridge","conditional","classic"]):
            handles[name],=ax.plot(curves["v"]/scale,curves[name+"_"+suffix]*(scale if suffix=="pdf" else 1),
                color=COLORS[name],ls=STYLES[name],lw=1.65 if name!="reference" else 2.7,label=LABELS[name],
                marker=MARKERS[name],markevery=(.03+.055*i,.23),markersize=4.5,markerfacecolor="white",markeredgewidth=1,
                gid=f"curve-{suffix}-{name}")
        distributions=[row["models"][name] for name in ["classic","conditional","bridge"]]+[row["reference"]["summary"]]
        low=min(d["quantiles"][0]-2.5*d["sd"] for d in distributions)
        high=min(0,max(d["quantiles"][2]+2.5*d["sd"] for d in distributions))
        ax.set(xlim=(low/scale,high/scale),xlabel="axial gradient v / (sigma gamma)",
               ylabel="density" if suffix=="pdf" else "CDF",title="Normalized collision gradient "+suffix.upper())
        ax.grid(alpha=.15)
        order=list(COLORS)+(["two_anchor"] if suffix=="cdf" else [])
        ax.legend(handles=[handles[n] for n in order],fontsize=8)
    fig.savefig(folder/f"{key}.png",dpi=160)
    fig.savefig(folder/f"{key}.svg")
    plt.close(fig)
    row["curve_overlap"]=plot_differences(folder,key,curves,scale,(low/scale,high/scale))

def summary_population(folder):
    data=g.read(folder/"expanded/rays.json")
    screen=g.read(folder/"expanded/screen-small-effect.json")
    by={r["id"]:r for r in screen["records"]}
    unique={}
    for r in data["records"]:
        signature=tuple(r["birth_position"]+r["wi"]+[r["t"],r["f0"],r["g0"]]+r["birth_gradient"])
        unique.setdefault(signature,r["id"])
    rows=[by[i] for i in unique.values()]
    evaluated=[r for r in rows if "errors" in r]
    result=dict(path_attempts=sum(c["paths"] for c in data["counts"]),counts=data["counts"],
        small_age_excluded=sum(c["small_age_excluded"] for c in data["counts"]),
        retained_records=len(data["records"]),unique_states=len(rows),screen_reference_states=len(evaluated),
        bridge_changes={str(v):sum(r["bridge_change_ks"]>v for r in rows) for v in [1e-5,.001,.01,.05,.1]},
        coarse_negative_gain_vs_conditional=sum(r["gain_vs_conditional"]<0 for r in evaluated),
        unresolved_rare_or_grid_sensitive=sum(r["grid_ks"]>.005 or r["prefix"]["log_prefix_mass_over_endpoint_flux"]<-20 for r in evaluated),
        screening_threshold=screen["threshold"],screening_failure_count=len(screen["failures"]))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,name in zip(axes,["classic","conditional"]):
        ax.plot([0,1],[0,1],color="#777",lw=1)
        ax.scatter([r["errors"][name] for r in evaluated],[r["errors"]["bridge"] for r in evaluated],
                   s=10,alpha=.5,c=[r["bounce"] for r in evaluated],cmap="viridis",vmin=0,vmax=5)
        ax.set(xlim=(-.01,1.01),ylim=(-.01,1.01),xlabel=name+" KS error",ylabel="bridge KS error",
               title="Coarse screen: below diagonal = bridge closer")
        ax.grid(alpha=.15)
    fig.savefig(folder/"population.png",dpi=160);plt.close(fig)
    return result

def main(folder,plots_only=False):
    folder=folder.resolve();details=g.read(folder/"detail.json")
    cached=g.read(folder/"report-data.json") if plots_only else None
    validation=cached["validation"] if cached else covariance_checks(folder,details)
    population=cached["population"] if cached else summary_population(folder)
    previous={row["key"]:row for row in cached["cases"]} if cached else {}
    scene=g.read(g.ROOT/details["scene"])
    source=(g.ROOT/details["scene"]).parent/"shaderball.png"
    shutil.copyfile(source,folder/"shaderball.png")
    background=plt.imread(folder/"shaderball.png")
    rows=[]
    for key,relative,identity,category,title in SELECTION:
        row=g.read(folder/relative/f"ray-{identity}.json")
        curves=dict(np.load(folder/relative/f"ray-{identity}-curves.npz"))
        row.update(key=key,category=category,title=title,artifact_folder=relative)
        row["angle_statistics"]=previous[key]["angle_statistics"] if cached else angle_statistics(row,curves)
        row["anchor_ablation"]=anchor_ablation(row,curves)
        plot_case(folder,key,row,curves,background,scene)
        rows.append(row)
        print(f"Plotted {key}, {row['anchor_ablation']}",flush=True)
    report=dict(population=population,cases=rows,validation=validation,
        definition="Fixed birth jet, direction and collision distance. Axial gradient CDF error; finite-prefix reference, not image error.",
        executable_sha256=hashlib.sha256((g.ROOT/"build/Release/fp_gradient_rays.exe").read_bytes()).hexdigest())
    g.write(folder/"report-data.json",report)
    with (folder/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["key","id","pixel","sample","hit","source","classic_ks","conditional_ks","bridge_ks","gain_vs_classic","gain_vs_conditional","grid_change","cdf_se"]
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in rows:
            r=row["record"];ref=row["reference"]
            writer.writerow(dict(key=row["key"],id=row["id"],pixel=str(r["pixel"]),sample=r["sample"],hit=r["bounce"]+1,source=r["source"],
                classic_ks=row["errors"]["classic"]["ks"],conditional_ks=row["errors"]["conditional"]["ks"],bridge_ks=row["errors"]["bridge"]["ks"],
                gain_vs_classic=row["gains"]["classic"]["value"],gain_vs_conditional=row["gains"]["conditional"]["value"],
                grid_change=ref.get("grid_ks_medium_vs_fine",ref.get("grid_ks_207_vs_415")),cdf_se=ref["max_pointwise_cdf_se"]))
    def table_row(row):
        r=row["record"];e=row["errors"]
        return f'<tr><td><a href="#{row["key"]}">{row["key"]}</a></td><td>{r["pixel"]}</td><td>{r["sample"]} / {r["bounce"]+1}</td><td>{e["classic"]["ks"]:.6f}</td><td>{e["conditional"]["ks"]:.6f}</td><td>{e["bridge"]["ks"]:.6f}</td><td>{row["gains"]["conditional"]["value"]:+.6f}</td></tr>'
    cards=[]
    for row in rows:
        key=row["key"];r=row["record"];ref=row["reference"];a=row["anchor_ablation"]
        overlap=row["curve_overlap"]
        controls="".join(f'<label style="color:{color}"><input type="checkbox" checked onchange="toggleCurve(\'{key}\',\'{name}\',this.checked)">{label}</label>'
            for name,color,label in [(name,COLORS[name],LABELS[name]) for name in COLORS]+[("two_anchor","#9a5de2","exact 2-anchor (CDF)")])
        grid=ref.get("grid_ks_medium_vs_fine",ref.get("grid_ks_207_vs_415"))
        special={"improved":"bridge 的增益超过参考的数值变化；P3 同时展示起点条件化与 bridge 的独立作用。",
                 "classic":"classic 的纵向梯度分布更接近参考。但 bridge 相对 conditional 仍是小幅改善；不能把 classic 的优势归因于 bridge 单独造成负优化。",
                 "control":"该条光线用于检验局部修正的适用范围。"}[row["category"]]
        if key=="F1": special="两个 anchor 的结果与当前 probit 很接近，却都远离整段 prefix 参考。主要问题是缺失中间的生存约束；归一化梯度修正没有修正这个碰撞距离的概率。"
        if key=="N1": special="整段生存约束几乎不筛掉终点梯度；bridge 在这里基本没有改变分布。"
        angles="，".join(f'{LABELS[name]} {row["angle_statistics"][name]["mean_degrees"]:.3f}°' for name in COLORS)
        cards.append(f'''<article id="{key}" data-category="{row['category']}"><h2>{key} · {row['title']}</h2>
<p>{special}</p><p class="meta">来源 {r['source']}；像素 {r['pixel']}；sample={r['sample']}（从 0 开始）；第 {r['bounce']+1} 次介质碰撞；t/ell={r['t']*r['gamma']:.6f}；f₀={r['f0']:.6g}；g₀={r['g0']:.6g}。</p>
<div class="curve-controls" aria-label="{key} 曲线显示开关">{controls}<button type="button" onclick="resetCurves('{key}')">显示全部</button></div>
<p class="meta">重合检查（两条曲线之间的最大 CDF 差）：classic ↔ conditional = {overlap['classic_vs_conditional']:.3g}；conditional ↔ bridge = {overlap['conditional_vs_bridge']:.3g}。曲线接近时会互相遮挡；取消勾选其他曲线即可单独查看。不同形状的空心标记位于各自原始曲线上。</p>
<div class="case-plot" role="img" aria-label="{key} 的光线位置、SDF、梯度 PDF 和 CDF">{inline_svg(folder/f'{key}.svg',key)}</div>
<p class="meta"><a href="{key}.svg">下载 SVG</a> · <a href="{key}.png">下载 PNG</a></p>
<details><summary>查看重合曲线的 CDF 差值图</summary><p class="meta">纵轴独立缩放，数值表示两种方法之间的差异，并非它们相对参考的误差。小于 10⁻¹² 的差异不再放大。</p><a href="{key}-differences.svg"><img src="{key}-differences.png" alt="{key} 两对曲线的 CDF 差值"></a></details>
<p class="meta">参考节点 {ref['points']}；每个重复 2^{ref['power']} 个 Sobol 样本，共 {ref['replicates']} 个重复。最后两级网格 CDF 差 {grid:.3g}；最大点态 CDF 标准误 {ref['max_pointwise_cdf_se']:.3g}。
精确双 anchor → prefix 的 KS={a['exact_two_anchor_vs_reference_ks']:.6f}；当前 probit → 精确双 anchor 的 KS={a['current_probit_vs_exact_two_anchor_ks']:.6f}。</p>
<details><summary>法线倾角与可复现状态</summary><p>θ=atan2(|g⊥|, −v)，0° 为正入射，90° 为掠射。使用条件横向高斯与共同 QMC 样本得到的平均倾角：{angles}。这是补充量，不用于上表的排名。</p>
<p>起点 {html.escape(str(r['birth_position']))}<br>方向 {html.escape(str(r['wi']))}<br>碰撞点 {html.escape(str(r['position']))}</p>
<a href="{row['artifact_folder']}/ray-{row['id']}.json">完整输入和参考统计 JSON</a></details></article>''')
    p=population
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball 梯度修正 · 逐光线诊断</title>
<style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f8;color:#1c2938;margin:0}}main{{max-width:1260px;margin:auto;padding:28px}}h1{{font-size:30px}}h2{{font-size:22px}}article,.panel{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:22px;margin:22px 0}}img{{width:100%;height:auto}}.case-plot svg{{width:100%;height:auto;display:block}}.curve-controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}}.curve-controls label{{cursor:pointer;font-size:14px;font-weight:600}}.curve-controls input{{accent-color:currentColor;margin-right:6px}}a{{color:#176ac3}}.meta{{font-size:13px;color:#526376}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:9px 12px;border-bottom:1px solid #dce3eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}button{{padding:8px 14px;border:1px solid #a7bad1;border-radius:7px;background:white;margin:5px;cursor:pointer}}.notice{{border-left:4px solid #367bd4;padding:12px 18px;background:#eaf1fc}}code,pre{{background:#edf1f5;padding:3px 6px;border-radius:4px}}pre{{overflow:auto;padding:16px}}details{{margin-top:14px}}.hidden{{display:none}}</style>
<main><h1>Shaderball 的 bridge 梯度修正是否有效？</h1><p>配置 ell=0.2、sigma=0.02；使用现有 256×256、128 spp 图像定位。这里比较固定碰撞条件下的<strong>纵向梯度分布</strong>，不是对这张图像的误差测量。</p>
<div class="notice"><strong>结论：</strong>找到了明显改善的光线，也找到了 classic 比“起点条件化＋bridge”更好的光线。目前没有找到经验证的“bridge 单独使 conditional 变差”的例子。三者不能混为同一种结论。</div>
<section class="panel"><h2>怎么比较</h2><p>每条记录固定同一个起点值/完整梯度、方向和碰撞距离。classic 使用无历史的负通量梯度；conditional 加入起点观测；bridge 使用当前生产代码中的双 anchor＋probit 修正。参考通过精确 SE 高斯条件化，对整段光线的多个内部点同时施加 F(s)&gt;0，并保留碰撞通量 −v。</p>
<p><code>p_ref(v) ∝ (−v) 1[v&lt;0] p(v | C₀,F(t)=0) P(∀j F(sⱼ)&gt;0 | C₀,F(t)=0,F′(t)=v)</code></p>
<p>误差取 <code>KS = max |CDF_model − CDF_reference|</code>，越小越好。固定碰撞后评估，才能区分梯度改进与自由程变化。当前 classic FP 与 conditional FP 在相同输入状态下共用这一个纵向梯度 Law。</p>
<p class="meta">这里的参考仍然是有限节点的 prefix 近似，不是连续首次穿越的严格真值，也没有加入这次 flight 之前的完整路径历史。不能据此证明最终图像一定变好。</p></section>
<section class="panel"><h2>筛选覆盖范围</h2><p>扩大扫描共 {p['path_attempts']:,} 条诊断路径，保留 {p['retained_records']:,} 个记录，去除两种采样源重复的首碰撞状态后共 <strong>{p['unique_states']:,}</strong> 个状态。{p['small_age_excluded']} 个归一化距离小于 0.05 的碰撞未进入 QR 参考。对 bridge 变化超过 10⁻⁵ 的 {p['screen_reference_states']:,} 个不同状态计算了初筛参考；其中变化超过 0.001 的有 {p['bridge_changes']['0.001']} 个（{100*p['bridge_changes']['0.001']/p['unique_states']:.2f}%），超过 0.01 的有 {p['bridge_changes']['0.01']} 个。</p>
<p>初筛未找到 bridge 的 KS 高于 conditional 的候选；这不是不会负优化的证明。初筛有 {p['unresolved_rare_or_grid_sensitive']} 个稀有或对网格敏感的状态，个例结论以加密复核为准。其他变化小于 10⁻⁵ 的状态，即便存在 KS 负优化，幅度也不超过两种 CDF 的最大差。</p>
<img src="population.png" alt="初筛的基线与 bridge KS 对照"><p class="meta">图中每点是一个固定状态，斜线下方表示 bridge 更接近初筛参考。诊断调用生产相机、介质距离和 mirror 相位采样；省略 NEE、roulette 和普通网格表面反射，并在地板/介质盒处停止。种子为 20261108/20261109，图像种子为 41；叠加路径只用于定位，不能解释为原渲染中该像素的精确路径或亮度贡献。</p></section>
<section class="panel"><h2>精选光线</h2><p>sample 从 0 开始，碰撞次序从 1 开始。最后一列为 conditional 误差减 bridge 误差，正值表示 bridge 改善。</p>
<table><thead><tr><th>案例</th><th>像素</th><th>sample / 碰撞</th><th>classic KS</th><th>conditional KS</th><th>bridge KS</th><th>bridge 增益</th></tr></thead><tbody>{''.join(table_row(row) for row in rows)}</tbody></table>
<p><a href="metrics.csv">下载数值表</a> · <a href="report-data.json">全部精选数据</a> · <a href="validation.json">数值验证</a></p></section>
<nav><button onclick="filter('all')">全部</button><button onclick="filter('improved')">明显改善</button><button onclick="filter('classic')">classic 更好</button><button onclick="filter('control')">无效与失效</button></nav>{''.join(cards)}
<section class="panel"><h2>实现与复现</h2><p>新增诊断程序和报告脚本，没有改变渲染模式。独立 QR 端点统计、anchor 统计、Python PDF/CDF 与生产 C++ Law 的比较，以及单 anchor 的参考控制均通过。图中紫色曲线把“双 anchor 选择”与“probit 切线近似”拆开检查。</p>
<pre>cmake --build build --config Release --target fp_gradient_rays --parallel 2
build/Release/fp_gradient_rays.exe scan outputs/shaderball-gradient-ray-diagnostic-20261108/expanded/spec.json outputs/shaderball-gradient-ray-diagnostic-20261108/expanded/rays.json
python -B tools/surface_compare/gradient_rays.py screen outputs/shaderball-gradient-ray-diagnostic-20261108/expanded --threshold 0.00001 --power 11 --workers 6 --output screen-small-effect.json
python -B tools/surface_compare/gradient_rays_report.py outputs/shaderball-gradient-ray-diagnostic-20261108</pre>
<p class="meta">精确选中状态已保存在 detail-input.json / detail.json；每个 ray-*.json 包含 QMC seed、节点数和重复次数。完整报告数据用于复核，不应把诊断采样频率当作最终辐射亮度权重。</p></section></main>
<script>function filter(kind){{document.querySelectorAll('article[data-category]').forEach(x=>x.classList.toggle('hidden',kind!=='all'&&x.dataset.category!==kind));}}
function toggleCurve(key,name,visible){{for(const kind of ['pdf','cdf']){{const line=document.getElementById(key+'_curve-'+kind+'-'+name);if(line)line.style.display=visible?'':'none';}}}}
function resetCurves(key){{document.querySelectorAll('#'+key+' .curve-controls input').forEach(x=>{{x.checked=true;x.dispatchEvent(new Event('change'));}});}}</script></html>'''
    (folder/"comparison.html").write_text(document,encoding="utf-8")
    print("Saved",folder/"comparison.html",flush=True)

if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("folder",type=Path)
    parser.add_argument("--plots-only",action="store_true",help="Reuse the saved numerical validation, population and angle statistics")
    args=parser.parse_args()
    main(args.folder,args.plots_only)
