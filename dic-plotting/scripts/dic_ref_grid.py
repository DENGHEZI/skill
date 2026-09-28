# -*- coding: utf-8 -*-
"""dic_ref_grid.py —— 期刊 Fig.7/8 风格：荷载进程(列) × 应变分量(行) 干净云图
（Structures 92 (2026) 112808 Zhong et al. 版式）

特征（按真实论文图反推）:
  - 行 = 应变分量 (默认 Exx / Eyy), 列 = 荷载进程 (默认 0/20/40/60/80/100 %)
  - **diverging rainbow, 中心绿=0** (紫-蓝-青-绿-黄-橙-红)
  - 试件块填色: 未测/低置信区按 0 显示 -> 绿底, **绝不出现黑色散斑点**
  - **默认 --tex-alpha 0 = 纯色场**: 0% 面板为整块实心绿, 与论文 Fig.7/8 完全一致
    （这是用户最终确认的版本: 保持参考图观感 + 黑点全除）
  - 归一化卷积平滑: 把点状稀疏场读成连续条带 (--sigma/--minsup)
  - 每行下方 **横向 colorbar**, 右侧标注 "Exx/%" / "Eyy/%"
  - 荷载标签置于面板**下方**
  - **--steel-bar**: 叠钢棒(加载端) + Loading end/Steel bar 标注, 还原期刊试件样式
  - **--smooth** + bicubic 插值: 去方块糙感(底层网格仅数十格, 放大显块)

输入: 目录内每阶段一个 stage_{idx:05d}.npz, 键 exx/eyy/ux/uy + mask(bool) + xs/ys
      （由 dic_raw_to_field.py 生成, 或 VIC-2D/Ncorr 导出后按此格式存）

用法（默认=纯净无点, 推荐）:
  python dic_ref_grid.py --fields ./dic_fields \
      --stages 0,10,20,30,40,49 --labels "0,20%,40%,60%,80%,100%" \
      --rows exx,eyy --out ref_exx_eyy.png
  # 裁剪框默认取各阶段 mask 并集的包围盒; 也可显式 --crop x0,x1,y0,y1
  # 期刊试件样式(钢棒+标注): 加 --steel-bar
  # 去糙(更高像素, 默认 dpi=600, bicubic + 轻量高斯): 默认即开 --smooth 1.2
  # 可选: 叠一层极淡试件纹理(仍无清晰黑点), 需给出照片:
  #   --tex stage_00000_0.tiff --tex-alpha 0.13 --tex-blur 2.2 --tex-ds 4
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
try:
    from matplotlib.cm import ScalarMappable
except ImportError:
    from matplotlib.colors import ScalarMappable

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# diverging rainbow: 紫(-) -> 蓝 -> 青 -> 绿(0) -> 黄 -> 橙 -> 红(+)  —— 论文 Fig.7/8 实测
EXY = LinearSegmentedColormap.from_list("exy", [
    (0.00, "#7a1fa2"), (0.12, "#3f51d6"), (0.28, "#1f8fd8"), (0.42, "#19c0d6"),
    (0.50, "#22b14c"), (0.60, "#8fd12b"), (0.72, "#f2c21a"), (0.85, "#f28a1a"),
    (1.00, "#e0241f")])


def smooth_field(f, m, sigma, minsup):
    """归一化卷积: 用有效样本平滑插值, 支撑不足处 NaN (边界不溢出到无数据区)
    注意: 先 nan_to_num, 否则未测区 NaN 经高斯卷积会污染整张图 -> 全绿。"""
    f = np.nan_to_num(f.astype(np.float64), nan=0.0)
    w = m.astype(np.float64)
    fs = gaussian_filter(np.where(m, f, 0.0), sigma)
    ws = gaussian_filter(w, sigma)
    out = fs / np.maximum(ws, 1e-9)
    out[ws < minsup] = np.nan
    return out


def load_tex(path):
    """读试件照片 -> 灰度 float [0,1]; 支持 tif/tiff/png/jpg"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path).astype(np.float32)
    else:
        import matplotlib.image as mpimg
        a = mpimg.imread(path).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    m = float(a.max()) or 1.0
    return a / m


def nearest_fill(f):
    """NaN 区用最近有限值填充; 用于 --dilate 时把裂缝核心向外连续填充(限制在裂缝区内)。"""
    f = f.astype(np.float64).copy()
    nan = ~np.isfinite(f)
    if nan.any():
        _, (iy, ix) = distance_transform_edt(nan, return_indices=True)
        f[nan] = f[iy[nan], ix[nan]]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True, help="含 stage_XXXXX.npz 的目录")
    ap.add_argument("--stages", default="0,10,20,30,40,49", help="列所选阶段索引(逗号分隔)")
    ap.add_argument("--labels", default="0,20%,40%,60%,80%,100%", help="列标签(逗号分隔)")
    ap.add_argument("--rows", default="exx,eyy", help="行应变分量(逗号分隔): exx,eyy,ux,uy")
    ap.add_argument("--crop", default=None, help="裁剪框 x0,x1,y0,y1(默认=mask并集包围盒)")
    ap.add_argument("--sigma", type=float, default=4.0, help="平滑核(网格cell), 越大越连续")
    ap.add_argument("--minsup", type=float, default=0.05, help="归一化卷积最小支撑")
    ap.add_argument("--pct", type=float, default=90, help="每行色标上限百分位")
    ap.add_argument("--unit", type=str, default="%", help="应变单位: %% (乘100) 或 ue")
    ap.add_argument("--tex", default=None, help="可选: 试件照片(tif/png/jpg), 叠极淡纹理")
    ap.add_argument("--tex-alpha", type=float, default=0.0,
                    help="纹理不透明度; 0=纯净无点(默认, 与论文一致), ~0.13=极淡试件感")
    ap.add_argument("--tex-blur", type=float, default=2.2, help="纹理预模糊(柔化散斑点, 避免成点)")
    ap.add_argument("--tex-ds", type=int, default=4, help="纹理降采样步长(与场网格对齐)")
    ap.add_argument("--dilate", type=int, default=0,
                    help="从裂缝核心向外膨胀迭代次数; 把开裂区连成连续重色带, 别处仍绿底")
    ap.add_argument("--steel-bar", action="store_true", dest="steel_bar",
                    help="叠钢棒(加载端) + Loading end/Steel bar 标注(期刊试件样式, Fig.7/8)")
    ap.add_argument("--smooth", type=float, default=1.2,
                    help="去糙: 场切片后轻量高斯平滑(网格cell), 0=关; 配合 bicubic 插值去方块感")
    ap.add_argument("--out", default="ref_grid.png")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()

    stages = [int(x) for x in a.stages.split(",")]
    labels = [s.strip() for s in a.labels.split(",")]
    if len(labels) != len(stages):
        labels = [str(s) for s in stages]
    rows = [q.strip().lower() for q in a.rows.split(",")]
    rowlab = {"exx": "Exx", "eyy": "Eyy", "ux": "Ux", "uy": "Uy"}
    scale = 100.0 if a.unit == "%" else 1.0

    data = {s: np.load(os.path.join(a.fields, f"stage_{s:05d}.npz")) for s in stages}
    key = stages[-1]
    xs = np.asarray(data[key]["xs"]); ys = np.asarray(data[key]["ys"])
    ext = [xs[0], xs[-1], ys[0], ys[-1]]

    def get_mask(s):
        return data[s]["mask"].astype(bool)

    if a.crop:
        cc = [float(v) for v in a.crop.split(",")]
        crop = dict(x0=cc[0], x1=cc[1], y0=cc[2], y1=cc[3])
    else:
        u = None
        for s in stages:
            m = data[s]["mask"].astype(bool)
            u = m if u is None else (u | m)
        iy, ix = np.where(u.any(1))[0], np.where(u.any(0))[0]
        pad = 40
        crop = dict(x0=float(xs[ix[0]] - pad), x1=float(xs[ix[-1]] + pad),
                    y0=float(ys[iy[0]] - pad), y1=float(ys[iy[-1]] + pad))

    def rowscale(q):
        v = []
        for s in stages:
            o = smooth_field(data[s][q], get_mask(s), a.sigma, a.minsup)
            vv = o[np.isfinite(o)]
            if vv.size:
                v.append(np.abs(vv))
        return float(np.percentile(np.concatenate(v), a.pct) * scale) if v else 0.1

    def field_pct(q, s):
        m = get_mask(s)                                       # 原始实测掩膜(权重/边界)
        real = smooth_field(data[s][q], m, a.sigma, a.minsup) * scale   # 实测(归一化卷积)
        if a.dilate > 0:
            # 从裂缝"核心"(已着色的实测点)向外膨胀几格, 仅在裂缝区内用最近邻连续填充,
            # 区外一律 -> 0(绿底)。=> 重颜色只在开裂处, 别处浅绿。
            core = np.isfinite(real)
            gate = binary_dilation(core, iterations=a.dilate)
            filled = nearest_fill(real)                       # 裂缝区内连续填充
            real = np.where(gate, filled, np.nan)
        real[~np.isfinite(real)] = 0.0     # 未测区 -> 0 -> 绿底(去黑点)
        return real

    # 可选纹理: 先读入并预模糊 -> 柔化成"底纹"而非清晰散斑点
    tex0 = None
    if a.tex and a.tex_alpha > 0:
        tex0 = load_tex(a.tex)[::a.tex_ds, ::a.tex_ds]
        if a.tex_blur > 0:
            tex0 = gaussian_filter(tex0, a.tex_blur)

    nr = len(rows)
    fig = plt.figure(figsize=(13.2, 3.4 * nr + 0.4))
    gs = fig.add_gridspec(2 * nr, len(stages),
                          height_ratios=([1, 0.05] * nr),
                          hspace=0.40, wspace=0.06,
                          top=0.93, bottom=0.055, left=0.05, right=0.95)
    for ri, q in enumerate(rows):
        vm = rowscale(q)
        for c, s in enumerate(stages):
            ax = fig.add_subplot(gs[2 * ri, c])
            fld = field_pct(q, s)
            if a.smooth > 0:
                fld = gaussian_filter(fld, a.smooth)   # 去方块糙感
            ax.imshow(fld, extent=ext, origin="lower", cmap=EXY,
                      vmin=-vm, vmax=vm, aspect="equal", interpolation="bicubic")
            if tex0 is not None:
                ax.imshow(tex0, cmap="gray",
                          extent=[0, tex0.shape[1], 0, tex0.shape[0]],
                          origin="lower", aspect="equal", alpha=a.tex_alpha)
            ax.set_xlim(crop["x0"], crop["x1"]); ax.set_ylim(crop["y0"], crop["y1"])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xlabel(labels[c], fontsize=11, labelpad=6)
            if a.steel_bar:
                # 钢棒(加载端): 上下各一段, 居中
                ax.add_patch(Rectangle((0.444, 0.905), 0.112, 0.095, transform=ax.transAxes,
                                       clip_on=False, fc="0.70", ec="0.35", lw=0.5))
                ax.add_patch(Rectangle((0.444, 0.000), 0.112, 0.095, transform=ax.transAxes,
                                       clip_on=False, fc="0.70", ec="0.35", lw=0.5))
                if c == 0:
                    ax.plot([0.0, 0.40], [-0.11, -0.11], transform=ax.transAxes,
                            color="0.2", lw=0.7, clip_on=False)
                    ax.text(0.0, -0.15, "Loading end", transform=ax.transAxes,
                            fontsize=8.5, ha="left", va="top")
                    ax.text(0.60, -0.15, "Steel bar", transform=ax.transAxes,
                            fontsize=8.5, ha="left", va="top")
        cax = fig.add_subplot(gs[2 * ri + 1, :])
        cb = fig.colorbar(ScalarMappable(Normalize(-vm, vm), EXY), cax=cax,
                          orientation="horizontal")
        cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.6)
        cax.text(1.006, 0.5, f"{rowlab.get(q,q)}/{a.unit}", transform=cax.transAxes,
                 va="center", ha="left", fontsize=12)
    footer = "荷载进程 × 全场应变（重色仅落在开裂/损伤区，其余低应变区为绿底；仅损伤区为实测值）"
    fig.text(0.5, 0.012, footer, ha="center", fontsize=9.5)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {a.out}")


if __name__ == "__main__":
    main()
