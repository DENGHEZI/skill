# -*- coding: utf-8 -*-
"""
DIC 全场云图（论文 Fig.12 风格 + 土木结构全场位移/应变模式）

两大场景（"训练"来源）:
  A) 界面直剪/粘结滑移（ECC-煤矸石混凝土、UHPC-NC 复合）:
       应变场 exx 用 VIC rainbow（深蓝远场 -> 青绿低 -> 黄 -> 红集中 -> 深红贯穿）
  B) 土木结构 DIC（损伤识别 / 振动 / 抗震 / 健康监测 SHM）:
       位移场 Ux/Uy 为"有符号量"，用 VIC 双向(diverging)色标 蓝(负) -> 白(0) -> 红(正)
       典型量: Ux, Uy（位移场）、εxx, εyy, γxy（应变场）、开裂位置由高应变带/位移间断定位

用法:
  # 1) 界面粘结合成演示(默认, 已按论文规律训练)
  python dic_strain_field.py --specimen CE-2-20-0-r --stages 0.20,0.65,1.00 --out fig12.png

  # 2) 真实 DIC 应变导出数据 (CSV: x,y,exx)
  python dic_strain_field.py --csv my_dic.csv --out real.png

  # 3) 土木结构: 三点弯曲梁 位移场 Ux（双向色标 + 开裂叠加）
  python dic_strain_field.py --sample beam --quantity ux --stages 0.30,0.70,1.00 --out beam_ux.png

  # 4) 土木结构: 2x2 全场网格 (Ux, Uy, εxx, γxy) —— 损伤识别典型成图
  python dic_strain_field.py --sample beam --grid ux,uy,exx,exy --out beam_grid.png

  # 5) 用检测出的开裂折线叠加 (CSV: x,y 折线)
  python dic_strain_field.py --sample beam --quantity exx --crack crack.csv --out beam_crack.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# --- 色标库 ---------------------------------------------------------------
# VIC rainbow: 深蓝(远场负) -> 青/绿(低) -> 黄 -> 红(集中) -> 深红(贯穿)  [应变/幅值量]
VIC_RAINBOW = LinearSegmentedColormap.from_list("vic", [
    (0.00, "#2c00aa"), (0.15, "#2255cc"), (0.30, "#00a2c8"),
    (0.45, "#3cb46e"), (0.60, "#c8d84a"), (0.75, "#f0a030"),
    (0.88, "#e03424"), (1.00, "#7a0a12"),
])
# VIC diverging: 蓝(负位移) -> 白(0) -> 红(正位移)  [有符号位移场 Ux/Uy]
VIC_DIVERGING = LinearSegmentedColormap.from_list("vicdiv", [
    (0.00, "#08306b"), (0.25, "#2171b5"), (0.48, "#9ecae1"),
    (0.50, "#f7f7f7"), (0.52, "#fdae6b"), (0.75, "#d94701"), (1.00, "#67000d"),
])


# --- 界面键槽几何 ---------------------------------------------------------
def keyed_interface_path(n_keys=3, key_w=0.10, key_d=0.12, H=1.0, x_mid=0.5):
    pts = [(x_mid, H)]
    seg = H / (n_keys + 1)
    y = H - seg
    for i in range(n_keys):
        pts += [(x_mid, y), (x_mid + key_w, y), (x_mid + key_w, y - key_d),
                (x_mid, y - key_d), (x_mid, y - seg)]
        y -= seg
    pts += [(x_mid, 0.0)]
    return np.array(pts)


# --- 场景A: 界面粘结应变场合成（按论文规律训练） -------------------------
def synthesize_stage(path, W=1.0, H=1.0, nx=300, ny=300, stage=0.65, mode="ductile", seed=7):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, W, nx); ys = np.linspace(0, H, ny)
    X, Y = np.meshgrid(xs, ys)
    f = {"mild": 0.25, "mid": 0.55, "peak": 1.0}.get(stage, stage) if isinstance(stage, str) else float(stage)
    exx = -1.2e-4 + rng.normal(0, 0.8e-4, X.shape)
    d = np.full(X.shape, 1e9)
    for i in range(len(path) - 1):
        (x1, y1), (x2, y2) = path[i], path[i + 1]
        t = np.clip(((X - x1) * (x2 - x1) + (Y - y1) * (y2 - y1)) / max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(X - (x1 + t * (x2 - x1)), Y - (y1 + t * (y2 - y1))))
    band_w = 0.038 - 0.014 * f
    amp = (1.5 + 8.5 * f) * (1.5 if mode == "brittle" else 1.0)
    exx += amp * np.exp(-(d / band_w) ** 2)
    if f > 0.8:
        crack = (d < band_w * 0.8) & (rng.random(X.shape) < 0.18 + 0.5 * f)
        exx[crack] *= 0.35
        hot = d < band_w * 0.35
        exx[hot] += 3.0 * f
    return xs, ys, exx


# --- 场景B: 土木结构三点弯曲梁全场（位移/应变）合成 -----------------------
def synthesize_beam(quantity="ux", W=1.0, H=1.0, nx=320, ny=320,
                    stage=0.7, seed=11):
    """三点弯曲梁 DIC 全场合成（损伤识别/抗震常用）。
    stage: 加载水平 0~1, 控制跨中裂缝高度 crack_frac 与量值。
    返回 (xs, ys, field), crack_path(折线点列, 随 stage 生长)
    """
    rng = np.random.default_rng(seed + int(stage * 100))
    xs = np.linspace(0, W, nx); ys = np.linspace(0, H, ny)
    X, Y = np.meshgrid(xs, ys)
    xc = 0.5  # 跨中
    crack_frac = min(0.92, 0.15 + 0.95 * stage)  # 裂缝从底部向上生长
    crack_top = H * (1 - crack_frac)
    mag = 1.0 + 4.0 * stage

    if quantity in ("ux", "uy"):
        # 有符号位移场: 以弯曲变形为主, 跨中剪切/开裂处位移间断
        # 竖向位移 Uy: 向下挠度, 跨中最大(负), 两端为0 -> 用抛物线型 + 开裂处陡降
        if quantity == "uy":
            field = -mag * 2.2e-3 * np.sin(np.pi * X / W) * (1 - (Y / H) ** 0.35)
            # 开裂处位移突变(上/下盘错动)
            d_crack = np.abs(X - xc)
            field += -mag * 1.6e-3 * np.exp(-(d_crack / 0.05) ** 2) * (Y < crack_top)
        else:  # Ux: 水平位移, 顶压底拉, 关于跨中反对称
            field = mag * 1.4e-3 * (0.5 - X) * np.cos(np.pi * Y / (2 * H))
            # 裂缝附近水平错动(张开) -> 跨中附近高梯度
            field += mag * 1.1e-3 * np.exp(-((X - xc) / 0.04) ** 2) * (Y < crack_top) * (0.5 - X)
        field += rng.normal(0, 4e-5, X.shape)
        field = field - field.mean()
    else:
        # 应变场: εxx 受拉集中在跨中底部(裂缝尖端正应变最大)
        if quantity == "exx":
            field = -mag * 5e-4 * (Y - H * 0.5) * 2   # 顶部压底部拉(线性)
            d_crack = np.abs(X - xc)
            ten = np.exp(-(d_crack / 0.06) ** 2) * np.exp(-((Y - 0) / (H * crack_frac)) ** 2)
            field += mag * 1.2e-2 * ten
            field += rng.normal(0, 1.5e-4, X.shape)
        elif quantity == "exy" or quantity == "gxy" or quantity == "gamma":
            # 剪应变 γxy: 跨中剪切带(垂直), 随裂缝发展增强
            d_crack = np.abs(X - xc)
            field = mag * 6e-3 * np.exp(-(d_crack / 0.05) ** 2) * (0.3 + 0.7 * (1 - Y / H))
            field += rng.normal(0, 2e-4, X.shape)
        else:
            field = rng.normal(0, 1e-4, X.shape)
    crack_path = np.array([(xc, 0.0), (xc, crack_top)])
    return xs, ys, field, crack_path


def fmt_e(v):
    return f"{v:.1e}".replace("e+0", "e+").replace("e-0", "e-")


# --- 土木梁 εyy(竖向应变) 合成（与 εxx 对称补充, 用于双行云图） ----------
def synthesize_beam_eyy(W=1.0, H=1.0, nx=320, ny=320, stage=0.7, seed=13):
    rng = np.random.default_rng(seed + int(stage * 100))
    xs = np.linspace(0, W, nx); ys = np.linspace(0, H, ny)
    X, Y = np.meshgrid(xs, ys)
    xc = 0.5
    crack_frac = min(0.92, 0.15 + 0.95 * stage)
    crack_top = H * (1 - crack_frac)
    mag = 1.0 + 4.0 * stage
    # 竖向应变: 弯曲主导(顶压底拉) + 跨中裂缝处竖向集中
    field = mag * 4e-4 * np.cos(np.pi * X / W) * (1 - (Y / H) ** 0.3)
    d_crack = np.abs(X - xc)
    ten = np.exp(-(d_crack / 0.06) ** 2) * np.exp(-((Y - crack_top) / (H * 0.25)) ** 2)
    field += mag * 1.0e-2 * ten
    field += rng.normal(0, 1.5e-4, X.shape)
    crack_path = np.array([(xc, 0.0), (xc, crack_top)])
    return xs, ys, field, crack_path


def draw_strain_grid(row_labels, col_labels, fields, masks=None,
                     outfile="grid.png", cmap=VIC_RAINBOW, symmetric=False,
                     W=1.0, H=1.0, outline=None, outline_mask=None, title=None):
    """荷载分级(列) × 应变分量(行) 双行/多行干净云图（论文 Fig.7/8 风格）。

    fields: dict (r,c)->2D ndarray, 无效区用 np.nan 表示
    masks : dict (r,c)->bool 无效(可选, 不传则按 isnan)
    outline      : (N,2) 折线, 叠加为黑实线(试件/夹具轮廓或裂缝)
    outline_mask : 2D bool 数组(与场同网格), 取其 0.5 等值线作浅灰试件边界
    - 每个 cell 用 cmap 上色, NaN -> 白(去散斑/黑点)
    - 每行统一 clim(共享色标), 右侧每行独立 colorbar(e 计数法)
    - 顶部=荷载级别标签, 左侧=应变分量标签
    """
    nrows = len(row_labels); ncols = len(col_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols + 1.4, 3.0 * nrows),
                             squeeze=False)
    cmap = cmap.copy(); cmap.set_bad("white")           # 无效区白色, 绝不出现黑点
    row_vmax = []
    for r in range(nrows):
        mx = 1e-6
        for c in range(ncols):
            f = fields[(r, c)]
            m = (masks[(r, c)] if masks else np.isfinite(f))
            vals = np.abs(f[m]) if m.any() else np.array([0.0])
            if vals.size:
                # 用 98 百分位作色标上限, 裁剪离群(裂缝尖/噪声)避免发白
                mx = max(mx, np.percentile(vals, 98))
        row_vmax.append(max(mx, 1e-6))
    for r in range(nrows):
        vmax = row_vmax[r]; vmin = -vmax if symmetric else -0.45 * vmax
        for c in range(ncols):
            ax = axes[r][c]
            f = fields[(r, c)]
            im = ax.imshow(f, extent=[0, W, 0, H], origin="lower", cmap=cmap,
                           aspect="equal", interpolation="bilinear",
                           vmin=vmin, vmax=vmax)
            if outline is not None:
                ax.plot(outline[:, 0], outline[:, 1], "k-", lw=1.4)
            if outline_mask is not None and outline_mask.shape == f.shape:
                ax.contour(outline_mask.astype(float), levels=[0.5],
                           colors="gray", linewidths=1.0, alpha=0.55,
                           extent=[0, W, 0, H])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == ncols - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
                cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_e(v)))
                cb.ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10, pad=4)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=11, labelpad=8)
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 荷载×分量网格 -> {outfile}")


def draw_panels(stages, titles, outfile, W=1.0, H=1.0,
                cmap=VIC_RAINBOW, symmetric=False, interface=None,
                crack=None, note=None, single_colorbar=True):
    """通用全场云图绘制。
    stages: list of (xs, ys, field)
    symmetric=True -> 双向色标(vmin=-vmax); 否则应变式 vmin=-0.45*vmax
    crack: (N,2) 折线(开裂位置叠加, 红色虚线)
    interface: (N,2) 折线(键槽/结合面叠加, 黑色实线)
    """
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 5.6))
    if n == 1:
        axes = [axes]
    vmax = max(max(np.abs(s[2]).max() if symmetric else s[2].max() for s in stages), 1e-6)
    vmin = -vmax if symmetric else -0.45 * vmax
    imgs = []
    for ax, (xs, ys, field) in zip(axes, stages):
        im = ax.imshow(field, extent=[0, W, 0, H], origin="lower", cmap=cmap,
                       aspect="equal", interpolation="bilinear")
        if interface is not None:
            ax.plot(interface[:, 0], interface[:, 1], "k-", lw=1.6)
        if crack is not None:
            ax.plot(crack[:, 0], crack[:, 1], "r--", lw=2.2, alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        imgs.append(im)
    for ax, im, t in zip(axes, imgs, titles):
        im.set_clim(vmin, vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.95)
        cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_e(v)))
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel(t, fontsize=11, labelpad=8)
    if note:
        axes[-1].annotate(note, xy=(0.72, 0.62), xycoords="axes fraction", fontsize=9,
                          color="white", ha="center",
                          bbox=dict(boxstyle="round,pad=0.3", fc="gray", alpha=0.55, lw=0))
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 全场云图 -> {outfile}")


def load_csv(fp):
    """读 DIC 导出 CSV（表头 x,y,量值 三列，按规则网格排列）。
    将一行行散点透视(pivot)成 2D 网格供 imshow 使用。
    """
    raw = np.genfromtxt(fp, delimiter=",", names=True, deletechars="")
    names = list(raw.dtype.names)
    val_col = [c for c in names if c.lower() not in ("x", "y")][0]
    xs = raw["x"].astype(float); ys = raw["y"].astype(float); e = raw[val_col].astype(float)
    ux = np.unique(xs); uy = np.unique(ys)
    nx, ny = len(ux), len(uy)
    if nx * ny != len(xs):
        # 非规则网格: 退化为最近规则网格重采样(简单线性索引)
        X, Y = np.meshgrid(np.linspace(xs.min(), xs.max(), nx),
                           np.linspace(ys.min(), ys.max(), ny))
        E = e.reshape(ny, nx) if e.shape == (ny * nx,) else e[:ny * nx].reshape(ny, nx)
        return X, Y, E
    E = np.full((ny, nx), np.nan)
    xmap = {v: i for i, v in enumerate(ux)}
    ymap = {v: i for i, v in enumerate(uy)}
    for xi, yi, val in zip(xs, ys, e):
        E[ymap[yi], xmap[xi]] = val
    X, Y = np.meshgrid(ux, uy)
    # 应变若以微应变给出(>1e-2)则折算为无量纲
    if np.abs(E[~np.isnan(E)]).max() > 1e-2:
        E = E / 1e6
    return X, Y, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--npz")
    ap.add_argument("--specimen", default="CE-2-20-0-r")
    ap.add_argument("--sample", default=None, choices=[None, "beam"],
                    help="合成样例: beam=三点弯曲梁(土木结构 DIC)")
    ap.add_argument("--stages", default="0.30,0.70,1.00",
                    help="加载水平, 逗号分隔; beam 默认 0.30,0.70,1.00")
    ap.add_argument("--quantity", default="exx",
                    choices=["exx", "ux", "uy", "exy", "gxy", "gamma"],
                    help="beam 样例的场量: exx应变 / ux,uy位移 / exy剪应变")
    ap.add_argument("--grid", default=None,
                    help="2x2 全场网格, 如 ux,uy,exx,exy（beam 样例专用）")
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--mode", default="ductile", choices=["ductile", "brittle"])
    ap.add_argument("--crack", default=None, help="开裂折线 CSV(x,y), 叠加为红虚线")
    ap.add_argument("--out", default="dic_strain_field.png")
    # ---- 新增: 荷载分级×应变分量 双行干净网格 ----
    ap.add_argument("--grid-style", action="store_true",
                    help="生成版式示意: 荷载分级(列)×Exx/Eyy(行) 干净云图(合成, 用于对齐论文样式)")
    ap.add_argument("--real-grid", default=None,
                    help="真实 DIC 导出目录: 内含 stage_XXXXX.npz(键 exx,eyy,mask,xs,ys)")
    ap.add_argument("--levels", default="0,12,25,37,49",
                    help="网格所选荷载阶段(列), 逗号分隔的 stage 索引")
    ap.add_argument("--rows", default="exx,eyy",
                    help="网格行应变分量: exx,eyy (可多选, 逗号分隔)")
    ap.add_argument("--col-labels", default=None,
                    help="荷载列标签(可选, 逗号分隔), 默认用 stage 索引; 例 0%%Pu,25%%Pu,...")
    a = ap.parse_args()

    # ---- 荷载×分量 双行网格 ----
    if a.grid_style or a.real_grid:
        rows = [q.strip().lower() for q in a.rows.split(",")]
        row_labels = {"exx": "εxx", "eyy": "εyy", "exy": "γxy", "ux": "Ux", "uy": "Uy"}
        rowlabels = [row_labels.get(q, q) for q in rows]
        levels = [int(x) for x in a.levels.split(",")]
        if a.col_labels:
            col_labels = [s.strip() for s in a.col_labels.split(",")]
            if len(col_labels) != len(levels):
                col_labels = [f"{lv}" for lv in levels]
        else:
            col_labels = [f"{lv}" for lv in levels]
        fields, masks = {}, {}
        union_mask = None
        if a.grid_style:
            for ci, lv in enumerate(levels):
                s = lv / max(levels) if max(levels) > 0 else 0.0
                for ri, q in enumerate(rows):
                    if q == "exx":
                        xs, ys, f, _ = synthesize_beam("exx", stage=s)
                    elif q == "eyy":
                        xs, ys, f, _ = synthesize_beam_eyy(stage=s)
                    elif q == "exy":
                        xs, ys, f, _ = synthesize_beam("exy", stage=s)
                    else:
                        xs, ys, f, _ = synthesize_beam("ux", stage=s)
                    # 合成场直接全量显示(无散斑黑点); 不人为造空白
                    f = np.where(np.isfinite(f), f, np.nan)
                    fields[(ri, ci)] = f
                    masks[(ri, ci)] = np.isfinite(f)
            draw_strain_grid(rowlabels, col_labels, fields, masks, a.out,
                            cmap=(VIC_DIVERGING if all(r in ("ux","uy") for r in rows) else VIC_RAINBOW),
                            symmetric=all(r in ("ux","uy") for r in rows),
                            title="荷载分级 × 应变分量（版式示意，合成）")
        else:
            base = a.real_grid
            for ci, lv in enumerate(levels):
                z = np.load(os.path.join(base, f"stage_{lv:05d}.npz"))
                xs, ys = z["xs"], z["ys"]
                W = float(xs.max()); H = float(ys.max())
                for ri, q in enumerate(rows):
                    f = z[q].astype(float).copy()
                    m = z["mask"].astype(bool) if "mask" in z.files else np.isfinite(f)
                    f[~m] = np.nan
                    fields[(ri, ci)] = f
                    masks[(ri, ci)] = m
                    union_mask = m if union_mask is None else (union_mask | m)
            draw_strain_grid(rowlabels, col_labels, fields, masks, a.out,
                            cmap=(VIC_DIVERGING if all(r in ("ux","uy") for r in rows) else VIC_RAINBOW),
                            symmetric=all(r in ("ux","uy") for r in rows),
                            W=W, H=H, outline_mask=union_mask,
                            title="荷载分级 × 应变分量（真实 DIC 导出）")
        return

    # ---- 网格模式 (beam) ----
    if a.grid and a.sample == "beam":
        quants = [q.strip() for q in a.grid.split(",")]
        labels = {"ux": "Ux", "uy": "Uy", "exx": "εxx", "exy": "γxy", "gxy": "γxy", "gamma": "γxy"}
        # 取最后一个 stage 作为代表
        stage = float(a.stages.split(",")[-1])
        panels, titles, cracks = [], [], []
        for q in quants:
            xs, ys, f, cp = synthesize_beam(q, stage=stage)
            panels.append((xs, ys, f)); titles.append(labels.get(q, q))
            cracks.append(cp)
        # 统一双向/应变色标
        sym = all(q in ("ux", "uy") for q in quants)
        cmap = VIC_DIVERGING if sym else VIC_RAINBOW
        draw_grid(panels, titles, a.out, cmap=cmap, symmetric=sym,
                  crack=(cracks[0] if a.crack is None else np.genfromtxt(a.crack, delimiter=",")))
        return

    # ---- 单量(可多阶段) ----
    if a.sample == "beam":
        labels = {"ux": "Ux", "uy": "Uy", "exx": "εxx", "exy": "γxy", "gxy": "γxy", "gamma": "γxy"}
        sym = a.quantity in ("ux", "uy")
        cmap = VIC_DIVERGING if sym else VIC_RAINBOW
        stages_lv = [s for s in a.stages.split(",")]
        panels, titles, crack_final = [], [], None
        for i, lv in enumerate(stages_lv):
            xs, ys, f, cp = synthesize_beam(a.quantity, stage=float(lv), seed=11 + i)
            panels.append((xs, ys, f)); titles.append(f"{lv} P")
            crack_final = cp
        crack = np.genfromtxt(a.crack, delimiter=",") if a.crack else crack_final
        draw_panels(panels, titles, a.out, cmap=cmap, symmetric=sym, crack=crack,
                    note="跨中开裂\n位移/应变集中" if sym is False else None)
        return

    # ---- 场景A: 界面粘结 ----
    path = keyed_interface_path(n_keys=a.keys)
    labels = [f"{s} $P_u$" for s in a.stages.split(",")]
    kinds = ["mild", "mid", "peak"][: len(labels)] if len(labels) >= 3 else ["mid"] * len(labels)
    if a.csv:
        x, y, e = load_csv(a.csv)
        draw_panels([(x, y, e)], [labels[-1]], a.out, interface=path)
    elif a.npz:
        z = np.load(a.npz)
        keys = [k for k in z.files if k.startswith("exx")]
        xs, ys = z["x"], z["y"]
        stages = [(xs, ys, z[k]) for k in keys]
        draw_panels(stages, labels[: len(stages)], a.out, interface=path)
    else:
        stages = [synthesize_stage(path, stage=k, mode=a.mode, seed=7 + i)
                  for i, k in enumerate(kinds)]
        draw_panels(stages, labels, a.out, interface=path,
                    note="部分开裂\n应变释放")


def draw_grid(panels, titles, outfile, W=1.0, H=1.0, cmap=VIC_RAINBOW,
              symmetric=False, crack=None):
    """2x2 全场网格（土木 DIC 损伤识别典型成图）"""
    n = len(panels)
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.0 * nr))
    axes = np.atleast_1d(axes).ravel()
    vmax = max(max(np.abs(p[2]).max() if symmetric else p[2].max() for p in panels), 1e-6)
    vmin = -vmax if symmetric else -0.45 * vmax
    for ax, (xs, ys, field), t in zip(axes, panels, titles):
        im = ax.imshow(field, extent=[0, W, 0, H], origin="lower", cmap=cmap,
                       aspect="equal", interpolation="bilinear")
        if crack is not None:
            ax.plot(crack[:, 0], crack[:, 1], "r--", lw=2.2, alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, fontsize=11, pad=4)
        im.set_clim(vmin, vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
        cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_e(v)))
        cb.ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 全场网格 -> {outfile}")


if __name__ == "__main__":
    main()
