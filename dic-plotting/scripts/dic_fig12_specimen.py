# -*- coding: utf-8 -*-
"""dic_fig12_specimen.py —— 直剪界面 Fig.12 风格：多试件(行) × 多时相(列) 应变云图

对标论文：
  赵卫平等《基于数字图像相关技术与声发射的 ECC-煤矸石混凝土界面黏结性能试验研究》
  《建筑结构学报》2025, 46(5) —— 图12「x 向应变场变化云图」(Fig.12 nephogram)

版式特征（逐项反推真实论文图）:
  - **行 = 试件**（如 10 个不同键槽参数的 Z 形直剪试件），每行 3 个时相面板
      · 未配受剪钢筋试件：0.90Pu / 0.95Pu / 1.00Pu
      · 配受剪钢筋试件：初裂 / 中间时刻 / 1.00Pu（峰值）
  - **每个面板右侧竖一条彩虹 colorbar**（不是每行一条！），刻度用 e 计数法（如 5.3e-1）
  - **黑色阶梯折线** = 试件/键槽界面轮廓（矩形 + 键齿凸起），对应 Z 形试件几何
  - 面板下方：**荷载/位移标签**（0.90 P_u）；组下方：**试件编号标题**（(a) 试件CE-2-2-0-0）
  - 底图：绿色=低应变/0，红黄=正应变集中（界面剪切带），蓝紫=负应变（部分开裂应变释放）
  - 可选：**青框 + 箭头 + 「部分开裂应变释放」注记**（标注应变释放区）

力学规律（"符合破坏的力学规律"）:
  - 加载初期应变分布均匀（全场近绿）
  - 进入裂缝发展阶段：结合界面附近出现**应变集中带**（红黄竖向条带）
  - 接近/达到峰值：主裂缝萌生贯通，应变集中带**更窄更强、贯穿界面**（贯穿条带）
  - 部分区域开裂后**应变释放**（转蓝紫）→ 可选注记框标出
  - 未配筋试件（脆性）：3 个时相都集中在峰值附近，破坏瞬间完成（0.90→1.00 急剧）
  - 配筋试件（延性）：应变带随荷载逐步增强、缝侧出现多条次生带

输入（两种，任选）:
  A) --manifest：一个 JSON/CSV，描述每个试件每时相用哪个场文件与标签（最灵活，推荐）
  B) --matrix：约定式目录 + 命名规则（见下）

用法示例：
  # 方式A: 用 manifest.json 精确控制每个面板
  python dic_fig12_specimen.py --manifest fig12.json --out fig12.png

  # 方式B: 直接给真实 DIC 场目录(单试件多阶段作 1 行), 快速试跑
  python dic_fig12_specimen.py --fields ./dic_fields \
      --stages 44,47,49 --labels "0.90 P_u,0.95 P_u,1.00 P_u" \
      --row-title "(a) 试件CE-2-2-0-0" --out fig12_demo.png

  # 叠加界面轮廓(黑色阶梯线, 数据坐标): --interface x0,y0,x1,y1[;key=...]
  # 青框+注记: --annotate "框中心x,框中心y,宽,高,文字"
"""
import os, json, argparse
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
try:
    from matplotlib.cm import ScalarMappable
except ImportError:
    from matplotlib.colors import ScalarMappable

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# diverging rainbow —— 严格按参考图(论文 Fig.12 实拍)逐点取色重建。
# 参考图色标条(y=48..404, x=613)实测自顶(vmax)→底(vmin)的 14 级:
#   #d09595 #d89c8e #e4e19a #afca84 #a7ce92 #abd1af #a0cdcc #95a5c5 #979dc0 #b192c4 ...
# 面板内实测主色(出现频率最高):
#   绿底 #58b058 (88,176,88) / #80b050 (128,176,80);  紫区 #8040a0 (128,64,160)
#   → 结论: 参考图是**中高饱和的 rainbow**(蓝-青-绿-黄-红), 不是低饱和灰绿。
# 中心 0 值 ≈ 草绿 #58b058; 正向(拉应变)走 黄→橙→红; 负向(压应变)走 青→蓝→紫。
EXY = LinearSegmentedColormap.from_list("exy_ref", [
    (0.000, "#6a3aa0"),   # vmin 深紫 (106,58,160)
    (0.060, "#7b48b8"),   # 紫
    (0.140, "#8a6dc8"),   # 紫罗兰
    (0.220, "#8f8fd0"),   # 蓝紫 (143,143,208)
    (0.300, "#88a8d8"),   # 蓝 (136,168,216)
    (0.380, "#8cc8d0"),   # 青 (140,200,208)
    (0.440, "#a0d8b8"),   # 青绿 (160,216,184)
    (0.500, "#58b058"),   # 0 值 = 草绿 (88,176,88)  ← 实测主色
    (0.560, "#90c860"),   # 黄绿
    (0.640, "#d8e070"),   # 黄 (216,224,112)
    (0.720, "#e8c060"),   # 橙黄
    (0.820, "#e09060"),   # 橙
    (0.920, "#d07068"),   # 橙红
    (1.000, "#c04848")])  # vmax 红 (192,72,72)
# 更亮/更饱和版: 供 --cmap vivid 使用
EXY_VIVID = LinearSegmentedColormap.from_list("exy_vivid", [
    (0.00, "#4a1a9e"), (0.12, "#3355d8"), (0.26, "#1f9fd0"), (0.38, "#20c8c0"),
    (0.50, "#2fbc4a"), (0.62, "#9ed12b"), (0.74, "#f2c81a"), (0.86, "#f2841a"),
    (1.00, "#e0241f")])
# 参考图实测主色常量(供 demo 合成场使用)
REF_GREEN = "#58b058"   # 面板绿底
REF_PURPLE = "#8040a0"  # 压应变紫区
REF_RED = "#c04848"     # 拉应变红带


# ----------------------------------------------------------------------------- helpers
def smooth_field(f, m, sigma, minsup):
    """归一化卷积平滑: 只对有效点加权, 支撑不足处 NaN。"""
    f = np.nan_to_num(f.astype(np.float64), nan=0.0)
    w = m.astype(np.float64)
    fs = gaussian_filter(np.where(m, f, 0.0), sigma)
    ws = gaussian_filter(w, sigma)
    out = fs / np.maximum(ws, 1e-9)
    out[ws < minsup] = np.nan
    return out


def nearest_fill(f):
    f = f.astype(np.float64).copy()
    nan = ~np.isfinite(f)
    if nan.any():
        _, (iy, ix) = distance_transform_edt(nan, return_indices=True)
        f[nan] = f[iy[nan], ix[nan]]
    return f


def load_npz_field(path, q, sigma, minsup, dilate, scale):
    """读 stage_*.npz -> 完成平滑/膨胀/换算的二维场(未测区=0 -> 绿底)。"""
    d = np.load(path)
    if q not in d:
        raise KeyError(f"{path} 无键 {q}, 可选 {list(d.keys())}")
    f = np.asarray(d[q], dtype=np.float64)
    m = np.asarray(d["mask"]).astype(bool) if "mask" in d else np.isfinite(f)
    real = smooth_field(f, m, sigma, minsup) * scale
    if dilate > 0:
        core = np.isfinite(real)
        gate = binary_dilation(core, iterations=dilate)
        real = np.where(gate, nearest_fill(real), np.nan)
    real[~np.isfinite(real)] = 0.0
    return real


def load_csv_field(path, q, scale):
    """读三列 CSV (x,y,<分量>) -> 规则网格 + 掩膜。"""
    import csv as _csv
    xs, ys, vs = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = _csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in rd.fieldnames}
        kx = cols.get("x"); ky = cols.get("y"); kq = cols.get(q)
        if not (kx and ky and kq):
            raise KeyError(f"{path} 需含 x,y,{q} 列, 实际 {rd.fieldnames}")
        for r in rd:
            xs.append(float(r[kx])); ys.append(float(r[ky])); vs.append(float(r[kq]))
    xs = np.asarray(xs); ys = np.asarray(ys); vs = np.asarray(vs)
    ux = np.unique(xs); uy = np.unique(ys)
    g = np.full((len(uy), len(ux)), np.nan)
    ix = np.searchsorted(ux, xs); iy = np.searchsorted(uy, ys)
    g[iy, ix] = vs * scale
    m = np.isfinite(g)
    return g, m, [ux[0], ux[-1], uy[0], uy[-1]]


def draw_stepped_interface(ax, pts, color="k", lw=1.4, closed=False):
    """画黑色阶梯轮廓线(试件/键槽界面)。pts=[[x,y],...] 数据坐标。"""
    if not pts:
        return
    p = np.asarray(pts, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, solid_joinstyle="miter", zorder=6)


def parse_interface(s):
    """'x,y;x,y;...' -> (n,2) float 数组; 'auto'/None/'' -> None(交给 demo 的 Z 形)。

    ⚠️ 真实场(--fields/--manifest)模式原先**没有接线**这个参数(只有 demo 分支设了 xint)，
       导致真实数据出图时**画不出**那条黑色界面/裂缝轮廓线 —— 已补齐。
    """
    if not s or s.strip().lower() == "auto":
        return None
    pts = []
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        xy = seg.replace(",", " ").split()
        if len(xy) < 2:
            continue
        pts.append([float(xy[0]), float(xy[1])])
    return np.asarray(pts, dtype=float) if pts else None


def auto_interface(x0, y0, x1, y1, keys=2, depth=0.16, inward=True):
    """按试件矩形生成 Z 形键槽阶梯轮廓(左边界内凹 keys 个键齿)。
    仅作默认示意; 真实试件请用 --interface 显式给点。"""
    pts = [(x1, y0)]
    W = x1 - x0; H = y1 - y0
    d = depth * W
    seg = H / (2 * keys + 1)
    yy = y0 + seg
    xb = x0
    pts.append((xb, y0))
    for k in range(keys):
        pts.append((xb, yy)); pts.append((xb + d, yy))
        pts.append((xb + d, yy + seg)); pts.append((xb, yy + seg))
        yy += 2 * seg
    pts.append((xb, y1))
    pts.append((x1, y1))
    return pts


def fmt_e(v):
    """e 计数法刻度: 0.53 -> '5.3e-1'; 近零 -> '0'"""
    if abs(v) < 1e-12:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    mant = v / (10 ** e)
    return f"{mant:.1f}e{e:+d}".replace("e+", "e").replace("e-0", "e-")


def _stepped_xy(xint, yline):
    """把 (x(y), y) 折线转成阶梯(水平+竖直)折线, 便于画键槽 Z 形轮廓。"""
    pts = []
    for i in range(len(yline)):
        if i == 0:
            pts.append((xint[i], yline[i]))
        else:
            # 先水平走到新 x, 再竖直上到新 y
            pts.append((xint[i], yline[i - 1]))
            pts.append((xint[i], yline[i]))
    return np.asarray(pts, dtype=float)


# ----------------------------------------------------------------------------- 合成演示
# 论文 10 个试件（表3）：Pu(kN), su(mm), τu(MPa), 破坏模式
PAPER_SPECIMENS = [
    ("CE-2-2-0-0",     152.1, 1.78, "C", ["0.90 P_u", "0.95 P_u", "1.00 P_u"]),
    ("NE-2-2-0-0",     178.1, 1.94, "A", ["0.90 P_u", "0.95 P_u", "1.00 P_u"]),
    ("CE-2-2-0-0-r",   302.2, 2.98, "A", ["0.20 P_u", "0.65 P_u", "1.00 P_u"]),
    ("CE-2-2-0-0-r(8)",233.5, 2.13, "C", ["0.10 P_u", "0.65 P_u", "1.00 P_u"]),
    ("CE-3-2-0-0-r",   374.3, 4.02, "C", ["0.15 P_u", "0.65 P_u", "1.00 P_u"]),
    ("CE-2-3-0-0-r",   332.1, 3.05, "B", ["0.28 P_u", "0.65 P_u", "1.00 P_u"]),
    ("CE-2-2-15-0-r",  305.0, 2.99, "B", ["0.29 P_u", "0.65 P_u", "1.00 P_u"]),
    ("CE-2-2-30-0-r",  313.6, 2.75, "A", ["0.22 P_u", "0.65 P_u", "1.00 P_u"]),
]


def make_synth_field(nx, ny, keys, mode, phase, Pu_frac, rng):
    """按直剪破坏力学规律合成 x 向应变场(单位: 归一化, 后续 ×1e-2)。

    严格对齐参考图(论文 Fig.12 实拍)单面板的真实结构:
      - 背景: 大面积 青绿/黄绿(中应变), 带明显斑驳低频起伏 (NOT 纯色);
      - 界面: 黑色粗阶梯折线(Z 形), 由 draw_stepped_interface 画;
      - 界面附近: **离散橙红小斑块**(高应变热点), 非连续红带 → 随荷载增强/增多;
      - 局部: 青/蓝色 负应变释放区(开裂处), 少量紫;
      - 峰值: 热点更亮更红、数目增加、局部贯通。
    mode: A键齿剪坏 / B键槽剪坏 / C键齿+键槽均剪坏。
    phase in [0,1] 荷载进程; Pu_frac 面板对应荷载比。"""
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    f = np.zeros((ny, nx), dtype=np.float64)
    # 试件左右两半: 结合界面位于 x≈0.5; 键槽使界面呈**阶梯**(Z 形)折线
    yv = np.linspace(0, 1, ny)
    step = 0.06
    saw = (yv * keys) % 1.0
    xint_1d = 0.5 + step * (saw - 0.5)
    xint = xint_1d[:, None]                                 # (ny,1) 广播
    dx = X - xint

    # ---- 背景: 中应变青绿底 + 轻微斑驳起伏 (参考图: 绿为主调, 背景较均匀)
    f += 0.0
    f += 0.06 * np.sin(2.1 * np.pi * X + 1.3 * Y) * np.sin(1.7 * np.pi * Y + 0.7)
    f += 0.05 * np.sin(5.3 * np.pi * X + 3.1 * np.sin(2.0 * Y))
    f += rng.normal(0, 0.018, f.shape)

    # ---- 界面附近的**离散橙红热点**(参考图: 沿黑阶梯线散布的小斑块, 密而小)
    #   热点数目随荷载增加; 位置紧贴界面(依阶梯线)上下交替
    n_hot = int(round(4 + 8 * phase))                      # 4 -> 12 个小热点
    spots = []
    for k in range(keys):
        yc = (k + 0.5) / keys
        if mode in ("A", "C"):
            spots.append((xint[0, 0] + 0.010, yc, 0.012, 0.020))   # 键齿根部
        if mode in ("B", "C"):
            spots.append((xint[0, 0] - 0.012, yc + 0.10, 0.011, 0.018))
    # 额外热点沿阶梯界面线随机密布(紧贴界面, 小尺寸)
    for _ in range(n_hot):
        yc = rng.uniform(0.06, 0.95)
        # 让热点跟随该高度的阶梯界面位置
        xl = 0.5 + step * (((float(yc) * keys) % 1.0) - 0.5)
        xo = rng.uniform(-0.012, 0.014)
        spots.append((xl + xo, yc, rng.uniform(0.008, 0.013),
                      rng.uniform(0.012, 0.020)))
    for (xc, yc, sx, sy) in spots:
        amp = (0.45 + 0.85 * phase) * float(Pu_frac)
        f += amp * np.exp(-(((X - xc) / sx) ** 2 + ((Y - yc) / sy) ** 2))

    # ---- 界面细脊(弱): 让界面线处略有应变, 但**不形成连续粗红带**
    f += (0.15 + 0.45 * phase) * np.exp(-(dx / 0.010) ** 2)

    # ---- 温和的界面梯度: 左半略正(绿), 右半略负(青/浅蓝) —— 参考图整体色相过渡
    f -= (0.05 + 0.12 * phase) * (1.0 / (1.0 + np.exp(-dx / 0.10)))

    # ---- 局部负应变(青/蓝 = 开裂释放区): 零散成块, 非整片
    #   主释放区偏下(参考图左下/右下有青蓝块), 块体更紧致
    rel = np.exp(-(((X - (xint + 0.05)) / 0.032) ** 2 + ((Y - 0.30) / 0.052) ** 2))
    f -= (0.30 + 0.75 * phase) * rel
    rel2 = np.exp(-(((X - (xint - 0.035)) / 0.030) ** 2 + ((Y - 0.22) / 0.048) ** 2))
    f -= (0.22 + 0.65 * phase) * rel2
    # 右下角小块深负(紫, 参考图右下的紫块)
    f -= (0.25 + 0.70 * phase) * np.exp(-(((X - (xint + 0.06)) / 0.036) ** 2
                                          + ((Y - 0.10) / 0.042) ** 2))
    # 上端轻微负应变(加载端)
    f -= 0.15 * (1 - 0.5 * phase) * np.exp(-((Y - 0.97) / 0.040) ** 2)
    return f, xint_1d


def _make_speckle(nx, ny, rng, density=0.22, dot=1):
    """生成随机散斑纹理(灰度 0~1, 白底黑点), 用于极淡叠加营造实测纹理感。"""
    t = np.ones((ny, nx), dtype=np.float32)
    mask = rng.random((ny, nx)) < density
    t[mask] = rng.uniform(0.15, 0.45, mask.sum())
    if dot > 0:
        t = gaussian_filter(t, 0.6)
    return t


def build_demo_rows(ncol, seed=7, texture=False):
    """生成论文 8 个代表性试件的 Fig.12 演示行。"""
    rng = np.random.default_rng(seed)
    nx, ny = 140, 220
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    rows = []
    letters = "abcdefghij"
    for i, (name, Pu, su, mode, labels) in enumerate(PAPER_SPECIMENS):
        keys = 3 if name.startswith("CE-3") else 2
        panels = []
        for j, lab in enumerate(labels):
            frac = float(lab.split()[0])
            phase = frac if mode in ("A",) and not name.endswith("-r") else 0.35 + 0.65 * frac
            f, xint = make_synth_field(nx, ny, keys, mode, phase, frac, rng)
            p = dict(fld=f * 1e-1, ext=[xs[0], xs[-1], ys[0], ys[-1]],
                     label=lab, xint=xint)
            if texture:
                p["tex"] = _make_speckle(nx, ny, rng)
            panels.append(p)
        rows.append(dict(title=f"({letters[i]}) 试件{name}", panels=panels,
                         keys=keys, mode=mode))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="直剪界面 Fig.12 风格: 多试件×多时相应变云图")
    # 数据源
    ap.add_argument("--manifest", default=None,
                    help="JSON 清单: [{title, panels:[{path,label,field}]}, ...]（最灵活）")
    ap.add_argument("--fields", default=None, help="单一试件的场目录(含 stage_XXXXX.npz)，配合 --stages")
    ap.add_argument("--stages", default="44,47,49", help="--fields 模式的阶段索引(逗号分隔)")
    ap.add_argument("--labels", default="0.90 P_u,0.95 P_u,1.00 P_u", help="每列标签(逗号分隔)")
    ap.add_argument("--row-title", default="(a) 试件CE-2-2-0-0", help="--fields 模式的组标题")
    ap.add_argument("--field", default="exx", help="应变分量: exx/eyy/exy(默认 exx = x 向)")
    ap.add_argument("--demo", action="store_true",
                    help="生成论文 Fig.12 合成演示(8 试件×3 时相, 按直剪破坏力学规律)")
    ap.add_argument("--texture", action="store_true",
                    help="(demo)叠加极淡散斑底纹(营造实测纹理感; 默认关=纯净纯色场)")
    ap.add_argument("--texture-alpha", type=float, default=0.18,
                    help="底纹不透明度(仅 --texture 时生效)")
    # 观感
    ap.add_argument("--sigma", type=float, default=4.0, help="归一化卷积平滑核(网格cell)")
    ap.add_argument("--minsup", type=float, default=0.05, help="归一化卷积最小支撑")
    ap.add_argument("--dilate", type=int, default=0, help="裂缝区膨胀连通(重色只连在开裂处)")
    ap.add_argument("--pct", type=float, default=92, help="每行色标上限百分位")
    ap.add_argument("--unit", type=str, default="", help="单位后缀: ''(原始) / '%%'(乘100) / 'ue'")
    ap.add_argument("--smooth", type=float, default=1.0, help="切片后轻量高斯(去方块)")
    ap.add_argument("--cmap", default="ref", choices=["ref", "vivid", "rainbow", "jet"],
                    help="色图: ref(按参考图取色, 柔和低饱和, 默认) / vivid / rainbow / jet")
    ap.add_argument("--symmetric", action="store_true", default=True,
                    help="色标以 0 对称(diverging 推荐, 默认开)")
    ap.add_argument("--vmax", type=float, default=None, help="手动色标上限(不填=按 pct 自动)")
    ap.add_argument("--vmin-rel", type=float, default=1.0,
                    help="下限 = -vmax*vmin_rel。1.0=对称(默认)；0.06~0.12 = 零应变贴到色标底部"
                         "→ 背景变紫(论文 Fig.12 的主导观感：紫底 + 界面绿肩红芯带)。"
                         "对称色标会让零应变落在色标正中→整版绿底，与参考图不符")
    # 界面轮廓/注记
    ap.add_argument("--interface", default=None,
                    help="黑色阶梯轮廓点 'x,y;x,y;...' (数据坐标); 或 'auto' 自动生成 Z 形")
    ap.add_argument("--crack-line", default=None,
                    help="真实裂纹中心线点列 'x,y' 每行一点(数据坐标 ds-px)，"
                         "画成粗黑锯齿线(参考图那种界面线；给了它就不再画 --interface 的平滑折线)")
    ap.add_argument("--crack-band", default=None,
                    help="真实缝宽带 'y,xleft,xright' 每行一点(数据坐标 ds-px)，"
                         "按每行实测缝宽填充成暗色带 —— 比 --crack-line 更接近照片里的缝(优先级最高)")
    ap.add_argument("--crack-line-lw", type=float, default=1.4,
                    help="真实裂纹中心线的线宽(参考图界面线是细线, 1.2~1.6; 默认 1.4)")
    ap.add_argument("--crack-band-color", default="#0d0d0d", help="缝带填充色")
    ap.add_argument("--crack-band-alpha", type=float, default=0.95, help="缝带不透明度")
    ap.add_argument("--interface-keys", type=int, default=2, help="auto 轮廓的键齿数")
    ap.add_argument("--annotate", action="append", default=None,
                    help="青框+箭头注记 'cx,cy,w,h,文字'（可多次）")
    ap.add_argument("--annotate-panel", type=int, default=0, help="注记画在第几列面板(默认0)")
    # 真实 DIC 观感: 散斑照片底图 + 半透明应变叠加
    ap.add_argument("--texture-dir", default=None, help="散斑 TIFF 目录(--texture-map 用)")
    ap.add_argument("--texture-map", default=None,
                    help="'阶段号=tiff文件名(不带扩展名)' 逗号分隔, 如 0=stage_00001_1,1=stage_00041_1")
    ap.add_argument("--tex-ds", type=float, default=4.0,
                    help="每 ds-px 对应多少原始 px(裁照片块用, 默认 4)")
    ap.add_argument("--tex-x0", type=float, default=None, help="照片裁剪带 x 起(ds-px)")
    ap.add_argument("--tex-x1", type=float, default=None, help="照片裁剪带 x 止(ds-px)")
    ap.add_argument("--tex-y0", type=float, default=None, help="照片裁剪带 y 起(ds-px)")
    ap.add_argument("--tex-y1", type=float, default=None, help="照片裁剪带 y 止(ds-px)")
    ap.add_argument("--tex-down", type=int, default=2, help="照片块降采样倍数(1=原分辨率)")
    ap.add_argument("--field-alpha", type=float, default=0.82,
                    help="应变场叠在照片上的不透明度(越低散斑越明显, 默认 0.82)")
    ap.add_argument("--y-flip", action="store_true",
                    help="y 轴反转(ds-px 的 y 是从照片顶部往下数的; 不翻转=试件上下颠倒, "
                         "叠照片底图时必须开)")
    # 布局
    ap.add_argument("--fig-w", type=float, default=None, help="整图宽(inch, 默认按列数)")
    ap.add_argument("--panel-h", type=float, default=2.05, help="单个面板高(inch)")
    ap.add_argument("--title-size", type=float, default=10.5)
    ap.add_argument("--out", default="fig12.png")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()

    scale = 100.0 if a.unit == "%" else 1.0
    cmap = {"ref": EXY, "vivid": EXY_VIVID, "rainbow": "rainbow", "jet": "jet"}[a.cmap]

    # ---- 组装 rows: [{title, panels:[dict(field2d, ext, label)]}]
    rows = []
    if a.demo:
        rows = build_demo_rows(ncol=None, texture=a.texture)
    elif a.manifest:
        with open(a.manifest, encoding="utf-8") as fh:
            man = json.load(fh)
        xint = parse_interface(a.interface)
        for grp in man:
            panels = []
            for p in grp["panels"]:
                q = p.get("field", a.field)
                if p["path"].lower().endswith(".npz"):
                    fld = load_npz_field(p["path"], q, a.sigma, a.minsup, a.dilate, scale)
                    d = np.load(p["path"]); xs = np.asarray(d["xs"]); ys = np.asarray(d["ys"])
                    ext = [xs[0], xs[-1], ys[0], ys[-1]]
                else:
                    g, m, ext = load_csv_field(p["path"], q, scale)
                    fld = np.nan_to_num(smooth_field(g, m, a.sigma, a.minsup), nan=0.0)
                panels.append(dict(fld=fld, ext=ext, label=p.get("label", "")))
            if xint is not None:                 # 真实场: 每块面板都叠同一条界面/裂缝轮廓
                for pnl in panels:
                    pnl["xint"] = xint
            rows.append(dict(title=grp.get("title", ""), panels=panels))
    else:
        xint = parse_interface(a.interface)
        stages = [int(x) for x in a.stages.split(",")]
        labels = [s.strip() for s in a.labels.split(",")]
        # 真实 DIC 观感: 应变场半透明叠在**散斑原图**上(失效区露散斑)。
        # --texture-map "0=stage_00001_1,1=stage_00041_1,..." 给出 阶段->tiff 文件名。
        tmap = {}
        if a.texture_map:
            for kv in a.texture_map.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    tmap[int(k)] = v.strip()
        tex_cache = {}
        panels = []
        ext = None
        for i, s in enumerate(stages):
            path = os.path.join(a.fields, f"stage_{s:05d}.npz")
            fld = load_npz_field(path, a.field, a.sigma, a.minsup, a.dilate, scale)
            d = np.load(path); xs = np.asarray(d["xs"]); ys = np.asarray(d["ys"])
            ext = [xs[0], xs[-1], ys[0], ys[-1]]
            pn = dict(fld=fld, ext=ext, label=labels[i] if i < len(labels) else str(s))
            if xint is not None:
                pn["xint"] = xint
            if s in tmap and a.texture_dir:
                import tifffile
                tp = os.path.join(a.texture_dir, tmap[s] + ".tiff")
                if tp not in tex_cache:
                    img = tifffile.imread(tp)
                    # 裁出面板对应的原始像素块(轴: [y, x], 原始px = ds-px × tex_ds)
                    y_a, y_b = int(a.tex_y0 * a.tex_ds), int(a.tex_y1 * a.tex_ds)
                    x_a, x_b = int(a.tex_x0 * a.tex_ds), int(a.tex_x1 * a.tex_ds)
                    crop = img[y_a:y_b, x_a:x_b]
                    if a.tex_down > 1:
                        crop = crop[::a.tex_down, ::a.tex_down]
                    tex_cache[tp] = crop
                pn["tex_img"] = tex_cache[tp]
                pn["tex_ext"] = [a.tex_x0, a.tex_x1, a.tex_y0, a.tex_y1]
            panels.append(pn)
        rows.append(dict(title=a.row_title, panels=panels))

    ncol = max(len(r["panels"]) for r in rows)
    nrow = len(rows)

    # ---- 统一裁剪框 = 所有 ext 的并集
    x0 = min(p["ext"][0] for r in rows for p in r["panels"])
    x1 = max(p["ext"][1] for r in rows for p in r["panels"])
    y0 = min(p["ext"][2] for r in rows for p in r["panels"])
    y1 = max(p["ext"][3] for r in rows for p in r["panels"])
    aspect = (y1 - y0) / max(x1 - x0, 1e-9)

    # ---- 色标范围(全局统一, 便于跨试件对比)
    if a.vmax is not None:
        vm = a.vmax
    else:
        vals = []
        for r in rows:
            for p in r["panels"]:
                v = p["fld"][np.abs(p["fld"]) > 0]
                if v.size:
                    vals.append(np.abs(v))
        vm = float(np.percentile(np.concatenate(vals), a.pct)) if vals else 0.1
    if vm <= 0:
        vm = 0.1
    vmn = -vm * a.vmin_rel          # 不对称下限: vmin_rel<1 时零应变贴底 -> 背景紫(参考图观感)

    # ---- 布局: 严格按参考图实测比例 (参考图 738x928, 面板 78x158, 色标 5, 间距 5)
    #   面板   w:h = 78:158 = 1:2.026
    #   色标宽 = 面板宽 * 5/78  = 面板宽 * 0.0641
    #   面-标距 = 面板宽 * 5/78  = 面板宽 * 0.0641
    #   组间横距 = 面板宽 * (119-85=34)/78 = 面板宽 * 0.436 (容下刻度数字)
    panel_h = a.panel_h                        # 面板高(inch)
    panel_w = panel_h / 2.03                   # 面板宽 —— 参考图实测 78:158 = 1:2.026
    cb_w = panel_w * 0.0641                    # 色标宽 —— 实测 1/15.6 面板宽
    gap_cb = panel_w * 0.0641                  # 面板↔色标 —— 紧贴
    gap_col = panel_w * 0.436                  # 面板组间距 —— 需容左侧刻度数字
    fig_w = a.fig_w or (ncol * (panel_w + gap_cb + cb_w + gap_col) + 0.55)
    row_h = panel_h + 0.30                     # 面板 + 相位标签 + 组标题(紧凑)
    fig_h = nrow * row_h + 0.20
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    cline = None
    if a.crack_line:
        cline = np.loadtxt(a.crack_line, delimiter=",", ndmin=2)   # [x, y] 数据坐标
        print(f"[crack-line] {len(cline)} 点  x=[{cline[:,0].min():.0f},{cline[:,0].max():.0f}]"
              f"  y=[{cline[:,1].min():.0f},{cline[:,1].max():.0f}]")
    cband = None
    if a.crack_band:
        cband = np.loadtxt(a.crack_band, delimiter=",", ndmin=2)   # [y, xl, xr] 数据坐标
        print(f"[crack-band] {len(cband)} 行  缝宽中位 {np.median(cband[:,2]-cband[:,1]):.1f} ds-px")

    left = 0.42 / fig_w
    top = 1 - 0.16 / fig_h
    for ri, r in enumerate(rows):
        ytop = top - ri * (row_h / fig_h)
        ybot = ytop - (panel_h / fig_h)
        for ci, p in enumerate(r["panels"]):
            xleft = left + ci * ((panel_w + gap_cb + cb_w + gap_col) / fig_w)
            w = panel_w / fig_w
            h = panel_h / fig_h
            ax = fig.add_axes([xleft, ybot, w, h])
            fld = p["fld"]
            if a.smooth > 0:
                fld = gaussian_filter(fld, a.smooth)
            # 真实 DIC 观感: 先铺散斑照片, 再叠 NaN 透明的应变场(失效区露散斑)
            if "tex_img" in p:
                ax.imshow(p["tex_img"], cmap="gray", vmin=0, vmax=255,
                          extent=p["tex_ext"], origin="lower", aspect="auto",
                          interpolation="bilinear", zorder=0)
                cmap_f = copy.copy(cmap)
                cmap_f.set_bad((1.0, 1.0, 1.0, 0.0))   # NaN -> 全透明, 露出照片
                ax.imshow(np.ma.masked_invalid(fld), cmap=cmap_f,
                          vmin=vmn, vmax=vm,
                          extent=p["ext"], origin="lower", aspect="auto",
                          alpha=a.field_alpha, interpolation="bicubic", zorder=1)
            else:
                ax.imshow(fld, extent=p["ext"], origin="lower", cmap=cmap,
                          vmin=vmn, vmax=vm,
                          aspect="auto", interpolation="bicubic")
            # 可选: 极淡散斑底纹(营造实测纹理感)
            if "tex" in p:
                ax.imshow(p["tex"], cmap="gray", vmin=0, vmax=1,
                          extent=p["ext"], origin="lower", aspect="auto",
                          alpha=a.texture_alpha, interpolation="bilinear")
            ax.set_xlim(x0, x1)
            # ⚠️ ds-px 的 y 是「从照片顶部往下数」的。默认 origin=lower + ylim(y0,y1)
            #    会把整块试件上下颠倒 —— 场和裂缝线内部自洽, 看不出来, 但一叠照片
            #    底图就露馅(缝的宽开口端跑到面板底下)。--y-flip 恢复物理正立。
            ax.set_ylim(y1, y0) if a.y_flip else ax.set_ylim(y0, y1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.25"); sp.set_linewidth(0.7)
            # 界面轮廓
            if cband is not None:
                # 真实缝宽填充带(优先级最高): 按每行实测缝宽填充, 形状/宽度都跟照片一致
                ax.fill(np.concatenate([cband[:, 1], cband[:, 2][::-1]]),
                        np.concatenate([cband[:, 0], cband[:, 0][::-1]]),
                        color=a.crack_band_color, lw=0,
                        alpha=a.crack_band_alpha, zorder=6)
            elif a.crack_line:
                # 真实裂纹中心线(从原始照片逐行抠出的锯齿线): 粗黑实线,
                # 形状跟照片一致(有横向锯齿/折拐), 不是 DIC 平滑轨迹的替身。
                ax.plot(cline[:, 0], cline[:, 1], color="k", lw=a.crack_line_lw,
                        solid_joinstyle="round", zorder=6)
            elif "xint" in p:
                xint = np.asarray(p["xint"], dtype=float)
                # 真实场回: 只有一条界面轨迹(裂缝/键槽线) -> 直接按点直连折线画,
                #           不要套 demo 那两条"左半外框 + 右半外框"的 Z 形键槽组合
                # ⚠️ 区分 1 维(demo: 存的是各行界面 x 坐标) 与 2 维(真实场: N×2 折线点)。
                #    1 维数组 shape[0]=ny 也会 >=3, 若不先判 ndim 会误走折线分支而 IndexError。
                if xint.ndim == 2 and xint.shape[0] >= 3:
                    # 真实场(裂缝轨迹/界面线): 按给定点直连折线
                    ax.plot(xint[:, 0], xint[:, 1], color="k", lw=1.4,
                            solid_joinstyle="miter", zorder=6)
                else:
                    # 点少(≤2)当作 demo 式的半边轮廓: 画左边界 + 顶部水平段 + 键槽阶梯线
                    yline = np.linspace(0.0, 1.0, max(xint.size, 2))
                    draw_stepped_interface(ax, [(0.0, 0.0), (0.0, 1.0)], lw=1.2)
                    draw_stepped_interface(ax, [(float(xint[0]), 0.0), (1.0, 0.0),
                                                (1.0, 1.0), (float(xint[-1]), 1.0)], lw=1.2)
                    sx = _stepped_xy(xint, yline)
                    ax.plot(sx[:, 0], sx[:, 1], color="k", lw=1.5, zorder=6,
                            solid_joinstyle="miter")
            elif a.interface == "auto":
                pts = auto_interface(x0, y0, x1, y1, keys=a.interface_keys)
                draw_stepped_interface(ax, pts)
            elif a.interface:
                pts = [[float(v.split(",")[0]), float(v.split(",")[1])]
                       for v in a.interface.split(";") if v.strip()]
                draw_stepped_interface(ax, pts)
            # 注记(仅指定列)
            if a.annotate and ci == a.annotate_panel:
                for spec in a.annotate:
                    parts = spec.split(",")
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = map(float, parts[:4]); txt = ",".join(parts[4:])
                    txt = txt.replace("|", "\n")     # 用 | 代替换行符(命令行传不了裸 \n)
                    # ⚠️ cx/cy/bw/bh 是**数据坐标**(面板 extent), 不是 0~1 分数:
                    #   早先写成 tx = min(cx + bw*0.78, 0.98) 会把文字甩到最左缘(ha=right 往左溢出)。
                    ax.add_patch(Rectangle((cx - bw / 2, cy - bh / 2), bw, bh,
                                           fill=False, ec="cyan", lw=1.6, zorder=8))
                    ax.add_patch(FancyArrowPatch((cx + bw * 0.72, cy + bh * 0.9),
                                                 (cx + bw * 0.34, cy + bh * 0.35),
                                                 arrowstyle="-|>", mutation_scale=11,
                                                 color="cyan", lw=1.4, zorder=8))
                    # 文字贴**面板内右缘**(数据坐标 x1 往里缩一点), 避免溢出到色标
                    tx = x1 - 0.012 * (x1 - x0)
                    ax.text(tx, cy + bh * 1.05, txt, color="k", fontsize=7.5,
                            ha="right", va="bottom", zorder=9,
                            bbox=dict(fc="white", ec="cyan", lw=0.8, alpha=0.92, pad=1.2))
            # 面板下方荷载/位移标签
            if p["label"]:
                ax.text(0.5, -0.045, p["label"], transform=ax.transAxes,
                        ha="center", va="top", fontsize=9.5)
            # 每面板右侧竖直色标(紧贴面板, 刻度在左, 参考图风格: 极细 + 离散分级 + 小字)
            cax = fig.add_axes([xleft + w + (gap_cb / fig_w), ybot, (cb_w / fig_w), h])
            cb = fig.colorbar(ScalarMappable(Normalize(vmn, vm), cmap), cax=cax)
            cb.ax.yaxis.set_ticks_position("left")
            cb.ax.yaxis.set_label_position("left")
            cb.ax.tick_params(labelsize=4.6, length=1.0, width=0.3, pad=0.4)
            cb.outline.set_linewidth(0.35)
            nticks = 9
            ticks = np.linspace(vmn, vm, nticks)
            # 0 若在量程内, 把离 0 最近的刻度精确置 0(等距位置不变, 参考图色标也有 0 刻度)
            if vmn < 0 < vm:
                ticks[int(np.argmin(np.abs(ticks)))] = 0.0
            cb.set_ticks(ticks)
            cb.set_ticklabels([fmt_e(t) for t in ticks])
        # 组标题(组下方居中)
        fig.text(left + (ncol * (panel_w + gap_cb + cb_w + gap_col) - gap_col) / (2 * fig_w),
                 ybot - (0.40 / fig_h),
                 r["title"], ha="center", va="top", fontsize=a.title_size)

    fig.savefig(a.out, dpi=a.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {a.out}  ({nrow} 试件 × {ncol} 时相, vmax={vm:.4g}, vmin={vmn:.4g})")


if __name__ == "__main__":
    main()
