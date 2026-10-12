#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从原始 TIFF 抠出**真实裂纹**（像素级锯齿 + 分支），产出可叠到 Fig.12 面板上的
黑色裂纹掩膜。

为什么要单独抠：DIC 的 crack_mask/trace 是在 59×66 的**粗网格**上按 "|du/dx| 峰值"
提的界面轨迹 —— 一条平滑细折线。但真实裂缝是照片里看到的锯齿暗缝，有实际开口
宽度、有分支、有局部大空洞。Fig.12 面板上画那条平滑折线跟这张照片对不上。

坐标
  ds-px -> 原始 px : ×4（由 detect_crack.py 的 DS=4 与 TIFF 4096×4504 核对得出）
  Fig.12 面板范围 : x ds-px [291,748]  y ds-px [40,968]
                    -> 原始 px x [1164,2992]  y [160,3872]
输出
  fig12_fields/crack_rgba.png   RGBA 黑裂纹（A=255 处为裂纹），2× 超采样
  fig12_fields/crack_mask.png   二值白底黑裂纹，便于人眼复核
"""
import os

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import (binary_closing, binary_opening, gaussian_filter,
                           label, median_filter)

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "in", "新建文件夹 (5)")
DST = os.path.join(BASE, "fig12_fields")

PX = 4                 # ds-px -> 原始 px
X0, X1 = 291, 748      # 面板 x 范围 (ds-px) —— 与 make_fig12_fields.py 一致
Y0, Y1 = 40, 968       # 面板 y 范围 (ds-px)
UGE = 4                # 用加载最大的那一帧(裂缝张开最明显)来定裂纹形状
SS = 2                 # 超采样倍数


def detect_crack(gray, drop=40.0, row_q=80.0, trace_ds=None):
    """gray: 2D uint8 原始像素 -> 布尔裂纹掩膜

    ⚠️ 坑: 一开始用 median_filter(size=(35,15)) 做局部背景, 结果只检出 22 px。
       原因是这道缝顶部是**上百像素宽的大暗区**, 中值窗(15 列)整个落在缝里面,
       缝内中值 ≈ 缝外中值 -> 对比度被自己抹平 -> 检了个寂寞。
       裂缝是**竖直贯穿**特征, 所以按行取高分位(默认 75%)当地面亮度更稳。
    """
    gm = gray.astype(np.float32)
    # ⚠️ 先高斯平滑再判暗: 散斑是 ~10px 的小点, 缝是 ~20~250px 的宽暗带。
    #    不平滑的话逐行取「最长暗段」会在**缝和散斑团之间来回跳**, 中心线锯齿乱飞。
    #    σ=6 把散斑衰减一个数量级, 缝基本保留。
    gm = gaussian_filter(gm, sigma=6.0)
    # 每行亮度基线 = 该行明亮散斑面(裂缝只占该行 <20% 宽度)
    bg = np.percentile(gm, row_q, axis=1, keepdims=True)
    dark = gm < (bg - drop)
    # 打通沿缝方向的断续, 并吃掉孤立亮斑造成的碎片
    dark = binary_closing(dark, np.ones((5, 3)))
    dark = binary_opening(dark, np.ones((3, 3)))
    # 只留「竖着贯穿」的连通域, 否则底部的黑横杠/阴影也会进来
    # --- 2) 以 DIC 界面轨迹为中心开宽窗, 每行只留「窗口内最长的一段暗区」----
    #    这么做的三个好处:
    #      (a) 照片里那条黑横标线(横向贯穿)被彻底排除 —— 单行长条根本不是缝;
    #      (b) 散斑噪点(孤立暗点)被排除;
    #      (c) 顺带反证 DIC 轨迹确实落在真实缝上(窗口 ±200 原始px ≈ ±50 ds-px)。
    if trace_ds is not None:
        W = 200                                   # 窗口半宽(原始px)
        # ⚠️ 中心线是**逐行**的(行->ds y->轨迹上的 ds x), 别写成逐列,
        #    否则 cen 长度变成列数, 循环里 cen_o[i] 直接 IndexError。
        ys_ds = Y0 + np.arange(dark.shape[0]) / PX
        cen = np.interp(ys_ds, trace_ds[:, 1], trace_ds[:, 0])   # 轨迹 x(y)
        cen_o = (cen - X0) * PX
        H = dark.shape[0]
        src = dark.copy()                      # 后面要用原图逐行找暗段, 别就地清零
        out = np.zeros_like(dark)
        cen = np.full(H, np.nan, dtype=float)

        # --- 2a) 种子: 窗口内**纵向贯穿度最大**的连通域(顶部斑块/横标线被排掉) ---
        lab, n = label(src)
        best_lab, best_score = -1, -1.0
        for i in range(1, n + 1):
            r = np.where(lab == i)[0]
            if not r.size:
                continue
            span_f = (r.max() - r.min()) / float(H)
            if span_f < 0.6:
                continue                       # 斑块: 横向成片, 纵向不到 6 成幅
            score = span_f * float((lab == i).sum()) ** 0.5
            if score > best_score:
                best_lab, best_score = i, score
        if best_lab > 0:
            src = (lab == best_lab)
            r0 = int(np.median(np.where(src.any(axis=1))[0]))
            xs0 = np.flatnonzero(src[r0])
            prev = float(np.median(xs0)) if xs0.size else float(cen_o[r0])
        else:                                  # 退路: 直接拿 DIC 轨迹中心起步
            r0 = int(np.clip(cen_o, 0, H - 1)[0])
            prev = float(cen_o[r0])

        # --- 2b) **连续性跟踪**: 从种子行向上下各推一行, 每行只接「离上一点最近的段」
        #    这比「每行取最长段」稳得多 —— 后者会在缝和散斑团之间来回跳(实测乱飞)。
        for i in list(range(r0, H)) + list(range(r0 - 1, -1, -1)):
            d = src[i]
            if not d.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(([0], d.astype(np.int8), [0]))))
            if not edges.size:
                continue
            s, e = edges[0::2], edges[1::2]
            centers = (s + e) / 2.0
            k = int(np.argmin(np.abs(centers - prev)))
            if abs(centers[k] - prev) > 150:          # 跳太远 = 跟丢了, 保持不动
                continue
            out[i, s[k]:e[k]] = True
            seg = gm[i, s[k]:e[k]]
            w = np.clip((bg[i, 0] - seg) / max(bg[i, 0] - seg.max(), 1e-3), 0, 1)
            cen[i] = (s[k] + np.sum(w * np.arange(seg.size)) / max(w.sum(), 1e-6)
                      ) if w.sum() > 1e-6 else centers[k]
            prev = cen[i]
        dark = out
    return dark, cen


def centerline_to_ds(cen, y_row):
    """逐行中心(原始px, 行号) -> (x_ds, y_ds) 折线点列"""
    ys_ds = Y0 + y_row / PX
    xs_ds = (cen / PX) + X0
    o = np.isfinite(xs_ds) & np.isfinite(ys_ds)
    return np.stack([xs_ds[o], ys_ds[o]], axis=1)


def main():
    os.makedirs(DST, exist_ok=True)
    imgs = {}
    for k in ["stage_00001_1", "stage_00035_0", "stage_00041_1",
              "stage_00054_1", "stage_00059_1"]:
        p = os.path.join(RAW, k + ".tiff")
        if os.path.exists(p):
            imgs[k] = tifffile.imread(p)
    key = "stage_00059_1" if "stage_00059_1" in imgs else sorted(imgs)[0]
    im = imgs[key]
    print(f"[src] {key}.tiff  {im.shape}  mean={im.mean():.1f}")

    # 裁出面板对应的原始像素块
    sub = im[Y0 * PX:Y1 * PX, X0 * PX:X1 * PX]
    H, W = sub.shape
    print(f"[crop] orig {W}x{H}  (ds-px {X1 - X0}x{Y1 - Y0}, 宽:高={W / H:.3f})")

    cm = np.load(os.path.join(BASE, "crack_mask_b.npz"))
    tr = np.stack([cm["trace_x"], cm["trace_y"]], axis=1)   # [x,y] ds-px
    m, cen = detect_crack(sub, trace_ds=tr)
    # 2× 超采样上采样(最近邻会有锯齿边, 裂纹这种细缝必须平滑)
    H2, W2 = m.shape[0] * SS, m.shape[1] * SS
    yi = (np.arange(H2) // SS).clip(0, H - 1)
    xi = (np.arange(W2) // SS).clip(0, W - 1)
    m2 = m[np.ix_(yi, xi)]

    yrow = np.arange(H)
    line_ds = centerline_to_ds(cen, yrow)
    # 面板上下边沿补齐: 照片里这条缝是**贯穿整块面板**的, 中心线底部(y≈850)悬空
    # 是因为最底下那几行跟踪跟丢了(缝里太黑), 沿末端方向外推到 y=Y1 即可。
    for i_end, y_edge in [(0, Y0), (-1, Y1)]:
        if abs(line_ds[i_end, 1] - y_edge) > 1.0 and len(line_ds) > 2:
            j = -2 if i_end == -1 else 1
            dx = line_ds[i_end, 0] - line_ds[j, 0]
            dy = line_ds[i_end, 1] - line_ds[j, 1]
            if dy != 0:
                t = (y_edge - line_ds[i_end, 1]) / dy
                line_ds = np.vstack([line_ds, [line_ds[i_end, 0] + dx * t, y_edge]])
    np.savetxt(os.path.join(DST, "crack_line.txt"), line_ds, fmt="%.1f,%.1f")
    print(f"[line] 中心线点数 {len(line_ds)}  x=[{line_ds[:,0].min():.0f},{line_ds[:,0].max():.0f}] ds-px")

    rgba = np.zeros((H2, W2, 4), np.uint8)
    rgba[..., 0:3] = 8
    rgba[..., 3] = np.where(m2, 235, 0)
    Image.fromarray(rgba, "RGBA").save(os.path.join(DST, "crack_rgba.png"))
    bin8 = np.where(m2, 0, 255).astype(np.uint8)
    Image.fromarray(bin8, "L").save(os.path.join(DST, "crack_visual.png"))

    # 自检: 裂纹掩膜的每行中心 x 是否落在 DIC 界面轨迹 (456~616 ds-px) 附近
    xs_ds = np.arange(X0, X1 + 1, 16.0)
    print(f"[crack] 像素数 {m.sum()}  占面板 {100.0 * m.mean():.2f}%")
    ys = np.where(m.any(axis=1))[0]
    if ys.size:
        print(f"[crack] y 范围 [0,{H - 1}] -> ds-py [{ys.min() / PX:.0f},{ys.max() / PX:.0f}]")
        # 沿列分布看主缝位置
        prof = m.mean(axis=0)
        wc = int(np.argmax(np.convolve(prof, np.ones(9), "same")))
        print(f"[check] 主缝列中心 orig x={wc} -> ds-px {X0 + wc / PX:.0f}"
              f" (DIC 界面轨迹 x=[456,616])")


if __name__ == "__main__":
    main()
