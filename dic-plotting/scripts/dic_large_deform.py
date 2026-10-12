#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dic_large_deform.py —— 大变形/非连续帧 的 DIC 位移-应变场计算

## 为什么需要它
`dic_raw_to_field.py` 假设相邻帧位移很小(逐帧跟踪, 搜索窗 ±6 ds-px), 只能处理**连续序列**。
但实际试验常给出**稀疏的关键阶段**(如 stage_00001/00041/00054/00059), 帧间跨越几十个荷载步,
位移可达 ±100 ds-px(数百原始像素), 远超小搜索窗 -> 匹配全部失败(valid=0, cval~0.2)。

## 算法: coarse-to-fine + reliability-guided 传播
1) **稀疏种子**: 在粗网格(step=--seed-step)上, 于大范围(±--seed-r)做 ZNCC 粗配准;
2) **外点剔除**: 用邻域中值滤波剔掉误配种子(ZNCC 低或与邻居不一致);
3) **插值预测**: 由可靠种子双线性插值出全网格的初始位移(捕获大位移趋势);
4) **局部精化**: 每点在小窗(±--refine-r)内以预测值为中心精搜 -> 亚像素位移;
5) **应变**: 位移场高斯平滑后求梯度 -> exx/eyy(同 dic_raw_to_field.py 约定)。

## 用法
    python dic_large_deform.py --indir DIR --outdir OUT --pattern 'stage_*_{ch}.tiff' --channel 1
关键参数:
    --ds        降采样倍数(默认4)
    --half      子集半宽(ds-px, 默认16; 需 > 散斑周期/ds 使模板唯一)
    --seed-step 种子间距(默认64)   --seed-r 粗搜半径(默认128)
    --refine-r  精化半径(默认8)    --step   输出网格步长(默认16)
    --rel-max   种子可靠性筛选比例(默认0.6, 越小越严格)

输出: outdir/stage_{idx:05d}.npz  (exx, eyy, ux, uy, mask, xs, ys)
      与 dic_raw_to_field.py 完全兼容, 可直接喂给 dic_ref_grid.py / dic_fig12_specimen.py
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import fftconvolve


# ------------------------------------------------------------------ ZNCC 核心
def integral(a):
    I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return I


def _win_mean_std(Ia, Ia2, k):
    """积分图 -> 每个 kxk 窗口的 mean/std (输出 shape = 原图 - k + 1)"""
    A, B = Ia[:-k, :-k], Ia[k:, k:]
    C, D = Ia[k:, :-k], Ia[:-k, k:]
    s = B - D - C + A
    n = float(k * k)
    mean = s / n
    if Ia2 is None:
        return mean, None
    A2, B2 = Ia2[:-k, :-k], Ia2[k:, k:]
    C2, D2 = Ia2[k:, :-k], Ia2[:-k, k:]
    sq = B2 - D2 - C2 + A2
    var = np.maximum(sq / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


def zncc_map(T, S, ts=None, tsum=None):
    """在 S 中对模板 T 做全窗口归一化互相关, 返回 corr 平面 (shape = S-T+1)"""
    ky, kx = T.shape
    if ts is None:
        ts = float(T.std())
    if tsum is None:
        tsum = float(T.sum())
    corr = fftconvolve(S, T[::-1, ::-1], mode="valid")
    Sm, Ss = _win_mean_std(integral(S), integral(S * S), ky)
    n = float(ky * kx)
    den = np.where(Ss * ts * n > 1e-6, Ss * ts * n, 1e-6)
    return np.clip((corr - Sm * tsum) / den, -1.0, 1.0)


def subpixel(C, yy, xx):
    """抛物线亚像素修正, 返回 (dy, dx) 修正量"""
    if 0 < yy < C.shape[0] - 1:
        d = (C[yy + 1, xx] - C[yy - 1, xx])
        e = C[yy + 1, xx] + C[yy - 1, xx] - 2 * C[yy, xx]
        dy = -d / (2 * e) if e != 0 else 0.0
    else:
        dy = 0.0
    if 0 < xx < C.shape[1] - 1:
        d = (C[yy, xx + 1] - C[yy, xx - 1])
        e = C[yy, xx + 1] + C[yy, xx - 1] - 2 * C[yy, xx]
        dx = -d / (2 * e) if e != 0 else 0.0
    else:
        dx = 0.0
    return dy, dx


# ------------------------------------------------------------------ 主流程
def load_ds(path, ds):
    a = np.asarray(Image.open(path)).astype(np.float64)
    return a[::ds, ::ds]


def match_seed(ref, cur, HALF, seed_step, seed_r, tstd_min, cval_min, peakgap):
    """阶段1: 稀疏种子粗配准"""
    h, w = ref.shape
    pad = HALF + seed_r + 2
    ys = np.arange(pad, h - pad, seed_step)
    xs = np.arange(pad, w - pad, seed_step)
    pts, Uy, Ux, Rel = [], [], [], []
    for gy in ys:
        for gx in xs:
            T = ref[gy - HALF:gy + HALF + 1, gx - HALF:gx + HALF + 1]
            ts = float(T.std())
            if ts < tstd_min:
                continue
            y0 = max(0, gy - pad)
            x0 = max(0, gx - pad)
            S = cur[y0:y0 + 2 * pad + HALF * 2 + 1, x0:x0 + 2 * pad + HALF * 2 + 1]
            if S.shape[0] < 2 * HALF + 1 or S.shape[1] < 2 * HALF + 1:
                continue
            C = zncc_map(T, S, ts, float(T.sum()))
            if C.size == 0:
                continue
            yy, xx = np.unravel_index(np.argmax(C), C.shape)
            cval = float(C[yy, xx])
            if cval < cval_min:
                continue
            C2 = C.copy()
            C2[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] = -9
            if cval - float(C2.max()) < peakgap:
                continue
            # corr 平面 yy/xx = 模板左上角在搜索窗 S 内的偏移,
            # 模板中心在 cur 中 = (y0 + yy + HALF), 位移 = 该值 - gy
            py = yy + HALF - pad
            px = xx + HALF - pad
            if max(abs(py), abs(px)) > seed_r:
                continue          # 落在搜索窗边界 -> 可能未收敛
            pts.append((gy, gx))
            Uy.append(py)
            Ux.append(px)
            Rel.append(cval)
    if not pts:
        return None
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
            np.array(Uy, float), np.array(Ux, float), np.array(Rel, float))


def filter_seeds(sy, sx, Uy, Ux, Rel, rel_keep=0.6):
    """阶段2: 邻域一致性筛选 —— 剔除与周围中值差异过大的误配种子"""
    if len(sy) < 5:
        return sy, sx, Uy, Ux, Rel, np.ones(len(sy), bool)
    ny, nx = sy.max() + 1, sx.max() + 1
    gun = np.full((ny + 1, nx + 1), np.nan)
    gun[sy, sx] = Uy
    gunx = np.full((ny + 1, nx + 1), np.nan)
    gunx[sy, sx] = Ux
    # 用最近邻填充后再取中值
    from scipy.interpolate import griddata
    pts = np.c_[sy, sx]
    try:
        gy_vals = griddata(pts, Uy, pts, method="nearest")
        gx_vals = griddata(pts, Ux, pts, method="nearest")
    except Exception:
        return sy, sx, Uy, Ux, Rel, np.ones(len(sy), bool)
    err = np.abs(Uy - gy_vals) + np.abs(Ux - gx_vals)
    thr = np.nanpercentile(Rel, 100 * (1 - rel_keep)) if len(Rel) > 3 else 0
    good = (Rel >= thr) & (err <= max(4.0, np.nanpercentile(err, 85)))
    return sy[good], sx[good], Uy[good], Ux[good], Rel[good], good


def interp_seeds(sy, sx, Uy, Ux, gys, gxs):
    """阶段3: 由种子双线性插值出全网格初始位移"""
    from scipy.interpolate import griddata
    pts = np.c_[sy, sx]
    GY, GX = np.meshgrid(gys, gxs, indexing="ij")
    tgt = np.c_[GY.ravel(), GX.ravel()]
    try:
        uy = griddata(pts, Uy, tgt, method="linear")
        ux = griddata(pts, Ux, tgt, method="linear")
    except Exception:
        return np.zeros(GY.shape), np.zeros(GX.shape)
    # linear 在凸包外为 nan -> 用 nearest 兜底
    m = ~np.isfinite(uy)
    if m.any():
        uy[m] = griddata(pts, Uy, tgt[m], method="nearest")
    m = ~np.isfinite(ux)
    if m.any():
        ux[m] = griddata(pts, Ux, tgt[m], method="nearest")
    return uy.reshape(GY.shape), ux.reshape(GX.shape)


def refine(ref, cur, HALF, gys, gxs, uy0, ux0, a_refine_r, tstd_min,
           cval_min, peakgap):
    """阶段4: 以插值预测为中心, 小窗内精搜 -> 亚像素位移 + mask"""
    h, w = cur.shape
    ngy, ngx = len(gys), len(gxs)
    Uy = np.full((ngy, ngx), np.nan)
    Ux = np.full((ngy, ngx), np.nan)
    cvals = np.zeros((ngy, ngx), float)
    for iy, gy in enumerate(gys):
        for ix, gx in enumerate(gxs):
            T = ref[gy - HALF:gy + HALF + 1, gx - HALF:gx + HALF + 1]
            ts = float(T.std())
            if ts < tstd_min:
                continue
            py = int(round(uy0[iy, ix]))
            px = int(round(ux0[iy, ix]))
            cy, cx = gy + py, gx + px
            pad = HALF + a_refine_r
            y0, x0 = cy - pad, cx - pad
            if y0 < 0 or x0 < 0 or cy + pad + 1 > h or cx + pad + 1 > w:
                continue
            S = cur[y0:cy + pad + 1, x0:cx + pad + 1]
            if S.shape[0] < 2 * HALF + 1 or S.shape[1] < 2 * HALF + 1:
                continue
            C = zncc_map(T, S, ts, float(T.sum()))
            if C.size == 0:
                continue
            yy, xx = np.unravel_index(np.argmax(C), C.shape)
            cval = float(C[yy, xx])
            if cval < cval_min:
                continue
            C2 = C.copy()
            C2[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] = -9
            if cval - float(C2.max()) < peakgap:
                continue
            sy_, sx_ = subpixel(C, yy, xx)
            # 模板中心在 cur 中 = (y0 + yy + HALF) = cy - pad + yy + HALF
            # 位移 = 该值 - gy = py - a_refine_r + yy   (pad = HALF + a_refine_r)
            oy = py - a_refine_r + yy
            ox = px - a_refine_r + xx
            Uy[iy, ix] = oy + sy_
            Ux[iy, ix] = ox + sx_
            cvals[iy, ix] = cval
    m = np.isfinite(Uy) & np.isfinite(Ux)
    return np.nan_to_num(Uy), np.nan_to_num(Ux), m, cvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    ap.add_argument("--indir")
    ap.add_argument("--pattern", default="stage_*_{ch}.tiff")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--outdir", default="dic_out")
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--half", type=int, default=16)
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--seed-step", type=int, default=64)
    ap.add_argument("--seed-r", type=int, default=128)
    ap.add_argument("--refine-r", type=int, default=8)
    ap.add_argument("--tstd-min", type=float, default=8.0)
    ap.add_argument("--cval-min", type=float, default=0.60)
    ap.add_argument("--peakgap", type=float, default=0.08)
    ap.add_argument("--rel-max", type=float, default=0.6)
    ap.add_argument("--incremental", action="store_true",
                    help="增量模式: 以相邻帧为参考做匹配, 位移累加(变形大/帧稀疏时推荐)")
    ap.add_argument("--strain-win", type=int, default=5,
                    help="应变窗边长(格): 局部最小二乘平面拟合求应变, 1=逐点差分(噪声大)")
    ap.add_argument("--nstage", type=int, default=-1)
    ap.add_argument("--smooth", type=float, default=1.0)
    a = ap.parse_args()

    # ---- 输入
    if a.zip:
        import zipfile
        import tempfile
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(a.zip).extractall(tmp)
        root = None
        for r, ds, fs in os.walk(tmp):
            if any(f.endswith(".tiff") for f in fs):
                root = r
                break
        indir = root
    else:
        indir = a.indir
    files = sorted(glob.glob(os.path.join(indir, a.pattern.format(ch=a.channel))))
    if not files:
        raise SystemExit(f"[ERR] 未找到 TIFF: {indir} pattern={a.pattern}")
    if a.nstage > 0:
        files = files[:a.nstage]

    os.makedirs(a.outdir, exist_ok=True)
    print(f"[INFO] {len(files)} 帧, ds={a.ds} half={a.half} "
          f"seed-step={a.seed_step} seed-r={a.seed_r} refine-r={a.refine_r}")

    ref = load_ds(files[0], a.ds)
    h, w = ref.shape
    pad0 = a.half + a.refine_r + 2
    gys = np.arange(pad0, h - pad0, a.step)
    gxs = np.arange(pad0, w - pad0, a.step)

    def do_strain(Uy, Ux, m):
        """位移场 -> 应变。默认用**局部窗最小二乘平面拟合**(strain window),
        比逐点差分稳健得多: 大位移噪声不会被微分放大成虚假大应变。"""
        if a.strain_win <= 1:
            def grad(U):
                Uc = np.where(m, U, 0.0)
                Uc = gaussian_filter(Uc, a.smooth)
                gx_ = np.gradient(Uc, gxs, axis=1)
                gy_ = np.gradient(Uc, gys, axis=0)
                return gx_, gy_
        else:
            k = a.strain_win
            off = k // 2
            def grad(U):
                Uc = np.where(m, U, np.nan)
                gy_ = np.full_like(Uc, np.nan)
                gx_ = np.full_like(Uc, np.nan)
                for iy in range(off, len(gys) - off):
                    for ix in range(off, len(gxs) - off):
                        blk = Uc[iy - off:iy + off + 1, ix - off:ix + off + 1]
                        yy = gys[iy - off:iy + off + 1]
                        xx = gxs[ix - off:ix + off + 1]
                        v = blk.ravel()
                        sel = np.isfinite(v)
                        if sel.sum() < max(4, k * k // 2):
                            continue
                        YY, XX = np.meshgrid(yy, xx, indexing="ij")
                        A = np.c_[np.ones(sel.sum()), XX.ravel()[sel],
                                  YY.ravel()[sel]]
                        try:
                            coef, *_ = np.linalg.lstsq(A, v[sel], rcond=None)
                        except np.linalg.LinAlgError:
                            continue
                        gx_[iy, ix] = coef[1]
                        gy_[iy, ix] = coef[2]
                return gx_, gy_
        gxx, _ = grad(Ux)
        _, gyy = grad(Uy)
        return np.nan_to_num(gxx), np.nan_to_num(gyy)

    # ---- 参考自身: 零位移
    z = np.zeros((len(gys), len(gxs)))
    np.savez(os.path.join(a.outdir, "stage_00000.npz"),
             exx=z, eyy=z.copy(), ux=z, uy=z.copy(),
             mask=np.ones_like(z, bool), xs=gxs, ys=gys,
             nok=np.ones_like(z, int))
    # 增量模式下 ref 逐帧推进
    ref_frame = ref
    acc_y = np.zeros((len(gys), len(gxs)))
    acc_x = np.zeros((len(gys), len(gxs)))

    for idx in range(1, len(files)):
        cur = load_ds(files[idx], a.ds)
        seeds = match_seed(ref_frame, cur, a.half, a.seed_step, a.seed_r,
                           a.tstd_min, 0.45, 0.04)
        if seeds is None:
            print(f"stage {idx:05d}: [FAIL] 粗配准无有效种子")
            continue
        sy, sx, Uys, Uxs, Rel = seeds
        sy, sx, Uys, Uxs, Rel, _ = filter_seeds(sy, sx, Uys, Uxs, Rel, a.rel_max)
        if len(sy) < 4:
            print(f"stage {idx:05d}: [FAIL] 筛选后种子不足({len(sy)})")
            continue
        uy0, ux0 = interp_seeds(sy, sx, Uys, Uxs, gys, gxs)
        dUy, dUx, m, cv = refine(ref_frame, cur, a.half, gys, gxs, uy0, ux0,
                                 a.refine_r, a.tstd_min, a.cval_min, a.peakgap)
        if a.incremental:
            # 注意: 传给 refine 的 uy0 必须是**相对于 ref_frame 的增量预测**,
            # 不能叠加 acc(否则搜索中心会被推到已变形位置, 与模板位置不匹配)。
            # 总位移 = 上一帧累积 + 本帧增量。
            Uy = np.where(m, acc_y + dUy, acc_y + uy0)
            Ux = np.where(m, acc_x + dUx, acc_x + ux0)
            acc_y = Uy.copy()
            acc_x = Ux.copy()
            ref_frame = cur                     # 下一帧以本帧为参考
        else:
            Uy, Ux = dUy, dUx
        frac = 100.0 * m.mean()
        exx, eyy = do_strain(Uy, Ux, m)
        exx = np.where(m, exx, np.nan)
        eyy = np.where(m, eyy, np.nan)
        np.savez(os.path.join(a.outdir, f"stage_{idx:05d}.npz"),
                 exx=exx, eyy=eyy, ux=Ux, uy=Uy, mask=m, xs=gxs, ys=gys,
                 nok=m.astype(int))
        print(f"stage {idx:05d}: 种子{len(sy)} 有效 {frac:5.1f}%  "
              f"Umed y={np.median(Uy[m]):+.1f} x={np.median(Ux[m]):+.1f} ds-px  "
              f"cval={np.median(cv[m]):.3f}  exx_std={np.nanstd(exx) * 100:.3f}%")

    print(f"[DONE] -> {a.outdir}")


def _glob(indir, pat):
    import glob as _g
    return sorted(_g.glob(os.path.join(indir, pat)))


import glob  # noqa: E402

if __name__ == "__main__":
    main()
