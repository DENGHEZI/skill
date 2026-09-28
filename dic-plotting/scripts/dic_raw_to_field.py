# -*- coding: utf-8 -*-
"""
dic_raw_to_field.py — 原始 DIC 散斑照片(TIFF 序列) -> 应变场(Exx/Eyy) 稳健计算
============================================================================
适用：用户直接给一批按阶段命名的散斑 TIFF（如 stage_00000_0.tiff ...）。
算法：固定参考图模板 + 逐帧小搜索(±seqr ds) + ZNCC，累积位移；大子集(half ds)
      跨越散斑周期使模板唯一；峰值比判据(主-次峰差<peakgap 判歧义剔除)自动把
      周期假匹配区掩膜为白，仅保留真实变形(裂缝)区。
输出：outdir/stage_{idx:05d}.npz  (exx, eyy, ux, uy, mask, xs, ys, nok)
依赖：numpy, scipy, tifffile, matplotlib(仅 rcParams, 可省)

典型：
  python dic_raw_to_field.py --zip D:/Images.zip --outdir ./dic_fields --channel 0
  python dic_raw_to_field.py --indir ./Images --outdir ./dic_fields --nstage 50
"""
import os, argparse, zipfile, glob
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import sliding_window_view

try:
    import tifffile
except ImportError:
    tifffile = None


def load_tiff(fp):
    return tifffile.imread(fp).astype(np.float32)


def load_ds_path(fp, ds):
    a = load_tiff(fp)
    return a[::ds, ::ds]


def match_point(T, tstd, tsum, cur, cyc, cxc, SEQ_R, HALF, cval_min, peakgap):
    """在 cur 中以 (cyc,cxc) 为预测中心, 用固定模板 T 做局部 ZNCC, 返回增量位移(ds-px)"""
    h, w = cur.shape
    ky, kx = T.shape
    pad = SEQ_R + HALF
    y0, y1 = cyc - pad, cyc + pad + 1
    x0, x1 = cxc - pad, cxc + pad + 1
    if y0 < 0 or y1 > h or x0 < 0 or x1 > w:
        return None
    if not (HALF <= cyc <= h - HALF and HALF <= cxc <= w - HALF):
        return None
    S = cur[y0:y1, x0:x1].astype(np.float64)
    corr = fftconvolve(S, T[::-1, ::-1], 'valid')
    patches = sliding_window_view(S, (ky, kx))
    Smean = patches.mean(axis=(2, 3))
    Sstd = patches.std(axis=(2, 3))
    n = ky * kx
    num = corr - Smean * tsum
    denom = Sstd * tstd * n
    C = num / np.where(denom > 1e-6, denom, 1e-6)
    C = np.clip(C, -1, 1)
    peak = np.unravel_index(np.argmax(C), C.shape)
    cval = C[peak]
    if cval < cval_min:
        return None
    C2 = C.copy()
    yy, xx = peak
    C2[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] = -9
    if cval - C2.max() < peakgap:        # 多峰歧义(周期散斑残余)
        return None
    def subp(c, p):
        if 0 < p < c.shape[0] - 1:
            d = (c[p + 1] - c[p - 1]) / 2.0
            e = (c[p + 1] + c[p - 1] - 2 * c[p])
            if e != 0:
                return p - d / e / 2.0
        return float(p)
    dy = (peak[0] - SEQ_R) + (subp(C[:, peak[1]], peak[0]) - peak[0])
    dx = (peak[1] - SEQ_R) + (subp(C[peak[0], :], peak[1]) - peak[1])
    return dy, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="原始 TIFF 压缩包(可选)")
    ap.add_argument("--indir", default=None, help="已解压的 TIFF 目录(与 --zip 二选一)")
    ap.add_argument("--pattern", default="stage_*_{ch}.tiff", help="TIFF 命名模式({ch}被--channel替换)")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--outdir", default="./dic_fields")
    ap.add_argument("--nstage", type=int, default=-1, help="处理阶段数(-1=全部)")
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--half", type=int, default=16, help="子集半宽(ds): >散斑周期(95px)使模板唯一")
    ap.add_argument("--seqr", type=int, default=6, help="逐帧搜索半宽(ds): ±24px")
    ap.add_argument("--step", type=int, default=12, help="网格步长(ds)")
    ap.add_argument("--tstd-min", type=float, default=8.0)
    ap.add_argument("--cval-min", type=float, default=0.60)
    ap.add_argument("--peakgap", type=float, default=0.08)
    ap.add_argument("--mask-frac", type=float, default=0.40)
    a = ap.parse_args()

    # 解析输入文件
    if a.zip:
        zf = zipfile.ZipFile(a.zip)
        names = sorted([n for n in zf.namelist()
                        if n.endswith(".tiff") and a.pattern.format(ch=a.channel).split("*")[0] in n
                        and ("_%d.tiff" % a.channel) in n])
        work = a.indir or os.path.join(a.outdir, "_zip_extract")
        os.makedirs(work, exist_ok=True)
        files = []
        for n in names:
            tgt = os.path.join(work, os.path.basename(n))
            if not os.path.exists(tgt):
                with open(tgt, "wb") as f:
                    f.write(zf.read(n))
            files.append(tgt)
    else:
        if not a.indir:
            print("[ERR] 需提供 --zip 或 --indir"); return
        files = sorted(glob.glob(os.path.join(a.indir, a.pattern.format(ch=a.channel))))
    if not files:
        print("[ERR] 未匹配到 TIFF 文件"); return
    nstage = a.nstage if a.nstage > 0 else len(files)
    nstage = min(nstage, len(files))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"[INFO] {len(files)} 个 TIFF, 处理前 {nstage} 阶段, channel={a.channel} ds={a.ds}")

    HALF, SEQ_R, STEP = a.half, a.seqr, a.step
    ref = load_ds_path(files[0], a.ds)
    h, w = ref.shape
    margin = HALF + SEQ_R + 2
    ys = np.arange(margin, h - margin, STEP)
    xs = np.arange(margin, w - margin, STEP)
    ngy, ngx = len(ys), len(xs)

    # 预计算参考模板
    templates, tstds, tsums, gpts = [], [], [], []
    for iy, gy in enumerate(ys):
        for ix, gx in enumerate(xs):
            T = ref[gy - HALF:gy + HALF + 1, gx - HALF:gx + HALF + 1]
            ts = T.std()
            templates.append(T if ts >= a.tstd_min else None)
            tstds.append(ts); tsums.append(T.sum())
            gpts.append((iy, ix, gy, gx))

    Ux = np.zeros((ngy, ngx)); Uy = np.zeros((ngy, ngx))
    nok = np.zeros((ngy, ngx), dtype=int)

    def save_stage(idx):
        m = nok >= max(2, int(a.mask_frac * max(1, idx)))
        def strain(U):
            Uc = np.where(m, U, 0.0)
            Uc = gaussian_filter(Uc, 1.0)
            gx = np.gradient(Uc, xs, axis=1)
            gy = np.gradient(Uc, ys, axis=0)
            return gx, gy
        exx, _ = strain(Ux); _, eyy = strain(Uy)
        exx[~m] = np.nan; eyy[~m] = np.nan
        np.savez(os.path.join(a.outdir, f"stage_{idx:05d}.npz"),
                 exx=exx, eyy=eyy, ux=Ux, uy=Uy, mask=m, xs=xs, ys=ys, nok=nok)
        return m

    save_stage(0)
    for idx in range(1, nstage):
        cur = load_ds_path(files[idx], a.ds)
        for k, (iy, ix, gy, gx) in enumerate(gpts):
            T = templates[k]
            if T is None:
                continue
            cyc = int(round(gy + Uy[iy, ix]))
            cxc = int(round(gx + Ux[iy, ix]))
            r = match_point(T, tstds[k], tsums[k], cur, cyc, cxc, SEQ_R, HALF,
                            a.cval_min, a.peakgap)
            if r is None:
                continue
            dy, dx = r
            Uy[iy, ix] += dy
            Ux[iy, ix] += dx
            nok[iy, ix] += 1
        m = save_stage(idx)
        v = np.hypot(Ux[m], Uy[m])
        print(f"stage {idx:02d}: valid={int(m.sum())} Umed={np.nanmedian(v):.2f} "
              f"Umax={np.nanmax(np.hypot(Ux,Uy)):.1f}ds-px", flush=True)
    print(f"[DONE] 应变场已写出 -> {a.outdir}")


if __name__ == "__main__":
    main()
