# -*- coding: utf-8 -*-
"""
DIC 图片检测 —— 从 DIC 应变云图/散斑图中检测应变集中带与开裂区
用途:
  1) QC 新生成的 DIC 云图: 检测红色高应变集中区(面积占比/位置/条带数)
  2) 检测论文/他人 DIC 图片的应变集中区, 提取开裂条带位置
  3) 与 skill 内置论文参考图(paper_fig12_strain_field.png)对比作图风格

原理: jet/rainbow 色标中 高应变=红/橙, 用 HSV 色相阈值分割 + 连通域标记
用法:
  python dic_image_detect.py 图.png --out annotated.png
  python dic_image_detect.py 图.png --hsv 5 45 --min-area 0.3   # 收紧红/橙阈值(色相度), 最小连通域0.3%
"""
import argparse, json, sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from matplotlib.colors import rgb_to_hsv


def detect(fp, hue_red=10.0, hue_orange=48.0, sat_min=0.45, val_min=0.35,
            min_area_pct=0.15, diverging=False, hue_blue=230.0):
    img = Image.open(fp).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(arr)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hr, ho = hue_red / 360.0, hue_orange / 360.0
    # 红(绕环) + 橙/红橙 = 高应变集中
    mask = (((H <= hr) | (H >= 1 - hr)) | ((H >= hr) & (H <= ho))) & (S > sat_min) & (V > val_min)
    if diverging:
        # 双向色标(位移场 Ux/Uy): 红=正极值、蓝=负极值 均为"变形集中", 一并标记
        hb = hue_blue / 360.0
        blue = ((H >= hb - 0.06) & (H <= hb + 0.06)) & (S > sat_min) & (V > val_min)
        mask = mask | blue

    h, w = mask.shape
    total = h * w
    ratio = mask.sum() / total * 100

    # 连通域 (4邻域 BFS, 降采样提速)
    ds = 2 if max(h, w) > 1200 else 1
    m = mask[::ds, ::ds]
    lab = np.zeros(m.shape, dtype=np.int32); cur = 0
    comps = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] and lab[i, j] == 0:
                cur += 1
                q = deque([(i, j)]); lab[i, j] = cur; pix = []
                while q:
                    y, x = q.popleft(); pix.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = cur; q.append((yy, xx))
                if len(pix) >= min_area_pct * 0.01 * m.size:
                    ys = [p[0] for p in pix]; xs = [p[1] for p in pix]
                    comps.append({
                        "bbox_px": [min(xs)*ds, min(ys)*ds, max(xs)*ds, max(ys)*ds],
                        "area_pct": round(len(pix) / m.size * 100, 2),
                        "centroid_px": [int(np.mean(xs))*ds, int(np.mean(ys))*ds],
                    })
    comps.sort(key=lambda c: -c["area_pct"])
    return img, mask, {"image": fp, "size_px": [w, h],
                       "high_strain_ratio_pct": round(ratio, 2),
                       "n_bands": len(comps), "bands": comps[:8]}


def annotate(img, mask, stats, out):
    vis = img.convert("RGB").copy()
    dr = ImageDraw.Draw(vis)
    for c in stats["bands"]:
        x1, y1, x2, y2 = c["bbox_px"]
        dr.rectangle([x1, y1, x2, y2], outline=(0, 200, 255), width=3)
        dr.text((x1 + 4, max(2, y1 - 14)), f"{c['area_pct']}%", fill=(0, 200, 255))
    vis.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hue-red", type=float, default=10.0)
    ap.add_argument("--hue-orange", type=float, default=48.0)
    ap.add_argument("--min-area", type=float, default=0.15)
    ap.add_argument("--diverging", action="store_true",
                    help="双向色标(位移场 Ux/Uy): 同时标记红色正极值与蓝色负极值")
    a = ap.parse_args()
    out = a.out or a.image.rsplit(".", 1)[0] + "_detect.png"

    img, mask, stats = detect(a.image, a.hue_red, a.hue_orange,
                              min_area_pct=a.min_area, diverging=a.diverging)
    annotate(img, mask, stats, out)
    stats["annotated"] = out
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
