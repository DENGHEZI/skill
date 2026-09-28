# -*- coding: utf-8 -*-
"""
DIC 配套 P-s 曲线（荷载-滑移/荷载-位移曲线, 论文 Fig.9 风格）
数据来源(内置): 论文表3《剪切试验主要结果》10 个试件的 Pu/su/τu/破坏模式 (真实试验值)
也可喂入任意真实 DIC/力学曲线 CSV: 列 specimen,slip,load

用法:
  # 内置论文表3数据, 直接复现 Fig.9
  python dic_load_slip.py --out fig9.png

  # 真实曲线 CSV (长表: specimen,slip,load) —— 自动适配 24 条曲线 / 坐标轴 / 图例
  python dic_load_slip.py --csv my_curves.csv --out p_s.png --legend-cols 4

  # 按试件前缀分组(p-/z-)画双栏
  python dic_load_slip.py --csv my_curves.csv --group pz --out p_s.png
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"   # 论文风格: 刻度朝内
plt.rcParams["ytick.direction"] = "in"
# SVG 导出时保留真实文本(而非路径), 便于 Illustrator/Inkscape 里改字改色
plt.rcParams["svg.fonttype"] = "none"

# 论文表3 (ECC-煤矸石混凝土直剪试验, 10 试件): 编号, Pu/kN, su/mm, 破坏模式
TABLE3 = [
    ("CE-2-20-0",      152.1, 1.78, "brittle"),
    ("NE-2-20-0",      178.1, 1.94, "brittle"),
    ("CE-2-20-0-r",    302.2, 2.98, "ductile"),
    ("CE-2-20-0-r(8)", 233.5, 2.13, "ductile"),
    ("CE-3-20-0-r",    374.3, 4.02, "ductile"),
    ("CE-2-30-0-r",    332.1, 3.05, "ductile"),
    ("CE-2-20-15-r",   305.0, 2.99, "ductile"),
    ("CE-2-20-30-r",   313.6, 2.75, "ductile"),
    ("EC-2-20-0-r",    285.6, 2.25, "ductile"),
    ("CE-0-0-0-r",     195.1, 2.04, "ductile"),
]

COLORS = (list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors))[:40]
MARKERS = ["o", "*", "s", "D", "v", "^", "<", ">", "P", "X", "h", "d"]


def _save(fig, out):
    """按扩展名保存。.svg = 矢量且文字可编辑(推荐二次修改); 其余按位图 dpi=200。"""
    ext = os.path.splitext(out)[1].lower()
    if ext == ".svg":
        fig.savefig(out, format="svg", bbox_inches="tight")
    elif ext in (".pdf", ".eps", ".ps"):
        fig.savefig(out, bbox_inches="tight")
    else:
        fig.savefig(out, dpi=200, bbox_inches="tight")


def synth_curve(pu, su, mode, n=140, seed=3):
    """按论文曲线形态合成: 上升段光滑 -> 峰值 -> 脆性竖直跌落 / 延性软化平台"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    rise = np.sin(t * np.pi / 2) ** 1.35
    s = t * su
    p = pu * rise
    if mode == "brittle":
        s = np.concatenate([s, np.full(12, su)])
        p = np.concatenate([p, np.linspace(pu, 0.08 * pu, 12)])
    else:
        t2 = np.linspace(0, 1, n // 2)
        s2 = su + 4.5 * su * t2
        p2 = pu * (0.78 * np.exp(-2.2 * t2) + 0.22) + rng.normal(0, 1.5, t2.size)
        p2 = np.clip(p2, 0.12 * pu, 0.95 * pu)
        s = np.concatenate([s, s2]); p = np.concatenate([p, p2])
    p = p + rng.normal(0, pu * 0.004, p.size)
    return s, p


def _axes_style(ax, xlabel, ylabel, xlim, ylim):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def plot_paper_style(series, out, xlabel="s/mm", ylabel="P/kN",
                     xlim=None, ylim=None, legend_ncol=2, title=None,
                     legend_loc="top"):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for i, (name, s, p) in enumerate(series):
        ax.plot(s, p, "-", color=COLORS[i % len(COLORS)], lw=1.1, alpha=0.9,
                marker=MARKERS[i % len(MARKERS)], ms=3.0, markevery=8,
                label=name)
    _axes_style(ax, xlabel, ylabel, xlim, ylim)
    if legend_loc == "bottom":
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                        ncol=legend_ncol, fontsize=8.5, framealpha=1.0,
                        borderpad=0.5, columnspacing=1.2, handlelength=2.0)
    else:
        leg = ax.legend(loc="upper center", ncol=legend_ncol, fontsize=8.5,
                        framealpha=1.0, borderpad=0.5, columnspacing=1.2,
                        handlelength=2.0)
    leg.get_frame().set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)
    print(f"[OK] P-s 曲线 -> {out}  ({len(series)} 条)")


def plot_grouped(series, out, prefix_map, xlabel="s/mm", ylabel="P/kN",
                 xlim=None, ylim=None):
    """按试件名前缀(如 p-/z-) 分栏"""
    groups = {}
    for name, s, p in series:
        pre = next((k for k in prefix_map if name.startswith(k)), "其他")
        groups.setdefault(pre, []).append((name, s, p))
    keys = list(prefix_map.keys())
    keys = [k for k in keys if k in groups] + [k for k in groups if k not in keys]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, pre in zip(axes, keys):
        for i, (name, s, p) in enumerate(groups[pre]):
            ax.plot(s, p, "-", color=COLORS[i % len(COLORS)], lw=1.1, alpha=0.9,
                    marker=MARKERS[i % len(MARKERS)], ms=3.0, markevery=8, label=name)
        _axes_style(ax, xlabel, ylabel, xlim, ylim)
        ax.set_title(pre + " 系列", fontsize=12)
        ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)
    print(f"[OK] 分组 P-s 曲线 -> {out}  ({n} 组)")


def plot_summary(series, out):
    """峰值荷载/峰值位移汇总柱状图(论文 Fig.10/11 风格), p-/z- 前缀双色"""
    names = [n for n, _, _ in series]
    pus = [float(p.max()) for _, _, p in series]
    sus = [float(s[np.argmax(p)]) for _, s, p in series]
    cols = ["#2980b9" if n.startswith("p-") else "#e67e22" for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 0.42 * len(names)), 5.2))
    for ax, vals, ylab in ((axes[0], pus, "峰值荷载 $P_u$/kN"), (axes[1], sus, "峰值位移 $s_u$/mm")):
        ax.bar(range(len(names)), vals, color=cols, edgecolor="black", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=12)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.grid(axis="y", ls="--", alpha=0.35)
    import matplotlib.patches as mpatches
    axes[0].legend(handles=[mpatches.Patch(color="#2980b9", label="p- 系列"),
                            mpatches.Patch(color="#e67e22", label="z- 系列")],
                   fontsize=9, framealpha=1.0)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)
    print(f"[OK] 峰值汇总柱状图 -> {out}")


def load_csv(fp):
    rows = list(csv.DictReader(open(fp, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"[ERR] CSV 为空: {fp}")
    cols = rows[0].keys()
    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None
    k_s = pick(["slip", "disp", "disp_avg", "displacement", "s", "位移", "位移平均值"])
    k_p = pick(["load", "load_kN", "P", "p", "force", "荷载", "荷载KN", "荷载(kN)"])
    if not k_s or not k_p:
        raise SystemExit(f"[ERR] CSV 缺少位移/荷载列, 现有列: {list(cols)}")
    by = {}
    order = []
    for r in rows:
        k = str(r["specimen"]).strip()
        if k not in by:
            by[k] = ([], []); order.append(k)
        try:
            by[k][0].append(float(r[k_s]))
            by[k][1].append(float(r[k_p]))
        except (ValueError, KeyError, TypeError):
            continue
    return [(k, np.array(by[k][0]), np.array(by[k][1])) for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="dic_load_slip.png")
    ap.add_argument("--xlim", nargs=2, type=float, default=None)
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    ap.add_argument("--legend-cols", type=int, default=2)
    ap.add_argument("--legend-loc", default="top", choices=["top", "bottom"])
    ap.add_argument("--summary", action="store_true", help="输出峰值荷载/位移汇总柱状图")
    ap.add_argument("--group", default=None, help="按前缀分组, 如 pz 表示按 p-/z- 分栏")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    if a.csv:
        series = load_csv(a.csv)
        if a.summary:
            plot_summary(series, a.out)
        elif a.group:
            prefix_map = {c + "-": c + "-" for c in a.group if c != "0"}
            plot_grouped(series, a.out, prefix_map, xlim=a.xlim, ylim=a.ylim)
        else:
            plot_paper_style(series, a.out, xlim=a.xlim, ylim=a.ylim,
                             legend_ncol=a.legend_cols, title=a.title,
                             legend_loc=a.legend_loc)
    else:
        series = []
        for i, (name, pu, su, mode) in enumerate(TABLE3):
            s, p = synth_curve(pu, su, mode, seed=3 + i)
            series.append((name, s, p))
        plot_paper_style(series, a.out, xlim=(0, 15), ylim=(0, 520),
                         legend_ncol=2, title="荷载-位移曲线 (论文表3)")


if __name__ == "__main__":
    main()
