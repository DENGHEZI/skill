---
name: dic-plotting
description: DIC（数字图像相关）科研作图技能：应变场云图、荷载-滑移(P-s)曲线、位移场(Ux/Uy)双向云图、2x2全场网格、DIC 图片检测（应变集中带/开裂区识别）。覆盖土木界面粘结(ECC/煤矸石、UHPC-NC)与土木结构损伤识别/振动/抗震/SHM 两类场景。触发词：DIC作图、DIC云图、应变场、位移场、荷载滑移曲线、P-s曲线、散斑应变、DIC图片检测、全场、损伤识别。
description_zh: DIC数字图像相关作图与图片检测（应变场/位移场云图/P-s曲线/开裂区识别），覆盖界面粘结与土木结构损伤识别
description_en: DIC plotting & image detection — strain/displacement field contours, load-slip curves, 2x2 full-field grids, strain-band & crack detection; covers bond-interface and civil structural (SHM/seismic) DIC
version: 1.3.3
agent_created: true
---

# DIC 作图

面向土木/材料 DIC 结果作图与图片检测，覆盖两类场景：
- **A. 界面粘结/直剪**（ECC-煤矸石混凝土、UHPC-NC 复合）：应变场 exx 用 VIC rainbow
- **B. 土木结构损伤识别/振动/抗震/SHM**：位移场 Ux/Uy（有符号，双向色标）、应变场 εxx/γxy、
  2×2 全场网格、开裂位置叠加

作图规范由真实 DIC 论文图"训练"反推（详见 `references/dic-plotting-conventions.md`），
并附论文表3实测数据与 `data/sample_civil_beam/` 土木梁样例，用于校准和验证。


## 何时用

- 要画 DIC 应变场云图（多加载阶段对比，VIC 软件风格）
- 要画界面直剪的荷载-滑移（P-s）曲线 / 荷载-位移曲线
- 要从一张 DIC 云图图片中**检测**应变集中带 / 开裂区（面积占比、条带位置）
- 要按论文风格批量出图（黑框内刻度、双列顶部图例、e 计数法色标）

## 快速开始

```bash
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
S="C:/Users/Lenovo/.workbuddy/skills/dic-plotting/scripts"
D="C:/Users/Lenovo/.workbuddy/skills/dic-plotting/data/sample_civil_beam"

# 1) 界面粘结合成演示(默认, 已按论文规律训练: 初裂均匀→界面集中→峰值贯穿)
"$PY" "$S/dic_strain_field.py" --specimen CE-2-20-0-r --stages 0.20,0.65,1.00 --out fig12.png

# 2) 真实 DIC 应变导出数据（CSV 三列表头 x,y,exx）→ 单阶段云图
"$PY" "$S/dic_strain_field.py" --csv my_dic_data.csv --out fig12_real.png

# 3) 土木结构: 三点弯曲梁 位移场 Ux（双向色标 + 开裂叠加）
"$PY" "$S/dic_strain_field.py" --sample beam --quantity ux --stages 0.30,0.70,1.00 --out beam_ux.png

# 4) 土木结构: 2x2 全场网格 (Ux,Uy,εxx,γxy) —— 损伤识别典型成图
"$PY" "$S/dic_strain_field.py" --sample beam --grid ux,uy,exx,exy --out beam_grid.png

# 5) 用检测出的开裂折线叠加 exx 云图
"$PY" "$S/dic_strain_field.py" --sample beam --quantity exx --crack "$D/crack.csv" --out beam_crack.png

# 6) P-s 曲线：内置论文表3实测 Pu/su，直接复现 Fig.9 风格
"$PY" "$S/dic_load_slip.py" --out fig9.png

# 7) DIC 图片检测：应变场识别红色高应变集中带；位移场加 --diverging 同时标红/蓝极值
"$PY" "$S/dic_image_detect.py" 某云图.png --out 检测结果.png
"$PY" "$S/dic_image_detect.py" beam_ux.png --diverging --out beam_ux_detect.png

# 8) 荷载分级 × 应变分量 双行干净云图（论文 Fig.7/8 风格, 去散斑/黑点）
#    8a) 版式示意(合成, 用于对齐论文样式): 列=荷载级别, 行=Exx/Eyy
"$PY" "$S/dic_strain_field.py" --grid-style --levels 0,10,20,30,40,49 --rows exx,eyy --out grid_exx_eyy.png
#    8b) 真实 DIC 导出 -> 同款干净网格(Exx/Eyy 等): 目录内放 stage_XXXXX.npz(键 exx,eyy,mask,xs,ys)
"$PY" "$S/dic_strain_field.py" --real-grid ./dic_fields --levels 0,10,20,30,40,49 \
      --rows exx,eyy --col-labels "0%,20%,41%,61%,82%,100%" --out real_grid.png
#        --col-labels 可换为实际荷载(如 0,5,10,15,20kN); 真实图自动叠加试件轮廓(各阶段并集)

# 9) 【端到端】原始散斑 TIFF 序列 -> 应变场(自动写 stage_XXXXX.npz), 供 8b 使用
#    9a) 直接给已解压目录
"$PY" "$S/dic_raw_to_field.py" --indir ./Images --outdir ./dic_fields --channel 0 --nstage 50
#    9b) 或直接给压缩包(自动解压)
"$PY" "$S/dic_raw_to_field.py" --zip D:/Images.zip --outdir ./dic_fields --channel 0
#    9c) 关键参数: --half 子集半宽(需>散斑周期, 默认16ds≈132px) --seqr 逐帧搜索半宽(默认6ds)
# 9c) 关键参数: --half 子集半宽(需>散斑周期, 默认16ds≈132px) --seqr 逐帧搜索半宽(默认6ds)
#        --ds 降采样(默认4) --step 网格步长(默认12) --peakgap 峰值比判据(默认0.08)

# 10) 【期刊 Fig.7/8 风格·最终版】荷载进程(列) × Exx/Eyy(行) 干净云图(中心绿=0 双向彩虹)
"$PY" "$S/dic_ref_grid.py" --fields ./dic_fields \
      --stages 0,10,20,30,40,49 --labels "0,20%,40%,60%,80%,100%" \
      --rows exx,eyy --out ref_exx_eyy.png
#      ★默认 --tex-alpha 0 = 纯色场: 0% 面板为整块实心绿、全场零黑点, 与论文 Fig.7/8 完全一致
#      裁剪框默认取各阶段 mask 并集包围盒; 可 --crop x0,x1,y0,y1 手动; --sigma/--minsup 控平滑
#      重色只在开裂/损伤区: --dilate 3 把裂缝核心连成连续重色带, 区外仍绿底(详见 A4)
#      可选叠极淡试件纹理(仍无清晰黑点): --tex stage_00000_0.tiff --tex-alpha 0.13 --tex-ds 4
#      期刊试件样式(钢棒+标注): 加 --steel-bar ; 去糙默认开(--smooth 1.2 + bicubic), 默认 dpi=600
```

### A2. 荷载分级 × 应变分量 双行网格（论文 Fig.7/8 风格, 去散斑/黑点）

- `--grid-style`：合成示意。列=荷载级别，行=应变分量（`--rows exx,eyy` 可多选，也支持 `ux,uy,exy`）。
  每个 cell 用 VIC rainbow（应变）或 VIC diverging（位移）；**无效区 `np.nan` → 白色**
  （`cmap.set_bad("white")`），因此出图绝无散斑黑点。`--col-labels` 可把列改成"0%Pu"等可读标签。
- `--real-grid <dir>`：吃真实 DIC 导出数据。目录内每阶段一个 `stage_{idx:05d}.npz`，
  键 `exx/eyy/ux/uy`（任意子集）+ `mask`(bool) + `xs/ys`。按 `--levels` 选列、`--rows` 选行，
  **每行统一 clim（98 百分位裁剪离群）、每行独立 colorbar（e 计数法）**，顶部列标签、左侧行标签；
  自动以各阶段 mask 并集作**浅灰试件轮廓**叠加（解决"裂缝区之外全白、缺少试件边界"的观感问题）。
- 适用范围：界面粘结/直剪 Exx 云图、土木结构全场 Exx/Eyy/Ux/Uy 的"荷载分级对比"成图。

### A3. 原始散斑照片 → 应变场（端到端, 新增脚本 `dic_raw_to_field.py`）

把"散斑 TIFF 序列"直接算成 Exx/Eyy，再交给 A2 出图，形成完整闭环：

```bash
"$PY" "$S/dic_raw_to_field.py" --zip D:/Images.zip --outdir ./dic_fields --channel 0
"$PY" "$S/dic_strain_field.py" --real-grid ./dic_fields --levels 0,10,20,30,40,49 \
      --rows exx,eyy --out real_grid.png
```

算法要点（针对"散斑大尺度周期 → 全场面应变不可靠"的实测坑）：
- **固定参考图模板** + **逐帧小搜索(±seqr ds)** + ZNCC，逐帧累积位移，避免链式漂移；
- 大子集（`--half` ds，默认 16ds≈132px）**跨越散斑周期(95px)** 使模板全局唯一；
- **峰值比判据**（主-次峰差 `< --peakgap` 默认 0.08 判为歧义剔除）→ 整图周期区自动掩膜为白，
  仅打破周期性的**裂缝/损伤区**保留真实应变（实测：中心点 stage49 真实位移 (-36,-44)px，Cmax=0.93）。
- 输出 `stage_{idx:05d}.npz`（exx,eyy,ux,uy,mask,xs,ys,nok），`mask`=该阶段累计匹配率达标点。

> ⚠️ 真实图中若整片空白：说明该数据散斑周期性过强/变形超出搜索窗。请改用 VIC-2D/Ncorr
> 导出真实 Exx/Eyy 再走 `--real-grid`，或重喷更细、更高频、无周期性的随机散斑后重跑本脚本。

### A4. 期刊 Fig.7/8 风格网格（`dic_ref_grid.py`，中心绿=0 双向彩虹，最贴近论文版式）

按真实论文图（Structures 92 (2026) 112808 Zhong et al.，图 7/8）**逐项反推**的版式，用于
"荷载进程 × 应变分量"对比图。与 A2 的区别见下：

| 要素 | A2（`--grid-style`/`--real-grid`） | A4（`dic_ref_grid.py`，论文风格） |
|---|---|---|
| 色标 | VIC rainbow（单向） | **diverging rainbow，中心绿=0**（紫-蓝-青-绿-黄-橙-红）|
| 无效区 | NaN → **白** | **按 0 显示 → 绿底**（试件成整块绿，最像论文 0% 面板）|
| colorbar | 每行**竖直**、右侧、e 计数法 | 每行**横向**、行下方、右侧标注 `Exx/%` |
| 列标签 | 顶部 | **面板下方**（0,20%,40%,60%,80%,100%）|
| 稀疏场 | 原样 | **归一化卷积平滑**（--sigma/--minsup）读成连续条带 |
| 裁剪 | 全图 | 默认取 mask 并集**包围盒**（去空白背景）|
| 黑点 | 无（白底）| 无（**绿底，零黑点**）；可选 `--tex-alpha` 叠极淡底纹（默认 0）|

```bash
"$PY" "$S/dic_ref_grid.py" --fields ./dic_fields \
      --stages 0,10,20,30,40,49 --rows exx,eyy \
      --crop 380,800,0,815 --sigma 3 --minsup 0.05 --pct 90 --out ref_exx_eyy.png
```
- `--pct`（默认 90）控制每行色标上限百分位；损伤区离群大值用 `--pct 85~92` 避免整幅饱和。
- 彩色块只出现在**实测区**（裂缝/损伤区），其余为绿底 → 视觉即"应变集中带随荷载增强"。
- 若轮廓不是矩形（试件形状不规则），后续可加 `--mask-shape` 用并集掩膜裁形（暂用矩形）。

#### ★ 期刊试件样式（钢棒 + 标注）+ 去糙（v1.3.3 新增）
还原 Structures 92 (2026) 112808 图 7/8 的"试件块 + 加载端钢棒 + Loading end/Steel bar 标注"观感：

```bash
"$PY" "$S/dic_ref_grid.py" --fields ./dic_fields \
      --stages 0,10,20,30,40,49 --rows exx,eyy --steel-bar --out ref_exx_eyy.png
```
- **`--steel-bar`**：每个面板上下各叠一段灰色钢棒（加载端），首列下方加 "Loading end" / "Steel bar" 标注，
  与论文试件样式一致（钢棒在 `transAxes`，与数据坐标解耦，任何裁剪框都对齐）。
- **去糙（默认开）**：底层应变场仅数十格分辨率，放大显方块 → 渲染改用 **bicubic 插值** +
  场切片后 **`--smooth 1.2` 轻量高斯**（网格 cell）消除方块糙感；`--smooth 0` 可关。
- **像素（默认 dpi=600）**：期刊级分辨率；要更小文件用 `--dpi 300`。
- 配合 `--dilate 3` 让裂缝区连成连续重色带、区外仍绿底（见上节"重色只在开裂处"）。

#### ★ 最终定版：纯色场（零黑点）＝ 默认
用户多轮迭代后的结论：**保持论文观感 + 黑点全除** ⇒ 默认 `--tex-alpha 0`（纯色场）。

- **表现**：未测/低置信区按 0 显示 → **整块实心绿**；0% 面板就是一整块绿（与论文 Fig.7/8
  的 0% 面板完全一致），场上**没有任何散斑点/黑点**。
- **为什么不是"叠试件照片"**：本样本的"试件"本身就是约 1.7 万个圆点组成的散斑簇；只要把照片
  叠上去（哪怕 alpha 0.13、先柔化），绿底上仍会浮出一圈圈**深绿圆点** → 仍被视为"没除干净"。
  故正确做法是**完全不叠照片**，让应变场自己说话。
- **可选极淡纹理**（仅当确需"试件感"且用户明确接受底纹）：`--tex <照片> --tex-alpha 0.13 --tex-ds 4`
  —— 会先把照片 `gaussian_filter(2.2)` 柔化成底纹，再低透明度叠加；**不是默认**。
- 试件几何靠**裁剪框**（`--crop`）界定为矩形，与论文试件块一致。

#### ★ 让应变场"更连续" + 重色只在开裂处（用户最终要求）
论文 Fig.7/8 的要点是：**重颜色（红/蓝）只落在开裂/损伤区，其余低应变区是浅绿**。
实测数据只有 14–23% 被可靠追踪（即"打破周期性的裂缝区"，见数据质量须知），其余判据剔除。
因此正确做法是**只在裂缝区着色 + 连续化，绝不对全试件外推**：

1. **平滑条带（保真，默认先看）**：加大 `--sigma`（如 3→7）让裂缝区实测点扩成平滑宽条带，
   未测区仍绿底（局部条带 + 绿底，最贴近论文"局部高应变带 + 整体低应变"观感）。
   ```bash
   "$PY" "$S/dic_ref_grid.py" --fields ./dic_fields --rows exx,eyy --sigma 7 --out smooth.png
   ```
2. **裂缝区连成连续重色带（首选，推荐）**：`--dilate N`（如 3），从已着色的裂缝核心向外
   **膨胀 N 格**并在裂缝区内用最近邻连续填充 → 重色带沿裂缝连成片，区外一律绿底。
   实测核验：裂缝区内 57–86% 为非绿、区外 0%（重色严格只在开裂处）。
   ```bash
   "$PY" "$S/dic_ref_grid.py" --fields ./dic_fields --rows exx,eyy \
         --sigma 7 --minsup 0.04 --dilate 3 --out crack.png
   ```
   - ⚠️ 切勿用"全试件外推"：曾试 `--inpaint`（把未测区最近邻外推填满全试件），会把裂缝高应变
     **外溢到非开裂区**，违背"重色只在开裂处"，已废弃。
   - `--sigma` 控条带粗细、`--dilate` 控裂缝带连通程度，可反复调。

#### 去黑点（散斑）原则
所有网格/全场面函数均把**无效/低置信区设为 NaN**；A4 进一步把 NaN **按 0 渲染成绿底**，
因此 A4 出图**结构上不可能出现黑点**（绿是最低值，不是黑）。若输入是原始散斑照片，
务必先经 DIC 计算得到应变场再作图——云图显示的是应变场本身，**天然不含散斑黑点**。
如仍有"黑点"观感，几乎都是**叠了试件照片纹理**所致 → 去掉 `--tex` / 设 `--tex-alpha 0`。

#### 数据质量须知（实测踩坑）
- 实测一批原始散斑 TIFF（4096×4504，白底+约 1.7 万个 ~11px 随机黑点，覆盖率 25%）。
  探针发现该类散斑在**大尺度(~200px)存在像素级周期重复**（ZNCC 次峰 C2=1.0 与主峰 Cmax=1.0 重合）。
  后果：**整图全场面应变 DIC 不可靠**——相关面出现多个等大假峰，位移被周期假匹配主导，
  而真实刚体平移≈0；只有打破周期性的**裂缝/损伤区**能可靠追踪。
- 应对：① 时序 DIC 用"固定参考模板 + 峰值比判据"使整图周期区自动掩膜为白、仅裂缝区保留真实应变
  （见 A3）；② 若需全场面应变，请用 VIC-2D/Ncorr 等成熟软件导出 Exx/Eyy 再用 `--real-grid` 渲染；
  ③ 重新制样时喷更细、更高频、无周期性的随机散斑。

## 三种模式

### A. 全场云图 `dic_strain_field.py`
- 场景A（界面粘结，默认）：`--specimen` + `--stages`，输出应变场 exx（VIC rainbow）
  + 黑色键槽界面折线 + "部分开裂 应变释放"标注；`--keys 3`、`--mode ductile|brittle`
- 场景B（土木结构）：`--sample beam` 进入三点弯曲梁模式
  - `--quantity exx|ux|uy|exy`：应变场用 rainbow；位移场 Ux/Uy 用双向色标(蓝-白-红)，clim 对称
  - `--grid ux,uy,exx,exy`：2×2 全场网格，每幅独立 colorbar（损伤识别典型成图）
  - `--crack crack.csv`：叠加红色虚线开裂折线
- 真实数据：`--csv x,y,exx`（或 ux/uy/exx 列名均可，脚本按列名自适应）；`--npz`(键 x,y,exx*)
- 内置样例：`data/sample_civil_beam/`（beam_ux/uy/exx/exy.csv + crack.csv）

### B. P-s 曲线 `dic_load_slip.py`
- 无参 = 复现论文 Fig.9（表3 的 10 个试件实测 Pu/su 驱动，脆性竖直跌落 vs 延性软化平台自动区分）
- `--csv` 长表列名自动识别：specimen + slip/disp/disp_avg/位移 + load/load_kN/P/荷载 均可
- 多曲线适配：`--legend-cols 6 --legend-loc bottom`（图例移到下方不遮挡）、`--xlim/--ylim` 手动范围
- `--group pz`：按试件名前缀（p-/z-）分组画双栏
- `--summary`：输出峰值荷载 Pu / 峰值位移 su 汇总柱状图（论文 Fig.10/11 风格，p-蓝 z-橙双色）

### C. 图片检测 `dic_image_detect.py`
- 输入任意 DIC 云图 PNG/JPG → HSV 色相分割（红≈高应变、橙≈中高）
- 输出：高应变面积占比 %、条带数 n_bands、各条带 bbox/质心/面积、青框标注图
- **`--diverging`**：双向色标位移场(Ux/Uy)用 → 同时标记红色正极值与蓝色负极值（变形集中区）
- 用途：① 质检新生成云图 ② 反推他人图片的应变/位移集中位置 ③ 对比 assets/ 论文参考图
- 参数：`--hue-red 10 --hue-orange 48`（色相度）、`--min-area 0.15`（最小连通域%）、`--diverging`

## 数据要求

- 云图 CSV/npz：坐标 mm，应变无量纲（若给微应变自动 ÷1e6）
- P-s CSV：slip/mm、load/kN
- 依赖：numpy、matplotlib、pillow、**scipy、tifffile**（`dic_raw_to_field.py` 需要 scipy/tifffile）；venv 已装好，路径见上

## 验证（用论文图片）

- 参考图1：`assets/paper_fig12_strain_field.png`（论文图12，9试件×3阶段应变云图）
- 参考图2：`assets/paper_fig9_load_slip.png`（论文图9，10试件P-s曲线）
- 自检方式：先生成合成图，再对生成图跑 `dic_image_detect.py`，
  应得到"高应变面积占比随阶段递增、峰值阶段出现贯穿条带"——即论文所述规律。

## 出图规范速查

1. 云图：VIC rainbow 色标、每幅独立竖直 colorbar（5.3e+1 式刻度）、黑折线勾界面、阶段标签在下方
2. P-s：黑框内刻度、marker 每隔几点一个、双列顶部图例、轴标签 `s/mm`、`P/kN`
3. 阶段选取：未配筋试件 0.90/0.95/1.00Pu；配筋试件 初裂/0.65Pu/1.00Pu
4. 中文标注字体 Microsoft YaHei（脚本已内置 rcParams）
5. **期刊 Fig.7/8 风格（首选，见 A4）**：2 行(Exx/Eyy) × 6 列(0/20/40/60/80/100 %Pmax)；
   中心绿=0 的 diverging rainbow；**默认纯色场 `--tex-alpha 0`** → 试件成**整块绿底**、
   0% 面板为一整块绿（与论文一致）、**全场零黑点**；裂缝区彩色带随荷载增强；行下方横向 colorbar
   并标注 `Exx/%`；荷载标签置面板**下方**；无标题。**不要叠试件照片**（会把散斑点重新带回来）。
   参考图：`assets/paper_fig7_fig8_ref.jpg`（用户提供的真实论文页）。
