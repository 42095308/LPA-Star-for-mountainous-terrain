# 山地无人机动态航路规划实验工程

本工程用于基于 DEM、OSM 人群暴露风险、安全飞行走廊和分层航路网络，评估动态扰动下的无人机路径规划与增量重规划方法。当前默认场景是华山，但流程已经按 `scenarios/*.json` 泛化：裁剪中心、目标点、虚拟配送站、风险关键词、通信参数和实验任务均来自场景配置，后续新增山体场景不需要改源码。

## 目录结构

| 路径 | 作用 |
|---|---|
| `article_planner/` | 通用工具包，包含场景配置、DEM 坐标、目标定位和输出规则。 |
| `data/raw/<scene_name>/` | 原始输入数据归档目录，按场景保存 DEM、OSM 和其他不可再生资料。 |
| `tools/locate_targets.py` | 通用目标点定位工具，替代旧的华山五峰定位实验脚本。 |
| `scenarios/huashan.json` | 华山场景配置，包含 DEM、裁剪范围、五峰目标、任务和风险关键词。 |
| `scenarios/template.example.json` | 新场景配置模板，复制后填写 DEM、目标点和可选 OSM 信息即可泛化实验。 |
| `init_graph.py` | 兼容入口：裁剪 DEM 并生成地理网格。 |
| `human_risk_osm.py` | 兼容入口：从 OSM 生成游客/人群暴露风险场。 |
| `safe_corridor.py` | 兼容入口：根据 DEM、风险和目标区生成安全走廊与三层飞行高度。 |
| `communication_risk.py` | 兼容入口：根据 DEM 视距生成三层通信风险场。 |
| `layered_graph.py` | 兼容入口：构建分层航路图和终端锚点。 |
| `task_generator.py` | 兼容入口：生成虚拟配送站、候选目标点和物流任务。 |
| `lpa_star.py` | 兼容入口：执行单次 LPA* 路径规划、区域扰动和增量重规划演示。 |
| `benchmark.py` | 兼容入口：运行 single 或 matrix benchmark。 |
| `benchmark_matrix.py` | matrix benchmark 的 M-P/M-A 事件流实验实现。 |
| `tools/plot_matrix_results.py` | 读取 matrix CSV 并输出论文 PDF 图。 |
| `tools/plot_generalization_results.py` | 读取多场景汇总 CSV 并输出 E1 跨地形泛化论文图。 |
| `tools/plot_ablation_results.py` | 读取 single benchmark 结构性消融 CSV 并输出 E2 消融论文图。 |
| `run_multi_scene.py` | 多场景流水线执行器。 |
| `data/raw/huashan/AP_19438_FBD_F0680_RT1.dem.tif` | 华山原始 DEM 输入。 |
| `data/raw/huashan/map.osm` | 华山本地 OSM 输入。 |
| `outputs/` | 运行输出根目录，按 `outputs/<scene_name>/` 归类；测试结果统一放在 `outputs/<scene_name>/tests/`。 |

## 环境准备

建议使用独立虚拟环境：

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

当前必需依赖为 `numpy`、`scipy`、`matplotlib`、`tifffile`、`pyproj`。`requirements.txt` 和 `pyproject.toml` 使用最低兼容版本，便于在新机器上安装；`constraints.txt` 锁定当前已验证环境版本，便于复现实验：

```powershell
pip install -r requirements.txt -c constraints.txt
```

旧定位实验使用过 `rasterio`，现在的 `tools/locate_targets.py` 已改用 `tifffile + pyproj`，默认不再需要 `rasterio`。

工程同时提供轻量 `pyproject.toml`。如果希望以包形式安装，可使用：

```powershell
pip install -e .
```

开发工具可选安装：

```powershell
pip install -r requirements-dev.txt
pip install -e ".[dev]" -c constraints.txt
```

## 场景配置

所有泛化入口都通过 `--scenario-config` 读取 JSON。华山配置位于 `scenarios/huashan.json`，新场景可从 `scenarios/template.example.json` 复制。

核心字段说明：

| 字段 | 含义 |
|---|---|
| `scene_name` | 场景名称，也用于默认输出目录占位符。 |
| `dem_path` | 原始 DEM 文件路径，相对路径以项目根目录为基准，建议放在 `data/raw/<scene_name>/`。 |
| `output_dir` | 场景输出目录，通常使用 `outputs/{scene_name}`。 |
| `crop` | DEM 裁剪参数：中心经纬度和裁剪边长。实际像元分辨率由 `init_graph.py` 从 GeoTIFF 像元尺度读取并写入 `Z_crop_meta.json`。 |
| `source_crs` | 可选字段；仅当 GeoTIFF 缺少 CRS 元数据时填写，例如 `EPSG:xxxx`。正常情况下不需要配置。 |
| `targets` | 任务目标点列表；每个目标可包含 `lon`、`lat`、`elev`、`display_name`。华山五峰只是这里的一组配置。 |
| `default_start` / `default_goal` | 单次 LPA* 演示默认起终点名称。 |
| `virtual_depots` | 虚拟配送站自动生成规则。 |
| `terrain_sampling` | 分层图节点采样规模和间距参数。 |
| `adaptive_corridor` | 安全走廊地形抬升、终端厚度、高风险禁行层等参数。 |
| `communication` | 通信风险建模参数和路径代价风险融合权重。 |
| `task_generation` | 自动生成物流任务的目标数量、任务数量和分层采样规则。 |
| `osm_file` | 可选 OSM 文件；建议放在 `data/raw/<scene_name>/`，缺失时可跳过人群风险生成。 |
| `osm_risk_keywords` | 场景专有 OSM 风险关键词；华山危险道路名不再写死在通用流程中。 |

## 单场景完整执行流

以下命令按单步执行华山场景，场景中间缓存写入 `outputs/huashan/`，测试或 benchmark 结果写入 `outputs/huashan/tests/`：

```powershell
python init_graph.py --scenario-config scenarios/huashan.json --workdir .
python human_risk_osm.py --scenario-config scenarios/huashan.json --workdir .
python safe_corridor.py --scenario-config scenarios/huashan.json --workdir .
python communication_risk.py --scenario-config scenarios/huashan.json --workdir .
python layered_graph.py --scenario-config scenarios/huashan.json --workdir .
python task_generator.py --scenario-config scenarios/huashan.json --workdir .
python benchmark.py --mode single --scenario-config scenarios/huashan.json --workdir . --trials 5 --skip-b1 --out-dir tests/benchmark_single
```

执行含义：

1. `init_graph.py`：读取原始 DEM，按 `crop` 中心和尺寸裁剪研究区，生成高程矩阵和经纬度网格。
2. `human_risk_osm.py`：读取本地 OSM，结合通用 OSM 标签规则和场景关键词，生成 L1-L4 人群暴露风险。
3. `safe_corridor.py`：融合地形坡度、山脊、风险和终端区域，生成安全飞行上下边界和三层中面。
4. `communication_risk.py`：基于地形视距和地面通信源生成三层通信风险。
5. `layered_graph.py`：对终端层、区域支路层和骨干层采样，进行碰撞检测并生成分层图。
6. `task_generator.py`：自动生成虚拟配送站、补充候选目标点和分层物流任务。
7. `benchmark.py --mode single`：比较 M-P/M-A/M-F/M-R/M-V 五类方法，其中 M-R 为“规则三层图 + LPA*”结构性消融；在 `--skip-b1` 下可跳过传统体素全局搜索。

也可以使用总入口一次跑完：

```powershell
python run_multi_scene.py --scenario-configs scenarios/huashan.json --benchmark-mode single --trials 5 --skip-b1 --disable-plots --skip-layered-plot --benchmark-out-name tests/benchmark_single
```

## 目标定位流

`tools/locate_targets.py` 是通用工具，不再绑定华山五峰。它读取场景 JSON 的 `targets`，优先使用裁剪缓存；如果缓存不存在，则直接读取源 DEM。

```powershell
python tools/locate_targets.py --scenario-config scenarios/huashan.json --workdir . --source auto
```

默认输出：

| 文件 | 含义 |
|---|---|
| `outputs/<scene_name>/target_locations.json` | 每个目标点的像元坐标、吸附后经纬度、DEM 高程、声明高程和高程误差。 |

当新场景目标不是“五峰”时，只需要在场景 JSON 中声明目标名称和经纬度即可。

## 单次 LPA* 演示流

在完成裁剪、风险、走廊和分层图之后，可运行：

```powershell
python lpa_star.py --scenario-config scenarios/huashan.json --workdir . --event-type no_fly --event-radius-km 0.8 --event-severity 1.0 --disable-seed-sweep
```

该流程输出初始路径、区域动态扰动、增量重规划路径、通信覆盖指标和可视化图片。它适合检查单条路径是否合理，不替代 benchmark 统计实验。

## 多场景与泛化实验流

新增场景步骤：

1. 复制 `scenarios/template.example.json` 为新文件，例如 `scenarios/new_mountain.json`。
2. 创建 `data/raw/new_mountain/`，把该山体的 DEM、OSM 等原始输入放进去。
3. 修改 `scene_name`、`dem_path`、`crop.center_lon`、`crop.center_lat`、`crop.crop_size_m` 和 `targets`；分辨率默认由 DEM 元数据自动读取。
4. 如有 OSM，填写 `osm_file` 和场景专有 `osm_risk_keywords`；没有 OSM 可在运行时使用 `--skip-osm-risk`。
5. 运行多场景入口：

```powershell
python run_multi_scene.py --scenario-configs scenarios/*.json --benchmark-mode single --trials 5 --skip-b1 --disable-plots --skip-layered-plot
```

`run_multi_scene.py` 会自动跳过 `*.example.json` 示例模板，避免 `scenarios/template.example.json` 被通配符纳入正式多场景实验。

所有场景输出进入 `outputs/<scene_name>/`。benchmark 测试结果默认进入 `outputs/<scene_name>/tests/benchmark_multi_scene/`，总汇总写入 `outputs/_summaries/multi_scene_summary.csv`。

## Benchmark 流

论文正文和结果图统一使用 M 系列方法编号，CSV 的 `baseline` 字段仍保留 B 系列内部代号便于检索：

| Method ID | Internal baseline ID | Figure label | Full method name | 作用 |
|---|---|---|---|---|
| M-P | B4_Proposed_LPA_Layered | Terrain-aware Layered LPA* (Proposed) | Terrain-aware three-layer airway network with LPA*-based incremental replanning | 本文主方法 |
| M-A | B2_GlobalAstar_Layered | Terrain-aware Layered A* | Terrain-aware three-layer airway network with global A* recomputation | 消融增量重规划 |
| M-F | B3_LPA_SingleLayer | Flat-graph LPA* | Flat graph with LPA*-based replanning | 消融三层航线结构 |
| M-R | B5_RegularLayered_LPA | Regular-layered LPA* | Regular three-layer graph with LPA*-based replanning | 消融地形驱动分层 |
| M-V | B1_Voxel_Dijkstra | Voxel Global Search | Coarse voxel graph with global search | 传统基线 |

论文实验编号统一使用 E 系列：

| Experiment ID | Recommended name | 对应输出 | 主要目的 |
|---|---|---|---|
| E1 | Cross-terrain Generalization and Baseline Comparison | `multi_scene_summary.csv` / `benchmark_summary.csv` | 验证跨华山、黄山、峨眉山的泛化能力，并进行综合基线对比。 |
| E2 | Structural Ablation Study | `benchmark_structural_ablation.csv` | 拆开验证增量机制、三层结构、地形驱动分层和传统体素基线。 |
| E3 | Event-driven Replanning Matrix Analysis | `experiment_A.csv` / `experiment_B.csv` / `experiment_C.csv` / `experiment_D.csv` | 在不同事件参数下分析 M-P 与 M-A 的增量重规划差异。 |
| E4 | Path-quality Consistency Analysis | `experiment_path_quality.csv` / `benchmark_trials.csv` | 验证 M-P 的速度优势不是通过牺牲路径质量获得。 |

`benchmark.py --mode single` 输出一次多基线统计表（含 M-R 结构性消融）：

```powershell
python benchmark.py --mode single --scenario-config scenarios/huashan.json --workdir . --trials 10 --skip-b1 --out-dir tests/benchmark_single
```

`benchmark.py --mode matrix` 输出论文式 E3.1-E3.4 事件驱动矩阵实验：

```powershell
python benchmark.py --mode matrix --scenario-config scenarios/huashan.json --workdir . --trials 10 --matrix-key-trials 30 --skip-b1 --disable-plots --out-dir tests/benchmark_matrix
```

如果想保持 full matrix 的广度，同时把论文主分析涉及的关键组合提升到约 30 次，可直接运行：
```powershell
python benchmark_matrix.py --scenario-config scenarios/huashan.json --workdir . --trials 10 --key-trials 30 --disable-plots --out-dir tests/benchmark_matrix_paper
```

该脚本会自动把 E3.1-E3.4 焦点组合识别为关键组合，并在结果表中额外给出 `median / p95 / 配对 speedup / 检验方法 / p 值`，同时在 `benchmark_discussion.md` 中写出对非单调现象的解释。

完整场景流水线也可以直接调用新版矩阵实验，不需要先生成场景再手动运行 `benchmark_matrix.py`：
```powershell
python run_multi_scene.py --scenario-configs scenarios/huangshan.json --benchmark-runner benchmark_matrix --trials 10 --key-trials 30 --benchmark-out-name tests/matrix_final
```

`run_multi_scene.py` 已直接支持矩阵参数，不需要全部塞进 `--benchmark-extra-args`，例如：
```powershell
python run_multi_scene.py --scenario-configs scenarios/huangshan.json --benchmark-runner benchmark_matrix --trials 10 --key-trials 30 --n-block-grid 2,4,6,8 --k-events-grid 1,3,5,7,10 --scales small,medium,large --scale-fractions small:0.55,medium:0.78,large:1.0 --focus-scale large --focus-k-intensity 5 --focus-n-block-cont 4 --benchmark-out-name tests/matrix_final
```

矩阵结果生成后，可直接输出论文 PDF 图：
```powershell
python tools/plot_matrix_results.py --result-dir outputs/huangshan/tests/matrix_final
```

E1 跨地形泛化图从 `run_multi_scene.py` 的汇总 CSV 生成：
```powershell
python tools/plot_generalization_results.py --summary-csv outputs/_summaries/E1_E2_three_mountain_single_final.csv --workdir .
```

E2 结构性消融图从 single benchmark 输出目录生成：
```powershell
python tools/plot_ablation_results.py --result-dir outputs/huangshan/tests/E1_E2_single_final
```

若需要继续使用矩阵绘图脚本的兼容入口，结果目录中需要包含 `benchmark_structural_ablation.csv`；可用 `benchmark.py --mode matrix` 自动补充 M-R 单事件消融，或对三场景分别运行 `benchmark.py --mode single` 后汇总。
单独给 single 输出目录画 M-R 消融图也可运行：
```powershell
python tools/plot_matrix_results.py --result-dir outputs/huangshan/tests/benchmark_single --ablation-only
```

E3 子实验含义：

| Sub-experiment ID | 输出文件 | 推荐名称 | 变化参数 | 固定/对比方法 |
|---|---|---|---|---|
| E3.1 | `experiment_A.csv` | Event-intensity Sensitivity | `intensity_index` / `n_block` | M-P vs M-A |
| E3.2 | `experiment_B.csv` | Consecutive-event Replanning | `K` | M-P vs M-A |
| E3.3 | `experiment_C.csv` | Graph-scale Sensitivity | `scale` | M-P vs M-A |
| E3.4 | `experiment_D.csv` | Workload Mechanism Analysis | workload metrics | M-P vs M-A |

## 输出结果说明

场景主目录 `outputs/<scene_name>/` 的关键输出：

| 文件 | 生成步骤 | 含义 |
|---|---|---|
| `Z_crop.npy` | `init_graph.py` | 裁剪后的 DEM 高程矩阵。 |
| `Z_crop_geo.npz` | `init_graph.py` | 与 `Z_crop.npy` 对齐的 `lon_grid`、`lat_grid`。 |
| `Z_crop_meta.json` | `init_graph.py` | 裁剪中心、源 DEM、源 CRS、像元尺度、实际分辨率、窗口行列范围和方向信息。 |
| `<scene_name>_final.png` | `init_graph.py` | 裁剪 DEM 的俯视和三维预览图。 |
| `risk_l1.npy` | `human_risk_osm.py` | 高风险危险路线或危险地物风险。 |
| `risk_l2.npy` | `human_risk_osm.py` | 主要游线、峰顶、景点等风险。 |
| `risk_l3.npy` | `human_risk_osm.py` | 索道、设施热点等高斯扩散风险。 |
| `risk_l4.npy` | `human_risk_osm.py` | 低等级道路和山脚设施风险。 |
| `risk_trail.npy` | `human_risk_osm.py` | 兼容旧流程的路线风险合并层。 |
| `risk_hotspot.npy` | `human_risk_osm.py` | 兼容旧流程的热点风险合并层。 |
| `risk_human.npy` | `human_risk_osm.py` | L1-L4 综合人群暴露风险。 |
| `osm_feature_summary.json` | `human_risk_osm.py` | OSM 解析、命中特征和风险统计。 |
| `osm_human_risk_preview.png` | `human_risk_osm.py` | OSM 风险可视化预览。 |
| `floor.npy` | `safe_corridor.py` | 每个像元的飞行走廊下边界。 |
| `ceiling.npy` | `safe_corridor.py` | 每个像元的飞行走廊上边界。 |
| `layer_mid.npy` | `safe_corridor.py` | 三层飞行中面高度，形状通常为 `3 x rows x cols`。 |
| `layer_allowed.npy` | `safe_corridor.py` | 三层像元可通行布尔掩码。 |
| `corridor_meta.json` | `safe_corridor.py` | 走廊厚度、风险区域比例和可通行比例。 |
| `corridor_vis.png` | `safe_corridor.py` | 安全走廊和三层高度可视化。 |
| `risk_comm.npy` | `communication_risk.py` | 三层通信风险栅格。 |
| `communication_summary.json` | `communication_risk.py` | 通信源、风险统计和覆盖比例。 |
| `graph_nodes.npy` | `layered_graph.py` | 分层图节点表，字段为 `x_km, y_km, z_m, layer_id`。 |
| `graph_edges.npy` | `layered_graph.py` | 分层图边表，字段为 `u, v, edge_type`。 |
| `graph_terminal_status.json` | `layered_graph.py` | 终端锚点接入状态和安全校验。 |
| `graph_node_roles.json` | `layered_graph.py` | 节点角色和采样元数据。 |
| `graph_vis.png` | `layered_graph.py` | 分层航路图可视化。 |
| `generated_depots.json` | `virtual_depots.py` / `layered_graph.py` / `task_generator.py` | 自动生成的虚拟配送站。 |
| `generated_tasks.json` | `task_generator.py` | 自动生成的配送任务、目标点和任务分层。 |
| `lpa_path_summary.json` | `lpa_star.py` | 单次初始规划和重规划指标摘要。 |
| `path_vis.png` | `lpa_star.py` | 初始路径俯视、三维和高度剖面图。 |
| `path_cost_profile.png` | `lpa_star.py` | 初始路径和重规划路径代价分布图。 |
| `lpa_result.png` | `lpa_star.py` | 三阶段 LPA* 规划与动态扰动对比图。 |

Benchmark 输出目录中的关键文件：

| 文件 | 含义 |
|---|---|
| `benchmark_trials.csv` | 每次 Monte Carlo trial、起终点、事件和各基线结果。 |
| `benchmark_summary.csv` | 按基线或实验组合聚合后的均值、成功率和图规模。 |
| `benchmark_pairwise.csv` | 成对基线统计比较。 |
| `benchmark_combo_status.csv` | matrix 模式下各组合接受 trial 情况。 |
| `benchmark_failure_reasons.csv` | matrix 模式下失败原因统计，用于解释 small scale 低成功率。 |
| `benchmark_trial_failures.csv` | matrix 模式下 trial 级失败原因，包含采样失败和事件后断路。 |
| `benchmark_events.csv` | matrix 模式下事件流采样记录。 |
| `benchmark_table.md` | 可直接放入论文草稿的结果表。 |
| `benchmark_table_four_baselines.md` | 四基线对比表。 |
| `benchmark_table_structural_ablation.md` | 含 M-R 的结构性消融对比表。 |
| `benchmark_structural_ablation.csv` | M-P/M-A/M-F/M-R/M-V 结构性消融 CSV，含 `method_id`、`internal_code`、`figure_label` 和 `full_method_name`。 |
| `fig_expA_event_intensity_time.pdf` | E3.1 事件强度时间图。 |
| `fig_E1_cross_terrain_success.pdf` | E1 跨地形成功率图。 |
| `fig_E1_cross_terrain_replan_time.pdf` | E1 跨地形重规划时间图。 |
| `fig_E1_cross_terrain_comm_coverage.pdf` | E1 跨地形通信覆盖图。 |
| `fig_E1_cross_terrain_risk_exposure.pdf` | E1 跨地形风险暴露图。 |
| `fig_E2_structural_ablation.pdf` | E2 结构性消融图。 |
| `fig_*.pdf` | `tools/plot_matrix_results.py` 生成的其他论文图。 |
| `benchmark_discussion.md` | matrix 模式生成的讨论要点。 |
| `fig*.png` | matrix 模式图表，可通过 `--disable-plots` 跳过。 |
| `benchmark_config.json` | 本次实验参数快照。 |
| `outputs/_summaries/multi_scene_summary.csv` | `run_multi_scene.py` 汇总所有场景的 benchmark 摘要。 |

## 测试流

语法检查：

```powershell
python -m compileall .
```

兼容入口检查：

```powershell
python init_graph.py --help
python human_risk_osm.py --help
python safe_corridor.py --help
python communication_risk.py --help
python layered_graph.py --help
python task_generator.py --help
python lpa_star.py --help
python benchmark.py --help
python benchmark_matrix.py --help
python run_multi_scene.py --help
python tools/locate_targets.py --help
python tools/plot_matrix_results.py --help
python tools/plot_generalization_results.py --help
python tools/plot_ablation_results.py --help
```

单场景 smoke：

```powershell
python run_multi_scene.py --scenario-configs scenarios/huashan.json --benchmark-mode single --trials 1 --skip-b1 --disable-plots --skip-layered-plot --benchmark-out-name tests/benchmark_smoke_refactor
```

Matrix smoke：

```powershell
python benchmark.py --mode matrix --scenario-config scenarios/huashan.json --workdir . --trials 1 --skip-b1 --skip-four-baseline --disable-plots --out-dir tests/benchmark_matrix_smoke --matrix-n-block-grid 2 --matrix-k-events-grid 1 --matrix-scales small --matrix-scale-fractions small:0.55
```

流水线运行后、清理中间结果前，可检查关键中间产物和测试结果：

```powershell
Test-Path outputs/huashan/Z_crop.npy
Test-Path outputs/huashan/risk_human.npy
Test-Path outputs/huashan/layer_mid.npy
Test-Path outputs/huashan/graph_nodes.npy
Test-Path outputs/huashan/generated_tasks.json
Test-Path outputs/huashan/tests/benchmark_smoke_refactor/benchmark_summary.csv
```

执行 `python tools/clean_outputs.py --outputs-dir outputs` 后，中间产物会被删除，此时应检查测试结果是否仍保留：

```powershell
Test-Path outputs/huashan/tests/benchmark_smoke_refactor/benchmark_summary.csv
Test-Path outputs/huashan/tests/benchmark_matrix_smoke/experiment_A.csv
Test-Path outputs/_summaries/multi_scene_summary.csv
```

根目录清洁检查：

```powershell
Get-ChildItem -File Z_crop*,risk_*,floor.npy,ceiling.npy,layer_mid.npy,layer_allowed.npy,graph_*,*.csv,*.png -ErrorAction SilentlyContinue
```

正常情况下，运行产物应位于 `outputs/<scene_name>/`，根目录不应再出现新的 `.npy/.png/.csv` 实验结果。

## 清理策略

根目录历史运行产物和 `benchmark_out_final/` 属于可再生输出，已从版本库清理；`outputs/` 和 `benchmark_out*/` 已加入 `.gitignore`。原始 DEM、OSM 已按场景归档到 `data/raw/<scene_name>/`，根目录不再直接放数据文件。

如果只想保留测试结果并删除场景中间缓存，运行：

```powershell
python tools/clean_outputs.py --outputs-dir outputs
```

清理后的结构为：

```text
outputs/
  _summaries/
    multi_scene_summary.csv
  huashan/
    tests/
      benchmark_smoke_refactor/
      benchmark_matrix_smoke/
```

如需复现已删除的历史结果，按 README 中的单场景或 benchmark 命令重新运行即可。
