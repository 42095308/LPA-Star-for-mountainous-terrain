"""
运行输出清单。

这些文件均可由流水线重新生成，不应再提交到版本库根目录。
"""

from __future__ import annotations


ROOT_GENERATED_FILES = [
    "Z_crop.npy",
    "Z_crop_geo.npz",
    "Z_crop_meta.json",
    "ceiling.npy",
    "floor.npy",
    "layer_allowed.npy",
    "layer_mid.npy",
    "corridor_meta.json",
    "corridor_vis.png",
    "graph_nodes.npy",
    "graph_edges.npy",
    "graph_node_roles.json",
    "graph_terminal_status.json",
    "graph_vis.png",
    "huashan_final.png",
    "lpa_result.png",
    "lpa_seed_sweep.csv",
    "lpa_seed_sweep_summary.json",
    "osm_feature_summary.json",
    "osm_human_risk_preview.png",
    "path_cost_profile.png",
    "path_vis.png",
    "risk_l1.npy",
    "risk_l2.npy",
    "risk_l3.npy",
    "risk_l4.npy",
    "risk_trail.npy",
    "risk_hotspot.npy",
    "risk_human.npy",
]

GENERATED_DIRECTORIES = [
    "outputs",
    "benchmark_out_final",
    "benchmark_out_matrix",
]
