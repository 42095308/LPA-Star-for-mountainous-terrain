# 原始数据归档规则

本目录只存放不可由程序重新生成的原始输入数据。每个场景使用独立子目录：

```text
data/
  raw/
    huashan/
      AP_19438_FBD_F0680_RT1.dem.tif
      map.osm
```

约定：

- `data/raw/<scene_name>/` 保存该场景的 DEM、OSM、手工采集坐标表等原始资料。
- `outputs/<scene_name>/` 只保存程序运行产物和测试结果，不放原始数据。
- `scenarios/<scene_name>.json` 中的 `dem_path`、`osm_file` 必须指向 `data/raw/<scene_name>/` 内的文件。
- 新增山体场景时，先创建 `data/raw/<scene_name>/`，再复制 `scenarios/template.example.json` 并更新路径。
