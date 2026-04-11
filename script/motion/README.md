# Motion Matching Scripts

## 目录职责
- `motion_matching_common.py`
  共享基础能力：数据加载、镜像、特征构建、弹簧模型、过渡拼接、Drake 导出。
- `build_single_skill_database.py`
  把现有 locomotion/skill 数据构成单技能检索数据库。
- `generate_single_skill_batch.py`
  生成 `locomotion -> skill -> locomotion` 的批量 BeyondMimic 轨迹。
- `suggest_skill_annotations.py`
  为 `s_k / e_k / H_k` 给出保守候选值。
- `infer_body_mapping.py`
  输出当前数据与 G1 URDF 的 body 对齐报告。

## 默认流程
1. 先检查或修改 `data/motion_matching/single_skill_catalog.json`。
2. 构建数据库：
   `python script/motion/build_single_skill_database.py`
3. 生成批量轨迹：
   `python script/motion/generate_single_skill_batch.py --num-trajectories 4`
4. 可视化生成结果：
   `python script/visual/visualize_lafan.py --input-dir output/motion_matching/generated/climb_15_z_scale_1.0 --terrain` 

## 当前默认假设
- 单技能只启用 `climb_15_z_scale_1.0`。
- `data/motion_matching/trajectory_generation_config.json` 里的 `start_pose`
  用来控制 0 帧站立姿态选择，以及是否强制把 0 帧 root 高度对齐到
  skill 进入高度。
- 同一个配置文件里的 `pre_skill.start_distance_meters` 用来控制起始点到
  skill 的距离；当前 `pre_skill_seconds` 是由距离和速度换算出来的结果，
  不再是主采样量。
- `terrain_root_offset` 目前是启发式占位值，仍然建议人工确认。
- 左右镜像默认开启。
