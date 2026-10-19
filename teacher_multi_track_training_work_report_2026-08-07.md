# G1 Climb Teacher 多轨迹训练工作与结果报告

> 日期：2026-08-07  
> 任务：`Tracking-Flat-G1-v0`  
> Skill：`climb_08_z_scale_1.0`  
> 当前推荐 checkpoint：`model_99.pt`（Repair-A2）  
> 当前状态：训练已完成，未启动完整 mesh 碰撞体微调

## 1. 执行摘要

本轮工作的目标是修复 teacher 多轨迹训练中的 motion–terrain 错位、完善多轨迹数据与采样机制、稳定 PPO 微调，并判断机器人碰撞体不完整是否是训练长期不达标的主要原因。

截至 2026-08-07，得到以下结论：

1. terrain offset 的核心 bug 已修复。每条 motion 现在使用各自 manifest 中的 terrain yaw 和 translation 做完整刚体对齐，日志中的 `anchor_error_m` 均为 `0.000000`。
2. 数据集包含 20 条轨迹；当前 Repair 训练选择了 10 条源轨迹并开启 mirror，因此训练的是一个共同 teacher policy，内部有效 motion variants 为 20 个，不是 20 个独立 teacher。
3. Safe PPO、固定第 0 帧／frontier／adaptive 三路 curriculum 和困难轨迹加权有效改善了 robust 指标。最终 robust 平均完成度达到 **16.48%**，平均 episode reward 达到 **5.236**。
4. 最终端到端成功率仍为 **0%**。640 个评估 episode 全部因 `anchor_pos` 终止，说明当前瓶颈是从第 0 帧开始的完整轨迹跟踪，而不是 PPO loss 发散。
5. 训练日志中接近 40% 的数字是 `adaptive_success_rate`，只代表从历史困难片段附近初始化后的条件成功率，不代表从第 0 帧完整越障成功率。
6. 碰撞体不完整不是当前不达标的主因。完整 mesh 凸包开启自碰撞会把完成度从 14.86% 降到 4.94%；关闭自碰撞后可以恢复，但 robust 指标仍比当前简化碰撞体低 2.10 个百分点。
7. 地形宽度、其他尺寸和 yaw 的随机化代码已经实现，但 Repair-A2 为保证 reference–terrain 严格配对，将独立 terrain 几何随机化关闭。当前运行并没有真正启用“只扰动 terrain、不同时更新 reference”的随机化。

因此，当前最合理的基线仍然是 Repair-A2 `model_99.pt`、简化碰撞体、关闭 push 的配置。下一阶段应重点修复端到端第 0 帧跟踪和最弱轨迹 `0006`、`0019`，而不是直接替换完整碰撞模型。

## 2. 数据与需求落实情况

数据 manifest：

```text
/home/sjtu/Desktop/motion_matching/motion_matching/output/motion_matching/
paper20/climb_08_z_scale_1.0/batch_manifest.json
```

### 2.1 数据集实际内容

完整 manifest 包含 20 条生成轨迹，组合关系为：

```text
2 个速度 × 5 个转向角 × 每个组合 2 条 = 20 条
```

实际统计如下：

| 项目 | 实际数据 |
|---|---|
| 速度等级 | `1.0 m/s` 10 条，`2.0 m/s` 10 条 |
| 转向角 | `-90°/-45°/0°/45°/90°`，每个角度 4 条 |
| pre-skill 时间 | `[0.1, 3.0] s`，10 个离散点，每个点 2 条 |
| 时间采样间距 | 约 `0.322 s`；manifest 目标字段为 `average_interval=0.3` |
| 实际 pre-skill 距离 | `[0.1, 6.0] m`，由速度与时间共同决定 |
| manifest seed | `20260806` |
| mirror | 训练侧启用 |

### 2.2 当前 Repair 训练使用的数据

Repair-A 和 Repair-A2 没有直接加载 manifest 的全部 20 个 npz，而是加载以下 10 条源轨迹：

```text
0000, 0002, 0004, 0006, 0008,
0011, 0013, 0015, 0017, 0019
```

这 10 条源轨迹恰好覆盖：

- 低速、高速各 5 条；
- 每种速度下完整覆盖五个角度；
- 再通过 `mirror_motions=True` 展开为 20 个内部 motion variants。

需要准确区分：

- 当前是 **1 个 teacher policy 联合跟踪 20 个 variants**；
- 不是 20 个 teacher；
- 也不是“20 个原始 npz 再 mirror 成 40 个 variants”。

当前 deterministic evaluator 会关闭 mirror，只评估上述 10 条源轨迹。因此最终 robust 报告尚未单独覆盖 10 条镜像轨迹，这是现有评估覆盖面的一个缺口。

### 2.3 用户需求对照

| 需求 | 状态 | 说明 |
|---|---|---|
| 每组 20 条 | 部分对齐 | manifest 有 20 条；当前训练用 10 个源 npz × mirror = 20 variants |
| 低速／高速 | 已对齐 | `1.0/2.0 m/s` |
| 五个角度 `[-90°, 90°]` | 已对齐 | 两种速度均覆盖五个角度 |
| 变化初始化距离 | 已对齐 | 实际 pre-skill 距离为 `0.1–6.0 m` |
| `[0.1, 3.0] s` 采样 | 已对齐 | 端点覆盖；实际离散间距约 `0.322 s` |
| mirror | 已在训练启用 | 以 terrain frame 为镜像基准 |
| 障碍宽度最大 `1.5 m` | 代码已实现，当前训练未启用 | paired 模式下为避免 reference 错位而禁用独立 terrain 扰动 |
| 其他尺寸 `±5 cm` | 代码已实现，当前训练未启用 | 同上 |
| 障碍 yaw `±45°` | 代码已实现，当前训练未启用 | 同上 |
| 干扰盒随机尺寸与姿态 | 已启用 | 两侧干扰盒按轨迹走廊布置，scale 与 episode pose 随机化 |

## 3. 完成的代码与训练工作

### 3.1 Motion–terrain 刚体对齐

修复内容包括：

- 不再只读取第一条 trajectory 的 terrain pose；
- 为每条 motion 计算独立 yaw offset 和 translation offset；
- 对 reference body position、orientation、linear velocity 和 angular velocity 使用一致的 yaw 变换；
- robot reset、reference reward 和实际 terrain collision 使用同一坐标关系；
- 支持逐 motion manifest 发现和 motion group 与 terrain group 的绑定；
- 运行时打印每条轨迹的 transform 和 `anchor_error_m`，本次 10 条源轨迹均为零误差。

这解决了 diagnosis 文档第一点描述的旧 terrain offset bug。

### 3.2 Motion runtime 与终止逻辑

已完成：

- 多 motion 加载和逐 motion offset；
- 完整 mirror，包括 root、body、关节和速度语义；
- 根据最长 motion 动态设置 episode 时长，本次为 `14.353 s`；
- 增加 `motion_finished` 终止项，避免走完 reference 后被误判；
- 逐帧 adaptive sampling；
- 增加从真实失败边界向前回看的 frontier sampling；
- 增加逐源轨迹 sampling weight；
- 增加 fixed／exact／frontier／adaptive 的独立训练指标。

### 3.3 Observation、reward 与评估

当前 actor observation 维度为 157，包含：

- motion command；
- anchor 相对位置和姿态；
- base linear/angular velocity；
- joint position/velocity；
- previous actions。

critic observation 维度为 227，并额外包含：

- global anchor position；
- global anchor linear velocity；
- 64 维 height scan。

保留论文风格 asymmetric actor–critic：height scan 目前仅进入 critic，没有直接进入 teacher actor。

评估脚本已补充：

- deterministic inference；
- clean／robust profile；
- 固定第 0 帧起步；
- 每条 motion 独立完成度、reward、episode length 和失败原因；
- 多 checkpoint 同口径批量评估；
- JSON 报告持久化。

### 3.4 Safe PPO 与 warm start

为避免从旧 checkpoint 继续训练时 optimizer 状态和大步更新破坏策略，加入了只加载 model／normalizer 权重的 warm start，并重置 optimizer 与 iteration counter。

Repair 阶段主要使用：

```text
learning_rate       = 1e-5
desired_kl          = 0.005
num_learning_epochs = 2
num_mini_batches    = 8
entropy_coef        = 0.001
clip_actions        = 1.0
push_robot          = disabled
```

Repair-A：

```text
action noise std = 0.30
exact frame 0    = 25%
frontier         = 50%
adaptive         = 25%
reset noise      = 0.50
iterations       = 100
```

Repair-A2：

```text
action noise std = 0.20
exact frame 0    = 40%
frontier         = 50%
adaptive         = 10%
reset noise      = 0.25
iterations       = 100
```

Repair-A2 对困难源轨迹使用的 sampling weights 为：

```text
[2, 1, 1, 3, 1, 2, 1, 1, 1, 3]
```

其中 `0006`、`0019` 权重最高。

### 3.5 Terrain 与 distractor randomization

已实现的 terrain event 参数为：

```text
maximum_width_m        = 1.5
other_dimension_delta = 0.05 m
yaw_range              = [-45°, 45°]
```

但该事件只在 legacy／unpaired 模式下启用。Repair-A2 使用 manifest pose 和 paired reference；如果只改变 terrain yaw／尺寸而不同时重生成 reference，会再次破坏已经修复的 motion–terrain 对齐。因此保存的 Repair-A2 `env.yaml` 中：

```yaml
randomize_terrain_geometry_and_pose: null
```

这不是代码遗漏，而是为避免重新引入 terrain offset 类错误所做的保护。若要完全满足论文式 terrain randomization，应预生成“terrain variant + 对应 reference transform”的配对数据，再按 env/motion group 一起采样。

干扰盒已真实启用：

- 在正常与 mirror 轨迹包围走廊两侧放置两个 box；
- 每个环境随机 scale；
- 每个 episode 随机局部位置和 yaw；
- clean evaluation 会关闭这些随机化，robust evaluation 保留。

### 3.6 碰撞体审计与 A/B 开关

碰撞体审计结果：

| URDF | links | 带直接 collision 的 links | colliders |
|---|---:|---:|---:|
| 当前简化训练 URDF | 40 | 14 | 29 个 primitive |
| 完整本地 G1 URDF | 51 | 45 | 47 个，以 mesh 为主 |

两份 URDF 的 29 个可动关节名称、顺序和 child link 顺序完全一致，因此完整 URDF 评估下降不是 actor 输入／动作错位导致的。

已经给 `train.py` 和 `play.py` 增加可逆参数：

```text
--robot_collision_urdf
--robot_collider_type {convex_hull,convex_decomposition}
--disable_robot_self_collisions
```

默认行为保持不变。

## 4. 训练与评估结果

### 4.1 评估口径

最终主要采用以下口径：

```text
policy mode          = deterministic inference
evaluation profile   = robust
fixed start frame    = 0
source motions       = 10
episodes per motion  = 64
total episodes       = 640
```

旧的 `model_3000` 和早期 Stage10 JSON 没有显式保存 `evaluation_profile` 字段，因此它们只能作为历史参考，不能与新 robust 报告做完全严格的统计比较。

### 4.2 各阶段结果

| 阶段 | checkpoint | profile | 平均完成度 | 平均 reward | 成功率 |
|---|---|---|---:|---:|---:|
| warm-start control | `model_3000.pt` | 旧口径未标注 | 8.08% | 1.941 | 0% |
| Stage10 safe | `model_600.pt` | 旧口径未标注 | 7.10% | 1.503 | 0% |
| Stage10 safe | `model_799.pt` | 旧口径未标注 | 6.83% | 1.405 | 0% |
| provenance repair | `model_399.pt` | 旧口径未标注 | 12.09% | 3.915 | 0% |
| provenance repair | `model_399.pt` | clean | 12.93% | 4.173 | 0% |
| Repair-A | `model_25.pt` | robust | 12.05% | 3.925 | 0% |
| Repair-A | `model_50.pt` | robust | 13.11% | 4.229 | 0% |
| Repair-A | `model_75.pt` | robust | 13.82% | 4.483 | 0% |
| Repair-A | `model_99.pt` | robust | 14.21% | 4.577 | 0% |
| Repair-A2 | `model_25.pt` | robust | 14.90% | 4.806 | 0% |
| Repair-A2 | `model_50.pt` | robust | 15.79% | 5.015 | 0% |
| Repair-A2 | `model_75.pt` | robust | 15.78% | 5.013 | 0% |
| **Repair-A2** | **`model_99.pt`** | **robust** | **16.48%** | **5.236** | **0%** |

结果说明：

- 最早的 Stage10 safe continuation 确实发生退化，说明仅从 `model_3000` 继续训练不能解决端到端问题。
- provenance repair 恢复了上升趋势。
- Repair-A robust 完成度从 12.05% 提升到 14.21%，增加 2.16 个百分点。
- Repair-A2 robust 完成度从 14.90% 提升到 16.48%，增加 1.58 个百分点。
- Repair-A2 的 reward 序列为 `4.806 → 5.015 → 5.013 → 5.236`，没有持续下跌；`model_50` 到 `model_75` 是平台期，`model_99` 又继续改善。
- 与旧 `model_3000` 控制评估相比，最终完成度约翻倍、reward 提升约 170%；但由于旧报告 profile 未显式标注，这一比较只能作为趋势参考。

### 4.3 Repair-A2 最终逐轨迹结果

| 源轨迹 | 完成度 | reward | 平均步数 | 主要失败 |
|---|---:|---:|---:|---|
| `0000` | 15.93% | 4.793 | 92.0 | anchor_pos |
| `0002` | 22.41% | 6.881 | 135.5 | anchor_pos |
| `0004` | 15.39% | 5.414 | 96.3 | anchor_pos |
| `0006` | 7.60% | 2.577 | 51.3 | anchor_pos |
| `0008` | 23.96% | 7.701 | 168.3 | anchor_pos |
| `0011` | 20.96% | 5.701 | 123.6 | anchor_pos |
| `0013` | 21.96% | 7.093 | 136.8 | anchor_pos |
| `0015` | 15.73% | 5.900 | 102.5 | anchor_pos |
| `0017` | 13.83% | 4.233 | 91.8 | anchor_pos |
| `0019` | 7.04% | 2.067 | 51.4 | anchor_pos |

最强的四条是 `0008`、`0002`、`0013`、`0011`；最弱的是 `0006` 和 `0019`。

此前设定的阶段 gate 为：

```text
overall completion >= 18%
0006 completion     >= 8%
0019 completion     >= 8%
```

最终结果分别为：

```text
overall = 16.48%  未通过，差 1.52 个百分点
0006    = 7.60%   未通过，差 0.40 个百分点
0019    = 7.04%   未通过，差 0.96 个百分点
```

### 4.4 Loss、reward 和“40% success”的正确解释

Repair-A2 最后一个训练 iteration 的主要日志为：

```text
mean value-function loss = 0.0338
mean surrogate loss      = -0.0017
mean reward              = 3.56
mean episode length      = 79.76
error_anchor_pos         = 0.4733 m
anchor termination       = 96.22%
joint termination        = 0.00%
motion_finished          = 3.89%
training success_rate    = 3.97%
adaptive_success_rate    = 38.71%
exact_start_success_rate = 0.00%
frontier success_rate    = 0.00%
```

由此可以判断：

1. value loss 和 surrogate loss 都很小，没有 PPO 数值发散或 optimizer collapse 的证据。
2. reward 与 episode length 相比 Repair-A 末尾的 `2.75/69.90` 提升到 `3.56/79.76`，说明局部跟踪能力仍在改善。
3. `error_anchor_pos=0.4733 m` 已接近 `0.5 m` 终止阈值，策略长期处在 termination 边界附近。
4. 约 40% 的 success 来自 adaptive 初始化。adaptive 往往直接从困难片段附近或片段中间开始，任务明显短于从第 0 帧完整执行。
5. 固定第 0 帧的 robust deterministic evaluation 仍为 0% success，并且所有 episode 都是 anchor failure。因此不能把 adaptive 的 38.71% 当作整条轨迹通过率。

训练 reward 的波动还会受每个 iteration 中 exact／frontier／adaptive 样本比例、motion 权重和 episode 长度影响。判断 checkpoint 好坏应优先看固定第 0 帧、相同 profile、相同 episode 数的 deterministic evaluation，而不是只看训练面板的瞬时 reward。

## 5. 碰撞体 A/B 结果

### 5.1 Clean smoke test

使用相同 `model_99.pt`、相同 10 条 motion、固定第 0 帧进行四组测试：

| 碰撞配置 | 平均完成度 | 平均 reward | 平均步数 |
|---|---:|---:|---:|
| 简化 primitive + 自碰撞开启 | 14.86% | 4.846 | 94.8 |
| 完整 mesh convex hull + 自碰撞开启 | 4.94% | 0.109 | 32.7 |
| 简化 primitive + 自碰撞关闭 | 14.62% | 4.796 | 93.2 |
| 完整 mesh convex hull + 自碰撞关闭 | 14.27% | 4.687 | 91.0 |

完整 mesh 开启自碰撞时，性能下降约三分之二；关闭自碰撞后基本恢复。这表明灾难性下降来自相邻完整 mesh 凸包之间的重叠／自碰撞，而不是关节顺序或 policy 接口问题。

### 5.2 Robust 64 次／轨迹测试

| 配置 | 完成度 | reward | 成功率 |
|---|---:|---:|---:|
| 当前简化碰撞体 + 自碰撞开启 | 16.48% | 5.236 | 0% |
| 完整碰撞体 + 自碰撞关闭 | 14.38% | 4.555 | 0% |

完整碰撞体方案下降 2.10 个百分点，因此没有达到启动微调的 gate。它对 `0006` 有提升，但严重损伤 `0008`，总体不值得在当前时间约束下替换默认碰撞模型。

结论：机器人碰撞体确实不完整，但这不是当前端到端失败的主要原因。直接补成完整 mesh 并保持自碰撞反而会制造更严重的问题。未来若要补齐，应使用 box／capsule 等简单 primitive，并过滤相邻 link 的自碰撞 pair。

## 6. 当前失败原因排序

### 6.1 主要原因：端到端 anchor 跟踪能力不足

证据：

- robust 640 个 episode 的 anchor failure rate 为 100%；
- joint failure rate 为 0%；
- exact start 和 frontier success 均为 0%；
- anchor error 长期贴近 `0.5 m` termination threshold。

策略已经学会部分局部片段，但不能把起步、接近障碍、接触转换和越障后恢复稳定串成完整 episode。

### 6.2 Curriculum 指标存在条件成功率偏高

adaptive start 可以从动作中间开始，因此训练 success 会显著高于从第 0 帧开始的真实通过率。这不会使算法本身错误，但会让训练面板显得过于乐观。

### 6.3 轨迹难度不均衡

`0006`、`0019` 长期只有约 7%，而 `0008` 接近 24%。Repair-A2 已对困难轨迹加权，但困难轨迹仍未跨过 8% gate，说明仅增加采样权重不足，还需要更精确的 per-motion frontier 或数据检查。

### 6.4 Terrain randomization 尚未以配对形式真正进入 Repair-A2

当前代码避免了错误的独立 terrain 扰动，但也意味着所要求的 width／dimension／yaw 鲁棒性还没有在最终 checkpoint 中得到验证。正确做法是生成配对 terrain-reference variants，而不是只改变碰撞几何。

### 6.5 碰撞体缺失是次要问题

现有脚、腿、躯干和手臂已有简化 collision proxy。完整碰撞 A/B 没有提高总体 robust 完成度，因此它不能解释此前训练长期不达标。

## 7. 建议的下一阶段方案

### 7.1 当前应保留的基线

```text
checkpoint:
/home/sjtu/running_man/new_teacherpolicy/logs/rsl_rl/g1_flat/
2026-08-07_16-00-11_climb08_repairA2_exact40_frontier50_warmA99/model_99.pt

collision:
default simplified primitive URDF, self-collision enabled
```

不要用完整 mesh checkpoint 覆盖当前基线，也不要从完整 mesh + self-collision 配置开始训练。

### 7.2 时间紧时的短程 Repair-A3

建议只做 20–50 iterations 的受控实验：

```text
warm start              = Repair-A2 model_99.pt
learning rate           = 1e-5
action noise std        = 0.15–0.20
reset noise scale       = 0.15–0.25
exact frame 0           = 50%
frontier                = 40%
adaptive                = 10%
push                     = disabled
collision               = current simplified model
```

重点不是继续增加 adaptive 成功率，而是提高 exact-start completion。对 `0006`、`0019` 保持较高权重，同时保留 `0008`、`0002` 的 replay，防止强轨迹退化。

停止／通过 gate 建议：

```text
overall robust completion >= 18%
0006 and 0019             >= 8%
model checkpoint success  > 0% preferred
```

如果 20–50 iterations 后 fixed-start 指标不再增长，应停止继续堆 iteration，转而检查 `0006/0019` 的 reference 接触、root height、第一失败帧和 terrain clearance。

### 7.3 补齐评估覆盖

下一轮应分别报告：

1. 10 条未镜像源轨迹；
2. 10 条镜像轨迹；
3. clean profile；
4. robust profile；
5. 每条轨迹第一 anchor failure frame。

这样才能确认训练内部的 20 variants 都被端到端验证，而不是只验证 10 条原始 motion。

### 7.4 正确加入 terrain randomization

建议离线生成多个配对 variant：

```text
terrain width: [reference minimum, 1.5 m]
other dimensions: reference ± 0.05 m
terrain yaw: [-45°, 45°]
```

每次 terrain 变化后同步更新对应 reference 的 rigid transform／接触关系，并把 terrain USD、manifest entry 和 motion npz 作为一个不可拆分的 group。只有这样才能同时满足鲁棒性要求和 diagnosis 第一项的对齐要求。

## 8. 关键产物

### 8.1 最佳 checkpoint 与评估

```text
/home/sjtu/running_man/new_teacherpolicy/logs/rsl_rl/g1_flat/
2026-08-07_16-00-11_climb08_repairA2_exact40_frontier50_warmA99/model_99.pt

/home/sjtu/running_man/new_teacherpolicy/logs/rsl_rl/g1_flat/
2026-08-07_16-00-11_climb08_repairA2_exact40_frontier50_warmA99/
deterministic_eval_repairA2_robust.json
```

### 8.2 碰撞体 A/B 报告

```text
/home/sjtu/Desktop/motion_matching/output/evaluations/collision_ab_20260807/
```

目录内包括四组 clean A/B 和一组 full-collision robust64 JSON。

### 8.3 相关代码

```text
/home/sjtu/running_man/new_teacherpolicy/scripts/rsl_rl/train.py
/home/sjtu/running_man/new_teacherpolicy/scripts/rsl_rl/play.py
/home/sjtu/running_man/new_teacherpolicy/scripts/rsl_rl/teacher_data.py
/home/sjtu/running_man/new_teacherpolicy/source/whole_body_tracking/
whole_body_tracking/tasks/tracking/mdp/commands.py
/home/sjtu/running_man/new_teacherpolicy/source/whole_body_tracking/
whole_body_tracking/tasks/tracking/mdp/events.py
/home/sjtu/running_man/new_teacherpolicy/source/whole_body_tracking/
whole_body_tracking/tasks/tracking/mdp/terminations.py
/home/sjtu/running_man/new_teacherpolicy/source/whole_body_tracking/
whole_body_tracking/tasks/tracking/tracking_env_cfg.py
```

## 9. 最终结论

本轮工作已经消除了多 motion terrain pose 只对齐第一条轨迹的硬 bug，并建立了可重复的 warm start、curriculum、robust evaluation 和碰撞模型 A/B 流程。Repair-A2 的 robust completion 与 reward 均稳定上升，说明改动方向有效，而且当前并不存在明显 PPO loss collapse。

但是 teacher 尚未达到可接受的端到端越障水平：最终 success 仍为 0%，所有失败均为 anchor tracking failure。当前约 40% 的 adaptive success 只能证明策略能完成部分局部片段，不能证明它能从第 0 帧完成整条轨迹。

因此，下一步的优先级应是：

1. 提高 exact-start 和 frontier 的端到端能力；
2. 单独修复 `0006`、`0019`；
3. 补测镜像轨迹；
4. 以配对数据方式加入 terrain randomization；
5. 最后再考虑选择性补齐 primitive collision。

在这些工作完成前，不建议把完整 mesh 碰撞模型作为新的默认训练配置，也不应把碰撞体缺失视为此前训练不达标的主要原因。
