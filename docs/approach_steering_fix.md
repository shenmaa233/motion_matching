# Pre-Skill Steering 修复说明

## 背景
在 `locomotion -> skill -> locomotion` 的单技能轨迹生成流程里，机器人在进入 obstacle 前会先绕大圈，再走向 climbing obstacle。

这个问题出现在 pre-skill approach 阶段，即 [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:640) 里的第一段 locomotion 检索循环。

## 问题分析
原始实现的 pre-skill 逻辑有两个特点：

- `pre_skill_distance_meters` 决定 approach 段总时长，随后换算出 `pre_skill_seconds` 和 `pre_skill_frames`。
- 真实驱动 locomotion 检索的命令向量是一个固定的 `heading_deg`，由配置中的 `pre_skill.heading_degrees` 随机采样得到。

也就是说，pre-skill 阶段虽然长度受距离控制，但方向并不是根据 skill 入口方向动态更新，而是沿用一个固定的随机 heading。

在 motion matching 框架里，这会导致一个典型问题：

- 局部检索是合理的，但全局 steering 很弱。
- 数据库里如果有较多带转向惯性的 locomotion 片段，固定 heading 会连续选中“继续转”的段。
- 最终表现为先绕圈，再接上 skill。

因此这次修复的重点不是把 `time` 换成 `distance`，而是给 pre-skill 增加一层轻量 steering。

## 修复目标
目标不是引入完整路径规划，而是在保持当前 motion matching 数据流不变的前提下，让 pre-skill approach 的命令方向能随着 skill entry 候选动态调整。

当前仓库里 obstacle 的世界坐标并不是在 pre-skill 开始前就固定好的：

- `terrain_world_pose` 是在 skill 片段真正接入后，根据选中的 skill clip 和 source frame 反推出来的。
- 因此 pre-skill 阶段无法直接使用“已知世界坐标下的 obstacle 点”做导航。

在这个约束下，最稳妥的做法是：

- 每个检索周期先估计一个最匹配的 `skill entry` 候选。
- 再读取这个候选在 skill clip 内进入前一小段时间的运动方向。
- 把这个方向转换成当前 pre-skill 的 approach command。

这样虽然没有显式的 obstacle world target，但 locomotion 会逐步朝“最可能接上 skill 的进入方向”收敛。

## 代码改动

### 1. 给 pre-skill 配置增加 steering 窗口参数
在 [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:54) 的 `PreSkillConfig` 中增加了：

- `approach_direction_window_frames`

对应的配置读取和校验在：

- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:77)

它来自：

- [trajectory_generation_config.json](/home/luzimeng/Desktop/motion_matching/data/motion_matching/trajectory_generation_config.json:1)

这个参数表示：当我们拿到一个 `skill entry` 候选帧后，往前回看多少帧，用这段位移来估计“进入 skill 前的局部方向”。

### 2. 新增 skill-entry-based steering 命令构造函数
新增函数：

- [\_approach_command_from_skill_entry_candidate](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:513)

函数逻辑：

1. 根据 `database_index` 找到对应的 skill clip 和 source frame。
2. 从该帧往前回看 `approach_direction_window_frames`。
3. 计算这段时间内 root 的位移。
4. 将位移转换到该帧 root 的局部坐标系。
5. 取其二维方向，归一化后乘上当前选定的 `speed_mps`。
6. 如果位移太小，则退化到使用该帧 root 局部速度。
7. 如果仍然不可用，则退回 `[speed_mps, 0.0]`。

这个函数本质上是在回答一个问题：

“如果当前最匹配的 skill entry 是这个候选，那在进入 skill 前，机器人应该朝哪个局部方向靠近它？”

### 3. 把 pre-skill approach 从固定 heading 改成动态 steering
主改动在：

- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:702)

原来的流程更接近：

1. 用固定 `approach_command`
2. 直接从 locomotion 数据库检索下一段

现在变成：

1. 先基于当前输出状态，构造一个 steering query。
2. 用这个 query 选出当前最匹配的 `skill entry` 候选。
3. 由这个候选反推当前应采用的 `approach_command`。
4. 再用新的 `approach_command` 去检索 locomotion 段。

因此，`approach_command` 不再是一个整段固定的随机 heading，而是每个 search interval 都会刷新一次。

### 4. 在输出 manifest 中记录 steering 信息
生成结果 manifest 增加了以下字段，便于调试：

- `command.initial_heading_deg`
- `command.approach_command_local_xy`
- `approach.steering_mode`
- `approach.steering_window_frames`
- `approach.last_steering_clip_name`
- `approach.last_steering_source_frame`

对应代码位置：

- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:885)
- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:894)

批量 manifest 也会记录：

- `pre_skill_config.approach_direction_window_frames`

对应位置：

- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:1018)

## 修复后的行为变化
修复前：

- `heading_deg` 只在 pre-skill 开始时采样一次。
- approach 阶段更像“朝一个固定方向做局部检索”。
- 如果数据库里连续命中了带转向惯性的片段，就容易出现大弯或绕圈。

修复后：

- 每个 approach 检索周期都会重新估计一次 `skill entry` 候选。
- `approach_command` 会向“最可能成功接入 skill 的进入方向”动态靠拢。
- locomotion 检索的全局方向约束更强，因此更容易直观地朝 obstacle 收敛。

## 为什么这个方案优先于只改 duration/distance
`pre_skill_distance_meters` 解决的是“从多远开始 approach”。

它确实重要，但它控制的是 approach 长度，不直接控制 approach 的全局方向。

绕大圈的问题本质上更接近：

- 方向约束不够强
- 没有 steering

因此，这次修复优先做 steering，而不是仅仅继续调整 time/distance 的参数化方式。

## 当前参数建议
主要关注：

- `pre_skill.start_distance_meters`
- `pre_skill.approach_direction_window_frames`
- `search.interval_frames`

经验上：

- `approach_direction_window_frames` 较大时，方向更平滑，但可能保留更多历史大弯趋势。
- `approach_direction_window_frames` 较小时，方向反应更快，但也可能更抖。
- 如果又开始出现大弯，可以先尝试减小这个窗口，比如从 `30` 降到 `10` 或 `15`。

## 限制与后续方向
这次修复仍然不是完整路径规划，仍有几个边界：

- pre-skill 阶段没有显式的 obstacle 世界坐标导航。
- steering 仍然依赖数据库中已有 locomotion / skill 的统计结构。
- 如果技能入口本身标注不准，steering 也会跟着偏。

如果后续还要继续增强，可考虑：

1. 给 skill / obstacle 建立固定世界锚点，让 pre-skill 直接朝世界目标导航。
2. 在接近 entry 时增加距离阈值和提前收敛逻辑，减少最后几步的摆动。
3. 为 locomotion 检索增加额外代价项，例如对横向偏移和多余 yaw 变化做惩罚。

## 本次结论
这次修复做的是一层轻量、兼容当前架构的 steering：

- 不改数据库格式
- 不引入完整路径规划
- 不改变 skill 接入和后处理主流程

只在 pre-skill locomotion 检索前，增加“基于当前最优 skill entry 候选的动态命令方向更新”。

它解决的是机器人在 skill 前“先绕大圈”的主因，也就是 pre-skill 阶段全局方向控制不足的问题。
