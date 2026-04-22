# Skill Transition Smoothing 修复说明

## 背景
在上一版修复里，pre-skill approach 的“大绕圈”问题已经通过动态 steering 明显缓解，但新的主要问题变成了：

- `locomotion -> skill` 接入处仍然存在可见突变
- 虽然方向已经更对，但进入 climbing 前的最后几步看起来不够自然
- 入口处的 root 轨迹、root 朝向变化和局部步态连续性仍然偏硬

这说明全局 steering 已经改善，但局部过渡质量还不够。

## 问题分析
当前系统本身已经有过渡机制：

- root 对齐：`yaw_align_qpos_sequence`
- 惯性平滑：`apply_inertialization_to_qpos_sequence`
- 统一入口：`compose_transition`

这些逻辑在 [motion_matching_common.py](/home/luzimeng/Desktop/motion_matching/script/motion/motion_matching_common.py:643) 到 [motion_matching_common.py](/home/luzimeng/Desktop/motion_matching/script/motion/motion_matching_common.py:722)。

但原始实现仍然有两个局限：

1. `skill entry` 的选择主要依赖 feature 最近邻，没有显式偏好 root 速度和 yaw 角速度连续的入口。
2. skill 段接入时虽然做了姿态 inertialization，但没有单独对 root 平移和 root yaw 做短窗口轨迹平滑。

这会导致一种典型现象：

- 技能入口在特征空间里是“像”的
- 但在 root 运动趋势上并不一定“顺”
- 结果就是人物整体能接上 skill，但脚下几步和 root 轨迹会显得突然拐一下

## 修复目标
这次修复的目标不是增加全局路径规划，而是让 `locomotion -> skill` 的局部接入更连续，重点改善：

- root 平移连续性
- root yaw 连续性
- 入口处速度趋势一致性

修复思路分成两层：

1. 在 `skill entry` 选择阶段加入连续性代价，优先选择更顺的入口。
2. 在真正拼接 skill 段时，对 root 做一段额外的短窗口 blend。

## 代码改动

### 1. 增加运行时 root 状态提取 helper
新增两个辅助函数：

- [\_runtime_pose_state_from_output](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:216)
- [\_current_root_yaw_rate_from_output](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:227)

作用分别是：

- 从当前已生成输出中提取运行时姿态特征
- 计算当前输出序列最后两帧的 root yaw rate

这些 helper 的目的，是在选择 skill entry 时，不只依赖静态 query feature，还能额外利用当前输出的 root 运动趋势。

### 2. 强化 `skill entry` 选择代价
修改了：

- [\_select_skill_entry_index](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:511)

新增输入：

- `current_root_local_velocity_xy`
- `current_root_yaw_rate`

新的候选代价不再只有 feature 最近邻距离，而是：

- `feature distance`
- `root local velocity difference`
- `root yaw rate difference`

当前实现里对应的组合方式是：

- `total_cost = feature_distance + 0.75 * velocity_cost + 0.2 * yaw_rate_cost`

这意味着：

- 首先仍然保证 motion matching 的特征相似性
- 但在多个候选都差不多的时候，会更偏向 root 运动趋势连续的 skill entry

这一步的重点不是让动作“更像数据库平均值”，而是让入口更像当前正在走的这条轨迹自然延续出来的结果。

### 3. 增加 root 专用过渡平滑函数
新增函数：

- [\_smooth_root_transition](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:568)

它是在 skill 片段已经经过原有 `compose_transition` 处理后，再额外对 root 做一层短窗口平滑。

核心逻辑：

1. 读取接入前的 `previous_qpos`
2. 如果可用，再读取 `pre_previous_qpos`
3. 估计当前输出在切换前的 root 线速度
4. 估计当前输出在切换前的 root yaw rate
5. 在 skill 片段前若干帧里，对 root 平移和 yaw 做平滑插值

具体做法是：

- root 平移：使用上一段速度外推得到一个“期望继续走下去”的 root 位置，再和 skill 片段的对齐结果做 smoothstep blend
- root yaw：使用上一段 yaw rate 外推得到一个“期望继续转动”的 root 朝向，再和目标 root yaw 做 smoothstep blend

这样做的结果是：

- root 不会一进 skill 就突然折线式拐弯
- 入口几帧的世界系轨迹看起来更顺

### 4. 扩展 `_append_clip_chunk` 支持 root blend
修改了：

- [\_append_clip_chunk](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:372)

增加参数：

- `root_blend_frames`

逻辑变成：

1. 先执行原有的 `compose_transition` 或 inertialization
2. 如果 `root_blend_frames > 0`
3. 则继续调用 `\_smooth_root_transition`

也就是说，这次没有替换掉原有过渡，而是在原有过渡之后，再加一层 root 专用修正。

这样可以保持：

- joints 仍由原有 inertialization 负责平滑
- root 由新逻辑额外增强连续性

## 主流程接入方式
主改动在：

- [\_generate_single_trajectory](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:762)

### 1. pre-skill 阶段的 steering 候选也开始参考 root 连续性
在 pre-skill approach 循环中，当前输出状态会先通过：

- `\_runtime_pose_state_from_output`
- `\_current_root_yaw_rate_from_output`

取出 root 速度和 yaw rate，再传入 `\_select_skill_entry_index`。

这样 steering 用来参考的 `skill entry` 候选本身也更连续，不只是 feature 上匹配。

### 2. skill 真正接入前再次使用 continuity-aware entry 选择
在 approach 结束、真正切入 skill 之前，也同样会：

- 取当前输出的 root local velocity
- 取当前输出的 yaw rate
- 传给 `\_select_skill_entry_index`

因此真正接 skill 的入口选择，比上一版更关注和当前 locomotion 的衔接。

### 3. skill 接入默认启用 root blend
新增：

- `skill_transition_root_blend_frames = max(6, search_interval_frames)`

并在 skill 片段接入时传入：

- `root_blend_frames=skill_transition_root_blend_frames`

对应位置：

- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:902)

这表示 skill 接入时默认会对前若干帧的 root 做额外平滑，窗口长度至少为 6 帧，并随检索间隔放大。

## 输出 manifest 的新增信息
为便于追踪当前使用的平滑窗口，`manifest.search` 里新增了：

- `skill_transition_root_blend_frames`

对应位置：

- [generate_single_skill_batch.py](/home/luzimeng/Desktop/motion_matching/script/motion/generate_single_skill_batch.py:1017)

这能帮助后续排查：

- 当前样本入口处用了多少 root blend 帧
- 突变改善是否和 blend 窗口有关

## 修复后的行为变化
修复前：

- skill entry 候选主要由 feature 最近邻决定
- skill 入口虽然经过对齐和 inertialization，但 root 轨迹仍可能突然折一下
- 最后几步和 climbing 的第一步之间常有“切片感”

修复后：

- skill entry 选择会更偏好 root 速度和 yaw 变化更连续的候选
- skill 接入前几帧的 root 平移和 root yaw 会显式平滑
- 入口看起来更像走路自然过渡到 climbing，而不是突然切换到另一段片段

## 为什么这版修复优先做这两件事
如果直接对整条输出轨迹做全局滤波，虽然也能减轻突变，但问题是：

- 会模糊步态细节
- 会影响 skill 本身的关键接触动作
- 容易把真正的问题藏起来

所以这次优先做的是：

1. 改 `skill entry` 的选择质量
2. 只在最容易出问题的过渡窗口上做 root 平滑

这比“整条轨迹后处理滤波”更局部、更可控，也更符合 motion matching 的工作方式。

## 当前限制
这次修复虽然改善了入口连续性，但还没有完全解决所有步态自然性问题。

仍然存在的边界包括：

- locomotion 检索本身还没有显式步态相位约束
- 左右脚支撑一致性还没有加入候选代价
- 如果数据库里的 skill entry 本身质量一般，平滑也只能部分补救

所以这次更准确地说，是修复了“skill 接入时的突变”，而不是完全解决了“整条轨迹的所有步态连续性问题”。

## 后续方向
如果后续还要继续提高自然性，建议按这个顺序往下做：

1. 给 locomotion 检索加入脚接触 / 步态相位一致性代价
2. 对 `locomotion -> skill` 入口增加接触状态约束
3. 只在必要时引入更轻量的 root velocity / yaw rate 后处理

## 本次结论
这次修复聚焦的是 `locomotion -> skill` 的局部过渡质量。

核心做法有两点：

- 让 `skill entry` 的选择显式考虑当前 root 运动趋势
- 让 skill 接入前几帧的 root 平移和 yaw 单独再平滑一次

它不是全局路径规划，也不是整条轨迹统一滤波，而是面向 skill 入口突变问题的一次局部增强。
