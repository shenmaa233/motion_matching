# Teacher 多 Track 训练崩溃问题诊断

> 更新状态（2026-08-06）：本文记录的是修改前的诊断。当前实现已经改为对所有 motion 做完整 terrain yaw/translation 刚性对齐，不再存在“只加载第一条 terrain pose、其余 motion 仅平移到各自 pose”的旧问题；同时已补齐逐帧 adaptive sampling、末帧终止、动态 episode 时长、完整镜像、terrain 几何变体和 distractor boxes。下文旧代码片段仅保留作问题背景。

本文总结 teacher model 在使用大量 tracks 训练时出现 reward/loss collapse、机器人无法越过障碍、卡在从障碍一侧爬到另一侧位置的可能原因，以及对应的代码和训练层面解决方案。

## PDF 目标方法对照

本项目目标参考 `new_teacherpolicy/whole_body_tracking_cp.pdf`，其核心 pipeline 是 BeyondMimic / Perceptive Humanoid Parkour 风格的 RL motion tracking。和本问题最相关的目标设定如下：

- **Per-skill expert training**：PDF 明确说明 PHP 不是训练一个 expert 覆盖所有动作，而是为每个 parkour skill 训练单独 expert，之后再蒸馏到 student。
- **Reference trajectories paired terrain assets**：完整 expert training pipeline 的输入是 reference trajectories 和配对 terrain assets。也就是说，motion reference、robot reset pose、terrain collision 必须保持同一个坐标关系。
- **Asymmetric actor-critic**：actor 接收硬件可用的标准观测，critic 接收 privileged observation，包括全局位置、全局速度、terrain heightmap。
- **Reward**：使用 DeepMimic 风格 tracking rewards，包括 body position/orientation/linear velocity/angular velocity，以及 anchor position/orientation，权重和 sigma 基本固定。
- **Early termination**：expert training 中 anchor/pelvis tracking error threshold 为 `0.5m`；student distillation 中可放宽到 `1.0m`。
- **Adaptive sampling**：困难技能中需要自适应采样，优先从历史失败帧初始化 episode。
- **Lightweight domain randomization**：expert 阶段使用轻量随机化，避免破坏动作质量。

因此，下面的诊断会区分两类建议：

- **忠实复现 PDF**：尽量保持论文/manual 的 MDP、观测、reward、termination、PPO 超参一致。
- **工程补救方案**：当当前数据或实现偏离目标 pipeline 时，为了先让训练收敛可以临时采用，但需要明确它不是严格复现。

## 现象

- 使用较多 tracks 训练 teacher 时，reward 和 loss 曲线容易 collapse。
- 机器人通常无法稳定爬过障碍。
- 卡住点多发生在机器人即将从障碍一侧跨到另一侧的关键过渡阶段。
- 当 tracks 数量减少到较少数量时，例如 3 tracks，问题明显缓解或消失。

## 代码层面的主要问题

### 1. 多轨迹训练时 terrain pose 和 motion offset 不一致

这是当前最值得优先检查和修复的问题。

相关代码：

- `new_teacherpolicy/scripts/rsl_rl/train.py`
- `_compute_manifest_motion_offset(...)`
- `_add_manifest_terrain(...)`

当前逻辑中，每条 motion 都会根据自己的 manifest entry 计算 motion offset：

```python
manifest_entries = [_load_manifest_entry(args_cli.manifest_file, motion_file) for motion_file in motion_files]
...
motion_offsets.append(
    _compute_manifest_motion_offset(
        manifest_entry,
        motion_file,
        args_cli.motion_root_body_idx,
        args_cli.terrain_use_manifest_pose,
    )
)
env_cfg.commands.motion.motion_position_offsets = motion_offsets
```

但是 terrain asset 只会按照第一条有效 manifest entry 的 `terrain_world_pose` 加载：

```python
terrain_manifest_entry = next((entry for entry in manifest_entries if entry is not None), None)
_add_manifest_terrain(env_cfg, args_cli.terrain_file, terrain_manifest_entry, args_cli.terrain_use_manifest_pose)
```

如果不同 tracks/trajectories 的 `terrain_world_pose` 不同，就会出现：

- 第 1 条 motion 的参考轨迹和 terrain 对齐。
- 其他 motion 的参考轨迹被移动到各自的 manifest terrain pose。
- 但仿真里实际 terrain 仍然只在第 1 条 motion 的 terrain pose。
- robot 实际接触到的障碍和 reward/reference 中要求跟踪的位置不一致。

这会直接导致机器人在障碍边缘失败，尤其在多 tracks 训练时更明显。

#### 解决方法

优先选择其中一种一致的坐标策略：

1. **推荐：把所有 motion 预处理到同一个 terrain local frame**
   - 训练时 terrain 固定放在同一个 pose，例如 origin。
   - 所有 motion 的 root/body positions 都转换到同一个 terrain local 坐标系。
   - 多轨迹训练时不再让每条 trajectory 使用不同的 `terrain_world_pose`。

2. **只混合同一个 terrain pose 的 trajectories**
   - 多轨迹训练前检查 manifest。
   - 如果 `terrain_world_pose` 不同，不要放在同一个 training run 中。

3. **为每个并行环境加载对应 active motion 的 terrain pose**
   - 这个方案更复杂，因为 terrain asset 通常在环境创建时固定。
   - 如果要做，需要让每个 env 的 terrain 和 active motion 绑定，reset 时不能只换 motion 不换 terrain。

短期建议：先禁用多 `terrain_world_pose` 混合，确认 24 tracks 中所有参考轨迹和实际 terrain 对齐后再训练。

### 2. Privileged observation 与 PDF 目标需要再核对

当前配置中，`height_scan` 只在 critic observation 里：

```python
class PrivilegedCfg(ObsGroup):
    ...
    height_scan = ObsTerm(...)
```

policy observation 里没有 `height_scan`：

```python
class PolicyCfg(ObsGroup):
    command = ObsTerm(...)
    motion_anchor_pos_b = ObsTerm(...)
    motion_anchor_ori_b = ObsTerm(...)
    base_lin_vel = ObsTerm(...)
    base_ang_vel = ObsTerm(...)
    joint_pos = ObsTerm(...)
    joint_vel = ObsTerm(...)
    actions = ObsTerm(...)
```

按 PDF 的 asymmetric actor-critic 设定，这一点本身不一定是 bug：actor 接收硬件可用的标准观测，critic 接收 privileged observation，包括 terrain heightmap。

当前已经在代码中补齐/修正的是：

- critic 已加入 pelvis/anchor global position，对应 `mdp.robot_anchor_pos_w`。
- critic 已加入 pelvis/anchor global linear velocity，对应 `mdp.robot_anchor_lin_vel_w`。
- observation helper 中旧的 `robot_anchor_vel_w` 错误引用已改成 `robot_anchor_lin_vel_w` / `robot_anchor_ang_vel_w`。

仍需要训练前核对的是：

- `height_scan` 是否确实扫描到了本地 terrain USD，而不是只扫默认平面。
- actor 的标准观测是否足以区分 reference phase 和当前 tracking error。

注意：这次改动只改变 critic observation，不改变 actor observation。旧 teacher actor checkpoint 仍可用于 student DAgger 的 teacher actor 加载；但如果要从旧 teacher checkpoint 继续训练 teacher，critic 输入维度已经变化，最好从头训练或只加载 actor 兼容部分。

#### 解决方法

- **忠实复现 PDF**：保持 `height_scan` 在 critic privileged observation 中，并使用已补齐的 global anchor position / linear velocity。
- **忠实复现 PDF**：确保 height scanner 的 target 包含实际加载的 terrain USD。你的 `_add_height_scan_terrain_target(...)` 已经尝试添加 `{ENV_REGEX_NS}/Terrain`，这个方向是对的。
- **工程补救方案**：如果坐标对齐修好后 24 tracks 仍不收敛，可以临时把 terrain-relative 信息或 phase 加到 actor，但这属于偏离原始 asymmetric actor-critic 的增强版，不应当作为第一优先复现方案。

结论：之前“直接给 teacher actor 加 height_scan”的建议需要降级为工程补救，不是严格复现 PDF 的首选。

### 3. Adaptive sampling 可能在早期过度采样失败片段

当前 `MotionCommand._adaptive_sampling(...)` 会统计失败的 `(motion_id, time_bin)`，并提高这些 bin 被采样的概率。

多 tracks 时，困难片段数量增加，尤其是从障碍一侧跨到另一侧的位置。训练早期 policy 尚未掌握基本爬越动作，如果 adaptive sampling 太早集中到失败位置，PPO rollout 会被大量低质量、频繁 reset 的样本主导，导致 value loss 和 policy loss 不稳定。

#### 解决方法

- 训练早期先使用更均匀的 sampling。
- 增大 `adaptive_uniform_ratio`，例如从 `0.1` 提到 `0.5` 或 `1.0`。
- 降低 `adaptive_alpha`，减慢失败统计对采样分布的影响。
- 或者前几千 iterations 禁用 adaptive sampling，等基础跟踪稳定后再开启 hard mining。

### 4. Termination 不一定是代码错误，但可作为调试手段

当前 termination 使用 anchor 全局位置误差：

```python
anchor_pos = DoneTerm(
    func=mdp.bad_anchor_pos,
    params={"command_name": "motion", "threshold": 0.5},
)
```

这个 `0.5m` 和 PDF 中 expert training 的目标设置一致，所以它不是复现错误。  
但是当 motion/terrain 对齐、轨迹质量或采样分布存在问题时，`0.5m` 会让 episode 很快 reset，从而把其他问题放大。

#### 解决方法

- **忠实复现 PDF**：expert 保持 `0.5m`。
- **调试/排错**：临时放宽到 `0.8` 或 `1.0`，用于判断失败是否来自过早 termination。
- **最终训练**：如果目标是复现 PDF，排除坐标和数据问题后应回到 `0.5m`。

## 训练层面的主要问题

### 1. 24 tracks 是否符合 per-skill expert training 需要确认

PDF 的目标不是 “一个 teacher 学所有技能”，而是 per-skill expert training。  
所以 24 tracks 是否合理，取决于这 24 条是不是同一个 skill 下的多样化 reference。

合理情况：

- 24 tracks 都是同一种 skill，例如同一种 climb，只是起点、速度、微小地形扰动或镜像不同。
- 每条 track 都和自己的 terrain/reference 配对一致。

不合理情况：

- 24 tracks 混合了不同 skill，例如 climb、vault、jump、walk 等。
- 24 tracks 使用不同 terrain pose，但训练环境只加载一个固定 terrain pose。
- 24 tracks 的 motion phase/接触模式差异过大，已经超过单一 expert 的目标范围。

#### 解决方法

- 按 skill 分组训练 expert。
- 同一 expert 内只混合同一 skill 的 reference variants。
- 多 skill 泛化交给 student distillation，不要提前塞进 teacher expert。

### 2. 直接从少量 tracks 跳到大量 tracks，任务分布变化太大

3 tracks 能成功，不代表 24 tracks 从零训练也能成功。24 tracks 带来更多：

- 起始姿态差异；
- 障碍边缘位置差异；
- 跨越速度差异；
- 落脚相位差异；
- 接触状态差异；
- 失败片段数量。

PPO 对这种突然扩大的分布比较敏感，尤其是运动模仿和障碍跨越任务。

#### 解决方法

使用 curriculum：

1. 先用 3 tracks 训练到稳定。
2. 从 checkpoint resume 到 6 tracks。
3. 再 resume 到 12 tracks。
4. 最后 resume 到 24 tracks。

不要一开始直接从随机初始化训练 24 tracks。

### 3. PPO 更新可能过激

当前默认配置：

```python
learning_rate = 1.0e-3
desired_kl = 0.01
num_learning_epochs = 5
num_mini_batches = 4
```

这些数值和 PDF 中 expert PPO 超参数一致，因此不是复现错误。  
但如果当前实现里存在 terrain/reference 错位，或者 24 tracks 的数据分布超出 per-skill expert 的假设，`1e-3` 会更容易把不稳定放大。

#### 解决方法

建议尝试：

- `learning_rate = 3e-4` 或 `5e-4`
- `desired_kl = 0.005`
- 增大 `num_mini_batches`
- 保持或进一步限制 `max_grad_norm`
- 观察 KL、entropy、value loss、episode length 是否同步异常

### 4. Domain randomization 和障碍爬越难度叠加

训练里有摩擦、关节默认位置、COM、push 等随机化。对于普通 tracking 这有利于鲁棒性，但在爬障碍早期训练时，随机化会显著增加难度。

#### 解决方法

先关闭或减弱随机化，等 teacher 学会基本爬越后再逐步打开：

- 暂时关闭 `push_robot`。
- 缩小摩擦随机范围。
- 缩小 COM randomization。
- 缩小 ankle joint default pos randomization。

### 5. 需要检查轨迹质量

多 tracks 训练失败也可能来自少数坏轨迹。只要 24 tracks 中混入几条物理上不可跟踪、地形对齐错误、速度突变或接触不合理的轨迹，PPO 就可能被这些失败轨迹拖垮。

#### 解决方法

对每条 track 单独 replay 和单独训练短 run：

- 检查 root/body position 是否和 terrain 对齐。
- 检查跨障碍边缘时脚底是否穿模或悬空。
- 检查 root height 是否连续。
- 检查 joint velocity 是否有尖峰。
- 检查单条 track 是否能在较小 num_envs 下稳定学会。

## 建议的修复优先级

### P0：先修坐标/terrain 对齐

这是最可能的硬 bug。确认每条 motion 的 reference、robot reset pose、实际 terrain collision 三者在同一个坐标系中。

建议先写一个检查脚本，对每条 trajectory 输出：

- `terrain_world_pose`
- `skill_anchor.root_translation`
- `skill_execution.output_start_frame`
- offset 后的 root position
- terrain 实际加载 pose

如果同一个 training run 里 terrain 实际加载 pose 只有一个，而 motion offset 使用多个 terrain pose，就不要继续训练。

### P1：对齐 PDF 的 per-skill expert 训练方式

确认 24 tracks 是否属于同一个 skill。如果不是，应拆成多个 teacher expert。  
如果是同一个 skill，也要确认它们的 terrain/reference 配对关系没有被当前单 terrain 加载逻辑破坏。

### P2：确认 critic privileged observation 生效

按 PDF，critic 应该接收：

- pelvis/anchor global position；
- pelvis/anchor global velocity；
- terrain heightmap。

当前代码已加入 global anchor position 和 global anchor linear velocity；下一步重点确认保存的 `env.yaml` 中 critic observation 维度和 term 列表符合预期。

### P3：只在必要时增加 actor 的 phase 或 terrain-relative 信息

如果 terrain 对齐、per-skill 分组、critic privileged observation 都正确后，24 tracks 仍然失败，再考虑增强 actor：

- 加 motion phase。
- 必要时加 track/skill embedding。
- 必要时加局部 terrain-relative 信息。

这属于工程补救或增强，不是严格复现 PDF 的第一选择。

### P4：使用 curriculum

按 tracks 数量逐步扩大：

```text
3 tracks -> 6 tracks -> 12 tracks -> 24 tracks
```

每一步从上一步 checkpoint resume。

### P5：调低训练激进程度

多 tracks 阶段建议更保守：

```text
learning_rate: 3e-4 或 5e-4
desired_kl: 0.005
anchor_pos termination threshold: 0.8 或 1.0 起步
adaptive_uniform_ratio: 0.5 或 1.0 起步
```

### P6：逐步恢复随机化

先让 teacher 在干净环境下学会爬越，再逐步加入：

- push；
- friction randomization；
- COM randomization；
- joint default position randomization。

## 验证 Checklist

训练 24 tracks 前，至少确认以下项目：

- 所有 motion 和 terrain collision 在同一坐标系。
- 多轨迹时没有把不同 `terrain_world_pose` 的 motion 混到一个固定 terrain pose 里。
- 单条 track replay 时，机器人初始 root 和障碍位置对齐。
- 单条 track 可以稳定训练通过障碍。
- 3 tracks checkpoint resume 到更多 tracks，而不是 24 tracks 从零开始。
- teacher actor 的标准观测能区分 reference phase 和当前 tracking error。
- 如果目标是严格复现 PDF，teacher actor 不应优先加入 privileged heightmap；terrain heightmap 应先确认在 critic privileged observation 中正确生效。
- 24 tracks 属于同一个 skill，而不是把多 skill 提前混到 teacher expert。
- early training 的 adaptive sampling 不会被失败片段完全主导。
- termination 不会在跨越阶段过早 reset。

## 最可能的根因排序

1. 多轨迹中 `terrain_world_pose` 不一致，但 terrain 只按第一条 trajectory pose 加载。
2. 24 tracks 可能偏离 PDF 的 per-skill expert training 假设，或 motion/terrain 配对没有保持一致。
3. Critic privileged observation 已补齐 global anchor position/linear velocity，但需要通过新训练日志确认生效。
4. Adaptive sampling 早期过度集中失败片段。
5. Termination threshold 本身符合 PDF，但会放大坐标错位或坏轨迹问题。
6. PPO learning rate / KL 设置符合 PDF，但在数据错位或分布过宽时会显得偏激进。
7. 少数 track 数据质量或物理可跟踪性有问题。
