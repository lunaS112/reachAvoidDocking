# DeepReach & MPC Evaluation Methodology: Why Hybrid Learning Can't Directly Compare

## Executive Summary

**用户的疑问**：为什么 Hybrid Learning 的结果无法和 MPC DeepReach 直接对比，即使两者都是 DeepReach？

**答案**：因为两个项目的 **ground truth 来源不同** 和 **evaluation metrics 定义不同**。

---

## 1. reachAvoidDocking (MPC DeepReach) 的 Evaluation 流程

### 1.1 Ground Truth 来源

```
Grid-based HJ solver (hj_reachability)
         ↓
Exact value function V_grid(t, state)  [ComboControl.py 生成]
         ↓
Cached in: gridBased6DImplementation/outputs/grid_cache/values_*.npy
         ↓
Used as "oracle" for training/comparison
```

**关键特征**：
- V_grid 是通过精确HJ PDE求解得到的
- 在grid点处精确，其他点通过插值
- 作为 DeepReach 的 supervised target（训练 SIREN network 拟合它）

### 1.2 Evaluation 方法（run_controller.py compare）

```python
# Step 1: Sample random initial conditions (IC)
ics = sample_initial_conditions(
    dynamics=docking_dynamics,
    n=500,
    seed=42,
    sampling_method='uniform'
)
# 过滤：avoid_fn(ic) > 0 AND reach_fn(ic) > 0
#       （即不在obstacle也不在target）

# Step 2: 对每个controller运行轨迹
for controller_name in ['grid_based', 'brat', 'vanilla_brat', 'mpc', ...]:
    trajectories = []
    for ic in ics:
        traj = controller.rollout(ic, t_max=10.0)
        trajectories.append(traj)
    
    # Step 3: 计算metrics（基于最终状态）
    success_count = 0
    for traj in trajectories:
        final_state = traj[-1]
        is_docked = reach_fn(final_state) <= 0  # 在target内
        no_collision = avoid_fn(final_state) > 0  # 不在obstacle
        
        if is_docked and no_collision:
            success_count += 1
    
    docking_rate = success_count / len(trajectories)
    print(f"{controller_name}: docking_rate={docking_rate:.1%}")
```

### 1.3 Evaluation 指标

```
docking_rate     = (docked & no_collision) / total
failure_rate     = (collision occurred) / total
timeout_rate     = (did not dock after t_max) / total

这些指标基于：
✓ 实际轨迹simulation
✓ 真实的reach_fn/avoid_fn（ComboControl.py中定义）
✗ NOT直接比对value function的accuracy
```

**关键点**：evaluation是 **behavior-based**，不是 **value-based**

---

## 2. cmpt720_hybrid_hj (Hybrid Learning) 的 Evaluation 问题

### 2.1 不同的 Ground Truth

```
reachAvoidDocking:
  V_gt = Grid-based HJ solution from ComboControl.py
       = exact on 6D grid (px, py, vx, vy, theta, omega)
       
cmpt720_hybrid_hj:
  V_gt = ??? (depends on which dynamics/artifact it uses)
       = might be different grid resolution
       = might be different state scaling
       = might be different dynamics class definition
```

### 2.2 可能的不兼容问题

**问题 1：State Space Definition**
```
reachAvoidDocking state = [px, py, vx, vy, theta, omega]
  px, py ∈ [-15, 15] m
  vx, vy ∈ [-1.5, 1.5] m/s
  theta ∈ [-π, π] rad (periodic)
  omega ∈ [-2, 2] rad/s

cmpt720_hybrid_hj state = [x, y, vx, vy, theta, omega] (maybe?)
  但state scaling可能不同
  或者normalize到[-1, 1]?
  或者使用不同的state ordering?
```

**问题 2：Dynamics Definition**
```
reachAvoidDocking: 使用 Docking_translational + Docking_rotational
  (from ComboControl.py, CW轨道动力学)
  
cmpt720_hybrid_hj: 使用自己的 Docking4DTranslational class (from YH branch)
  (可能复制+修改了ComboControl的dynamics)
  (可能有subtle差异)
```

**问题 3：Value Function Definition**
```
reachAvoidDocking:
  V(t, x) = min time to reach target while avoiding obstacle
          = -∞ if in obstacle, 0 if in target, >0 otherwise
  
cmpt720_hybrid_hj:
  V_hat(x) = teacher signal (supervised target for neural network)
           = might be scaled differently
           = might be time-integrated loss instead of min-time
```

**问题 4：Training Objective**
```
reachAvoidDocking (DeepReach):
  Loss = ||NN(x) - V_gt(x)||²_L2  (everywhere)
  
cmpt720_hybrid_hj (Hybrid Learning):
  Loss = ||NN(x) - V_hat(x)||²_L2  (smooth regions only, m_corner < 0.5)
       + PDE_residual(NN)(x)  (corner regions, m_corner > 0.5)
       
  这意味着：
  - NN的目标不是拟合V_gt，而是满足 LQR-like criterion
  - 部分区域用PDE loss替代supervised loss
  - 最终的NN(x)可能systematically不同于V_gt
```

---

## 3. 为什么无法直接对比

### 根本原因

| 方面 | reachAvoidDocking | cmpt720_hybrid_hj | 影响 |
|------|-------------------|-------------------|------|
| Ground truth源 | ComboControl grid (4.4GB) | ??? (check YH branch) | ✗ 不同的oracle |
| State定义 | CW轨道(px,py,vx,vy) | ??? (maybe不同scaling) | ✗ 不同的state space |
| Dynamics类 | ComboControl.Docking_* | YH branch Docking4D* | ✗ 可能subtle差异 |
| Training目标 | supervised MSE | hybrid supervised+PDE | ✗ 不同的NN学习目标 |
| NN输出scale | V ∈ [0, 10] (reaching time) | ??? (maybe normalized) | ✗ output scale不匹配 |
| Checkpoint来源 | 自己的external/deepreach | 自己的external/deepreach | ✗ 两个repo diverge |

### 直接对比失败的场景

```python
# 这样做会失败：
ic = np.array([1.0, 2.0, 0.1, -0.1, 0.0, 0.5])

# reachAvoidDocking DeepReach checkpoint
v_brat = brat_ckpt(ic)  # Maybe: 3.5 (需要3.5秒到达target)

# cmpt720_hybrid_hj Hybrid Learning checkpoint  
v_hybrid = hybrid_ckpt(ic)  # Maybe: 0.42 (scaled到[0,1]?)

# 无法比较！scale/semantics都不同
```

---

## 4. 怎样才能 **使之可对比**

### 方案 A：使用相同的 Ground Truth（最稳健）

```
1. 确认 cmpt720_hybrid_hj 使用了正确的ground truth:
   - 加载 reachAvoidDocking/gridBased6DImplementation/artifacts/values_4D_*.npy
   - 加载 reachAvoidDocking/gridBased6DImplementation/artifacts/values_2D_*.npy
   - 重构成6D: V_gt = max(V_4D, V_2D)

2. 修改 cmpt720_hybrid_hj 的 export_docking4d_from_reachavoid.py:
   - 确保加载的是同一个cached grid
   - 确保state ordering一致
   - 确保state scaling一致

3. 验证：
   query_point = np.array([1.0, 2.0, 0.1, -0.1, 0.0, 0.5])
   v_gt_reachavoid = reachavoid_interpolator(query_point, t=9.0)
   v_gt_hybrid = hybrid_interpolator(query_point, t=9.0)
   assert np.allclose(v_gt_reachavoid, v_gt_hybrid)  # 必须相同！
```

### 方案 B：使用相同的 Dynamics Class

```
1. 在 cmpt720_hybrid_hj 中导入 reachAvoidDocking 的 Docking_translational/Docking_rotational
2. 不要修改或复制它们
3. 在YH branch中统一指向ComboControl.py的原始定义

# In cmpt720_hybrid_hj/src/learning/dynamics_registry.py
from reachAvoidDocking.gridBased6DImplementation.ComboControl import (
    Docking_translational, 
    Docking_rotational
)
```

### 方案 C：验证 State Correspondence

```python
# 关键：state索引/scaling必须完全一致
reachavoid_state = np.array([px, py, vx, vy, theta, omega])
hybrid_state = reachavoid_state  # 必须完全相同

# 检查ordering：
# reachavoid: [px, py, vx, vy, theta, omega]
# hybrid:     [x,  y,  vx, vy, theta, omega]  (CW coordinates = same as px, py)

# 检查scaling：
# reachavoid: px ∈ [-15, 15]，未normalize
# hybrid:     都normalize到[-1, 1]?  ✗ 这会破坏对比
```

### 方案 D：Evaluation Pipeline 对齐

```python
# 都应该使用 reachAvoidDocking 的 run_controller.py compare 流程：

# For cmpt720_hybrid_hj checkpoint:
1. Convert NN checkpoint to reachAvoidDocking-compatible format
   (或创建wrapper来query它)

2. 和grid-based、MPC等controller一起在 run_controller.py compare 中运行
   python run_controller.py compare \
     --controllers grid_based brat vanilla_brat deepreach_hybrid \
     --n_rollouts 500 --seed 19

3. 输出统一的 comparison_results.json
```

---

## 5. 检查清单：你的组员应该做的

**立即检查**：

- [ ] YH branch 中的 Docking4DTranslational 和 ComboControl.Docking_translational 是否相同？
- [ ] Ground truth grid (v_hat_all.npy) 的加载路径指向哪里？
- [ ] State ordering/scaling 是否和 reachAvoidDocking 完全匹配？
- [ ] 最终NN checkpoint的output scale是什么？（reaching time 还是 normalized [0,1]）

**如果要对比**：

- [ ] 使用相同的 run_controller.py compare 流程
- [ ] 或者在 cmpt720_hybrid_hj 中创建 reachAvoidDocking-compatible controller wrapper
- [ ] 用相同的 500 个 ICs（固定 seed=19）
- [ ] 输出相同格式的 comparison_results.json

---

## 6. 你应该问你的组员的问题

1. **"你们的hybrid checkpoint在哪里？我怎样加载它？"**
2. **"你们用的ground truth grid来自reachAvoidDocking吗？还是自己生成的？"**
3. **"状态向量是 [px, py, vx, vy, theta, omega]吗？有没有改变顺序或scale?"**
4. **"你们的NN输出是reaching time（单位：秒）还是normalized值？"**
5. **"你能用 run_controller.py compare 跑你的checkpoint吗？"**

---

## 7. 我的建议

**为了让你的 10D quadrotor 和 8D aircraft 系统将来能正确地对比 Hybrid Learning 和 Native DeepReach**，建议：

1. **现在就对齐** cmpt720_hybrid_hj 中的实现
   - 确保使用相同的 ground truth
   - 确保 state 定义完全一致
   - 创建 reachAvoidDocking-compatible wrapper

2. **为将来的系统制定标准**
   - 分别创建 `src/dynamics/` classes
   - 分别创建 `src/evaluation/` metrics
   - 共享 `artifacts/ground_truth/` grid

3. **不要依赖于 "直接对比" 的假设**
   - 始终使用 rollout simulation 来评估
   - 始终使用相同的采样方法（ICs）
   - 始终使用相同的metrics定义

