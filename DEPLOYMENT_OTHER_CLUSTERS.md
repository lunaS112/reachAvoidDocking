# 在其他集群上部署 MPC Evaluation Framework

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/lunaS112/reachAvoidDocking.git
cd reachAvoidDocking
```

### 2. 环境检查
```bash
python3 scripts/check_environment.py
```

### 3. 安装依赖
```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 4. 调整 sbatch 脚本

编辑 `compare_hybrid_10s.sbatch` 和 `grid_baseline_17s.sbatch`：

**修改项目**（根据你的集群调整）：

```bash
# 集群账户和分区
#SBATCH --account=YOUR_ACCOUNT          # 改成你的账户
#SBATCH --partition=YOUR_GPU_PARTITION   # 改成你的GPU分区（可选）

# GPU 类型（根据集群调整）
#SBATCH --gres=gpu:h100:1                # h100、v100、a100、rtx_a5000 等

# 资源需求
#SBATCH --cpus-per-task=8                # CPU 核心数
#SBATCH --mem=32G                        # 内存

# 时间限制（根据你的集群调整）
#SBATCH --time=04:00:00                  # 最大运行时间
```

### 5. 准备模型和数据

**必需文件**：
- `runs/Docking6D_RA/training/checkpoints/model_final.pth` - BRAT 模型
- `runs/Docking6D_Vanilla/training/checkpoints/model_final.pth` - Vanilla 模型
- `./outputs/grid_cache/` - 缓存的网格基础真值（自动生成）

**文件结构**：
```
reachAvoidDocking/
├── deepReachMPCReachAvoid/
│   ├── runs/
│   │   ├── Docking6D_RA/
│   │   │   └── training/checkpoints/model_final.pth
│   │   └── Docking6D_Vanilla/
│   │       └── training/checkpoints/model_final.pth
│   └── outputs/
│       └── grid_cache/     # 自动创建
├── compare_hybrid_10s.sbatch
└── grid_baseline_17s.sbatch
```

### 6. 提交任务

```bash
cd reachAvoidDocking

# 提交对比任务（10.0s 时间限制）
sbatch compare_hybrid_10s.sbatch

# 提交基线任务（17.0s 时间限制）
sbatch grid_baseline_17s.sbatch

# 查看任务状态
squeue -u $USER | grep Hybrid_Compare
```

### 7. 检查结果

任务完成后，结果保存在：
```bash
deepReachMPCReachAvoid/outputs/
├── hybrid_vs_baseline_10s/
│   ├── comparison_results.json      # 10s 对比指标
│   ├── trajectory.png
│   └── simulation_data.png
└── grid_baseline_17s/
    ├── comparison_results.json      # 17s 基准指标
    ├── trajectory.png
    └── simulation_data.png
```

**查看结果**：
```python
import json

# 加载 10s 对比结果
with open('deepReachMPCReachAvoid/outputs/hybrid_vs_baseline_10s/comparison_results.json') as f:
    results_10s = json.load(f)

# 加载 17s 基准结果
with open('deepReachMPCReachAvoid/outputs/grid_baseline_17s/comparison_results.json') as f:
    results_17s = json.load(f)

# 显示指标
for controller, metrics in results_10s.items():
    print(f"{controller}:")
    print(f"  Docking Rate: {metrics['docking_rate']*100:.1f}%")
    print(f"  Failure Rate: {metrics['failure_rate']*100:.1f}%")
    print(f"  Timeout Rate: {metrics['timeout_rate']*100:.1f}%")
```

---

## 常见集群配置

### Compute Canada (Alliance)
```bash
#SBATCH --account=def-YOUR_PI_CODE
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
```

### XSEDE/ACCESS
```bash
#SBATCH -A YOUR_PROJECT_ID
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu_4
```

### 本地 Slurm 集群
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

### AWS ParallelCluster
```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
```

---

## 故障排除

### 问题：模块找不到（ModuleNotFoundError）

**解决方案**：
```bash
# 检查 Python 版本
python3 --version  # 需要 3.8+

# 手动安装缺失的包
pip install torch jax jaxlib numpy matplotlib

# 或使用提供的 requirements.txt
pip install -r requirements.txt
```

### 问题：CUDA 相关错误

**解决方案**：
```bash
# 检查 CUDA 可用性
python3 -c "import torch; print(torch.cuda.is_available())"

# 加载 CUDA 模块（如果没有自动加载）
module load cuda/12.2  # 或你的集群版本
module load cudnn      # 如果需要
```

### 问题：任务超时

**解决方案**：增加时间限制
```bash
#SBATCH --time=06:00:00  # 从 04:00:00 改为 06:00:00
```

### 问题：内存不足（OOM）

**解决方案**：
```bash
#SBATCH --mem=64G  # 从 32G 增加到 64G
```

### 问题：找不到检查点文件

**检查清单**：
```bash
# 确保模型文件存在
ls deepReachMPCReachAvoid/runs/Docking6D_RA/training/checkpoints/model_final.pth
ls deepReachMPCReachAvoid/runs/Docking6D_Vanilla/training/checkpoints/model_final.pth

# 如果不存在，从原始repo 下载或从你的 MPC 项目拷贝
```

---

## 输出格式

### comparison_results.json
```json
{
  "grid_based": {
    "docking_rate": 0.98,
    "failure_rate": 0.01,
    "timeout_rate": 0.01,
    "mean_control_effort": 245.3,
    "mean_wall_time": 3.2
  },
  "brat": {
    "docking_rate": 0.95,
    ...
  },
  ...
}
```

### 关键指标说明
- **docking_rate**: 成功对接的比例 (0-1)
- **failure_rate**: 发生碰撞的比例 (0-1)
- **timeout_rate**: 超时未对接的比例 (0-1)
- **mean_control_effort**: 平均控制力
- **mean_wall_time**: 平均实际运行时间 (秒)

---

## 总结：两个关键数字

**对比完后，你会得到**：

1. **10s 对比** (`hybrid_vs_baseline_10s/`)
   - hybrid (trained@17s) vs grid_based vs brat
   - 验证方法是否可行

2. **17s 基准** (`grid_baseline_17s/`)
   - grid_based ground truth
   - 用来对比 hybrid 的改进

**最后计算改进比例**：
```python
improvement = (hybrid_docking_rate - grid_baseline_docking_rate) / grid_baseline_docking_rate * 100
print(f"Hybrid vs Grid Baseline: {improvement:+.1f}%")
```

---

## 需要帮助？

1. 检查 `.sbatch` 文件的参数是否符合你的集群要求
2. 查看 job logs: `tail -f job_logs/Hybrid_Compare_10s-*.out`
3. 对标你的集群文档调整 SBATCH 参数
