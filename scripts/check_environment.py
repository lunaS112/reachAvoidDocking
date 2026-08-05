#!/usr/bin/env python3
"""
Environment check script for MPC Evaluation Framework.
Validates Python version, required packages, and system capabilities.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Verify Python 3.8+"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (需要 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (未安装)")
        return False

def check_required_packages():
    """Check essential packages"""
    print("\n✓ Checking required packages...")

    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
    ]

    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False

    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\n✓ Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用")
            print(f"  ✓ GPU 数量: {torch.cuda.device_count()}")
            print(f"  ✓ GPU 型号: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"  ✗ CUDA 不可用（需要 GPU）")
            return False
    except ImportError:
        print(f"  ✗ PyTorch 未安装")
        return False

def check_repository_structure():
    """Check if required directories exist"""
    print("\n✓ Checking repository structure...")

    repo_root = Path(__file__).parent.parent
    required_dirs = [
        "deepReachMPCReachAvoid",
        "gridBased6DImplementation",
    ]

    required_files = [
        "compare_hybrid_10s.sbatch",
        "grid_baseline_17s.sbatch",
        "DEPLOYMENT_OTHER_CLUSTERS.md",
    ]

    all_ok = True
    for dirname in required_dirs:
        dirpath = repo_root / dirname
        if dirpath.is_dir():
            print(f"  ✓ {dirname}/")
        else:
            print(f"  ✗ {dirname}/ (未找到)")
            all_ok = False

    for filename in required_files:
        filepath = repo_root / filename
        if filepath.is_file():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (未找到)")
            all_ok = False

    return all_ok

def check_model_checkpoints():
    """Check if model checkpoints exist"""
    print("\n✓ Checking model checkpoints...")

    repo_root = Path(__file__).parent.parent
    checkpoints = [
        "deepReachMPCReachAvoid/runs/Docking6D_RA/training/checkpoints/model_final.pth",
        "deepReachMPCReachAvoid/runs/Docking6D_Vanilla/training/checkpoints/model_final.pth",
    ]

    all_found = True
    for ckpt in checkpoints:
        ckpt_path = repo_root / ckpt
        if ckpt_path.is_file():
            size_mb = ckpt_path.stat().st_size / (1024**2)
            print(f"  ✓ {ckpt} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠ {ckpt} (未找到，任务运行前需要下载或拷贝)")
            all_found = False

    return all_found

def main():
    print("=" * 60)
    print("MPC Evaluation Framework - Environment Check")
    print("=" * 60)

    checks = [
        ("Python 版本", check_python_version),
        ("必需包", check_required_packages),
        ("CUDA/GPU", check_cuda),
        ("仓库结构", check_repository_structure),
        ("模型检查点", check_model_checkpoints),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ 检查 {name} 时出错: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 需要修复"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！可以提交任务")
        print("  运行:")
        print("    sbatch compare_hybrid_10s.sbatch")
        print("    sbatch grid_baseline_17s.sbatch")
    else:
        print("✗ 部分检查失败，请修复后重试")
        print("\n建议步骤:")
        print("  1. 安装依赖: pip install -r requirements.txt")
        print("  2. 下载模型检查点")
        print("  3. 调整 sbatch 脚本中的集群参数")
        print("  4. 重新运行此脚本验证")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
