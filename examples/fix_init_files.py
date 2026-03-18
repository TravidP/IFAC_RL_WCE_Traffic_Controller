#!/usr/bin/env python3
import os

# ===== 1. 设置你的项目路径 =====
PROJECT_ROOT = "/home/sdc_joran/flow"

# 你的 multiagent 环境文件目录
TARGET_PATH = os.path.join(PROJECT_ROOT, "flow/examples/exp_configs/rl/multiagent")

def ensure_init_files(path):
    """确保从 path 到项目根的所有目录都有 __init__.py（自动建目录）"""
    created = []
    current = path
    while current.startswith(PROJECT_ROOT):
        # 如果目录不存在就创建
        if not os.path.exists(current):
            os.makedirs(current, exist_ok=True)
            print(f"📂 创建目录: {current}")

        init_file = os.path.join(current, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# auto-generated for Python package recognition\n")
            created.append(init_file)

        # 向上一级
        current = os.path.dirname(current)
        if current == os.path.dirname(PROJECT_ROOT):
            break
    return created


if __name__ == "__main__":
    print(f"🔍 检查并修复路径: {TARGET_PATH}")
    created_files = ensure_init_files(TARGET_PATH)

    if created_files:
        print("\n✅ 已创建以下 __init__.py 文件：")
        for f in created_files:
            print("   ", f)
    else:
        print("✅ 所有目录下的 __init__.py 文件已存在，无需创建。")

    print("\n💡 运行前请设置包根路径：")
    print("   export PYTHONPATH=/home/sdc_joran/flow")
    print("然后运行：")
    print("   python run_checkpoint_eval_direct2.py")
