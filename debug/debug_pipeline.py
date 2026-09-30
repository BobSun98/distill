"""IDE 入口：真实 checkpoint、真实 OWT，直接进入数据准备/训练/推理函数。"""

import sys
from pathlib import Path

# 从任意工作目录运行，不需要先安装项目，也不启动其他 Python 进程。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill.pipeline import main


if __name__ == "__main__":
    main(default_config="configs/owt_debug.yaml")
