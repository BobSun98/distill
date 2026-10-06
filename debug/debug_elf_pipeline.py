"""真实 ELF-B/L 单卡 debug，当前进程直接进入 T5 编码、蒸馏与原版采样。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.elf.pipeline import main

if __name__ == "__main__":
    main(default_config="configs/elf_b_debug.yaml", allow_distributed=False)
