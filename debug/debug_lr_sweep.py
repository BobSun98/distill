"""真实 LangFlow-OWT LR sweep：单卡同进程直接训练与测 PPL。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.lr_sweep import main


if __name__ == "__main__":
    main(default_config="configs/owt_lr_sweep_debug.yaml", allow_distributed=False)
