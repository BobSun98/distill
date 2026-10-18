"""真实 LangFlow 6→11 层训练与 teacher 统一评测，单卡同进程断点入口。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.depth_sweep import main


if __name__ == "__main__":
    main(default_config="configs/owt_depth_sweep_unified_debug.yaml", allow_distributed=False, debug=True)
