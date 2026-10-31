"""真实 12 层 LangFlow 的十组尾部低秩学生：单卡、同进程采集/训练/评测。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.rank_sweep import main


if __name__ == "__main__":
    main(default_config="configs/owt_tail_rank_sweep_debug.yaml", allow_distributed=False, debug=True)
