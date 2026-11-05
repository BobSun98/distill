"""单卡直接进入真实 80% checkpoint -> 75% -> 70% 的采集/截断/训练/评测流程。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

import torch
from distill.progressive_rank import main


if __name__ == "__main__":
    if not {"--dry-run", "--help", "-h"}.intersection(sys.argv) and not torch.cuda.is_available():
        raise RuntimeError("真实逐级蒸馏验证需要服务器 CUDA；本地只运行 --dry-run")
    main(default_config="configs/owt_progressive_rank_debug.yaml", allow_distributed=False, debug=True)
