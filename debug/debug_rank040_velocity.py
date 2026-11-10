"""单卡全层低秩 checkpoint 的 KL+velocity 续训与统一评测，架构与可见卡由入口指定。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4").split(",")[0]

import torch
from distill.off_policy import main


if __name__ == "__main__":
    if not {"--dry-run", "--help", "-h"}.intersection(sys.argv) and not torch.cuda.is_available():
        raise RuntimeError("真实低秩 checkpoint 续训验证需要服务器 CUDA；本地只运行 --dry-run")
    main(default_config="configs/owt_rank040_velocity_debug.yaml", allow_distributed=False, debug=True)
