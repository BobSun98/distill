"""单卡原始 Teacher -> 真实全层SVD -> 纯MSE训练 -> 统一评测，同进程直接调用工程函数。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

import torch
from distill.svd_mse import main


if __name__ == "__main__":
    if not {"--dry-run", "--help", "-h"}.intersection(sys.argv) and not torch.cuda.is_available():
        raise RuntimeError("真实 Teacher SVD/MSE 验证需要服务器 CUDA；本地只运行 --dry-run")
    main(default_config="configs/owt_svd_pure_mse_debug.yaml", allow_distributed=False, debug=True)
