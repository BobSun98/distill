"""单卡真实模型的随机 blocks 蒸馏，同进程调用采集、训练、PPL/entropy 与 GM。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.off_policy import main


if __name__ == "__main__":
    main(default_config="configs/off_policy_random_langflow_owt_debug.yaml", allow_distributed=False, debug=True)
