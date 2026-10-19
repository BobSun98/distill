"""单卡真实 off-policy checkpoint 的 rollout、查询 teacher、反传及评测。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.on_policy.pipeline import main


if __name__ == "__main__":
    main(default_config="configs/on_policy_langflow_owt_debug.yaml", allow_distributed=False, debug=True)
