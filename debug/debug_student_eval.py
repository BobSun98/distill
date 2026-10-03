"""真实 student 的单卡评测调试：直接进入 Flow NLL、采样及 GPT-2 评分函数。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.student_eval import main


if __name__ == "__main__":
    main(default_config="configs/langflow_student_eval_debug.yaml", allow_distributed=False)
