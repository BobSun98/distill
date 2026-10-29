"""单卡真实 Teacher 的全部 block 投影 SVD、四种学生、KL 训练与统一评测，同进程进入断点。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.rank_sweep import main


if __name__ == "__main__":
    main(default_config="configs/owt_rank_sweep_debug.yaml", allow_distributed=False, debug=True)
