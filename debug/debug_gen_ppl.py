"""真实模型和 GPT-2-Large 的统一评测，单卡当前进程可进入采样与评分断点。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.gen_ppl import main


if __name__ == "__main__":
    main(default_config="configs/gen_ppl_debug.yaml", allow_distributed=False, debug=True)
