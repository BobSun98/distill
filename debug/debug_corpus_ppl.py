"""真实 GPT-2-Large 单卡 debug；当前进程直接进入数据采样和 GPT NLL。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from distill.corpus_ppl import main


if __name__ == "__main__":
    main(default_config="configs/gpt2_corpus_ppl_debug.yaml", allow_distributed=False)
