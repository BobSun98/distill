"""四卡真实80%学生短验证，覆盖生成、DDP反向和梯度累积；每个worker直接调用工程函数。"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("此入口使用torchrun启动四个worker，请通过compile-debug-ddp作业运行")
    # 先在当前计算节点验证临时目录，提前暴露Triton创建PTX时的目录错误。
    with tempfile.NamedTemporaryFile(suffix=".ptx", dir=os.environ["TMPDIR"]):
        pass
    print(f"[runtime] rank={os.environ['RANK']} TMPDIR={tempfile.gettempdir()}", flush=True)
    from distill.rank_sweep import main

    main(default_config="configs/owt_rank080_isambard_ddp_debug.yaml", allow_distributed=True, debug=True)
