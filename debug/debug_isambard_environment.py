"""真实 CUDA 运算和工程双通信组检查，不加载模型、不启动训练。"""

import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from distill import distributed
from distill.config import write_json


if __name__ == "__main__":
    owns_group = distributed.setup("cuda")
    try:
        distributed.setup_tensor_group("cuda", timeout_seconds=60)
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        # 小型真实 CUDA 运算只验证环境；正式模型的峰值显存由 compile-debug 验证。
        values = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
        assert (values @ values).eq(16).all().item()
        rows = distributed.gather_main({"rank": distributed.rank(), "device": device,
            "gpu": properties.name, "total_memory_gib": properties.total_memory / 1024**3,
            "bf16": torch.cuda.is_bf16_supported()})
        if distributed.is_main():
            report = {"python": platform.python_version(), "architecture": platform.machine(),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "world_size": distributed.world_size(),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
                "cuda_bf16_matmul": "passed", "project_collectives": "passed", "devices": rows}
            write_json(sys.argv[1], report)
            print(json.dumps(report, indent=2), flush=True)
    finally:
        if owns_group:
            distributed.close()
