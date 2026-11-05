"""单卡真实 LangFlow/SVD/训练/生成验证；所有工程步骤都在当前 Python 进程。"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

from safetensors import safe_open
import torch
from distill.config import write_json
from distill.rank_sweep import main


if __name__ == "__main__":
    config_only = bool({"--dry-run", "--help", "-h"}.intersection(sys.argv))
    environment = None
    if not config_only:
        if not torch.cuda.is_available():
            raise RuntimeError("编译验证入口需要服务器 CUDA；本地只运行 --dry-run")
        # 环境写入启动日志；即使首次编译失败，也能回看显卡与初始可用显存。
        gpu = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        environment = {"python": sys.version.split()[0], "pytorch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu": gpu.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": total_bytes, "gpu_free_memory_bytes_at_start": free_bytes,
            "torchinductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
            "torch_logs": os.environ.get("TORCH_LOGS")}
        print("[compile-debug] environment: " + json.dumps(environment, ensure_ascii=False), flush=True)
    directory = main(default_config="configs/owt_tail_rank_sweep_compile_debug.yaml", allow_distributed=False, debug=True)
    if directory is not None:
        launches = 0
        evaluated_models = []
        for stage in ("before", "after"):
            receipt = json.loads((directory / f"{stage}_evaluation.json").read_text())
            report = json.loads((Path(receipt["directory"]) / "metrics.json").read_text())
            launches += sum(row.get("observed_cuda_graph_launches", 0) for row in report["models"])
            evaluated_models.extend({"stage": stage, **{key: row.get(key) for key in (
                "name", "execution_backend", "compile_mode", "compile_target", "cuda_graphs_verified",
                "observed_cuda_graph_launches", "compile_warmup_seconds", "mean_step_ms")}}
                for row in report["models"])
        checkpoint_count = 0
        for checkpoint in directory.glob("tail_*/checkpoints/*/model.safetensors"):
            with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
                if any("_orig_mod" in name for name in weights.keys()):
                    raise RuntimeError("编译包装污染了 checkpoint 参数名")
            checkpoint_count += 1
        if checkpoint_count == 0:
            raise RuntimeError("真实训练流程未保存 safetensors checkpoint")
        if launches == 0:
            raise RuntimeError("实际流程未观察到 CUDA Graph replay")
        write_json(directory / "cuda_graph_verification.json", {"status": "passed", "environment": environment,
            "evaluated_models": evaluated_models, "observed_cuda_graph_launches": launches,
            "cuda_graph_replay_verified": launches > 0, "checkpoint_count": checkpoint_count,
            "parameter_names_preserved": True, "pipeline": "real teacher -> SVD -> train -> unified generation/PPL/GM"})
        print(f"[compile-debug] CUDA Graph replay 验证通过，结果: {directory}", flush=True)
