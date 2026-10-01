"""各 rank 的阶段日志：在长耗时操作之前写盘，阻塞时仍能定位最后一步。"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_log_path = None
_rank = int(os.environ.get("RANK", "0"))
_started = time.perf_counter()


def configure(run_dir, rank):
    global _log_path, _rank, _started
    _rank, _started = rank, time.perf_counter()
    directory = Path(run_dir) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    _log_path = directory / f"rank_{rank:02d}.jsonl"
    # 只指定输出位置，日志级别仍由 NCCL_DEBUG 控制；每个进程使用独立文件。
    os.environ.setdefault("NCCL_DEBUG_FILE", str(directory / "nccl.%h.%p.log"))


def emit(stage, **fields):
    event = {"time": datetime.now().astimezone().isoformat(timespec="seconds"),
             "rank": _rank, "pid": os.getpid(), "stage": stage,
             "elapsed_seconds": time.perf_counter() - _started, **fields}
    if _log_path is not None:
        with _log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"[rank {_rank}] {stage} {json.dumps(fields, ensure_ascii=False)}", flush=True)


@contextmanager
def phase(name, **fields):
    emit(name + ".begin", **fields)
    started = time.perf_counter()
    try:
        yield
    except Exception as error:
        emit(name + ".failed", error=f"{type(error).__name__}: {error}", **fields)
        raise
    else:
        emit(name + ".end", seconds=time.perf_counter() - started, **fields)
