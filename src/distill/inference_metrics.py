"""统一生成评测的实际前向耗时；CUDA events 不逐步阻塞采样器。"""

import math
import time

import torch


class ForwardTimer:
    def __init__(self, model, warmup=5):
        self.model, self.warmup = model, warmup
        self.device = next(model.parameters()).device
        self.calls, self.measured_calls, self.total_ms = 0, 0, 0.0
        self.pending = []

    def __enter__(self):
        self.previous_instance_forward = self.model.__dict__.get("forward")
        self.original = self.model.forward

        def timed_forward(*args, **kwargs):
            self.calls += 1
            if self.calls <= self.warmup:
                return self.original(*args, **kwargs)
            if self.device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream(self.device)
                start.record(stream)
                result = self.original(*args, **kwargs)
                end.record(stream)
                self.pending.append((start, end))
            else:
                start = time.perf_counter()
                result = self.original(*args, **kwargs)
                self.total_ms += (time.perf_counter() - start) * 1000
            self.measured_calls += 1
            return result

        # LangFlow.generate_samples 直接调用 self.forward，普通 forward hook 无法覆盖。
        self.model.forward = timed_forward
        return self

    def flush(self):
        if self.pending:
            torch.cuda.synchronize(self.device)
            self.total_ms += math.fsum(start.elapsed_time(end) for start, end in self.pending)
            self.pending.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.previous_instance_forward is None:
            del self.model.forward
        else:
            self.model.forward = self.previous_instance_forward
        if exc_type is None:
            self.flush()

    def result(self):
        return {"forward_time_ms": self.total_ms, "timed_forward_calls": self.measured_calls,
                "warmup_forward_calls": min(self.calls, self.warmup), "total_forward_calls": self.calls,
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "method": "cuda_events" if self.device.type == "cuda" else "cpu_perf_counter"}


def aggregate_timing(shards):
    calls = sum(row["timed_forward_calls"] for row in shards)
    milliseconds = math.fsum(row["forward_time_ms"] for row in shards)
    # 各卡按实际前向次数加权；不把总生成时间、八卡 wall time 或 GPT scorer 混入分母。
    return {"mean_step_ms": milliseconds / calls if calls else None,
            "timed_forward_calls": calls, "forward_time_ms": milliseconds,
            "timing_device_names": sorted({row.get("device_name", "unknown") for row in shards}),
            "warmup_forward_calls": sum(row["warmup_forward_calls"] for row in shards),
            "total_forward_calls": sum(row["total_forward_calls"] for row in shards)}
