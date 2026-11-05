"""可选的实际 forward 编译；保留原模型结构、权重名和采样公式。"""

import gc
import time
from contextlib import contextmanager

import torch

from . import distributed, progress


_active_sessions = 0


def execution_options(value=None, *, debug=False):
    result = {"backend": "eager", "mode": "reduce-overhead", "warmup_forwards": 3,
              "debug_compile": False, **(value or {})}
    if result["backend"] not in ("eager", "compile"):
        raise ValueError("execution.backend 只支持 eager / compile")
    if result["mode"] not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ValueError("execution.mode 不是有效的 torch.compile 模式")
    if type(result["warmup_forwards"]) is not int or result["warmup_forwards"] < 3:
        raise ValueError("execution.warmup_forwards 至少为 3，覆盖首次编译与 CUDA Graph 预热")
    if type(result["debug_compile"]) is not bool:
        raise ValueError("execution.debug_compile 必须是布尔值")
    if debug and not result["debug_compile"]:
        result["backend"] = "eager"
    return result


def execution_metadata(value=None):
    settings = execution_options(value)
    compiled = settings["backend"] == "compile"
    return {"execution_backend": settings["backend"],
            "compile_mode": settings["mode"] if compiled else None,
            "cuda_graphs_requested": compiled and settings["mode"] in ("reduce-overhead", "max-autotune")}


def tensor_group_timeout(config):
    timeout = config["distributed"]["tensor_timeout_seconds"]
    return max(timeout, 900) if execution_options(config.get("execution"))["backend"] == "compile" else timeout


def mark_step_begin():
    if _active_sessions:
        torch.compiler.cudagraph_mark_step_begin()


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype),
                str(value.device), value.requires_grad)
    if isinstance(value, dict):
        return tuple((key, _signature(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_signature(item) for item in value)
    return (type(value).__name__, value)


@contextmanager
def generation_context(model):
    state = getattr(distributed.unwrap(model), "_distill_execution_state", None)
    if state is None:
        yield
        return
    previous = state.generating
    state.generating = True
    try:
        yield
    finally:
        state.generating = previous


def forward_preparer(model):
    state = getattr(distributed.unwrap(model), "_distill_execution_state", None)
    return state.prepare if state is not None else None


def model_execution_metadata(model):
    state = getattr(distributed.unwrap(model), "_distill_execution_state", None)
    return state.metadata() if state is not None else execution_metadata()


class _CompiledForward:
    def __init__(self, model, settings, role):
        self.model, self.settings, self.role = model, settings, role
        self.device = next(model.parameters()).device
        self.previous_forward = model.__dict__.get("forward")
        self.original = model.forward
        self.target = getattr(model, "backbone", model)
        self.target_name = "backbone.forward" if self.target is not model else "forward"
        self.previous_target_forward = self.target.__dict__.get("forward")
        self.original_target = self.target.forward
        self.previous_generate = model.__dict__.get("generate_samples")
        self.original_generate = getattr(model, "generate_samples", None)
        self.generating = False
        self.primed = False
        self.ready = set()
        self.warmup_seconds, self.warmup_calls = 0.0, 0
        self.graph_launches = 0
        # 编译真正会被 self.forward 调用的方法，不创建 _orig_mod 参数前缀。
        self.compiled = torch.compile(self.original_target, mode=settings["mode"], dynamic=False, fullgraph=False)
        if self.target is not model:
            self.target.forward = self.compiled
        model.forward = self.forward
        model.__dict__["_distill_execution_state"] = self
        if self.original_generate is not None:
            model.generate_samples = self.generate

    def generate(self, *args, **kwargs):
        with generation_context(self.model):
            return self.original_generate(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self.compiled(*args, **kwargs) if self.target is self.model else self.original(*args, **kwargs)

    def prime(self, *args, **kwargs):
        if self.target is not self.model:
            self.target.forward = self.original_target
        try:
            output = self.original(*args, **kwargs)
            del output
        finally:
            if self.target is not self.model:
                self.target.forward = self.compiled
        self.primed = True

    def prepare(self, *args, **kwargs):
        if not self.generating:
            return
        key = (self.model.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
               torch.is_autocast_enabled(), _signature(args), _signature(kwargs))
        if key in self.ready:
            return
        started = time.perf_counter()
        with progress.phase("execution.warmup", role=self.role, mode=self.settings["mode"], variant=len(self.ready) + 1):
            # 真实输入预热位置缓存和两种 SC 分支；额外前向不消耗正式采样/训练的 RNG。
            with torch.random.fork_rng(devices=[self.device.index]):
                self.prime(*args, **kwargs)
                self.warmup_calls += 1
                for _ in range(self.settings["warmup_forwards"]):
                    mark_step_begin()
                    output = self.invoke(*args, **kwargs)
                    del output
                    self.warmup_calls += 1
                torch.cuda.synchronize(self.device)
                if self.settings["debug_compile"] and execution_metadata(self.settings)["cuda_graphs_requested"]:
                    # 专用真实模型 debug 只观察热态调用，避免把编译期 autotune 的图误当成模型重放。
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        mark_step_begin()
                        output = self.invoke(*args, **kwargs)
                        torch.cuda.synchronize(self.device)
                        del output
                    launches = sum(event.count for event in profiler.key_averages() if "graphlaunch" in event.key.lower())
                    self.warmup_calls += 1
                    if launches == 0:
                        raise RuntimeError("热态 forward 未观察到 CUDA Graph replay；请检查 TORCH_LOGS=perf_hints 的跳过原因")
                    self.graph_launches += launches
        self.warmup_seconds += time.perf_counter() - started
        self.ready.add(key)

    def forward(self, *args, **kwargs):
        if self.generating:
            self.prepare(*args, **kwargs)
            # 采样前一步的 logits 已经用于 eager softmax/transport；在下一个 forward 前可回收图输出。
            mark_step_begin()
        elif not self.primed:
            # 真实输入先用 eager 建立 RoPE 等模块缓存，避免缓存长期持有图私有池中的 Tensor。
            with torch.random.fork_rng(devices=[self.device.index]), torch.no_grad():
                self.prime(*args, **kwargs)
        return self.invoke(*args, **kwargs)

    def metadata(self):
        return {**execution_metadata(self.settings), "compile_target": self.target_name,
                "cuda_graphs_verified": True if self.graph_launches else None,
                "observed_cuda_graph_launches": self.graph_launches,
                "compile_warmup_seconds": self.warmup_seconds,
                "compile_warmup_forward_calls": self.warmup_calls, "compiled_input_variants": len(self.ready)}

    def restore(self):
        model = self.model
        if self.target is not model:
            if self.previous_target_forward is None:
                self.target.__dict__.pop("forward", None)
            else:
                self.target.forward = self.previous_target_forward
        for name, previous in (("forward", self.previous_forward), ("generate_samples", self.previous_generate)):
            if name == "generate_samples" and self.original_generate is None:
                continue
            if previous is None:
                model.__dict__.pop(name, None)
            else:
                setattr(model, name, previous)
        model.__dict__.pop("_distill_execution_state", None)
        # Timer/闭包可能还引用本对象，显式断开编译图，评分 GM 前释放图私有内存池。
        self.compiled = self.original = self.original_generate = self.model = None
        self.target = self.original_target = self.previous_target_forward = None


class ExecutionSession:
    def __init__(self, models, value=None, *, run_dir, generation=False):
        self.models = models
        self.settings = execution_options(value)
        self.run_dir, self.generation = run_dir, generation
        self.states, self.owned, self.previous_generation = {}, [], {}
        self.saved_metadata = {}
        self.active = False

    def __enter__(self):
        global _active_sessions
        if self.settings["backend"] == "eager":
            if any(getattr(model, "_distill_execution_state", None) is not None for model in self.models.values()):
                raise ValueError("当前模型已有活动编译上下文，不能将其实际执行标记成 eager")
            return self
        for model in self.models.values():
            if next(model.parameters()).device.type != "cuda":
                raise RuntimeError("compile + CUDA Graph 后端需要真实 CUDA 模型；本地/debug 请用 eager")
        if any(not callable(getattr(torch.compiler, name, None)) for name in ("reset", "cudagraph_mark_step_begin")):
            raise RuntimeError("当前 PyTorch 缺少 compiler.reset / cudagraph_mark_step_begin，请升级服务器 PyTorch")
        try:
            if _active_sessions == 0:
                torch.compiler.reset()
            _active_sessions += 1
            self.active = True
            for role, model in self.models.items():
                state = getattr(model, "_distill_execution_state", None)
                if state is None:
                    state = _CompiledForward(model, self.settings, role)
                    self.owned.append(state)
                elif state.settings != self.settings:
                    raise ValueError("同一模型的嵌套执行后端必须一致")
                self.states[role] = state
                self.previous_generation[role] = state.generating
                state.generating = self.generation or state.generating
            progress.emit("execution.enabled", **execution_metadata(self.settings), roles=list(self.models))
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def metadata(self, role):
        if role in self.saved_metadata:
            return self.saved_metadata[role]
        if role in self.states:
            return self.states[role].metadata()
        return {**execution_metadata(self.settings), "compile_warmup_seconds": 0.0,
                "cuda_graphs_verified": False, "observed_cuda_graph_launches": 0,
                "compile_target": None, "compile_warmup_forward_calls": 0, "compiled_input_variants": 0}

    def __exit__(self, exc_type, error, traceback):
        if not self.active:
            return False
        if error is not None:
            # 编译异常/显存不足也快速结束本作业，避免其他 rank 在 collective 等两小时。
            distributed.abort_worker(error, self.run_dir, reason="oom" if isinstance(error, torch.OutOfMemoryError) else "execution")
        try:
            self._close(synchronize=error is None)
        except BaseException as cleanup_error:
            distributed.abort_worker(cleanup_error, self.run_dir, reason="execution")
            raise
        return False

    def _close(self, synchronize):
        global _active_sessions
        if synchronize:
            torch.cuda.synchronize()
        for role, state in self.states.items():
            self.saved_metadata[role] = state.metadata()
            state.generating = self.previous_generation[role]
        for state in reversed(self.owned):
            state.model.zero_grad(set_to_none=True)
            state.restore()
        self.states.clear()
        self.owned.clear()
        self.models = {}
        _active_sessions -= 1
        if _active_sessions == 0:
            torch.compiler.reset()
            gc.collect()
        self.active = False
