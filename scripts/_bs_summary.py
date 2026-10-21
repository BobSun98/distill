"""汇总 /tmp/onp_bs_test/logs 里 <tag>_<key>_<value>.log 的探测结果。"""
import ast
import re
import sys
from pathlib import Path

tag = sys.argv[1]
rows = []
for path in Path("/tmp/onp_bs_test/logs").glob(f"{tag}_*.log"):
    value = int(path.stem.rsplit("_", 1)[1])
    text = path.read_text(errors="ignore")
    steps = {}
    for line in text.splitlines():
        if line.startswith("[train] {") and "'phase': 'train'" in line:
            event = ast.literal_eval(line[len("[train] "):])
            steps[event["step"]] = event
    rollout = [float(x) for x in re.findall(r'on_policy\.rollout\.end \{"seconds": ([0-9.]+)', text)]
    oom = "out of memory" in text.lower()
    if 6 in steps and 2 in steps:
        per_step = (steps[6]["elapsed_seconds"] - steps[2]["elapsed_seconds"]) / 4
        status = f"ok  peak={steps[6]['rank0_peak_memory_gib']:.1f}GiB  step={per_step:.2f}s"
        rows.append((value, status, per_step, steps[6]["rank0_peak_memory_gib"], rollout))
    else:
        rows.append((value, "OOM" if oom else "FAILED/timeout", None, None, rollout))
print(f"{'value':>6}  {'status':<36} {'samples/s':>10}  rollout_s")
for value, status, per_step, peak, rollout in sorted(rows):
    rate = f"{value / per_step:10.1f}" if per_step else f"{'-':>10}"
    print(f"{value:>6}  {status:<36} {rate}  {[round(x, 1) for x in rollout[:2]]}")
