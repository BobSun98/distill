`model.py`、`layers.py`、`sampling_utils.py`、`generation_utils.py`、`metrics.py` 复制自项目的
`ELF/src/modules/` 和 `ELF/src/utils/`，保持原版模型结构及 ODE/SDE/decoder 公式。
上游 PyTorch 端口及 checkpoint 信息见 `ELF/README.md`。

模型/采样仅调整包内导入和类型注解；metrics 仅保留 Gen. PPL 使用的四个类。
RoPE 支持截取已有位置编码前缀以运行较短 debug 序列；
不缩小 teacher/student 的宽度或额外减少 debug 模型层数。外部 `ELF/` 不修改。
