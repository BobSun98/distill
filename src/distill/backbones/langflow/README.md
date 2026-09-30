本目录的 `model.py`、`config.py`、`__init__.py` 复制自项目中的
`LangFlow/langflow/`，上游为 https://github.com/nealchen2003/LangFlow 。

本项目只调整 autocast 的设备处理及 CPU JIT fusion 开关，使真实模型可以在
CPU 上调试；CUDA 仍沿用原始 bf16 backbone、前向参数化与 Euler-EDM sampler。
外部 `LangFlow/` 与 `ELF/` 不作修改。checkpoint 配置从实际预训练模型读取。
