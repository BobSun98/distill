本目录的 `model.py`、`config.py`、`__init__.py` 复制自项目中的
`LangFlow/langflow/`，上游为 https://github.com/nealchen2003/LangFlow 。

本项目只调整 autocast 的设备处理及 CPU JIT fusion 开关，使真实模型可以在
CPU 上调试；CUDA 仍沿用原始 bf16 backbone、前向参数化与 Euler-EDM sampler。
外部 `LangFlow/` 与 `ELF/` 不作修改。checkpoint 配置从实际预训练模型读取。

`flow_nll.py` 的模型类复制自 `LangFlow/eval_ppl.py`，保持原版 NLL/散度计算。
`reference_gen.py` 的生成、entropy、评分函数复制自 `LangFlow/gen_ppl.py`，
只新增可选下载缓存路径。`reference_data.py` 的 `_group_texts` 复制自
`LangFlow/duo/dataloader.py`；OWT 分支保留两次 batched map、每篇 EOS、每个 block
BOS/EOS 以及各 map batch 丢弃余数的规则，不复用蒸馏训练的连续 buffer packing。
`text.py` 的 LM1B detokenizer 同样复制自 `LangFlow/duo/dataloader.py`。
LM1B wrapped 兼容原代码 `encode(special_token)[0]` 的边界取值方式；BERT 会自动
前置 CLS，因此此规则的 EOS 也取 CLS，不另行改为 SEP。
