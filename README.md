# Continuous DLM Capacity Distillation

第一版使用 **真实 OWT 文本加噪 → 冻结 teacher 查询 → 删层 student 学习 posterior**。
默认 teacher 为 `Continuous-Rivals-Discrete/langflow-owt`，student 从其 12 层中保留
`[0, 2, 4, 7, 9, 11]` 共 6 层，并复制其余模块权重。共享 embedding 和 Gumbel
schedule 冻结；其余 student 参数由 AdamW 更新。删掉一半 block 不代表总参数减半，
每次实验会记录总参数、非 embedding 参数与可训练参数。

模型和采样实现复制在 `src/distill/backbones/langflow/`，外部 `ELF/`、`LangFlow/`
保持原样。项目目标见 [DistillationContinuesDLM.md](DistillationContinuesDLM.md)。

## 服务器运行

在项目根目录安装依赖。服务器需事先安装适配驱动的 CUDA PyTorch；下面命令使用
现有环境，不自动更换 CUDA 版本。

```bash
/proj/gpu_mtk53742/.conda/envs/distill/bin/python -m pip install -e .

# 先跑真实模型的小样本流程：准备数据、训练 3 步、保存重载、生成比较。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh configs/owt_debug.yaml debug

# 初步效果实验：1000 个 optimizer steps，每步累积 8 个单样本 micro-batches。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh configs/owt_kd.yaml all
```

初版为单卡、单进程。第一次准备 OWT 可能下载整个 Parquet 源数据集；文档数限制
减少的是 tokenization 和训练开销，不是 Hugging Face 的源数据下载量。
已有 `Dataset.save_to_disk()` 格式的原始 OWT（含 `text` 列）可以直接复用：

```bash
bash scripts/run_experiment.sh configs/owt_debug.yaml debug \
  --set data.local_dataset=/path/to/raw_owt
```

该目录应是完整、有原始顺序的 OWT `Dataset`，不能是 `DatasetDict` 或已经 packing 的
LangFlow `.dat` 缓存。默认划分依照现有 LangFlow：最后 100000 篇作为保留验证集，
训练取前 `train_documents` 篇，验证取保留部分的前 `valid_documents` 篇。
可设置 `DATA_CACHE` 或 `data.hf_cache_dir` 复用 Hugging Face 下载缓存。

使用本地 teacher 权重时，目录需包含 `config.json` 和 `model.safetensors`：

```bash
bash scripts/run_experiment.sh configs/owt_kd.yaml all \
  --set model.teacher=/path/to/langflow-owt
```

单独的 safetensors 文件也支持，但需同时设置
`model.teacher_config=/path/to/config.json`。只读取配置与权重，不执行远端模型代码。

## 配置与对照实验

`configs/owt_kd.yaml` 包含模型、数据、loss、训练与评测配置；debug 配置继承它，
只减少样本、序列长度、训练步数和生成开销，teacher/student 架构保持相同。
CLI 的 `--set key=value` 可覆盖已有配置字段：

```bash
# CE-only：同一删层初始化、数据、噪声与 SC 规则，学习真实 token。
bash scripts/run_experiment.sh configs/owt_kd.yaml all \
  --set run.name=owt_ce_12to6 --set loss.kd_weight=0 --set loss.ce_weight=1

# KD + CE。
bash scripts/run_experiment.sh configs/owt_kd.yaml all \
  --set run.name=owt_kd_ce_12to6 --set loss.ce_weight=0.1

# 仅准备数据 / 仅训练。
bash scripts/run_experiment.sh configs/owt_kd.yaml prepare
bash scripts/run_experiment.sh configs/owt_kd.yaml train

# 单独评测一个训练好的 student，仍比较 teacher/pruned/trained 三组。
bash scripts/run_experiment.sh configs/owt_kd.yaml evaluate \
  --checkpoint run/<实验目录>/checkpoints/step_001000
```

每次运行创建 `run/<时间>_<名称>/`，保存有效 `config.yaml`、`models.json`、
`metrics.jsonl`、`status.json`、checkpoint 和生成结果。
token 数据缓存位于 `run/data/`，同一预处理配置会复用。临时脚本和检查位于 `tmp/`。
运行产物和下载资产不提交 Git。

checkpoint 保存可重新加载的模型、tokenizer、层映射与 loss 配置。当前不保存 optimizer
及数据迭代位置，不提供训练过程的精确断点续跑。

## 训练状态与评测含义

加噪只执行一次，teacher/student 接收同一个 `(z, gamma, self_cond)`；沿用代码约定
`alpha=sqrt(sigmoid(-gamma))`、`sigma=sqrt(sigmoid(gamma))`。
OWT 文档尾部添加 EOS，连续 packing 后每个 block 再添加 BOS/EOS，不使用 padding。
缓存与 tokenizer、长度、原始数据源和划分规则一起记录，避免误用 debug 数据。

以 0.25 概率启用 self-conditioning：student 在当前加噪状态先做一次无梯度、无 dropout
预测，得到 clean embedding 估计，detach 后同时传给 teacher/student；否则传零。
这是一条明确的 off-policy 蒸馏规则，还不是实际 student rollout 的上一时刻状态。

KD 使用 temperature=1 的完整词表 forward KL，先按词表求和，再按 token 平均。
CE-only 不在训练 batch 上查询 teacher posterior，验证时仍报告 KL。
验证随机噪声固定，且不扰动后续训练 RNG。`ce` 是 **denoising CE**，不能将其指数
直接解释成整个扩散模型的 PPL。

评测分别生成 teacher、删层未训练 student、训练后 student 的样本；使用相同初始随机
种子、batch size、长度、冻结 schedule 和原始 Euler-EDM sampler。`num_steps` 等于
当前 sampler 的模型前向次数（NFE，含最终 token readout）。
默认由 `gpt2-large` 计算 token 加权的生成 PPL，同时报告 `distinct_2` 和逐样本 token
entropy，并保存 token IDs 和解码文本。这里的生成 PPL 聚合方式与论文的逐样本 PPL
平均不完全相同，因此应使用本项目同一协议下的三组结果作比较。

debug 默认不加载 scorer，少量样本和 8 NFE 的结果只用于流程验证，不能作为生成质量
结论。初步实验完成后，正式结果应增加生成样本量并补充 MAUVE 等指标。

## IDE 调试与后续扩展

IDE 直接运行 `debug/debug_pipeline.py`，可在 `prepare_data`、`off_policy_state`、
`batch_loss`、`train`、`generate` 内设置断点。所有步骤在当前 Python 进程内执行，
没有 subprocess，也不构造 tiny/mock 模型或自造 tokenizer。

本地 CPU 小样本调试可用：

```bash
PY=/Users/bobsun/miniconda3/envs/common/bin/python \
  bash scripts/run_experiment.sh configs/owt_debug.yaml debug --set model.device=cpu
```

主要模块为 `models.py`、`data.py`、`states.py`、`losses.py`、`train.py`、`evaluate.py`。
后续 on-policy 采集可以产生同一个 `DistillationState(z, gamma, self_cond)`，复用
`posterior_logits` 与 loss；local flow loss 可在 `losses.py` 添加。当前不引入通用 trainer、
模型注册表、插件系统或 replay buffer，等实际需要时再实现。

算法依据：
[LangFlow 训练算法与实验协议](https://arxiv.org/html/2604.11748v3#A1)。
