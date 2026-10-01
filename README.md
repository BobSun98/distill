# Continuous DLM Capacity Distillation

第一版使用 **真实 OWT 文本加噪 → 冻结 teacher 查询 → 删层 student 学习 posterior**。
默认 teacher 为 `Continuous-Rivals-Discrete/langflow-owt`，student 从其 12 层中保留
`[0, 2, 4, 7, 9, 11]` 共 6 层，并复制其余模块权重。共享 embedding 和 Gumbel
schedule 冻结；其余 student 参数由 AdamW 更新。删掉一半 block 不代表总参数减半，
每次实验会记录总参数、非 embedding 参数与可训练参数。

模型和采样实现复制在 `src/distill/backbones/langflow/`，外部 `ELF/`、`LangFlow/`
保持原样。项目目标见 [DistillationContinuesDLM.md](DistillationContinuesDLM.md)。

## 服务器运行

在项目根目录安装依赖。服务器需事先安装适配驱动、满足 `pyproject.toml` 版本要求的
CUDA PyTorch，然后安装本项目。

```bash
/proj/gpu_mtk53742/.conda/envs/distill/bin/python -m pip install -e .

# 先跑真实模型的小样本流程：准备数据、训练 3 步、保存重载、生成比较。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh configs/owt_debug.yaml debug

# 初步效果实验：默认使用 CUDA 0-7，八进程 DDP 训练和并行评测。
bash scripts/run_experiment.sh configs/owt_kd.yaml all
```

正式入口默认 `NGPU=8`、`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，每卡各有一份
冻结 teacher 与 student。student 使用 DDP 同步梯度，teacher 只做本卡前向。
数据仅由 rank 0 准备；训练分片不重复，每个 epoch 至多丢弃 7 个尾部 block，
并在下一个 epoch 重新 shuffle。只由 rank 0 写实验日志和 checkpoint。

`training.batch_size` 为 **每卡 micro-batch**。默认每卡 4、累积 2 次，八卡全局 batch
为 `8 × 4 × 2 = 64` 个 block，每步约 65536 个训练 token。`max_steps` 仍表示
optimizer 更新次数；相同更新步数下，当前默认处理的样本数是旧单卡默认的 8 倍。
不会自动按卡数缩放学习率。`models.json` 记录实际卡数和名义全局 batch，
`metrics.jsonl` 记录全局 token 平均 loss、实际 token 数以及 rank 0 的峰值显存。

可以调整并行度与每卡 batch，保持实验预算明确：

```bash
# 两张指定 GPU。
NGPU=2 CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_experiment.sh configs/owt_kd.yaml train

# 每卡 8、累积 1 次，八卡全局 batch 仍为 64；需在服务器上确认实际显存开销。
bash scripts/run_experiment.sh configs/owt_kd.yaml train \
  --set training.batch_size=8 --set training.grad_accumulation=1

# 多个实验并行启动时使用不同的端口。
MASTER_PORT=29501 bash scripts/run_experiment.sh configs/owt_kd.yaml train
```

梯度累积期间只在最后一次 backward 同步，并按全局有效 token 数归一化。
self-conditioning 的无梯度预测直接调用原始 student，避免各 rank 的随机 SC 开关
导致额外 DDP 前向通信。验证分片不补齐，指标按真实 token 数跨卡汇总。

准备数据、下载权重、广播路径、等待 checkpoint 使用 CPU/Gloo 控制通信；
DDP 梯度和 GPU 指标使用独立的 NCCL 组。等待磁盘/网络时不会调用 NCCL barrier。
模型文件由 rank 0 解析/下载并共享路径，各 rank 并行加载；全部模型就绪后才进入 DDP。
训练前会执行 1 MiB 的 all_reduce 与 broadcast，提前检查真实多卡通信。
GPU 通信超时默认为 180 秒，可用 `--set distributed.tensor_timeout_seconds=600` 调整；
CPU 准备阶段允许长下载，超时单独配置。

## 启动阻塞与通信诊断

如果 GPU 利用率持续 100% 但每卡显存只有几百 MB，且没有模型加载/训练日志，
优先检查 NCCL 通信等待。利用率不能证明训练已经开始，最终应以各 rank 的阶段日志定位。
先停止旧的前台 torchrun（Ctrl+C），同步代码后单独检查八卡通信：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO \
  bash scripts/run_experiment.sh configs/owt_kd.yaml check-distributed
```

成功时每个 rank 应打印 `communication.all_reduce.end`、
`communication.broadcast.end` 和 `communication.check.completed`。
此命令不下载数据或模型，也不执行训练。
如果仍停在 NCCL 初始化/all_reduce，做一次关闭 P2P 的对照：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 \
  bash scripts/run_experiment.sh configs/owt_kd.yaml check-distributed
```

若只在关闭 P2P 时通过，提示问题与 GPU P2P 传输路径有关，需要结合驱动、NCCL 版本和
GPU 拓扑进一步定位。该参数只用于对照或临时绕过，不默认关闭 P2P，以免影响正常性能。
依据：[NVIDIA GPU 通信排查](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/gpu_troubleshooting.html)。

通信检查通过后，先用真实模型验证八卡两步训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO \
  bash scripts/run_experiment.sh configs/owt_kd.yaml train \
  --set training.max_steps=2 --set training.log_every=1 --set training.validation_batches=1
```

确认 `ddp.init.end`、`train.micro_batch.backward_done` 和训练 loss 输出后，再运行 `all`。
如果使用临时 P2P 绕过，应在对应的训练命令中显式设置相同环境变量。

启动日志和所有异常堆栈保存在 `run/launch/*.log`；实验目录中的
`logs/rank_00.jsonl` 到 `logs/rank_07.jsonl` 记录每个 rank 的操作开始/结束、PID 和耗时。
`logs/nccl.<主机名>.<PID>.log` 保存 NCCL 原生日志（级别由 `NCCL_DEBUG` 控制）。
单看 rank 0 的最后一条日志还不够，应比较所有 rank 的最后阶段：

```bash
# 将路径替换为本次新实验目录。
tail -n 3 run/<实验目录>/logs/rank_*.jsonl
```

`communication.all_reduce.begin` 表示在测试 GPU 通信；`model.files.resolve.begin`
表示正在解析/下载文件；`model.weights.load.begin` 表示 CPU 权重加载；
`model.to_device.begin` 表示搬到 GPU；`ddp.init.begin` 表示 DDP 的参数同步；
`validation.first_batch.begin` 和 `train.micro_batch.forward_done` 可区分前向与反向阻塞。

## 数据准备与本地权重

第一次准备 OWT 可能下载整个 Parquet 源数据集；文档数限制
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

每次运行创建 `run/<时间>_<名称>/`，保存有效 `config.yaml`、`distributed.json`、`models.json`、
`metrics.jsonl`、`status.json`、每 rank 阶段日志、checkpoint 和生成结果。
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

评测分别生成 teacher、删层未训练 student、训练后 student 的样本；同一个 rank 使用
相同初始随机种子、batch size、长度、冻结 schedule 和原始 Euler-EDM sampler。`num_steps` 等于
当前 sampler 的模型前向次数（NFE，含最终 token readout）。
默认由 `gpt2-large` 计算 token 加权的生成 PPL，同时报告 `distinct_2` 和逐样本 token
entropy，并保存 token IDs 和解码文本。这里的生成 PPL 聚合方式与论文的逐样本 PPL
平均不完全相同，因此应使用本项目同一协议下的三组结果作比较。

生成样本总数 `evaluation.num_samples` 在各 rank 之间分配，默认每卡生成 batch 为 4。
每个 rank 的种子为 `evaluation.seed + rank`，rank 0 按 rank 顺序合并样本并计算整体
多样性。scorer 同样在各卡并行，先汇总 NLL 与 token 数再计算 PPL。
评测协议会记录 world size 和每卡生成 batch；改变卡数或生成 batch 会改变随机样本，
对照实验应固定这些设置。

debug 默认不加载 scorer，少量样本和 8 NFE 的结果只用于流程验证，不能作为生成质量
结论。初步实验完成后，正式结果应增加生成样本量并补充 MAUVE 等指标。

## IDE 调试与后续扩展

debug 脚本直接启动一个 Python 进程，只暴露指定的第一张 GPU（默认 GPU 0）；
不会调用 torchrun。IDE 直接运行 `debug/debug_pipeline.py`，可在 `prepare_data`、`off_policy_state`、
`batch_loss`、`train`、`generate` 内设置断点。所有步骤在当前 Python 进程内执行，
没有 subprocess，也不构造 tiny/mock 模型或自造 tokenizer。
debug 入口也会拒绝 `WORLD_SIZE > 1` 的启动方式。

本地 CPU 小样本调试可用：

```bash
PY=/Users/bobsun/miniconda3/envs/common/bin/python \
  bash scripts/run_experiment.sh configs/owt_debug.yaml debug --set model.device=cpu
```

主要模块为 `models.py`、`data.py`、`states.py`、`losses.py`、`train.py`、`evaluate.py`。
`distributed.py` 只负责必要的进程组和指标通信，单进程下这些调用保持本地行为。
后续 on-policy 采集可以产生同一个 `DistillationState(z, gamma, self_cond)`，复用
`posterior_logits` 与 loss；local flow loss 可在 `losses.py` 添加。当前不引入通用 trainer、
模型注册表、插件系统或 replay buffer，等实际需要时再实现。

算法依据：
[LangFlow 训练算法与实验协议](https://arxiv.org/html/2604.11748v3#A1)。
