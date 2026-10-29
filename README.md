# Continuous DLM Capacity Distillation

第一版使用 **真实 OWT 文本加噪 → 冻结 teacher 查询 → 删层 student 学习 posterior**。
默认 teacher 为 `Continuous-Rivals-Discrete/langflow-owt`，student 从其 12 层中保留
`[0, 2, 4, 7, 9, 11]` 共 6 层，并复制其余模块权重。共享 embedding 和 Gumbel
schedule 冻结；其余 student 参数由 AdamW 更新。删掉一半 block 不代表总参数减半，
每次实验会记录总参数、非 embedding 参数与可训练参数。

LangFlow 和 ELF 模型/采样实现分别复制在 `src/distill/backbones/langflow/`、
`src/distill/backbones/elf/`，外部 `ELF/`、`LangFlow/` 保持原样。
ELF-B/L 的运行方式见下方独立说明。项目目标见 [DistillationContinuesDLM.md](DistillationContinuesDLM.md)。

## 统一 Gen PPL：后续实验正式对比入口

正式对比统一使用 **`scripts/eval_gen_ppl.sh` / `gen_ppl_v1`**，固定 seed=42、
每套权重生成 1024 条、GPT-2-Large scorer、GPT 评分上限 1024 token，默认 CUDA 0-7。
默认只加载给定架构和权重做无条件生成；不额外加载 teacher 或 T5 encoder。
可选 GM 还会加载固定的真实文本参考集，并对 GPT-2-Large 做反向传播。
旧的原版复现入口与训练中诊断 PPL 保留原协议；历史数值需要通过新入口重评后再同表比较。

```bash
# 默认配置示范同一任务内评测 LangFlow OWT 和 LM1B 两套真实权重。
bash scripts/eval_gen_ppl.sh configs/gen_ppl.yaml

# 新训练保存的 checkpoint 自动附带单模型清单。
bash scripts/eval_gen_ppl.sh run/<实验>/checkpoints/step_005000/gen_ppl_eval.yaml

# depth / LR sweep 自动导出多权重清单，一次重评全部已完成条件。
bash scripts/eval_gen_ppl.sh run/<sweep实验>/gen_ppl_models.yaml
```

一个 YAML 的 `models` 可以放任意多套权重，每项独立设置步数、长度，必要时覆盖生成 batch：

```yaml
extends: gen_ppl.yaml
models:
  - name: student6_step1000
    architecture: run/<实验>/checkpoints/step_001000/config.json
    weights: run/<实验>/checkpoints/step_001000/model.safetensors
    generation:
      num_steps: 1024
      sequence_length: 1024
  - name: student6_step5000
    architecture: run/<实验>/checkpoints/step_005000/config.json
    weights: run/<实验>/checkpoints/step_005000/model.safetensors
    generation:
      num_steps: 128
      sequence_length: 1024
```

配置文件保存在 `configs/` 时可直接 `extends: gen_ppl.yaml`；其他目录需提供正确的相对
继承路径，或复制完整清单。模型文件字符串路径相对项目根目录；HF 文件用
`{repo_id: ..., filename: ...}` 明确指定。更多示例见 `configs/gen_ppl_models_example.yaml`。
student 必须使用自己的架构 JSON；safetensors 自身不包含完整架构。
tokenizer 优先读取权重旁保存的真实 tokenizer；LangFlow OWT/LM1B 可由词表识别 GPT-2/BERT，
ELF 使用架构中 `encoder` 的 tokenizer 引用，不加载 encoder 模型权重。

本项目 ELF student 可直接使用保存的 `config.json`。原生 ELF checkpoint 可使用
`configs/architectures/elf_b.json` / `elf_l.json` 加实际权重路径；默认选择存在的
`ema_params1`，否则 `params`/直接 state dict，也可通过模型项的 `weights_key` 明确指定。
需要修改过噪声/SC 设置的架构，应在 JSON 的 `diffusion`、`sampling` 中如实保存推理配置。
读取原生 PyTorch 文件使用 `weights_only=True`，自定义 pickle 对象需先导出模型 state dict。

噪声和采样约定：

- sample_id 固定为 0–1023，按编号分卡，不补齐、不重复。CPU FP32 generator 由
  `(seed, sample_id, noise_stream)` 定位，不依赖模型顺序、卡数或 batch 划分。
- 每个 latent 维度固定占 1024 个随机位置，较短长度/较小维度共享完整噪声的重叠前缀。
  不同潜空间只能共享标准正态 epsilon；各模型的坐标定义和噪声缩放仍来自其架构。
- LangFlow 使用原 Euler-EDM；ELF 使用原 SDE/ODE 公式，时间网格固定并共享，各 SDE step
  也使用可寻址噪声。ELF 生成步数之外还有 decoder 前向；实际采样参数随结果保存。
- 生成参数 FP32、CUDA backbone BF16，scorer 固定 FP32/eager、逐条评分、禁用 KV cache。
  固定噪声不等于跨 GPU/软件版本逐位相同的输出；结果记录 Torch 版本和运行卡数。

统一文本/评分约定：

- 每条序列先去掉至多一个开头 BOS，再在 tokenizer 定义的第一个 EOS 截断，解码时去掉
  special tokens，禁用额外 whitespace cleanup。这样 GPT-2 的 BOS=EOS 不会误清空整条样本。
- 用同一个 GPT-2-Large tokenizer 重新 tokenize，不加人工 BOS/EOS，最多前 1024 token。
  shifted next-token CE 排除 padding，首 token 不评分，正文 EOS 仍按真实内容正常评分。
- `Gen. PPL = exp(所有有效目标 token 的 NLL 总和 / 有效目标 token 总数)`，不平均 PPL。
  空白/不足两个 GPT token 的样本不重采样，记录 coverage；全部不可评分时 PPL 为 null。
- 同时报告 GPT-token unigram entropy、重复 4-gram 比例和空样本数，保留实际文本。
  各权重的长度/步数可以不同，但预算差异会影响对比，应核对结果中的生成设置。

统一接口还输出 `num_parameters` 和 `mean_step_ms`：前者为生成模型的唯一参数总数，
包括冻结 embedding/proposal，GPT scorer 不计入；后者为 **一次 batch 的生成模型前向平均毫秒数**，
按八卡实际计时前向次数加权。CUDA 使用 events，各卡各模型跳过前 5 次实际前向，
不额外采样；CPU 使用 `perf_counter`。计时覆盖 LangFlow 的直接 `self.forward` 调用和
ELF 的实际模型调用（含 terminal decoder），排除加载、采样器 transport、文本处理、
GPT 评分和 GM。`effective_generation_batch_size`、`timed_forward_calls` 和设备名称同时记录，
比较耗时应保持 batch、生成长度、硬件和精度一致。此数值是 batch 前向延迟，未除以 batch size。

产物在 `run/<时间>_unified_gen_ppl/`：`protocol.json`、汇总 `metrics.json`/`summary.csv`，
每个模型目录中的 `samples.jsonl`、含逐样本 NLL 的 `metrics.json`、真实 tokenizer，
以及 `initial_noise_sample_0.pt`（未施加模型噪声缩放的标准 epsilon）。每完成一套权重即更新汇总。

中断后可在相同配置下逐模型续评，已完成模型的质量、参数和耗时结果原样复用：

```bash
bash scripts/eval_gen_ppl.sh run/<已有统一评测目录>/config.yaml all \
  --resume-dir run/<已有统一评测目录>
```

已有模型 `metrics.json` 即为完成记录，包含空文本导致的合法 null 结果；
仅保存 `samples.jsonl` 的未完成模型会重新生成、评分和计时。
续评要求模型、协议、GM、batch 和计时预热设置一致，并复用原运行已记录的 GM 真实参考文本。
分布式 OOM 会保存 `fatal_rank_XX.json` 后立即退出故障 worker，由 torchrun 停止本作业其他 worker，
避免在 NCCL/Gloo 清理中等待超时；单卡 debug 保留普通异常堆栈。

```bash
# 单卡、真实模型、当前进程断点调试；seed/scorer 相同，只减少样本、长度和步数。
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_gen_ppl.sh configs/gen_ppl_debug.yaml debug
```

debug 结果标为 `gen_ppl_v1_debug`，正式入口拒绝样本数不是 1024 的配置，避免误报为正式结果。

### 可选 Gradient Moment（GM）

```bash
# 默认八卡，原统一 PPL/entropy/重复率 + GM；两个语料各评测 1024 条生成样本。
bash scripts/eval_gen_ppl.sh configs/gen_ppl_gm.yaml

# 等价开关。
bash scripts/eval_gen_ppl.sh configs/gen_ppl.yaml all --set gradient_moment.enabled=true

# 单卡同进程真实 debug：4 条生成样本、4 条真实参考、2 个梯度对。
bash scripts/eval_gen_ppl.sh configs/gen_ppl_gm_debug.yaml debug
```

GM 使用 GSF 引用的 [D-MMD 第 5 节、公式 14](https://arxiv.org/html/2603.20155#S5)：
对两个独立样本分别计算“生成文本梯度 − 真实文本梯度”，取内积后对所有梯度对平均。
GPT-2-Large 参数保持不更新，使用全部独立参数（共享 embedding 只计算一次），
FP32 前向/梯度、FP64 内积。loss 为每条文本的有效 token 平均 shifted CE，
与本项目统一 PPL 使用相同的 GPT tokenizer、前 1024 token 截断和 padding 排除规则。
这不是逐样本梯度范数的平均，也不是学生模型或 latent 的梯度。

`gradient_moment.reference_config` 默认指向 `configs/gpt2_corpus_ppl.yaml`，
沿用真实 PPL 的 held-out 范围：OWT 原始 train 最后 100000 条保留区、LM1B 官方 test。
每个语料使用 seed=42 固定抽取 1024 条可评分文本，按文档/句子计样本，不 packing；
debug 随生成样本数减少参考数量。真实 PPL 自身的 512 条默认值不变。
可在该数据配置中填写原始文本 `local_dataset`，复用服务器已有缓存。

每个 `models` 项开启 GM 时必须显式填写 `corpus: owt` 或 `corpus: lm1b`，
不从模型名称或 tokenizer 猜参考集。自定义 student 清单可继承 `gen_ppl_gm.yaml`
并覆盖 `models`；旧清单需补充 `gradient_moment` 配置和每项的 `corpus` 后开启。
默认 `gen_ppl.yaml` 及通用 checkpoint 导出的清单仍关闭 GM。
随机 blocks 四组实验的训练后清单由其配置显式开启 GM，见下方运行说明。

每 2 条生成文本及对应真实文本组成一个梯度对，正式共 512 对。
配对由全局 sample_id 决定，八卡按 pair_id 分工，不同步或写盘完整参数梯度。
同一运行内各模型复用对应语料的参考文本，生成样本也复用于原 PPL/entropy，不重复生成。
GM 原始数据及范围保存在 `gradient_moment/reference_owt.json`、`reference_lm1b.json`。

`summary.csv` / `metrics.json` 新增 `gm`、`gm_standard_error`、`gm_num_pairs`、
`gm_num_skipped_pairs`、`gm_status`、`gm_reference_corpus`；每个模型的
`gradient_moment.json` 记录逐对结果和 `gpt2_large_gm_v1` 协议。
GM 越低代表平均梯度越接近参考数据；有限样本无偏估计可以为负，保留原值和标准误。
空白或不足两个 GPT token 的生成文本使对应整对跳过，不重采样，状态标为
`partial_coverage`；所有对无效时 GM 为 null，不能当作零。
横向比较需使用相同的参考范围、文本长度和 loss 归一化，当前数值不宣称与论文表格逐位对齐。

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

## 训练曲线与定期生成 PPL

LangFlow OWT/LM1B 和 ELF-B/L 的正式训练配置默认开启结束后绘图及定期生成 PPL：

```yaml
training:
  plot_loss: true
  ppl_evaluation:
    enabled: true
    interval: 100
    num_samples_per_rank: 32
    batch_size: 32
```

`interval` 按 **optimizer 更新次数**计数。默认每 100 步暂停训练，各卡用当前 student
生成 32 条，八卡共 **256 条**；`batch_size=32` 表示每卡一次生成 32 条，可调小并分批生成，
总样本数仍由 `num_samples_per_rank` 决定。仅评测当前 step 的内存权重，每次重置同一生成种子。
interval 为正时，到达 max_steps 或平台期早停时，即使不是 interval 的整倍数，也补测一次停止 step。
设置 `training.ppl_evaluation.interval=-1` 则关闭训练循环内的全部自动 PPL 评测，
包括停止 step 的补测；`all` 命令结束后的独立评测仍正常执行。旧的正数 interval 行为不变。

PPL 指 **生成文本在 GPT-2-Large 下的 Gen. PPL**。采样步数、生成长度、scorer 和评分长度
继承当前 `evaluation`；ELF 还继承 SDE/ODE 与 SC-CFG 设置。ELF 沿用原版 BF16/EOS-mask
和 rank 0 整组评分，LangFlow 沿用本项目 `all` 入口的 FP32 scorer、有效 token 掩码和
多卡 NLL 汇总。它们与独立 LangFlow benchmark 的 FP16 或真实语料 PPL 入口是不同数值口径，
对比时请检查结果中记录的精度、掩码、生成 batch 和样本数。

评测不更新 student；结束后恢复训练模式、Python/Torch RNG 和 matmul precision。
GPT scorer 首次加载后缓存 CPU 权重，下一次复用，避免训练阶段常驻 GPU；较快的 rank
通过 CPU/Gloo 等待，不提前进入下一步 NCCL 梯度通信。早停仍依据原有内部验证 loss。
评测次数越多、`evaluation.num_steps` 越大，训练总耗时越长；这里沿用正式采样步数。

在服务器更新绘图依赖后，使用现有启动脚本：

```bash
/proj/gpu_mtk53742/.conda/envs/distill/bin/python -m pip install -e .

# ELF-B 示例，默认每 100 步、八卡各 32 条。
bash scripts/run_elf_experiment.sh configs/elf_b_kd.yaml all

# 自定义间隔；每卡分 4 批生成，样本总数仍为 8 × 32 = 256。
bash scripts/run_elf_experiment.sh configs/elf_l_kd.yaml train \
  --set training.ppl_evaluation.interval=200 \
  --set training.ppl_evaluation.batch_size=8

# LangFlow 同样支持；仅关闭训练期间 PPL，结束绘图仍保留。
bash scripts/run_experiment.sh configs/lm1b_kd.yaml train \
  --set training.ppl_evaluation.enabled=false
```

每次实验目录增加：

- `loss_curve.png`：日志中记录的 train/validation 总 loss，横轴是 optimizer step。
- `ppl_curve.png`：当前 student 的定期生成 PPL；无可评分文本时保留断点，不能画成零。
- `ppl/step_000100/` 等目录：该 step 的 `student_texts.json`、`student_tokens.pt` 和 `metrics.json`。
- `metrics.jsonl` 中 `phase=gen_ppl` 的行：step、PPL、entropy、实际样本数及结果目录。

`ppl/step_*/` 记录当前 step，而结束后的 `evaluation/trained_*` 仍来自验证 loss 最佳
checkpoint；两者可能是不同权重/样本。每次保存的采样协议可用于核对比较条件。

debug 默认不加载 GPT scorer；若要在同一个真实进程里调试“评测→继续训练”的路径：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_elf_experiment.sh configs/elf_b_debug.yaml debug train \
  --set training.ppl_evaluation.enabled=true \
  --set training.ppl_evaluation.interval=1 \
  --set evaluation.scorer=gpt2-large
```

此时仍使用真实 teacher/student，每步只生成 2 条、每批 1 条。旧实验可以直接重绘曲线：

```bash
PYTHONPATH=src /proj/gpu_mtk53742/.conda/envs/distill/bin/python \
  -m distill.training_monitor run/<已有实验目录>
```

## LangFlow-OWT 6→11 层 sweep 与 teacher 统一评测

新配置 `configs/owt_depth_sweep_unified.yaml` 按 **6、7、8、9、10、11 层**执行。
每个 student 独立从同一个原始 12 层 teacher 均匀删层初始化，每组完整训练 **5000 步**，
初始 LR 都为 **0.001**，沿用无 warmup 的 cosine scheduler 衰减至 **0.0001**。
每卡 batch=4、累积 2 次，八卡 global batch=64；每 100 步记录 train/val loss。
关闭早停和训练中 PPL（`interval=-1`），六组共 30000 次 optimizer 更新。

训练完成后释放训练模型，**直接调用与 `scripts/eval_gen_ppl.sh` 相同的 `run_gen_ppl` 工程函数**，
依次重载原始 teacher 和六套第 5000 步 student 权重进行八卡评测。teacher 仅评测，不训练。
七个条件共用 `gen_ppl_v1`：**seed=42、每个模型 1024 条、GPT-2-Large FP32/eager**，
生成长度和采样前向均为 1024。每卡生成 batch=8、共 128 条；同一 sample_id 使用相同初始噪声。

```bash
# 服务器项目根目录；默认 CUDA 0-7、八卡训练及评测。
bash scripts/run_owt_depth_sweep_unified.sh

# 显式指定新配置。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_owt_depth_sweep_unified.sh configs/owt_depth_sweep_unified.yaml all

# 真实模型单卡 debug：六个深度各 3 步，每个模型评测 2 条、长度 128、采样 8 步。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_owt_depth_sweep_unified.sh configs/owt_depth_sweep_unified_debug.yaml debug
```

根目录 `run/<时间>_langflow_owt_depth_sweep_lr1e-3_unified/` 保存 `summary.json`、
`summary.csv`、`depth_ppl_curve.png`；曲线包含 6–11 层 student，以及 12 层 teacher
的星形点和基准虚线，并附 entropy 面板。汇总记录 `role`、层数、checkpoint、实际评分覆盖率
及协议元数据。`student_06_layers/` 等目录保留各自的 loss 图、训练日志和最终权重。
`<时间>_unified_gen_ppl/` 保存统一入口的协议、指标及七个模型的文本、token IDs、逐样本 NLL。

根目录还导出包含 teacher 和六个 student 的 `gen_ppl_models.yaml`，之后可用
`bash scripts/eval_gen_ppl.sh run/<本次实验>/gen_ppl_models.yaml` 单独重评。
debug 必须通过单卡 debug 入口进入，产物明确标为 `gen_ppl_v1_debug`。
旧的 `owt_depth_sweep.yaml` 和启动脚本继续沿用原来的配置及评测流程。

## LangFlow-OWT 独立均匀删层实验（旧入口）

`configs/owt_depth_sweep.yaml` 固定同一个 12 层 LangFlow-OWT teacher，依次训练
11、10、9、8、7、6 层 student。**每个 student 独立从原始 teacher 初始化**，
不继承上一阶段的训练权重或 optimizer。每个深度完整训练 **5000 个 optimizer step**，
关闭早停与训练中间 PPL（`interval=-1`），随后重载第 5000 步 checkpoint，仅评测该 student。
完整实验共 6 × 5000 = 30000 次更新，数据、种子、学习率和全局 batch 保持一致。

均匀方案在原始编号 0–11 上等间距选取保留层：
`round(i * (12 - 1) / (student_depth - 1))`，保留首尾层。各目标深度分别计算，不要求
层集合嵌套；只做深度缩减，宽度、tokenizer、冻结 embedding/schedule 均沿用真实 teacher。

| Student 层数 | 保留的原始 teacher 层（从 0 开始） |
| --- | --- |
| 11 | `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]` |
| 10 | `[0, 1, 2, 4, 5, 6, 7, 9, 10, 11]` |
| 9 | `[0, 1, 3, 4, 6, 7, 8, 10, 11]` |
| 8 | `[0, 2, 3, 5, 6, 8, 9, 11]` |
| 7 | `[0, 2, 4, 6, 7, 9, 11]` |
| 6 | `[0, 2, 4, 7, 9, 11]` |

```bash
# 服务器项目根目录，依次使用全部八卡执行六个独立训练条件。
bash scripts/run_owt_depth_sweep.sh

# 单卡真实 debug：保留同一 12 层 teacher 和六种 student 结构，每种只训练 3 步。
# 当前 Python 进程直接调用训练、保存重载、生成和真实 GPT-2-Large 评分。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_owt_depth_sweep.sh configs/owt_depth_sweep_debug.yaml debug
```

默认每卡训练 batch=4、累积 2 次，八卡全局 batch=64。每个 student 训练结束后生成
**256 条**（八卡各 32 条，生成 batch=32），长度 1024、采样前向 1024 次、seed=2026+rank，
使用本项目 LangFlow `all` 入口的 GPT-2-Large FP32/有效 token 掩码口径。各深度的
生成和评分设置一致，不额外重复评测 teacher 或未训练 pruned student。

根目录 `run/<时间>_langflow_owt_uniform_depth_sweep/` 保存 `summary.json`、`summary.csv`
及 `depth_ppl_curve.png`，曲线横轴为 student 层数，并附 entropy 面板；每完成一个深度
就更新一次。`student_11_layers/` 等阶段目录保存配置、独立训练日志、`loss_curve.png`、
最终 checkpoint 和 `evaluation/` 中的文本、token IDs、逐样本评分。图上的每个点都对应
该 student 的最后一步权重。

可用 `--set evaluation.num_samples=512` 增加每个条件的生成样本，或用
`--set evaluation.batch_size=8` 分批生成。预算/采样协议会写入结果，不改变既有实验配置。

## LangFlow-OWT 12→6 层 LR sweep

`configs/owt_lr_sweep.yaml` 固定 12 层 teacher 与均匀保留 `[0,2,4,7,9,11]` 的 6 层
student，扫描五个初始学习率。每组重新从原始 teacher 初始化、重新创建 optimizer/scheduler，
固定训练数据、随机种子、八卡 global batch=64、5000 个 optimizer 更新，不启用早停。

| 初始 LR | 第 5000 次更新使用的 LR |
| --- | --- |
| `1e-5` | `1e-6` |
| `3e-5` | `3e-6` |
| `1e-4` | `1e-5` |
| `3e-4` | `3e-5` |
| `1e-3` | `1e-4` |

这些是覆盖现有 `3e-5` 起点的候选值，实际效果由本次实验比较。scheduler 使用无 warmup
的余弦衰减，从第一次更新的初始 LR 到最后一次更新的 `lr/10`，按 optimizer step 推进，
不按 micro-batch 推进。旧配置未设置 `lr_scheduler` 时仍使用原有恒定 LR。

```yaml
training:
  lr_scheduler:
    name: cosine
    final_ratio: 0.1
  log_every: 100
  validate_every: 100
  ppl_evaluation:
    enabled: true
    interval: 1000
    num_samples_per_rank: 32
    batch_size: 32
```

每组在 step **1000、2000、3000、4000、5000** 计算当前 student 的生成 PPL，每次八卡
各生成 32 条，共 256 条，GPT-2-Large 的生成/评分口径与现有 LangFlow `all` 一致。
不在结束后重复额外生成一遍。train/val loss 每 100 步记录；另外保留 step=0 验证和
首个训练步的诊断日志，汇总 loss 图的 train 曲线按 100 步取点。

```bash
# 服务器项目根目录：五组依次占用 CUDA 0-7，总计 5 × 5000 次 optimizer 更新。
bash scripts/run_owt_lr_sweep.sh

# 单卡真实 debug：相同 teacher、6 层 student 和五个候选 LR，各训练 3 步。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_owt_lr_sweep.sh configs/owt_lr_sweep_debug.yaml debug
```

结果根目录 `run/<时间>_langflow_owt_12to6_lr_sweep/` 保存：

- `ppl_comparison.png`：5 条 PPL 曲线，横轴 optimizer step，图例标注各自初始 LR。
- `loss_comparison.png`：5 条 train + 5 条 val，共 10 条曲线；同 LR 同色、train 实线/val 虚线。
- `summary.json`、`summary.csv`：每组初始/最终 LR、最后一步 PPL、entropy、checkpoint 等。
- `lr_*/`：各自完整配置、训练日志、loss/PPL 曲线、最终 checkpoint 和逐次 PPL 的文本/评分。

每完成一组就更新汇总图，五组完成后得到完整的 5/10 条曲线。`metrics.jsonl` 的 train
行记录该次 optimizer 更新实际使用的 `learning_rate`；PPL 行也记录对应 LR。
`training.json` 保存 scheduler 配置与实际最终 LR，便于核对衰减终点。
比较结果时同时检查 entropy 和文本，避免仅按低 PPL 判断生成质量。

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

单独的 safetensors 文件也支持，默认读取同目录 `config.json`；也可显式设置
`model.teacher_config=/path/to/config.json`。只读取配置与权重，不执行远端模型代码。

## LM1B 蒸馏与评测

LM1B 入口复用同一删层初始化、KL/CE loss、八卡 DDP 与早停逻辑，不修改 OWT 的
实验配置。`configs/lm1b_kd.yaml` 使用 `Continuous-Rivals-Discrete/langflow-lm1b`、
`bert-base-uncased`、训练/生成长度 128、正式生成 128 次前向。默认每卡 32、梯度
累积 2 次，八卡名义全局 batch 为 512 blocks（约 65536 token/optimizer step）。
这是初始资源配置，尚未在 A6000 实测；可通过 `training.batch_size` 调整。

```bash
# 在服务器项目根目录：先单卡真实模型调试，再八卡蒸馏。
bash scripts/run_experiment.sh configs/lm1b_debug.yaml debug
bash scripts/run_experiment.sh configs/lm1b_kd.yaml all

# 只训练；仍保存最佳和最后一步模型。
bash scripts/run_experiment.sh configs/lm1b_kd.yaml train
```

数据默认使用模型卡列出的 `dvruette/lm1b` Parquet 分支。训练只读取官方 `train`，
在其最后 100000 条文本中保留内部验证，取其中 1024 条用于验证/早停；训练取前
200000 条。官方 `test` 不参与模型选择。`train_documents`/`valid_documents` 在 LM1B
指句子/文本记录数，最终按固定长度 packing 成 blocks。缓存目录使用 `lm1b_` 前缀，
不会误用已有 OWT token 缓存。本地原始缓存可为 train `Dataset`，或含 `train`/`test`
的 `DatasetDict`，通过 `data.local_dataset` 指定。

BERT 的 BOS/EOS 自动关联已有 CLS/SEP，不扩充词表；文本先按原版
`lm1b_detokenizer` 规范化。需要注意原版 wrapped 实现使用
`tokenizer.encode(special_token)[0]`：BERT 自动前置 CLS，所以其分块的 EOS 实际也
取 CLS。训练与独立 Flow 评测沿用这条已有约定，不另行替换成 SEP。
独立评测优先复用原版缓存，保持其每 1000 条 map 分块与丢弃余数的边界。

LM1B student 独立 benchmark 使用同一个八卡脚本，默认先测 Gen. PPL：

```bash
STUDENT=/path/to/lm1b_student/best/model.safetensors

# 初步检查 64 条，仍使用长度 128、采样前向 128、GPT-2-large CUDA FP16。
bash scripts/eval_langflow_student.sh "$STUDENT" \
  --config configs/langflow_lm1b_eval.yaml --set gen_ppl.num_samples=64

# 原版 LM1B Gen. PPL 默认 512 条、每卡 batch=8、seed=42+rank。
bash scripts/eval_langflow_student.sh "$STUDENT" \
  --config configs/langflow_lm1b_eval.yaml

# 官方 test 的 Flow PPL，先用相同的前 256 个 block 对比 teacher/student。
bash scripts/eval_langflow_student.sh "$STUDENT" \
  --config configs/langflow_lm1b_eval.yaml \
  --set 'metrics=[flow_ppl]' --set flow_ppl.first_n=256
```

完整 Flow PPL 设置 `flow_ppl.first_n=0`，读取官方 `test`，长度 128、128 个积分步、
Heun2、自条件开启；原版缓存名称为 `lm1b_test_bs128_wrapped.dat`。可通过
`flow_ppl.validation_cache` 显式指定。benchmark 和训练内的轻量生成评测都保留
特殊 token，再由 GPT-2-large 重新 tokenize，不在 LM1B 上额外删除 `[CLS]`。

IDE 训练 debug 直接运行 `debug/debug_pipeline.py --config configs/lm1b_debug.yaml`。
评测 debug 直接运行 `debug/debug_student_eval.py --config configs/langflow_lm1b_eval_debug.yaml
--checkpoint /path/model.safetensors`。两者均单卡、同进程使用真实工程函数。
本次 LM1B 适配仅做静态检查，没有在本地下载或执行模型，实际验证留给服务器。

## 与原版对齐的独立 student 评测

`scripts/eval_langflow_student.sh` 输入 student 的 safetensors 文件，默认使用 CUDA 0-7
共八个进程。模型架构自动读取权重旁的 `config.json`，与评测协议 YAML 分开。
本项目保存的 safetensors metadata 仅有 `format=pt`，不能单凭张量文件完整恢复模型
结构；请一起保留 `config.json`（包括 student 的 `n_blocks`、hidden size、heads、SC 等）。

```bash
# 新增 flow NLL 依赖后先更新安装。
/proj/gpu_mtk53742/.conda/envs/distill/bin/python -m pip install -e .

# 默认 OWT 验证集 PPL，对齐 run_all_eval_8gpu.sh 的 owt_ppl。
bash scripts/eval_langflow_student.sh run/<实验目录>/checkpoints/best/model.safetensors

# 权重被单独移动时，显式提供 student 架构 JSON。
bash scripts/eval_langflow_student.sh /path/student.safetensors \
  --model-config /path/student_config.json

# 生成 PPL，对齐 LangFlow/run_gen_ppl_8gpu.sh。
bash scripts/eval_langflow_student.sh run/<实验目录>/checkpoints/best/model.safetensors \
  --set 'metrics=[gen_ppl]'

# 同时计算两种 PPL。
bash scripts/eval_langflow_student.sh run/<实验目录>/checkpoints/best/model.safetensors \
  --set 'metrics=[flow_ppl,gen_ppl]'
```

协议在 `configs/langflow_student_eval.yaml`：

| 指标 | 原版对齐设置 |
| --- | --- |
| `flow_ppl` | OWT 最后 100000 篇验证文本，wrapped 长度 1024，积分网格 128 步，Heun2，SC 开启，每卡 batch=1，seed=1+rank，重构+flow+prior NLL 按 token 汇总 |
| `gen_ppl` | 512 条，长度 1024，采样前向 1024 次，每卡 batch=8，seed=42+rank，GPT-2-large CUDA FP16，评分上限 1024，保留特殊 token，逐样本重评分后按 token 汇总 |

Flow 的 `num_steps=128` 沿用原脚本的积分网格参数（旧日志称 NFE），不能理解为
总共仅调用模型 128 次；Heun2 与 SC 会产生额外前向。生成的 `num_steps=1024`
则是原 sampler 的模型前向次数。两个指标在 `metrics.json` 中分开保存。

默认优先读取 `DATA_CACHE/openwebtext-valid_validation_bs1024_wrapped.dat`，也可通过
`--set flow_ppl.validation_cache=/path/to/original.dat` 明确指定**原版完整 token 缓存**。
这样连 map batch 丢弃余数和多进程分块边界都一致。没有缓存时，使用同一 OWT 的
Parquet 转换构建原版规则的缓存；首次准备可能需要下载完整源数据。此入口不使用
蒸馏训练的 `run/data` 缓存，那个 continuous buffer packing 不是原版评测分块。

`flow_ppl.first_n=0` 为完整验证集。首次服务器检查可以设置 32，只用于流程验证；
正式对比 teacher/student 时，必须使用相同缓存、样本范围、卡数、batch、种子及
积分/采样参数。这个入口也可输入原 teacher 的权重，建议在相同环境重跑 teacher
作为基准，而不是假设不同软件版本/硬件的历史数字能够逐位一致。

运行日志、完整模型配置、评测协议、结果和生成文本保存在 `run/<时间>_langflow_student_eval/`。
原有训练 `all` 的轻量 Gen. PPL 保持原设置，与这份独立 benchmark 分开。

IDE 运行 `debug/debug_student_eval.py --checkpoint /path/model.safetensors`，默认只做
少量 Flow PPL，数据 workers=0，同进程直接调用真实模型。可用 `--set model.device=cpu`
本地调试，或 `--set 'metrics=[gen_ppl]'` 进入真实 GPT-2-large 评分；debug 只减少开销，
生成长度和 Flow 缓存长度仍需显式匹配。debug 结果不能作为正式质量结论。

## 真实数据的 GPT-2-Large PPL（LM1B / OWT）

这个入口直接将真实文本输入预训练 GPT-2-Large，分别计算 LM1B 和 OWT 的 PPL。
默认每个语料抽取 **512 条可评分文本**，固定 seed=42；LM1B 从官方 `test` 抽取句子，
沿用项目中的 LM1B detokenizer，OWT 从 `train` 最后 100000 篇验证保留区抽取文档。
一个句子/文档算一个样本，不拼接、packing，也不添加 BERT/T5 边界 token。

```bash
# 服务器项目根目录：默认 CUDA 0-7，八卡并行，GPT 模型每卡只加载一次。
bash scripts/eval_gpt2_corpus_ppl.sh

# 单卡真实 GPT-2-Large debug（8 条/语料，长度上限 64，直接进入工程函数）。
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_gpt2_corpus_ppl.sh configs/gpt2_corpus_ppl_debug.yaml debug

# 可使用原始文本缓存；不能使用仅含 input_ids 的 ELF 数据或 wrapped token 缓存。
bash scripts/eval_gpt2_corpus_ppl.sh configs/gpt2_corpus_ppl.yaml all \
  --set data.lm1b.local_dataset=/path/to/raw_lm1b_test \
  --set data.owt.local_dataset=/path/to/raw_owt_train
```

缓存支持 `Dataset.save_to_disk` 或含配置中 `split` 的 `DatasetDict`，必须有 `text` 列。
OWT 若提供的是已经单独保存的验证尾部，请同时设置 `--set data.owt.tail_documents=null`。
没有本地缓存时从配置的 Hugging Face Parquet 分支加载，复用 `DATA_CACHE` 或 HF 默认缓存。
首次准备数据/下载 GPT 权重的时间单独记录在阶段日志中。

GPT 前向默认使用 FP16，交叉熵使用 FP32，NLL 汇总使用 FP64。标准自回归口径为
`PPL = exp(全部有效目标 token 的 NLL 总和 / 有效目标 token 总数)`，排除 padding，
每条文本的首 token 不评分；真实 EOS token 若在正文中仍正常评分。两个语料均最多取
前 1024 个 GPT token，超过长度的部分截断，因此结果是这 512 条文本前缀上的 PPL，
不是整个语料、也不是每条完整长文档的 PPL。分卡不重复样本，不平均各卡或逐样本 PPL。

结果位于 `run/<时间>_gpt2_real_corpus_ppl/metrics.json`：分别读取
`results.lm1b.ppl`、`results.owt.ppl`。同时记录 NLL、有效 token 数、逐样本评分，
`lm1b/samples.json` 和 `owt/samples.json` 保存实际文本、来源索引和实际送入 GPT 的 token IDs。
此入口使用标准 padding mask；ELF 原版 BF16/EOS-mask 评分属于另一套数值口径，
直接比较时应统一精度、文本长度和掩码。

配置见 `configs/gpt2_corpus_ppl.yaml`；可通过 `--set evaluation.num_samples=...`、
`--set evaluation.batch_size=...`、`--set evaluation.max_length=...` 调整预算。
IDE 可直接运行 `debug/debug_corpus_ppl.py`，在 `select_samples`、`score_samples`、`token_nll`
设置断点；debug 不创建其他 Python 进程，也不替换预训练模型。

## ELF-B / ELF-L 蒸馏

ELF 沿用原版 T5-small tokenizer、冻结 encoder 和 latent 归一化，使用原版
`embedded-language-flows/openwebtext-t5` 的预 tokenized Arrow 数据。训练直接对真实
文本 latent 加噪，不先做 teacher 文本生成。teacher 默认从官方 PyTorch checkpoint
读取 `ema_params1`（与原版生成一致）；student 复制 teacher 权重并删层，保留宽度和输出头。

| 配置 | Teacher 骨干 | Student | 每卡 batch × 累积 | 八卡全局 batch |
| --- | --- | --- | --- | --- |
| `configs/elf_b_kd.yaml` | 12 层，宽度 768，12 heads | 6 层 | 4 × 2 | 64 |
| `configs/elf_l_kd.yaml` | 32 层，宽度 1280，16 heads | 16 层 | 1 × 8 | 64 |

L 默认开启 student gradient checkpointing。上述 batch 是 A6000 的初步起点，实际峰值
显存记录在 `metrics.jsonl`；本地未执行模型或 GPU 验证。编码、数据、损失和评测放在
`src/distill/elf/`，共用现有 DDP、优化、日志与平台期早停流程。

在服务器项目根目录运行，先更新依赖，再分别验证真实 B/L 流程：

```bash
/proj/gpu_mtk53742/.conda/envs/distill/bin/python -m pip install -e .

# 单卡、当前进程调用真实 T5/ELF：短序列、训练 3 步、保存重载和生成。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_elf_experiment.sh configs/elf_b_debug.yaml debug
CUDA_VISIBLE_DEVICES=0 bash scripts/run_elf_experiment.sh configs/elf_l_debug.yaml debug

# 各自占用 CUDA 0-7；两条依次执行。
bash scripts/run_elf_experiment.sh configs/elf_b_kd.yaml all
bash scripts/run_elf_experiment.sh configs/elf_l_kd.yaml all
```

`all` 执行数据准备、八卡 DDP 训练、最佳 checkpoint 重载及三组生成比较。
默认最大 5000 个 optimizer step；每 100 步验证，至少训练 500 步后，连续 5 次验证
下降不足 0.001 则停止。早停依据内部验证 loss，不依据 Gen. PPL。
`train` 只训练，`prepare` 只准备数据；`check-distributed` 仅检查多卡通信。
所有产物在 `run/`，启动日志在 `run/launch/`。

首次运行需要在服务器下载真实模型及原版 Arrow 数据集，debug 也使用这些资产。
已有 ELF 数据缓存时建议直接配置 `data.local_dataset`，目录应为原版
`Dataset.save_to_disk` 输出且包含 `input_ids`、`sequence_length`。两个正式实验共享
预处理缓存；debug 使用独立的短序列缓存。训练前缀和尾部内部验证保留区分开，
不使用 Gen. PPL 选 checkpoint。

### ELF 不只支持 MSE

ELF 去噪头输出连续 latent，decoder 头输出词表 logits。它们是两个独立输出头，
decoder logits 的期望 embedding 并不定义 ELF 的去噪输出。

| 配置项 | 默认 | 目标 |
| --- | --- | --- |
| `loss.flow_weight` | 1 | 同一 `(z,t,SC)` 上 teacher/student 的 velocity MSE |
| `loss.kd_weight` | 1 | decoder 状态上 KL(teacher ‖ student)，温度 1 |
| `loss.ce_weight` | 0 | decoder 状态上真实 token CE |
| `loss.native_flow_weight` | 0 | 真实 latent 的原生 velocity MSE，含原版 SC-CFG 目标修正 |

`decoder_probability=0.2` 时每条样本独立选择 decoder 状态，其余为去噪状态。
去噪使用 `z=t*x0+(1-t)*2*noise`，velocity 为 `(xhat-z)/max(1-t,0.05)`；
decoder 使用原版逐 token logit-normal 噪声、`t=1` 和零 SC。
SC 来自当前 student 的无梯度预测，detach 后给双方相同输入。
两种分支沿用原版混合 batch 的有效 token 总数作分母；因此损失权重还受分支概率影响。
日志的分项也是对总有效 token 的贡献，不是各分支独立归一化的均值。

```bash
# 仅做 teacher velocity MSE；不采样 decoder 状态。
bash scripts/run_elf_experiment.sh configs/elf_b_kd.yaml train \
  --set loss.kd_weight=0 --set diffusion.decoder_probability=0

# 仅 decoder KL；训练的输入是 decoder 原生噪声状态。
bash scripts/run_elf_experiment.sh configs/elf_b_kd.yaml train \
  --set loss.flow_weight=0 --set diffusion.decoder_probability=1

# 原生目标对照：velocity MSE + 真实 token CE。
# 使用同一 student/优化预算，优化器等并非完整复现原版从头训练。
bash scripts/run_elf_experiment.sh configs/elf_b_kd.yaml train \
  --set loss.flow_weight=0 --set loss.kd_weight=0 \
  --set loss.native_flow_weight=1 --set loss.ce_weight=1
```

### ELF 生成评测与 checkpoint

正式 `all` 默认对 teacher、未训练删层 student、训练后最佳 student 各生成 **512 条**，
八卡并行生成、每卡 batch=1。B 使用原版第一组 SDE 32 步、gamma=1.5；L 使用第二组
SDE 64 步、gamma=1.0；两者 SC-CFG=3、CFG=1、logit-normal schedule。
步数指去噪步数，另有一次 decoder 前向（默认总 NFE 分别 33/65）。
三组分别重置同一 rank 种子 `42 + rank * 1000003`，固定相同卡数、batch、长度和采样参数。

生成和 GPT-2-large Gen. PPL/entropy 的实现复制自本地原版 ELF，保留 BF16、
EOS 处理、整组文本 retokenize/padding 和 token 加权 PPL。与原版一样，在 rank 0
集中评分，其他卡通过 CPU/Gloo 等待。结果保存在 `evaluation/metrics.json`，
同时保存文本和 token IDs。内部 `denoising` 指标中的 MSE/KL/CE 不能当作 PPL。
可增加到原版默认的 1000 条，保持三组协议一致：

```bash
# 路径替换为实际 ELF-B 实验的最佳模型目录或其中的 model.safetensors。
bash scripts/run_elf_experiment.sh configs/elf_b_kd.yaml evaluate \
  --checkpoint run/<实验目录>/checkpoints/best/model.safetensors \
  --set evaluation.num_samples=1000
```

ELF checkpoint 的 `model.safetensors` 只含 student 权重；同目录 `config.json` 含
student 架构、encoder 引用、diffusion 与采样信息，并保存 tokenizer 和层映射。
移动权重时请保留整个目录。评测配置须对应 B/L、原数据和训练时 latent 设置；生成协议
由当前 YAML 控制。当前 student 保存原始优化权重，不额外维护 EMA；teacher 使用原版 EMA。

IDE 可直接运行 `debug/debug_elf_pipeline.py --config configs/elf_b_debug.yaml`（或 L），
在 `elf.data.prepare_data`、`elf.losses.batch_loss`、共用 `train`、`elf.evaluate.generate`
设置断点。debug 不改变 teacher/student 架构，不加载 PPL scorer。

## 半层学生：只随机初始化 Transformer blocks 的 off-policy 对照

四组配置分别为 `configs/off_policy_random_langflow_owt.yaml`、
`configs/off_policy_random_langflow_lm1b.yaml`、`configs/off_policy_random_elf_b.yaml`、
`configs/off_policy_random_elf_l.yaml`。前三组为 12→6 层，ELF-L 为 32→16 层。
它们沿用真实语料加噪的 off-policy 流程；LangFlow 为 KL，ELF 为 decoder KL + velocity MSE，
所有 loss 权重和 decoder/SC 概率保持原实验设置。

`model.block_initialization: random` 只让 Transformer blocks 保留模型构造器的原生初始化。
embedding、latent/SC 投影、时间编码、输出头、噪声 schedule 等仍复制 teacher；
LangFlow 的 AdaLN 零门控和各模型的 norm 常数初始化也保留。
**仅改变初始化范围，不改变可训练参数范围**：LangFlow 的词嵌入与 proposal 继续冻结，
ELF 的 T5 encoder 继续冻结，其余 student 模块按原规则训练。
旧配置默认 `block_initialization: teacher`，仍复制所选 teacher blocks。

```bash
# 只读配置并记录运行计划，不加载模型或数据。
bash scripts/run_random_blocks.sh --dry-run

# 服务器：四个任务依次使用 CUDA 0–7；每组训后自动计算统一 PPL、entropy 和 GM。
bash scripts/run_random_blocks.sh

# 单独运行某组，或只训练。
bash scripts/run_random_blocks.sh --model elf_b
bash scripts/run_random_blocks.sh train --model langflow_owt

# 只重跑 ELF-B/L：对齐当前 LangFlow-OWT 的 LR 调度、全局 batch 和固定训练预算。
bash scripts/run_random_blocks.sh --elf-only

# 单卡真实 debug：相同 teacher/student 架构，3 次更新、128 token、2 条评测样本。
bash scripts/run_random_blocks.sh debug --model elf_l
```

四组初始 lr=1e-3、max_steps=5000。LangFlow-OWT 当前每卡 14、累积 2，
八卡 global batch=224，cosine 衰减到 1e-4，关闭早停并跑满 5000 步。
ELF-B/L 的重跑配置对齐这套优化与预算：B 每卡 7、累积 4；L 每卡 1、累积 28，
全局 batch 同为 224，cosine→1e-4、关闭早停、每 100 步验证。
ELF 保留自己的 KL + velocity MSE、SC 概率 0.5、decoder/latent 和采样参数；
其 micro-batch 按已有显存记录保守设置，L 继续使用 gradient checkpointing。
两组每 500 步保留 checkpoint 便于检查退化起点，最终统一评测第 5000 步权重，
新的 run.name 带 `cosine_gbs224`，原恒定 LR 试跑记录仍保留在 `run/`。
LangFlow-LM1B 继续恒定 LR、global batch=512，最少 500 步后按原平台期规则早停，
最终选择验证 loss 最佳的 `best/`。
可用 `--set training.max_steps=10000` 等调整配置，调整后应另记为不同训练预算的实验。

此批实验关闭旧版训练中间 PPL；`all` 在训练模型与优化器释放后，直接调用
`scripts/eval_gen_ppl.sh configs/gen_ppl.yaml` 对应的 `run_gen_ppl`，使用同一套生成、
文本清理、GPT-2-Large FP32/eager 评分和 token 加权汇总。
四份配置默认 `gradient_moment.enabled: true`，`reference_config` 指向
`configs/gpt2_corpus_ppl.yaml`，并将 `evaluation.corpus` 显式传给统一接口：
LangFlow-LM1B 为 `lm1b`，其余三组为 `owt`。GM 与 PPL/entropy 共用本次生成文本，
使用统一接口的真实参考文本和 GPT-2-Large 反向传播，不更新模型参数。
每组正式评测 **1024 条、seed=42**，生成长度/NFE/ELF 采样参数沿用对应模型的原设置，
只评测本次训练的 student。LM1B 配置的 `evaluation.seed` 仍用于训练内部验证噪声，
正式统一生成种子由 `gen_ppl_v1` 固定为 42。
正式 GM 每组使用 1024 条对应真实语料参考，组成 512 个梯度对，按八卡分工；
单卡 debug 为 2 条生成/真实参考、1 个梯度对，仍直接调用真实工程的前向和反向。

结果在 `run/<时间>_off_policy_random_*/`：训练日志与 checkpoint、可独立复跑的
`gen_ppl_eval.yaml`（已开启 GM），以及下级 `*_unified_gen_ppl/summary.csv`、`metrics.json` 和生成文本。
汇总包含 `gen_ppl`、`sample_entropy`、`gm`、`gm_standard_error` 和覆盖率；
模型子目录的 `gradient_moment.json` 保存逐对结果。
每个 checkpoint 的 `distillation.json` 记录 `block_initialization`。
启动计划和队列状态在 `run/launch/random_blocks_*/`，计划同时记录 GM 开关及参考语料；
任一任务失败时停止后续任务并保留日志。
单卡 IDE 入口为 `debug/debug_random_blocks.py --config configs/off_policy_random_elf_b_debug.yaml`，
直接调用原工程函数，不通过 subprocess 启动训练或评测。

```bash
# 训练后独立重测，仍使用统一接口。
bash scripts/eval_gen_ppl.sh run/<本次实验目录>/gen_ppl_eval.yaml
```

## LangFlow-OWT：同层数 SVD 低秩学生 sweep

`configs/owt_rank_sweep.yaml` 从原始 LangFlow-OWT teacher 独立构造四个 **12 层**学生，
保留 hidden size=768、12 heads、cond_dim=128、原激活与残差。
每个 block 内全部五个 Linear 都做 SVD：融合 `attn_qkv`、`attn_out`、FFN 的 `mlp.0`
和 `mlp.2`，以及条件调制投影 `adaLN_modulation`。
对权重 `[out,in]`，使用 `r=max(1,floor(ratio*min(out,in)))`，
ratio 表示矩阵秩比例；总参数量包含其他模块，独立记录在结果中。

| rank 比例 | QKV / attention 输出 / FFN rank | AdaLN rank |
| --- | --- | --- |
| 80% | 614 | 102 |
| 60% | 460 | 76 |
| 40% | 307 | 51 |
| 20% | 153 | 25 |

前四个矩阵的最小维度为 768，AdaLN 权重 `[4608,128]` 的最小维度为 128，
所以按各自维度计算实际 rank。LayerNorm 和 bias 是向量，不分解。
block 外的 `self_cond_proj`、时间编码、最终输出层及词嵌入保留原结构。

每个目标矩阵只做一次 FP32 SVD，60 个矩阵分给八卡计算，保存在本实验的 `svd/`。
四组截取同一分解的前 r 项，使用平衡因子 `U_r sqrt(S_r)` 和 `sqrt(S_r) Vh_r`。
两个因子之间没有新增激活，原 bias 放在第二个 Linear；其他参数复制 teacher。
`svd/summary.json` 记录满秩重构误差、实际 rank 和保留奇异值能量，rank 选择始终按数量比例。
默认还临时构造一次 100% rank 学生，用一条真实验证文本的至多 128 个位置进行单步对照，
在 `full_rank_check.json` 记录 posterior KL 和 logits 误差。此检查保留全部真实层和宽度，
BF16 的两次因子矩阵乘法允许与 dense teacher 存在数值差异；不额外训练、保存或完整生成
100% rank 模型，四个正式学生的训练/生成长度仍为 1024。

```bash
# 服务器：默认 CUDA 0–7，新实验构建 → Teacher/初始学生评测 → 四组 KL 蒸馏 → 复评。
bash scripts/run_owt_rank_sweep.sh

# 只检查配置、保存运行计划，不加载模型/数据，不初始化 GPU。
bash scripts/run_owt_rank_sweep.sh --dry-run

# 同进程单卡 debug：完整 12 层 teacher、全部 block 投影和四种低秩学生。
bash scripts/run_owt_rank_sweep.sh debug

# 只构建 SVD 起点，先检查 checkpoint。
bash scripts/run_owt_rank_sweep.sh build

```

本轮使用新的 `langflow_owt_svd_block_rank_sweep` 实验名，每次启动默认创建独立目录，
重新评测 Teacher 和新版学生，不使用 `--run-dir`，不复用旧版三矩阵学生或其初评结果。

正式训练每组最多 5000 次更新、纯 forward KL，初始 LR=1e-4、cosine 按 5000 步上限衰减到 1e-5。
每卡 batch=16、累积 2 次，八卡全局 batch=256；生成评测每卡 batch=16。
四组使用相同数据、训练 RNG、学习率和训练上限，默认按验证 KL 平台期早停：
每 100 步验证，500 步起累计连续无有效改善的检查；相对上次有效改善的 KL 累计下降
至少 0.001 时清零计数，连续 5 次未达到则停止。每次真实验证最优均保存 `best`，
统一复评使用验证 KL 最低的权重，同时保留停止时的权重。
各组实际训练步数可能不同：单组最多 1,310,720,000 token 位置曝光，
实际值记录在 `training.json` 和总表，不能再视为固定曝光预算对照。
LangFlow 的词嵌入与 proposal 继续冻结，其余学生参数（包括两组低秩因子）按原规则更新。
若需要固定 5000 步对照，可传入 `--set training.early_stopping.enabled=false`。

正式统一评测为 Teacher 一次、SVD initial 四次、trained 四次，共 9 套权重；
seed=42、每模型 1024 条、生成长度/NFE=1024、同一 GPT-2-Large FP32/eager 评分，
默认开启 OWT GM（1024 条真实参考，512 个梯度对）。同时输出 entropy、重复率、
覆盖率、`mean_step_ms` 和 `num_parameters`，全部复用 `eval_gen_ppl.sh` 工程函数。

运行结果位于 `run/<时间>_langflow_owt_svd_block_rank_sweep/`：

- `rank_080/`、`rank_060/`、`rank_040/`、`rank_020/`：每组配置、训练日志、`loss_curve.png`，以及
  `checkpoints/initial/`、验证最佳的 `checkpoints/best/` 和停止步的 `checkpoints/step_XXXXXX/`。
  `training.json` 记录停止原因、实际训练步数、最佳步数、总 token 曝光 `tokens_seen`
  和所选 checkpoint 对应的曝光 `checkpoint_tokens_seen`。
- checkpoint 的 `config.json` 保存每个目标模块的实际 rank；safetensors 保存两组因子，
  统一模型构造器会先恢复因子结构，再严格加载权重。
- `gen_ppl_before.yaml`、`gen_ppl_after.yaml` 和最终的 `gen_ppl_models.yaml` 可独立复测。
- `summary.csv` / `summary.json` 汇总 Teacher、before、after 的质量/耗时/参数指标和实际训练预算；
  `rank` 是四个主干投影的 rank，`conditioning_rank` 是 AdaLN 投影的 rank；
  `rank_quality_curve.png` 比较 rank 与蒸馏前后 PPL/entropy/GM，缺失指标保留空值。

```bash
# 任意初始/训练后的低秩 checkpoint，仍使用统一八卡评测。
bash scripts/eval_gen_ppl.sh run/<本次rank实验>/rank_080/checkpoints/initial/gen_ppl_eval.yaml
```

IDE 入口 `debug/debug_rank_sweep.py` 直接调用同一套 SVD、采集、训练、生成、GM 和计时函数，
不启动其他 Python 进程。debug 保留四种 12 层学生和全部 block 投影，仅缩短为 3 次更新、128 token、8 NFE、
4 条评测样本，并关闭绘图与早停；本地开发验证不下载/加载预训练权重，真实 GPU 验证由服务器执行。

## On-policy：真实 student rollout 的 CE/KL/MSE 蒸馏

**对齐第一阶段 loss，默认只补跑 ELF-B 和 ELF-L：**

```bash
# 查看实际起点和 loss，不加载模型或启动 GPU 任务。
bash scripts/rerun_on_policy_loss_matched.sh --dry-run

# 默认八卡依次重训 ELF-B/L，并用统一 Gen PPL 比较 off-policy 起点与新 student。
bash scripts/rerun_on_policy_loss_matched.sh

# 只训练；稍后对导出的 gen_ppl_eval.yaml 运行 scripts/eval_gen_ppl.sh。
bash scripts/rerun_on_policy_loss_matched.sh train

# 仅在确实需要时，显式要求额外重跑 LangFlow OWT/LM1B。
bash scripts/rerun_on_policy_loss_matched.sh --all-models
```

脚本保留选中的 `configs/on_policy_*.yaml` 中现有的起点、学习率、batch 和 rollout 调优，
读取每个 off-policy checkpoint 的 `distillation.json` 来生成实际 loss：
LangFlow 将 `kd_weight` 映射到 `kl_weight`；ELF 将 `kd_weight`、`flow_weight` 分别
映射到 decoder KL、velocity MSE，CE 权重为 0。若第一阶段 `config.yaml` 存在，
同时继承其 decoder 分支概率。状态仍来自 student 真实 rollout。

计划、完整有效 YAML、各任务日志和 `status.tsv` 保存在
`run/launch/on_policy_loss_matched_<时间>_<PID>/`。选中模型的权重/架构在第一项训练前统一检查，
拒绝将旧 on-policy checkpoint 当作起点；`--dry-run` 只检查 loss 元数据，
本地没有同步 safetensors 时也能查看计划。旧配置、模型和结果保持原样，
新实验名称含 `loss_matched`，每组都新建 optimizer 并从原 off-policy 权重开始。

在当前停止 teacher 和状态梯度的实现中，soft CE = forward KL + teacher entropy，
对 student 的梯度相同。显式 KL 统一了 loss 数值与日志，不能把 CE→KL 本身当作
新的监督方法；ELF 相较旧 MSE-only 恢复 decoder KL 是实质性的目标变化。

第二阶段从第一阶段的 off-policy student 权重继续训练。student 自己从噪声开始生成，
在采样器实际查询的位置保存 `(z, time, student_previous_self_cond)`；teacher 只在抽中的
相同状态上提供监督，下一步仍由 student 自己决定。训练时重新前向 student 计算梯度，
采集路径和 teacher 目标均停止梯度。这一版实现的是 on-policy 状态蒸馏，
proposal 的 paired trajectory MSE 还属于后续独立实验。

先在以下四份配置中填写 `model.student_weights`。它可以是第一阶段 `best/` 目录，
也可以是其中的 `model.safetensors`。模型深度直接从该 checkpoint 的架构读取，
因此同一配置可以接对应模型的不同删层比例，不会再次删层或从 teacher 重建 student。

| 配置 | 原始 teacher | 默认 loss | rollout 与评测步数 / 长度 |
| --- | --- | --- | --- |
| `configs/on_policy_langflow_owt.yaml` | LangFlow-OWT | teacher soft CE | 1024 / 1024 |
| `configs/on_policy_langflow_lm1b.yaml` | LangFlow-LM1B | teacher soft CE | 128 / 128 |
| `configs/on_policy_elf_b.yaml` | ELF-B EMA | velocity MSE | 32 / 1024 |
| `configs/on_policy_elf_l.yaml` | ELF-L EMA | velocity MSE | 64 / 1024 |

```yaml
model:
  student_weights: /path/to/off_policy/checkpoints/best/model.safetensors
  # 默认自动读取权重旁的 config.json；若单独保存架构，填其路径。
  student_config: null
```

safetensors 只含权重，必须保留配套架构 JSON。尽量保留整个 checkpoint 目录，
其中 tokenizer 和 `distillation.json` 会用于恢复 tokenizer、层映射及初始化来源。
LangFlow 的 embedding/Gumbel schedule 继续冻结并检查与 teacher 一致；
ELF 检查 T5 latent 坐标与噪声定义，本阶段不加载或训练 T5 encoder。

在服务器项目根目录执行；每条正式命令默认占 CUDA 0–7，四条依次执行：

```bash
bash scripts/run_on_policy.sh configs/on_policy_langflow_owt.yaml all
bash scripts/run_on_policy.sh configs/on_policy_langflow_lm1b.yaml all
bash scripts/run_on_policy.sh configs/on_policy_elf_b.yaml all
bash scripts/run_on_policy.sh configs/on_policy_elf_l.yaml all

# 推荐先运行对应的单卡真实模型 debug；它继承上方填写的权重路径。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_on_policy.sh configs/on_policy_elf_b_debug.yaml debug

# 只训练并导出统一评测配置；不立即执行最终两组 1024 样本评测。
bash scripts/run_on_policy.sh configs/on_policy_langflow_owt.yaml train
```

`all` 自动用统一 `gen_ppl_v1` 比较 `off_policy_initial` 与 `on_policy_student`，
原始 off-policy 权重不修改。训练目录中的 `evaluation.json` 指向最终评测目录，
其中 `summary.csv` / `metrics.json` 含 PPL、entropy、重复率和覆盖率。
`gen_ppl_eval.yaml` 也可稍后交给 `scripts/eval_gen_ppl.sh` 独立执行。
训练中的 PPL 也直接调用统一接口的生成、评分和聚合函数，评测 optimizer 更新后的
当前内存权重；正式协议固定 seed=42、全局 1024 条、GPT-2-Large FP32/eager 单条评分，
不使用各模型旧的 native PPL monitor。每次结果位于 `ppl/step_XXXXXX/`，包含统一的
`metrics.json`、`summary.csv`、`samples.jsonl`、`protocol.json`。这部分计算开销较大，
可设 `--set training.ppl_evaluation.interval=-1` 关闭训练内评测，最终 `all` 仍统一比较两组。

`loss.ce_weight`、`loss.kl_weight`、`loss.mse_weight` 分别控制目标。
`kl_weight` 默认为 0，保留旧 CE/MSE 实验；loss-matched 入口根据第一阶段元数据设置它：

- CE 默认是 `-sum(p_teacher * log p_student)`；`loss.ce_target=hard` 则使用 teacher
  的 argmax token 标签。rollout 没有配对的真实 token 标签，不能当作原文本的 supervised CE。
- KL 为 `KL(teacher || student)`，温度 1，词表求和后按 token 平均，与第一阶段的
  `kd_weight` 对应；固定 teacher 目标上，其梯度与 soft CE 一致。
- MSE 比较同一状态下的 velocity。LangFlow 沿用共享 embedding 的 VP gamma 路径，
  ELF 沿用 `(xhat-z)/max(1-t,0.05)`，对 token/latent 维度取均值。
- ELF 开启 CE 或 KL 时，采集实际 rollout 终点的 decoder-mode 状态：`t=1`、零 SC。
  同时启用 posterior/MSE 的混合概率由 `on_policy.decoder_probability` 控制，分项均按整个
  batch 归一化。只有 CE/KL 都关闭的 MSE-only 才冻结 decoder 输出头和 decoder mode tokens。

```bash
# ELF 同时使用 velocity MSE 与实际生成终点的 decoder CE。
bash scripts/run_on_policy.sh configs/on_policy_elf_b.yaml train --set loss.ce_weight=1

# LangFlow 使用 teacher hard token CE。
bash scripts/run_on_policy.sh configs/on_policy_langflow_owt.yaml train --set loss.ce_target=hard

# proposal 的确定性 ODE 实验；all 对 off-policy/on-policy 两组同时应用该采样设置。
bash scripts/run_on_policy.sh configs/on_policy_elf_b.yaml all \
  --set evaluation.sampling.sampling_method=ode
```

每卡独立采集并用 CPU 缓存访问状态，DDP 只同步 student 梯度。各模型的 rollout batch、
状态抽样数、缓存容量以当前配置为准，每 16 个训练 micro-batch 用当前 student 刷新一轮，按容量
淘汰旧状态；这会混合当前及历史 student 的访问分布。ELF SDE 记录实际送入模型的
`z_back/t_back`，保留真实 SC 和 SC-CFG scale，不用外层网格状态冒充模型输入。
本阶段不读取 OWT/LM1B 原始数据，也不缓存整个词表的 teacher logits。

默认最多 5000 个新的 optimizer step，初始 LR 取当前 on-policy 配置（现为 1e-4），cosine 下降到 0.1 倍。
每 100 步验证，保留最大步数和平台期早停。验证用固定随机种子的当前 student 新轨迹，
不使用或修改训练缓存。soft CE 含随状态变化的 teacher entropy，因此记录原始 CE
训练 loss，同时默认用 `error = ce_weight*(CE-teacher_entropy) + kl_weight*KL + mse_weight*MSE` 选最佳模型；
hard CE 的 entropy 项为零。它仍是拟合指标，生成质量需看独立 PPL/entropy/重复率及文本。
`training.json` 记录实际训练步数、停止原因、best step；不是精确恢复第一阶段 optimizer。

四份 `on_policy_*_debug.yaml` 保留实际 teacher/student 结构，只减少至训练 3 步、
rollout 8 次前向、序列 128、最终每组 4 条样本。IDE 可直接运行 `debug/debug_on_policy.py`
并指定对应 debug 配置，在 `collect_states`、`StateBuffer.sample`、`OnPolicyLoss.__call__`
及共用 `train` 中设置断点。整个流程在当前进程直接调用，没有 subprocess 或 mock 模型。

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
  --checkpoint run/<实验目录>/checkpoints/best
```

正式配置默认按验证平台期自动停止，`max_steps` 为硬上限：

```yaml
training:
  max_steps: 5000
  validate_every: 100
  early_stopping:
    enabled: true
    metric: loss
    min_steps: 500
    patience: 5
    min_delta: 0.001
```

`metric` 可选验证 `loss`、`kl` 或 `ce`，都以越低越好判定。默认 `loss` 与当前训练目标
一致：KD-only 时为 KL，CE-only 时为 CE，混合时为加权之和。验证使用跨卡汇总的指标。
500 步之前记录最佳指标但不累计平台次数；到 500 步开始，连续 5 次验证没有达到至少
0.001 的有效下降便停止。小幅下降会与上次有效改善累计比较；`patience` 按验证次数
计数，不是训练步数。硬上限优先，即使 `min_steps` 大于 `max_steps` 也按上限结束。

rank 0 广播停止决定，八个进程一起退出训练。最佳权重反复写入 `checkpoints/best/`，
避免为每次改善保存一份大模型；这里保留实际最低值，即使单次改善小于 `min_delta`。
初始 step 0 也作为候选，若训练始终没有改善，可选择删层初始化模型。
停止时额外保存 `checkpoints/step_<实际步数>/`。`training.json` 记录实际步数、
`stop_reason`（`plateau` 或 `max_steps`）、最佳步数/指标以及两份 checkpoint 路径。
`all` 最后重载**最佳验证指标 checkpoint**做 Gen. PPL 和多样性评测。

当前平台判定基于去噪验证指标，不是周期生成 PPL；它控制 off-policy 拟合的训练预算，
生成质量仍需看结束后的 Gen. PPL 与多样性。debug 默认关闭早停，保持指定的小步数
便于人工断点调试。可通过 CLI 调整上限或关闭早停：

```bash
bash scripts/run_experiment.sh configs/owt_kd.yaml all --set training.max_steps=2000
bash scripts/run_experiment.sh configs/owt_kd.yaml all --set training.early_stopping.enabled=false
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
