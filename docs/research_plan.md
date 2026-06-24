# SCB 课堂行为研究实验计划（ 2x A100 40G ）

## 1. 研究目标

### 1.1 问题定义

这个研究要解决的不是通用视频理解，而是一个更具体、也更现实的问题：

在课堂行为识别场景中，人工逐框标注成本高、类别边界细、跨课堂域偏移明显。那么，是否可以先让多模态大模型为目标框裁剪图生成伪标签，再利用这些伪标签把 CLIP 从零样本状态适配到课堂行为任务，从而以较低人工成本获得可观的性能提升。

换句话说，这个工作要验证的是下面这条链路是否成立：

`多模态 LLM 自动标注 -> 伪标签过滤提纯 -> CLIP 领域微调 -> 超过零样本基线`

### 1.2 研究目标

本研究的主目标是验证这条低标注成本训练路径在 SCB 课堂行为数据上的可行性，具体包含三步：

1. 用多模态 LLM 对 YOLO bbox 裁剪图做自动分类，生成伪标签。
2. 用一致性过滤和有效输出过滤，分析伪标签质量与覆盖率的权衡。
3. 用伪标签微调 CLIP，评估其是否能稳定超过零样本基线，并逼近真实标签上界。

### 1.3 核心研究问题

论文真正要回答的是三类问题：

1. 多模态 LLM 在课堂行为 bbox 裁剪图上的自动标注准确率是否达到“可用于训练”的水平。
2. 双模型一致性过滤是否能显著提升伪标签纯度，以及这种提升是否值得其带来的样本量损失。
3. 用伪标签微调后的 CLIP，是否能在三个 SCB 子任务上稳定优于零样本 CLIP。

### 1.4 可检验假设

如果这个研究是成立的，至少应满足下面几条经验假设：

1. `agreement` 策略的伪标签准确率高于 `none`，但保留率更低。
2. 在至少部分数据集上，`agreement` 伪标签训练得到的 CLIP 会优于零样本基线。
3. `gt` 上界会显著高于伪标签训练结果，说明当前瓶颈主要仍在标注质量，而不是模型容量。

### 1.5 论文定位

这个工作的合适定位不是“提出一个新的视觉模型”，而是：

- LLM-assisted weak supervision
- LLM-bootstrapped CLIP fine-tuning
- Annotation-efficient classroom behavior recognition

### 1.6 贡献边界

这项研究的预期贡献应当控制在下面这个范围内：

1. 提供一条适用于课堂行为识别的低人工成本标注与适配流程。
2. 定量分析多模态 LLM 伪标签在教育场景中的可用性。
3. 说明伪标签过滤策略与下游微调收益之间的关系。

同时要明确，这个工作暂时不试图证明：

1. 该方法优于所有监督学习方法。
2. 该流程可以替代完整人工标注。
3. 该结论可直接泛化到所有教育视频任务。

## 2. 当前机器约束与资源分配

本机资源是 2 张 A100 40G，所以不能沿用之前 “GPU0/1 跑 LLM，GPU2/3 跑训练” 的设定。

推荐采用分阶段运行，而不是同时混跑：

 - 阶段 A：GPU0 跑 Qwen，GPU1 跑 Llava，完成全部 LLM 标注。
- 阶段 B：停止 Ollama 服务，释放两张卡。
- 阶段 C：用 GPU0,1 跑 CLIP 微调。

这样比边标注边训练更稳，原因有三点：

1. 双 A100 只有两张卡，同时跑两个视觉 LLM 再做 CLIP 训练会直接抢显存。
2. 标注阶段主要受单样本推理延迟影响，和训练并行不会明显缩短总墙钟时间。
3. 训练阶段可以稳定用双卡 DataParallel，复现实验更简单。

## 3. 实验模块与依赖关系

当前研究更适合按 analysis modules 理解，而不是按论文叙事中的连续 phase 理解。代码目录里保留 `phase*` 结果路径和 `step*` 脚本名，是为了记录历史运行边界、输入输出依赖和日志归档，不表示所有实验都是线性递进的补丁阶段。

实际依赖关系是：数据集可以直接支持 zero-shot、MLLM annotation、cross-model validation 和 teacher-student 等独立分析；主 MLLM annotation 产物会被 filtering analysis 转成伪标签文件；CLIP adaptation 读取固定的伪标签或 GT；mechanism/strategy audits 则读取已经冻结的 annotation、filtering 或 adaptation 产物做边界检查。换言之，多数分析在固定输入存在后可以独立复跑。

### Module A: Protocol-matched baseline

先建立协议一致的 zero-shot CLIP 参照，用于回答“不训练时的起点是多少”，也用于后续 AnchorProxy 的 per-class 诊断。

- 脚本：`step0b_cape_zeroshot.py`
- 主要产物：`results/phase0_zero_shot/canonical_20260425_231347/phase0_zero_shot_results.json`
- 日志：`logs/phase0/`

### Module B: MLLM annotation and validation

主标注分析用 Qwen2-VL-7B 和 LLaVA-1.5-7B 对 bbox 裁剪图生成原始伪标签。三个子数据集保留 train/val 两个 split：val 用于估计伪标签质量，train 用于生成下游微调数据。

交叉模型验证增加 Qwen2.5-7B、Qwen2.5-32B 和 Gemma-3-27B，不替代主双模型设置，而是检查类别级失败模式是否依赖特定 annotator pair。

- 主标注入口：`run_phase123_full_pipeline.sh`
- 主标注脚本：`step1_llm_annotate.py`
- 主标注产物：`results/phase1_annotations/`
- 交叉模型入口：`run_cross_model_qwen25_teacher_bg.sh`、`run_cross_model_remaining_bg.sh`
- 交叉模型脚本：`cross_model_annotate.py`
- 交叉模型产物：`results/cross_model_validation/default/`

### Module C: CLIP adaptation under label-source choices

该模块包含两个可复现步骤：先把 raw annotation JSONL 转成可审计质量表和下游训练 JSONL，再在同一 bbox-crop 协议下比较三种标签来源和两种训练方式。

- 标签来源：`none`、`agreement`、`gt`
- 训练方式：`linear`、`lora`

`none` 使用全量 Qwen 伪标签，`agreement` 使用 Qwen 与 LLaVA 一致子集，`gt` 是人工标签上界。每个数据集形成 6 个主条件。LP 主矩阵使用五个随机种子，LoRA 包含完整 sweep 和针对关键条件的重复种子复核。

- 过滤与主线入口：`run_phase123_full_pipeline.sh`
- LoRA 与重复种子入口：`run_phase3_lora_sweep_2gpu.sh`、`run_phase3_repeated_seeds.sh`
- 过滤脚本：`step2_filter_analysis.py`
- 训练脚本：`step3_clip_finetune.py`
- 过滤产物：`results/phase2_filtering/`
- 训练产物：`results/phase3_finetune/`

### Module D: Mechanism and strategy audits

这组分析不改变主结果矩阵，而是解释主结果为什么成立、边界在哪里。它们读取固定产物做诊断，因此更接近独立 audit，而不是新的主线阶段。

- Selective routing：只在 TeacherBehavior 上检验类别同时包含高 anchor 和低 anchor 子集时，统一伪标签训练是否是错误抽象。当前 Scheme B 使用 oracle routing，因此只作为 upper-bound diagnostic，不作为 deployable pipeline。
- Retention-ratio curves：用 25\%、50\%、75\%、100\% 随机保留比例和 agreement endpoint 比较，区分“样本量下降”与“agreement 子集偏置”两类机制。
- Strategy audits：用 cross-model consistency、CLIP-assisted confidence filtering、train-split teacher-student self-training 检查主结论是否依赖单一训练策略或单一 anchoring proxy。

- 机制诊断入口：`run_phase45_diagnostics.sh`、`run_phase5_teacher_bg.sh`
- 策略审计入口：`run_phase6_strategy_audit.sh`、`run_phase6_strategy_audit_bg.sh`
- 核心脚本：`step4_selective_annotation.py`、`step5_retention_curve.py`、`cross_model_consistency.py`、`confidence_filtering.py`、`teacher_student_self_training.py`
- 主要产物：`results/phase4_selective_annotation/default/`、`results/phase5_retention_curve/default/`、`results/phase6_strategy_audit/`

## 4. 按双 A100 40G 推荐的训练参数

原脚本里 `batch_size=128` 偏激进，尤其是 `ViT-L/14` 加双卡 DataParallel 时，对输入裁剪分布和 worker 开销比较敏感。

更稳妥的起点：

- `batch_size=64`
- `epochs=20`
- `lr=1e-4`
- `lora_rank=4`

如果前几个 epoch 显存稳定，且吞吐还不错，可以再尝试：

- `batch_size=96`

不建议第一轮就上 `128`，因为一旦不同数据集 bbox 裁剪尺寸分布更极端，容易在长任务中段 OOM。

## 5. 当前版本里已修正的环境问题

为了让计划和代码一致，当前脚本已经按这个环境做了两类修正：

1. 数据根目录不再写死到旧机器路径，优先使用仓库下的 `datasets_scb/`。
2. 训练默认改成使用 `CUDA_VISIBLE_DEVICES=0,1`，不再假设存在 `GPU2,3`。

另外补了一条安全约束：

- 当 `pseudo_strategy` 不是 `gt` 时，如果伪标签文件不存在，训练脚本现在会直接报错，而不是悄悄回退成 GT 训练，避免实验结果被污染。

## 6. 建议复现顺序

下面是复现实验时的输入输出顺序，不是论文叙事顺序。论文建议按四个 analysis modules 讲；执行时则按数据依赖先生成固定产物，再复跑下游矩阵和审计分析。

1. 用 `step0b_cape_zeroshot.py` 生成协议一致的 zero-shot baseline。
2. 设置新的 `RUN_TAG`，用 `run_phase123_full_pipeline.sh` 跑完整的 `val/train` 标注、过滤分析和主线 LP 条件。
3. 检查 `results/phase1_annotations/${RUN_TAG}/`、`results/phase2_filtering/${RUN_TAG}/` 和 `results/phase3_finetune/` 里的 JSONL/CSV/JSON 产物。
4. 运行 `run_cross_model_qwen25_teacher_bg.sh`、`run_cross_model_remaining_bg.sh` 完成交叉模型验证；该分析只需要数据集和固定模型配置，不依赖 CLIP 训练完成。
5. 运行 `run_phase3_lora_sweep_2gpu.sh` 和 `run_phase3_repeated_seeds.sh` 补齐 LoRA 与重复种子复核。
6. 运行 `run_phase45_diagnostics.sh` 和必要的 `run_phase5_teacher_bg.sh` 生成 selective-routing 与 retention-ratio 诊断。
7. 运行 `run_phase6_strategy_audit.sh` 或 `run_phase6_strategy_audit_bg.sh` 生成 cross-model consistency、confidence filtering 和 teacher-student strategy-audit 结果；这些是读取冻结输入的 audit analyses，不改变主矩阵。

## 7. 预期论文主结果

论文主表可以组织成下面这类结构：

| Dataset | Label Source | Training Mode | Val Acc |
|---|---|---|---|
| BowTurnHead | Zero-shot CLIP | none | ? |
| BowTurnHead | Pseudo none | linear | ? |
| BowTurnHead | Pseudo none | lora | ? |
| BowTurnHead | Pseudo agreement | linear | ? |
| BowTurnHead | Pseudo agreement | lora | ? |
| BowTurnHead | GT | lora | ? |

另外再补一张过滤策略表，对应 `step2` 输出。

## 8. 当前判断

这个研究设计本身是成立的，而且和现有数据组织方式匹配。真正要控制好的不是“能不能做”，而是两点：

1. 伪标签质量是否足够支撑 CLIP 超过零样本基线。
2. 一致性过滤带来的精度提升，能否覆盖样本量下降带来的损失。

如果后续确实需要更细粒度编排，可以再补 module-named wrappers（例如主标注、主适配矩阵、机制诊断、策略审计），但不建议为了表面命名去移动已经用于论文追溯的结果目录。

