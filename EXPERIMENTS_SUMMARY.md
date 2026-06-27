# 实验代码与运行脚本汇总（快速复盘）

下面按文件列出仓库中与实验流程直接相关的 Python 脚本与 Shell 启动脚本，并给出一行功能说明。

- [download_dataset.py](download_dataset.py): 从 HF 镜像下载并恢复式续传实验所需的数据集压缩包与 yaml。

（核心实验代码）

- [step0_zeroshot_eval.py](step0_zeroshot_eval.py): CLIP 零样本评估（bbox crop + ViT-L-14），支持多 GPU DataParallel。
- [step0b_cape_zeroshot.py](step0b_cape_zeroshot.py): 使用 CAPE 提示词集的零样本 ensemble 评估（与 step0 协议对齐）。
- [step1_llm_annotate.py](step1_llm_annotate.py): 用 Hugging Face 多模态模型（Qwen / LLaVA 等）对 bbox 裁剪图像进行自动标注并输出 JSONL。
- [step2_filter_analysis.py](step2_filter_analysis.py): 标注质量分析与四种过滤策略比较（生成伪标签文件供微调使用）。
- [step2b_distribution_bias.py](step2b_distribution_bias.py): 计算类别分布偏移（KL/TV）并输出 per-class 统计与 summary。
- [step3_clip_finetune.py](step3_clip_finetune.py): 用伪标签微调 CLIP（Linear Probe / LoRA），包含数据加载、训练与导出结果 JSON。
- [step4_selective_annotation.py](step4_selective_annotation.py): Selective Annotation（方案B）：高锚点类别微调 + 低锚点使用零样本，合并评估。
- [step5_retention_curve.py](step5_retention_curve.py): retention curve 实验（采样 25/50/75% 点）用于 none 伪标签线性 probe 分析。
- [confidence_filtering.py](confidence_filtering.py): CLIP 辅助的置信度过滤实验（不同阈值、保留率与微调效果汇总）。
- [cross_model_annotate.py](cross_model_annotate.py): 扩展到多个 HF 多模态模型的 cross-model 标注与输出合并文件。
- [cross_model_consistency.py](cross_model_consistency.py): 计算不同模型间一致性、成对协议与与零样本性能的相关性分析并绘图（可选）。
- [teacher_student_self_training.py](teacher_student_self_training.py): teacher-student 自训练实验（在 train-split 上的保留与再训练流程）。
- [summarize_repeated_seeds.py](summarize_repeated_seeds.py): 对重复随机种子实验输出的结果 JSON 汇总统计（均值/标准差等）。
- [strategy_audit_utils.py](strategy_audit_utils.py): 多个脚本共享的工具集（数据加载、crop、编码、训练辅助、路径解析等）。
- [paper/generate_evidence_figures.py](paper/generate_evidence_figures.py): 从各阶段结果中生成论文图表与示例图（绘图样式与导出逻辑）。

（运行脚本）

- [run_phase123_full_pipeline.sh](run_phase123_full_pipeline.sh): 一键运行 Phase1/Phase2/Phase3 的完整流水线（注：会安装依赖并按顺序执行）。
- [run_phase123_full_legacy.sh](run_phase123_full_legacy.sh): 早期保留的全量实验脚本（参考用，推荐使用上面的 pipeline 脚本）。
- [run_phase3_repeated_seeds.sh](run_phase3_repeated_seeds.sh): 在多种 seed/数据集/策略/模式下并行或顺序提交第3阶段微调（并汇总结果）。
- [run_phase3_repeated_seeds_bg.sh](run_phase3_repeated_seeds_bg.sh): 后台启动 `run_phase3_repeated_seeds.sh` 的 launcher（nohup + pidfile）。
- [run_phase3_lora_sweep_2gpu.sh](run_phase3_lora_sweep_2gpu.sh): 逐项做 LoRA sweep（2 GPU DataParallel）并记录日志。
- [run_phase3_teacher_linear_bg.sh](run_phase3_teacher_linear_bg.sh): 针对 TeacherBehavior 的 linear 微调后台启动脚本（agreement / gt 两个并发）。
- [run_phase5_teacher_bg.sh](run_phase5_teacher_bg.sh): 后台运行 step5 retention curve（TeacherBehavior）的脚本 wrapper。
- [run_phase45_diagnostics.sh](run_phase45_diagnostics.sh): 依次运行 step4（selective）和 step5（retention），含失败回退设置。
- [run_phase6_strategy_audit.sh](run_phase6_strategy_audit.sh): 运行 Phase6 的策略审计（cross-model consistency / confidence filtering / teacher-student）。
- [run_phase6_strategy_audit_bg.sh](run_phase6_strategy_audit_bg.sh): 后台启动 Phase6 审计的 wrapper（nohup + pidfile）。
- [run_cross_model_remaining_bg.sh](run_cross_model_remaining_bg.sh): 运行剩余的 cross-model 验证任务（gemma / qwen_32b 等）。
- [run_cross_model_qwen25_teacher_bg.sh](run_cross_model_qwen25_teacher_bg.sh): 专门并行跑 Qwen2.5 的 TeacherBehavior train/val 两份任务并输出日志/pid。

---

如何使用这份汇总

- 若需逐文件的更详细复盘（函数/类结构、关键函数签名、依赖项），回复“逐文件详细复盘”，我会为每个文件生成 6-12 行的结构化摘要。
- 若需我把任一脚本的运行示例（包含 env 变量和可直接复制的命令）写成独立 README 或运行脚本，我可以接着生成。

_已自动从文件顶部 docstring / 脚本注释提取说明并手工校对。_
