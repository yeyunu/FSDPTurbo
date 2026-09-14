# 附录

## 术语表

| 术语 | 说明 |
|---|---|
| FSDP2 | PyTorch基于DTensor的全分片数据并行（Fully Sharded Data Parallel）实现，参数、梯度、优化器状态均按DeviceMesh分片 |
| TP（Tensor Parallelism） | 张量并行，将单层内的权重矩阵按列/行切分到多个设备 |
| EP（Expert Parallelism） | 专家并行，面向MoE模型，将专家按专家维切分到多个设备 |
| CP（Context Parallelism） | 上下文并行，将序列维切分到多个设备以支持超长序列训练 |
| Ulysses | 一种基于注意力头维切分的上下文并行方案 |
| EFSDP | 专家FSDP2，对EP切分后的专家参数进一步做全分片 |
| DeviceMesh | PyTorch设备网格，描述多维并行拓扑（如`dp × fsdp × tp × ep`） |
| Dispatcher | 专家并行中的专家计算调度器，决定token分发与专家计算的组织方式 |
| 重计算（Recompute） | 梯度检查点技术，前向时不保存激活、反向时重算以降低显存占用 |
| 分块损失（Chunk Loss） | 将损失计算按分块流水处理，避免大词表场景下logits全量物化 |
| MXFP8 | Microscaling FP8量化格式，支持E4M3/E5M2/HIF8（hifloat8）等变体 |
| MC2 | 昇腾提供的通信计算融合加速算子库，EP路径下用于加速专家计算 |
