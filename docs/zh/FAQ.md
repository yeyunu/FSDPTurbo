# FAQ

本文档收集FSDPTurbo使用过程中的常见问题与解决方法，按安装与环境、训练启动、并行策略、精度与断点续训分类。

## 安装与环境

### 依赖版本有什么要求？

Python >= 3.10（推荐3.11及以上），PyTorch >= 2.9.0，transformers >= 5.2.0；NPU环境还需配套版本的torch_npu（与PyTorch同版本）与triton-ascend >= 3.2.2。完整依赖清单与已验证版本组合见[软件安装](./training/install_guide.md)。

### NPU环境报CANN/驱动相关错误怎么办？

每个新终端执行训练前需先加载CANN环境变量，未加载时业务代码无法执行：

```bash
# root用户默认路径
source /usr/local/Ascend/cann/set_env.sh
# 非root用户默认路径
source ${HOME}/Ascend/cann/set_env.sh
```

### 如何验证FSDPTurbo安装成功？

```bash
python -c "import fsdp_turbo; print(fsdp_turbo.__file__)"
```

NPU环境可进一步执行`npu-smi info`确认NPU状态，执行以下命令确认torch_npu可用（输出`True`表示正常）：

```bash
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

## 训练启动

### 多机训练如何组网？

通过环境变量配置集群参数后，在所有节点同时执行`bash run.sh`：`MASTER_ADDR`（主节点IP）与`MASTER_PORT`各节点保持相同，节点序号（torchrun的`--node_rank`）各节点不同：

```bash
# 节点0执行
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 bash run.sh
# 节点1执行
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=1 bash run.sh
```

多机组网失败或初始化卡住时，依次检查：

- `MASTER_PORT`是否被占用，可更换端口重试。
- 节点间网络连通性与防火墙端口放通情况。
- 大集群HCCL连接超时，可将`HCCL_CONNECT_TIMEOUT`增大（推荐`1800`）。

集群变量与NPU性能调优环境变量详见[环境变量说明](./features/environment_variable.md)。

### 报OOM（显存不足）怎么办？

按以下顺序处理：

1. 降低`data.batch_size`或`max_seq_length`；实际卡数少于示例默认（16卡）时相应调整并行度。
2. 开启激活重计算`memory.recompute: true`（最常用、收益最稳定）。
3. 显存仍不足时叠加激活值交换`memory.swap_activation: true`或`AdamWSwap`（优化器状态占用大时优先）。
4. 整体显存紧张再开启`fsdp_plan.cpu_offload: true`。

各手段的显存收益与耗时影响对比见[模型性能调优](./training/tuning/performance_tuning.md)。

### 训练权重保存到哪里？

保存在`checkpoint.output_dir`目录，按`save_steps`间隔保存。保存内容包括模型参数、优化器状态、学习率调度器状态与训练步数，详见[断点续训](./features/checkpoint_resume.md)。

## 并行策略

### 配置了`data_parallel_size`为什么不生效？

`data_parallel_size`与`expert_data_parallel_size`为自动推导项，配置值会被覆盖：

- `data_parallel_size` = 总卡数 ÷（`fully_shard_parallel_size` × `tensor_parallel_size`）
- `expert_data_parallel_size` = 总卡数 ÷（`expert_parallel_size` × `expert_fully_shard_parallel_size`）

用户只需配置`fully_shard_parallel_size`、`tensor_parallel_size`、`ulysses_parallel_size`、`kv_allgather_parallel_size`、`expert_parallel_size`、`expert_fully_shard_parallel_size`，且需满足整除约束（各组合乘积整除总卡数），否则初始化报错。

### 训练前需要做权重切分或格式转换吗？

不需要。FSDPTurbo直接加载HuggingFace格式权重，无需转换为Megatron等格式；调整并行度也无需重新切分权重，直接在`distributed`段配置并行度即可。需要说明的是，权重加载后训练脚本会按`torch_dtype`配置对模型参数与buffer做dtype转换（`convert_model_dtype`），该处理在脚本内自动完成，无需离线操作。

### MoE模型EP报错或专家模块不匹配怎么办？

常见原因与处理：

- **专家模块结构要求**：EP要求experts模块持有3D张量参数（`gate_up_proj`/`down_proj`，形状`[num_experts, out_dim, in_dim]`），ModuleList形式的独立专家子模块需先转换为3D张量聚合结构。
- **整除约束**：`expert_parallel_size`需能整除每层专家总数。
- **dispatcher选型**：可使用 `all2all`（`use_fused_ops: true`）或 `mc2`；专家计算语义非标准时需实现自定义EP forward。

详见[专家并行](./features/expert_parallel.md)。

### `fixed_router`有什么用？

调试手段：开启后以固定均匀路由（轮转均分）替代模型实际路由输出，用于避免负载不均带来的性能与精度问题；仅在`expert_parallel_size > 1`时生效。

## 精度与断点续训

### 如何排查精度异常？

- 确认精度配置：`model.torch_dtype`（权重精度）、`fsdp_plan.param_dtype`（前向计算精度，如`bf16`）、`fsdp_plan.reduce_dtype`（梯度归约精度，如`fp32`）。
- 需要复现确定性结果时开启`run.deterministic`。
- EP与特殊并行组合下出现梯度错误时，检查`ep_plan.gradient_divide_factor`是否需要手动指定。

### 断点续训如何使用？

将`checkpoint.resume_from_checkpoint`指向`checkpoint.output_dir`下对应checkpoint目录，启动训练后各rank自动加载各自的分片，从保存的step继续训练，学习率与优化器状态同步恢复。详见[断点续训](./features/checkpoint_resume.md)。

### 恢复训练后step从0开始了？

FSDPTurbo恢复时会校验数据集指纹（由`dataset_path`/`dataset_config`/`split`/`batch_size`构成）：若与保存checkpoint时不一致，`global_step`与`consumed_samples`会重置为0并打印warning，避免训练数据错乱。确认数据集配置未变化即可正常续训。

### 可以更换并行度后恢复训练吗？

权重使用PyTorch分布式Checkpoint（DCP）格式保存，支持不同并行度下的加载（参数自动重新分片），但需保证模型结构一致；建议保持并行配置相同以避免兼容性问题。
