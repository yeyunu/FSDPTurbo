# 上下文并行

FSDPTurbo支持Ulysses风格的上下文并行（Context Parallel，CP），将序列维切分到多个设备，支持超长序列训练。

## 配置方法

在`distributed`段配置上下文并行度与函数替换：

```yaml
distributed:
  ulysses_parallel_size: 2
  cp_plan:
    ulysses_function_patches:
      - target_functions:
          - transformers.models.qwen3_5.modeling_qwen3_5.eager_attention_forward
        type: full_attention
      - target_functions:
          - transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward
        type: gated_delta_net
    loss_function_patches:
      - target_functions:
          - transformers.loss.loss_utils.ForCausalLMLoss
        type: causal_lm_loss
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `ulysses_parallel_size` | Ulysses上下文并行度 |
| `kv_allgather_parallel_size` | KV AllGather上下文并行度 |
| `ulysses_function_patches` | 注意力函数替换列表：`target_functions`指定被替换函数（全限定名），`type`指定替换类型（`full_attention`、`gated_delta_net`等，也支持Callable自定义注意力） |
| `kv_allgather_function_patches` | KV AllGather场景的函数替换 |
| `loss_function_patches` | 损失函数替换（如`causal_lm_loss`），保证CP切分后loss正确同步 |

## 配合module_patches

Ulysses CP通常需要替换模型主干的forward（以改变注意力调用与序列切分方式）：

```yaml
module_patches:
  - target: transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model.forward
    replacement: fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5.qwen3_5_model_forward
```

FSDPTurbo已为部分主流模型提供替换实现（见`fsdp_turbo/models/`）。

## CP Loss同步机制

CP将序列切分到多个rank，每个rank只持有部分序列片段。为保证loss计算正确：

1. **全局标签移位**：loss计算前需在完整序列上执行标签移位`shifted_labels[..., :-1] = labels[..., 1:]`，再将移位后的标签切分到各CP rank，以保留跨rank边界的训练样本。
2. **Loss函数替换**：通过`loss_function_patches`将原生损失函数（如`ForCausalLMLoss`）替换为CP感知版本，避免重复进行序列内移位。
3. **跨rank同步**：CP组内的loss已在forward中完成同步，无需额外的all-reduce。

CP loss兼容不传递额外参数的旧版模型，以及Transformers 5.17起Qwen3.5向loss传递`**kwargs`的调用方式。模块加载时读取`torch.nn.functional.cross_entropy`的函数签名，仅转发其明确支持关键字传参的参数，并排除CP已传入的`input`、`target`、`ignore_index`、`reduction`。`use_cache`等模型前向参数不会传入交叉熵；保留`weight`、`label_smoothing`及旧式`size_average`、`reduce`参数的原有行为，`reduction`参数仍由CP指定。

若提供`shift_labels`，它必须是完整序列上已移位的标签，loss直接pad并切分，不再移位；未提供时沿用完整`labels`的全局移位流程。`num_items_in_batch`支持Tensor或整数，表示全局归一化计数；未提供时沿用CP组内有效token计数。

## 注意事项

- **序列长度对齐**：`max_seq_length`应能被`ulysses_parallel_size`整除，避免序列切分时出现不等长片段。
- **数据加载**：CP不改变数据加载方式，仍由各DP rank加载完整样本，序列切分在模型forward内完成。
- **与Chunk Loss互斥**：开启Ulysses CP后不能同时设置`enable_chunk_loss`，模型会直接报错，详见[分块损失计算](./chunk_loss.md)。
- **与微批次切分互斥**：开启CP后不能同时配置`memory.chunk_batch`，该组合无显式报错守护、行为未定义，详见[微批次切分](./chunk_batch.md)。
- **典型长序列配置**：对于32k+ 长序列训练，建议`ulysses_parallel_size=2~4`，配合重计算与FSDP2使用。
