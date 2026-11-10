# 上下文并行

FSDPTurbo支持Ulysses、KV AllGather等上下文并行（Context Parallel，CP），将序列维切分到多个设备，支持超长序列训练。Kimi K3还支持KCP与Hybrid P2P。

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
    qsa_cp_query_chunk_size: null
    qsa_sparse_gather: false
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `ulysses_parallel_size` | Ulysses上下文并行度 |
| `kv_allgather_parallel_size` | KV AllGather上下文并行度 |
| `kcp_parallel_size` | KCP状态并行度，默认1 |
| `kda_p2p_parallel_size` | Hybrid P2P的KDA P2P并行度（K），默认1 |
| `kda_backend` | Kimi K3的KDA后端：`kcp`（默认）或`hybrid_p2p` |
| `mla_backend` | Hybrid P2P的MLA后端：`ulysses`（默认）或`kv_allgather` |
| `ulysses_function_patches` | 注意力函数替换列表：`target_functions`指定被替换函数（全限定名），`type`指定替换类型（`full_attention`、`gated_delta_net`等，也支持Callable自定义注意力）。模型已内置Ulysses forward时显式设为`[]`（如GLM-5.3-Flash）；其他模型在`ulysses_parallel_size > 1`时须配置对应替换 |
| `kv_allgather_function_patches` | KV AllGather场景的函数替换 |
| `loss_function_patches` | 损失函数替换（如`causal_lm_loss`），保证CP切分后loss正确同步 |
| `qsa_cp_query_chunk_size` | Qwen3.8 QSA在CP下的query分块大小；`null`使用模型默认值128 |
| `qsa_sparse_gather` | Qwen3.8是否只收集QSA选中的KV；长序列建议设为`true` |

## 配合module_patches

Ulysses CP通常需要替换模型主干的forward（以改变注意力调用与序列切分方式）：

```yaml
module_patches:
  - target: transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model.forward
    replacement: fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5.qwen3_5_model_forward
```

FSDPTurbo已为部分主流模型提供替换实现（见`fsdp_turbo/models/`）。

## Kimi K3

Ulysses、KCP和Hybrid P2P共用[K3配置](../../../examples/kimi_k3/config.yaml)：

| 模式 | `kda_backend` | `ulysses_parallel_size`（U） | `kcp_parallel_size` | `kda_p2p_parallel_size`（K） |
|---|---|---|---|---|
| Ulysses | `kcp` | >1 | 1 | 1 |
| KCP | `kcp` | 1 | >1 | 1 |
| Hybrid P2P | `hybrid_p2p` | ≥1 | 1 | ≥1 |

Ulysses和KCP的函数替换配置分别见[Ulysses示例](../../../examples/kimi_k3/README.md#开启-ulysses-cp)和[KCP示例](../../../examples/kimi_k3/README.md#开启-kcp)。
KCP的KDA层通过AllGather同步状态，MLA层使用Ulysses；序列按顺序切分，长度须不小于K且能被K整除。

Hybrid P2P在U组切分head，在K组通过P2P归并状态，完整CP大小为`P=U*K`，K=`kda_p2p_parallel_size`。
U=1退化为P2P，K=1退化为Ulysses，U=K=1无跨进程CP通信。

Hybrid P2P配置示例（完整设置`cp_plan`，其余配置按需保留）：

```yaml
model:
  attn_implementation: eager
distributed:
  ulysses_parallel_size: 2
  kcp_parallel_size: 1
  kda_p2p_parallel_size: 4
  cp_plan:
    kda_backend: hybrid_p2p
    mla_backend: ulysses
    ulysses_function_patches: []
    kcp_function_patches: []
    kv_allgather_function_patches: []
    loss_function_patches: []
module_patches: []
```

Kimi K3基础模型与Hybrid CP工厂统一位于`fsdp_turbo/models/kimi/modeling_kimi_k3.py`，通过同一模块选择模型类。
KDA专用P2P状态通信位于`distributed/context_parallel/kda_hybrid/p2p.py`；顶层`hybrid_cp.py`提供公共运行时与绑定，`kcp/`保留一维KCP实现。

Hybrid P2P无需函数替换，patch列表设为`[]`。`mla_backend: ulysses`使用顺序切分，
`kv_allgather`使用首尾切分；MLA均在完整P域通信，后者无需开启独立KV-AllGather并行轴。Ring暂不支持。

Hybrid P2P使用限制：

- Hybrid P2P的`kcp_parallel_size`保持1；Ulysses/KCP的`kda_p2p_parallel_size`保持1。
- U、K须为2的幂（含1），进程数须能被P整除，并满足其余FSDP/EP并行度约束。
- KDA head数须能被U整除；MLA使用Ulysses时，其query head数须能被P整除。
- `data.batch_size=1`、`tensor_parallel_size=1`、`kv_allgather_parallel_size=1`；不支持chunk batch、chunk loss、packed输入和cache；仅支持一个连续有效token区间的padding。
- NPU需匹配的`fla_npu`分阶段接口，Q/K/V为BF16、head_dim=128、chunk_size=64；模型的`gate_lower_bound`须在`[-5,0)`，不得修改已训练模型门控来绕过限制。

## Qwen3.8-Flash-Next

运行Qwen3.8示例前必须安装线性注意力依赖：

```bash
pip install fla-core==0.5.2 flash-linear-attention==0.5.2
```

Qwen3.8的CP实现在`fsdp_turbo/models/qwen/modeling_qwen4_exp.py`中，
在多模态融合后切分序列，并覆盖QSA全注意力与GatedDeltaNet线性注意力。
`examples/qwen3_8/train.py`会将`cp_plan`中的QSA参数传入模型文本配置。

当`kv_allgather_parallel_size > 1`时，Qwen专用的
`qsa`与`gated_delta_net` forward已内置KV-AllGather，
无需配置通用`kv_allgather_function_patches`。`ulysses_parallel_size`可以设为
`1`，此时仍可单独使用KV-AllGather。

## CP Loss同步机制

CP将序列切分到多个rank，每个rank只持有部分序列片段。为保证loss计算正确：

1. **全局标签移位**：loss计算前需在完整序列上执行标签移位`shifted_labels[..., :-1] = labels[..., 1:]`，再将移位后的标签切分到各CP rank，以保留跨rank边界的训练样本。
2. **Loss函数替换**：通过`loss_function_patches`将原生损失函数（如`ForCausalLMLoss`）替换为CP感知版本，避免重复进行序列内移位。模型已内置CP loss时无需配置替换。
3. **跨rank同步**：CP组内的loss已在forward中完成同步，无需额外的all-reduce。

CP loss兼容不传递额外参数的旧版模型，以及Transformers 5.17起Qwen3.5向loss传递`**kwargs`的调用方式。模块加载时读取`torch.nn.functional.cross_entropy`的函数签名，仅转发其明确支持关键字传参的参数，并排除CP已传入的`input`、`target`、`ignore_index`、`reduction`。`use_cache`等模型前向参数不会传入交叉熵；保留`weight`、`label_smoothing`及旧式`size_average`、`reduce`参数的原有行为，`reduction`参数仍由CP指定。

若提供`shift_labels`，它必须是完整序列上已移位的标签，loss直接pad并切分，不再移位；未提供时沿用完整`labels`的全局移位流程。`num_items_in_batch`支持Tensor或整数，表示全局归一化计数；未提供时沿用CP组内有效token计数。

## KV AllGather下的Engram边界token

DeepSeek-V4.1的KV AllGather CP将序列切成连续等长分片，Q保持本地、KV全局可见。开启Engram时，n-gram哈希需要回看当前位置左侧最多`max_ngram_size - 1`个token。为避免分片边界处的n-gram被截断，每个CP rank不再all-gather完整序列，而是通过一次变长all-to-all：

1. rank r把本分片尾部`max_ngram_size - 1`个token发给rank r+1；
2. rank 0没有左邻居，其halo用Engram pad token填充——与`NgramHashState`在n-gram回看越过序列起点时的替换值完全一致；
3. 各rank在`[halo | 本地分片]`上计算哈希，再切回本地行。

该做法与全量all-gather后哈希逐位等价（已在确定性8卡训练中对拍验证），且交换载荷与CP并行度、序列长度无关。KV AllGather CP当前仅支持训练模式下的纯文本等长分片，不支持padding、图像输入与decode缓存。

## 注意事项

- **序列长度对齐**：`max_seq_length`应能被`ulysses_parallel_size`整除，避免序列切分时出现不等长片段。
- **DeepSeek-V4 compressor CP 单轴限制**：使用`compressor_full_attention`或`compressor_full_attention_transposed`时，`kv_allgather_parallel_size`和`ulysses_parallel_size`仅允许一个大于1，另一个必须为1；双轴同时启用会在配置阶段报错。DeepSeek-V4 attention 的 mask/compressor/attention 语义位于 `fsdp_turbo/models/deepseek/deepseek_v4/context_parallel.py`；CP 通信侧通过 `fsdp_turbo/distributed/context_parallel/patch.py` 中的通用 capability installer 接入，通用 CP attention wrapper 不直接承载模型实现。若启用了该计划但模型中没有 Transformers `DeepseekV4Attention`，初始化会显式报错。KCP 暂不支持该 DeepSeek-V4 compressor attention 适配，配置阶段会拒绝该组合。
- **数据加载**：CP不改变数据加载方式，仍由各DP rank加载完整样本，序列切分在模型forward内完成。
- **Qwen3.8 CP及Kimi K3 Ulysses/KCP支持Chunk Loss**：CP下可同时设置`memory.chunk_loss: true`，此时损失走`cp_chunk_lm_loss`分块路径，不物化`[local_seq, vocab]`全量logits（1M序列、CP=8、vocab≈160k时约200 GiB/rank，分块后降至~2 GiB），详见[分块损失计算](./chunk_loss.md)。
- **与微批次切分互斥**：开启CP后不能同时配置`memory.chunk_batch`，启动时会显式报错拒绝该组合，详见[微批次切分](./chunk_batch.md)。
- **典型长序列配置**：对于32k+ 长序列训练，建议`ulysses_parallel_size=2~4`，配合重计算与FSDP2使用。
