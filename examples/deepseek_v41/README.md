# DeepSeek-V4.1-Flash FSDP 训练示例

本目录独立适配 `deepseek-ai/DeepSeek-V4.1-Flash`
（`model_type=deepseek_v41`），不会修改或覆盖 `examples/deepseekv4` 中的旧版
DeepSeek-V4-Flash 示例。

官方模型仓库当前只发布了 `inference/model.py` 参考实现，没有发布可训练的
Transformers `deepseek_v41` 建模模块。所需参考源码已固定复制到
`fsdp_turbo/models/deepseek/deepseek_v41/`：

- `inference/model.py`
- `inference/engram.py`
- `inference/vision.py`
- `inference/image_processor.py`

源码对应 revision `dba1be0a40aa45a94ad051997016db3960a90277`，运行示例时不再
动态下载或加载模型 Python 文件。模型结构优先读取 `model.config_path`，未设置时读取
`model.model_name_or_path/config.json`；tokenizer 从 `model.tokenizer_name_or_path` 读取。

语言模型适配代码按职责拆分在同一模型目录内：

| 文件 | 职责 |
|---|---|
| `model_args.py` | `ModelArgs` 模型结构参数 |
| `layers.py` | 基础层、词表输出头、MoE 和 MHC 解码层 |
| `engram/engram.py` | Engram 残差模块、哈希布局和 n-gram 哈希状态 |
| `engram/host_offload_engram.py` | `HostOffloadEngram`（ElasticBuffer）与 `TorchHostOffloadEngram`（PyTorch）存储后端 |
| `engram/engram_optimizer.py` | 稠密与稀疏优化器编排、活跃行 Adam、调度器及 checkpoint 工具 |
| `runtime.py` | 跨层注意力缓存 |
| `utils.py` | 模型内共用的序列边界、RoPE 频率拼接和临时 dtype 工具 |
| `compressor.py` | KV 压缩投影与池化 |
| `indexer.py` | 稀疏索引与候选块选择 |
| `attention.py` | 滑动窗口和压缩稀疏注意力 |
| `dspark.py` | DSpark 推测解码组件与采样 |
| `model.py` | Transformer 装配、前向及原有类和函数的兼容导出 |

组件导入统一使用 `from fsdp_turbo.models.deepseek.deepseek_v41.model import ...`。
拆分不改变参数名称与 checkpoint 路径；各组件共用 `runtime.py` 的跨层注意力缓存。`attention_mask`、`sfa_layout` 和 `cu_seqlens` 通过前向参数显式传递，不再依赖全局上下文。

`fsdp_turbo.models.deepseek.deepseek_v41` 已移除参考实现自身的 inference tensor parallel 分支及
对应的 `world_size`、`rank` 全局状态，通过 FSDPTurbo `ops/` 语义层分发可反向传播的 attention/mHC 算子（含 CPU 兜底和
NPU 实现），并提供图文 causal-LM loss 接口。模型专用 processor 按官方协议生成
`images` 和 `token_types`，视觉塔与语言模型可联合反向。训练并行由 FSDPTurbo 统一
管理：共享专家 SwiGLU 使用 TP，打包后的路由专家使用 EP，其余模块使用 FSDP2。

```bash
cd examples/deepseek_v41
bash run.sh
```

训练 head 的计算精度由 YAML 中的 `model.head_compute_dtype` 控制。A3、A5 和
TND 示例均设为 `bf16`，使用 BF16 head + FP32 CE；未配置该字段时框架默认 `fp32`：

```yaml
model:
  head_compute_dtype: "bf16"
```

`head_compute_dtype` 只控制词表投影的矩阵乘法精度，不改变参数存储或优化器配置。
logits 转为 FP32 后计算交叉熵；BF16 模式返回的 logits 为 BF16。
BF16 投影存在舍入差异，启用后应对比 loss 曲线、梯度范数及训练吞吐。

在昇腾 NPU 上还可以设置 `head_compute_dtype: "hf32"`：输入和 head 权重在投影时转为
FP32，head 前向及反向矩阵乘临时启用
`torch_npu.npu.matmul.cube_math_type = torch_npu.npu.CubeMathType.USE_HF32`，
计算完成或发生异常后恢复原设置。HF32 是矩阵乘计算模式，返回的 logits 和交叉熵仍为
FP32，参数存储和优化器配置不变。该模式要求 NPU 及支持上述接口的 `torch_npu`，
在 CPU/CUDA 或接口缺失时明确报错。HF32 同样存在精度差异，需要在目标设备上验证训练表现。
接口语义参见[昇腾文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu-matmul-cubeMathType.md)。

模型结构参数优先从 `model.config_path` 指定的 JSON 文件读取；未设置时读取
`model.model_name_or_path/config.json`。选中的文件不存在时直接报错，不回退到仓库内置配置。
`config_path` 是文件路径，`model_name_or_path` 是目录路径；相对路径按进程工作目录解析。
默认不读取官方发布的 FP8/FP4 权重，模型参数以 BF16 随机初始化；启用
`model.load_pretrained_weights` 时则从已反量化的 BF16 HuggingFace safetensors 初始化。

```yaml
model:
  model_name_or_path: /path/to/DeepSeek-V4.1-Flash
  config_path: /path/to/custom_model_config.json  # 可省略，优先于模型目录中的 config.json
```

TND 示例配置使用 `data.dataset_path` 指向 packed JSONL 数据集；每条记录的
`documents` 包含一个或多个已预处理文档。准备好 `messages + images` 格式的
JSONL 后，用本目录的 `pack_deepseek_v41_tnd.py` 打包。图像路径应为绝对路径，或相对于
运行时工作目录可解析。

```json
{"messages":[{"role":"user","content":"描述这张图片"},{"role":"assistant","content":"图片内容是……"}],"images":["/path/to/image.jpg"]}
```

## 使用虚构图文数据测试

`generate_mock_data_for_vlmodel.py` 改编自 MindSpeed-MM 的同名脚本，可生成固定图片尺寸、
图片数量和文本长度的重复样本，用于验证训练链路与测量吞吐。它使用本模型的 tokenizer、
对话模板和图片网格规则，统计图片展开后的序列长度。

在仓库根目录执行（先安装 FSDPTurbo 依赖）：

```bash
python examples/deepseek_v41/generate_mock_data_for_vlmodel.py \
  --tokenizer_path /path/to/DeepSeek-V4.1-Flash \
  --model_name_or_path /path/to/DeepSeek-V4.1-Flash \
  --pic_width 512 --pic_height 512 --num_pics 2 \
  --text_length 1024 \
  --num_samples 512 --save_dir ./data/mocked_v41
```

默认输出 JSONL 和一张供所有样本复用的随机 JPEG。每条样本的 `images` 都引用该图片的
绝对路径，因此训练进程需要能访问生成目录；多机运行时应将其放在共享存储上。生成器会输出
实际文本 token 数和图文展开后的总 token 数。`--text_length` 是 user 与 assistant
重复文本合计的 token 下限；生成器沿用 MindSpeed-MM 的重复次数公式，并按实际分词结果
补足长度。生成器不限制图文展开后的总长度；超出训练 YAML 的 `data.max_seq_length` 时，
数据加载器会截断文本并保留完整图片。若图片和必要的对话标记本身
超过训练长度，数据加载器会报错。可用 `--output_format json` 输出数组格式。

将生成器输出的 JSONL 绝对路径填入 `data.dataset_path`，保持 `data.is_multimodal: true`，
再运行 `bash examples/deepseek_v41/run.sh`。该数据由重复文本和随机图片组成，仅用于
功能与性能测试。

训练时，`data.max_seq_length` 同时控制纯文本 tokenizer 的截断长度和图文样本在图片
展开后的长度。图文样本超长时，DeepSeek-V4.1 processor 按 prompt/response 长度分配
预算，截去超出的文本；图片 token 区间会完整保留，并同步更新图片在序列中的位置。
若图片和必要的对话标记本身已超过预算，数据处理会报错，需要减少图片数量或尺寸，
或增大 `max_seq_length`。

## 数据格式转换
如果使用数据格式为原生的llava格式，不是上述中json格式，如原llava_instruct.json格式为：
```json
[
  {
    "id": "000",
    "image": "000.jpg",
    "conversations": [
      {
        "from": "human",
        "value":"描述这张图片"
      },
      {
        "from":"gpt",
        "value":"图片内容是……"
      }
    ]
  },
]
```
可以使用下面的脚本进行转换
```bash
bash convert_llava_for_fsdp_turbo.sh
```

## TND 与 BSND 训练

原有 `config.yaml` 保持原样，用于原有非打包数据；`config_tnd.yaml` 用于 TND，
开启 SparseFlashMla 和 `fixed_router`。`data.sfa_layout: TND` 将同一记录里的文档拼接到 token 维，向
SparseFlashMla 传递逐文档的 `actual_seq_len`，并隔离注意力、位置编码及 loss。
对 packed 数据显式设置 `BSND` 时，同一数据集按 batch 维填充，便于比较。两种布局都要求
`model.use_sparse_flash_attn: true`；多文档 TND 当前不能与 fused compressor 或
fused lightning indexer 同时使用。

```bash
python examples/deepseek_v41/pack_deepseek_v41_tnd.py \
  --input /path/to/normalized.jsonl \
  --output /path/to/packed.jsonl \
  --tokenizer examples/deepseekv4/tokenizer \
  --model-name-or-path /path/to/DeepSeek-V4.1-Flash \
  --max-seq-length 1024

# TND 配置默认使用 ./data/packed.jsonl；可在启动时传入实际数据路径。
CONFIG_FILE=config_tnd.yaml bash examples/deepseek_v41/run.sh \
  --dataset-path /path/to/packed.jsonl
```

使用同一 packed 数据集比较 TND/BSND 时，请另建本地 BSND 配置副本，显式设
`data.sfa_layout: BSND` 和 `model.use_sparse_flash_attn: true`；不要改动原有
`config.yaml`。若比较 loss 或梯度，两份临时配置还应都关闭 `fixed_router`，
并保持种子和其余训练参数一致：当前固定路由按展平后的 token 序号分配专家，
BSND 的 padding 会改变序号。在 A2/A3 上使用没有压缩 KV 的
SparseFlashMla 路径时，设置 `FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV=3`；A5
使用默认值。

## 当前支持范围

| 能力 | 状态 |
|---|---|
| BF16 图文 causal-LM 训练 | 支持 |
| FSDP2 | 支持 |
| 共享专家 SwiGLU tensor parallel | 支持 |
| 路由专家 expert parallel | 支持 |
| Padding attention mask | 支持 |
| Packed TND/BSND SparseFlashMla 训练 | 支持；需配套 CANN 算子包 |
| 官方 FP8/FP4 checkpoint 加载 | 暂不支持 |
| V4.1 attention tensor parallel | 暂不支持 |
| V4.1 eager KV-AllGather context parallel（小算子 CP） | 支持非 packing BSND 纯文本、图文及右侧 padding，可配合 Block 重计算；TP/Ulysses/KCP=1，关闭全部融合开关、Triton RoPE、chunk_batch 和 chunk_loss |
| CSA2 跨层压缩状态与稀疏索引结构 | 支持构造与结构验证 |
| CANN Compressor/CompressorGrad 融合压缩算子 | 可选；设置 `model.use_fused_compressor: true` |
| Lightning Indexer 融合算子 | 可选；支持 Atlas A3（Ascend 910_93）、Ascend 910B/910C 和 Ascend950DT |
| MhcPreSinkhornPremix/MhcPost 融合算子 | 可选；设置 `model.use_fused_mhc: true` |
| Clipped SwiGLU 前反向融合 | 可选；设置 `model.use_fused_clipped_swiglu: true`，模型的 `swiglu_limit` 必须大于 0 |
| Inplace-RoPE Triton算子 | 可选；设置 `model.use_triton_inplace_rope: true` |
| DeepGEMM attention 输出投影 | 可选；设置 `model.use_deepgemm_wo_a: true`，仅优化 BF16 NPU 的 `wo_a` |
| 视觉 FlashAttentionScore/Grad 融合算子 | 可选；设置 `model.use_fused_vision_flash_attn: true`；仅支持 BF16 |
| CSA2 activation recompute | eager CP 路径支持；普通共享缓存路径暂不支持 |
| Engram BF16 训练、哈希表常驻行切分 | 支持（CPU 两进程已验证） |
| Engram Host BF16 + ActiveRowAdam FP32 moments | 已接入可选后端；ElasticBuffer 需要配套 CANN 算子包 |
| DSpark 训练 | 暂不支持 |

未支持的路径在适配器或配置中会显式拒绝/关闭，避免把旧 V4 的实现误用于 V4.1。

融合索引开关开启后会读取当前 NPU 型号并自动选择实现：Atlas A3（Ascend 910_93）、
Ascend 910B/910C 使用 `LightningIndexerV2/SparseLightningIndexer`，Ascend950DT
使用 `QuantLightningIndexer/QuantSparseLightningIndexer`。A3/910B/910C 路径要求
安装提供 candidate source/consumer 算子的 `cann_ops_transformer` 或
`cann_ops_transformer_custom`；Ascend950DT 路径需要 Python 3.11、`cannbotdsl`、
`cannbot_arena_net_ops` 和配套 `cann_ops_transformer`，AOT profile 固定
`index_topk=512`，并支持 64/128 的 PA page size。默认保持关闭，关闭时继续运行原
BF16 `einsum + topk` 实现。当前融合路径仅接入训练/整段 prefill，尚未接入逐 token
decode 的累计序列元数据。

MHC 融合开关同时替换每个 Attention/FFN 前的 projection、Sinkhorn、premix，以及
子层输出后的 residual/post 混合。训练反向分别调用
`MhcPreSinkhornBackward` 和 `MhcPostBackward`；跨子层传递的 `h_pre` 梯度通过等价的
小型 pre projection 保留。该路径要求算子包包含 experimental
`MhcPreSinkhornPremix` 以及正式的 MHC backward/post 算子，并要求
`hc_mult=4`、`hc_sinkhorn_iters=20`。当前 master 版本在 Ascend 950 上仅接受隐藏维
4096 或 7168，因此默认的 5120 配置需等待配套算子扩展规格后才能在该平台开启。

融合压缩开关开启后，压缩率大于 1 的 KV source 层调用训练版 CANN
`Compressor`，反向调用配套的 `CompressorGrad`；压缩率 1 仍保留普通线性投影。
当前模型没有 compressor positional bias，因此向算子传入零 `ape`。该路径要求
`cann_ops_transformer` 与 CANN/torch-npu 版本配套；默认保持关闭，关闭时继续运行
原 FP32 投影、softmax pooling 实现。

小算子 CP 使用模型原生接口，配置如下：

```yaml
distributed:
  kv_allgather_parallel_size: 2
```

支持非 packing BSND 文本、图文及右侧 padding。示例 Trainer 按 CP 数与各层压缩比
要求补齐完整序列，补齐后不得超过 `data.max_seq_length`；仅接受前缀连续有效的
二值 mask。labels 保持完整未移位，padding 目标被屏蔽，全局移位保留分片边界的预测目标。
不支持左侧 padding、mask 空洞或 TND；TP/Ulysses/KCP 必须为 1，关闭全部融合开关、
Triton RoPE、chunk_batch 和 chunk_loss。非 CP 训练可使用下文的 chunk loss。

CP 可配合框架现有的非重入重计算，按 Block 开启：

```yaml
memory:
  recompute: true
  recompute_plan:
    - model.layers.{*}
```

CP 的跨层压缩 KV、索引和候选状态随层输入/输出显式传递，重放使用该层前向时的
状态，不读取后续层覆盖的共享缓存。普通路径仍沿用原实现。

图文数据设置 `data.is_multimodal: true` 并使用现有 processor。完整视觉融合后再切分
语言序列，允许图片跨分片，保留 MoE 图像路由，Engram 历史排除图片和 padding。
支持混合纯文本及整批无图；视觉编码仍在各 CP rank 重复执行，不承诺视觉显存随 CP 缩减。

Clipped SwiGLU 融合开关开启后，`swiglu_limit > 0` 的共享专家和路由专家
调用 CANN `ClippedSwiglu`，反向调用配套的 `ClippedSwigluGrad`，融合裁剪与激活计算。
该路径要求 `cann_ops_nn` 同时提供前反向接口及配套 CANN 内核。使用 EP 时，
内置 `all2all`、`allgather`、`chunkmoe`、`mc2`、`domino` 及
`custom_native_eager_forward` 已接入该开关。配置层不限制 dispatcher，
也不要求 `use_fused_ops` 或 `use_swiglu` 为 `true`；其他自定义或 callable dispatcher
需自行接入融合激活。开关开启且 limit 为正时，已接入路径显式选择 clipped SwiGLU。
`custom_native_eager_forward` 保持 FP32 激活、在 down projection 前乘路由权重；内置路径沿用 grouped GEMM
的激活 dtype，在 down projection 后的 combine 阶段乘路由权重。
开关不改变各路径的 dtype、权重乘入位置或参数布局。默认保持关闭，关闭时继续使用原裁剪与
SwiGLU 实现；`swiglu_limit <= 0` 时保留各 dispatcher 的原激活路径。
MC2 遇到任一源 rank 无输入时，所有 rank 统一回退通用 AllToAll。当前融合路径仅支持 eager
autograd 一阶梯度，不包含 TorchAir 图模式或高阶梯度。

DeepGEMM 输出投影通过训练 YAML 的 `model` 配置开启：

```yaml
model:
  use_deepgemm_wo_a: true
```

该开关默认为 `false`，关闭时使用原 `torch.einsum`；开启后仅替换 BF16 NPU 的
`wo_a` 前向和一阶反向，`wo_b` 保持原实现。需要环境中已有兼容的 DeepGEMM Ascend；
显式开启但未安装 `deep_gemm` 时，在 rank 0 告警一次，并回退到原 `torch.einsum`，
前反向仍在输入原设备执行。依赖探测结果按进程缓存；首次调用及变长输入可能触发 JIT 编译。
已安装但依赖损坏、接口不兼容、非 BF16、JIT 或设备执行失败仍报错，不会被回退掩盖。
该选项不能与当前原生 KV-AllGather CP 同时开启，配置初始化会拒绝该组合。

默认 8 卡拓扑为 `FSDP=4, TP=2, EP=4, expert DP=2`。注意力包含 attention
sink 与 indexed sparse-attention 的特殊头布局，因此当前保持复制，不能直接套用通用
Colwise/Rowwise 规则；`cp_plan` 也保持为空。

## Chunk Loss

在所用 YAML 的 `memory` 下开启分块损失，默认关闭：

```yaml
memory:
  chunk_loss: true
  chunk_loss_size: 512
```

`chunk_loss_size` 是每块的序列 token 数，必须为正整数。该功能沿用旧版
DeepSeek-V4-Flash 的通用 `chunk_loss()`，逐块计算输出投影、交叉熵及其梯度，
按整批有效标签数归一化。它支持 `head_compute_dtype` 的 FP32、BF16 和 HF32
路径，交叉熵保持 FP32；HF32 仍需要配套的 Ascend 环境。

分块计算保留 padding、`-100` 标签及 packed TND 的文档边界屏蔽，也可用于
图文训练。整批标签均被屏蔽时返回零 loss 和零梯度。它不能与 CP 同时开启，
配置会在启动时拒绝该组合。

Recompute 与 activation swap 的 plan 可继续匹配 decoder 层，但不能覆盖
`model.head` 或根模块（包括匹配它们的通配符）。分块求导使用的 `torch.func`
不支持这些 wrapper 的 saved-tensor hooks；配置会提前拒绝该组合。

有 `labels` 时，分块模式返回 `loss`，`logits=None`，不生成全量 logits，也
不需要收集全部中间 hidden states。没有 `labels` 时仍沿用普通 logits 路径。
adapter 初始化时直接调用已有
`apply_chunkloss_module(head)`，forward 通过
`self.model.head(hidden_states, self.loss_function, labels)` 进入模型自己的 loss 方法。
adapter 保留原有的因果移位、padding 和 TND 边界屏蔽；普通投影复用通用
`calculate_lm_loss`，HF32 分支沿用现有算子。原 head 类型、初始化分布、
`model.head.weight` 参数名与 FSDP forward 边界保持不变。
`ParallelHead` 继承 `nn.Linear`，不使用 bias，权重仍在 FSDP 分片后初始化。

模型工厂继续使用原有 `chunk_loss`、`chunk_loss_size` 参数，训练入口与 YAML
保持不变，不额外传递工厂或 loss 回调；不新增独立 loss 文件。
`fsdp_turbo/memory/chunk_loss/chunkloss_lm_head.py` 保持主仓原样。

沿用主仓通用实现，跨块权重梯度按传入 head 权重的 dtype 累加。
BF16 head 投影配合 BF16 梯度累加时，该缓冲区占用为
FP32 的一半；logits 仍转换为 FP32 计算交叉熵，loss 也保持 FP32。
多分块低精度累加会引入舍入误差，需要在实际训练中对比 loss 曲线和梯度范数。
FP32/HF32 投影的权重及其梯度累加仍使用 FP32，不具备上述缓冲区减半收益。

## Engram 切分

官方发布配置中的 Engram 位于第 1、14 层（从 0 开始）。每个位置查询
`(4 - 1) × 8 = 24` 个哈希桶，每个桶向量宽 256。两张表分别有
384,006,168 和 384,016,682 行，权重分别约 98.306B 和 98.308B 参数。
Engram 不再保留参考实现的 FP8 `ParallelEngramEmbedding` 路径。存储后端直接决定
表实现；当前支持 PyTorch Host offload 和 CANN `ElasticBuffer` 两种实现。
哈希表沿 EP 域切行，TP/EDP 副本共享相同行分片；小型 `wkv`、`q_weight`、
`k_weight` 仍由所属 Block 的 FSDP 管理。FSDPTurbo 自动把表参数加入 FSDP
的忽略集合，不需要修改 `fsdp_plan`，也不要给表配置通用 TP。

当前训练配置在第 1 层放置一张 384,006,168 行的发布规模 Engram 表；其 BF16
纯权重约为 183.11 GiB，实际运行必须结合切分后的梯度和优化器状态评估容量。

Host Engram 始终先在 meta 上创建表，无需在构造模型时传入 mesh。
其他参数和运行时 buffer 先在 CPU 创建。
`prepare_deepseek_v41_model_for_fsdp` 将普通参数转换为统一 dtype 的 meta 参数，
只搬运 buffer 并保留其值和 dtype，不把完整模型参数搬到设备。
随后 `FSDPTurbo` 根据 EP/FSDP mesh 分片参数并分配本地 Engram 权重，
未加载已有权重时，`initialize_deepseek_v41_model` 统一物化各 rank 的本地参数 shard 并随机初始化；
初始化逻辑在 `adapter.py` 中按职责分为三层：

- `_materialize_parameters` 只为尚未物化的本地参数 shard 分配存储，保留共享参数别名和所有 buffer。
- `_init_weights` 只初始化当前模块的直属参数，集中维护 DeepSeek 的 embedding、head、视觉投影、mHC 和按层深度缩放的专家初始化规则；
  专家 `gate_up_proj` 按全局行偏移仅缩放本地 shard 中的 up 部分，无需重新分片。
  视觉投影和归一化类型每个模块只判断一次；完整参数路径仅在 gate/shared-expert 后缀匹配时构造。
- 公共入口 `initialize_deepseek_v41_model` 先物化，再按父模块先于子模块的原有顺序调用 `_init_weights`，
  保持固定种子、固定并行拓扑下的随机采样顺序。Host Engram 仍由自身的 hook 分配和初始化。

初始化采用“先切分、后物化”的流程，不采用整模型 `to_empty()`，并保留 DeepSeek 专用的权重初始化规则。
`_materialize_parameters` 和 `_init_weights` 为内部实现，使用方仍调用原有公共入口。

`device` 仅用于物化 meta 参数；已物化的参数不重新分配，共享参数保持同一对象。初始化函数始终执行随机初始化。
加载已有权重时直接调用 `from_pretrained`。
CPU Offload 开启时，本地参数 shard 在 CPU 物化；buffer 仍在计算设备上。
示例的 `build_model` 已采用此流程：

```python
model = build_deepseek_v41_model(
    tokenizer=tokenizer,
)
model = prepare_deepseek_v41_model_for_fsdp(
    model, device=device, parameter_dtype=torch.bfloat16,
)
model = FSDPTurbo(config, model)
if config.model.load_pretrained_weights:
    model.model.from_pretrained(config.model.model_name_or_path)
else:
    initialize_deepseek_v41_model(
        model.model,
        device=torch.device("cpu") if config.distributed.fsdp_plan.cpu_offload else device,
    )
```

不要在切分前对 meta 表调用普通 `model.to()` 或整模型 `to_empty()`；请使用上面的
prepare 和 initialize 函数。优化器应在本地参数物化、初始化或加载后创建。
prepare 会丢弃已有参数值，不适用于已经加载权重的模型；RoPE、哈希常量和已物化的
Host Engram 表不会被 initialize 清空。模型构造入口不接收 mesh，
切分统一由 FSDPTurbo 管理。延后初始化按全局种子、层号和行起点派生随机种子，
在固定拓扑、相同初始种子下可复现；
不保证与先初始化完整表再切分的随机值逐元素一致。

加载已反量化为 BF16 的 HuggingFace safetensors 权重时，同样必须先应用并行：

```yaml
model:
  model_name_or_path: /path/to/DeepSeek-V4.1-Flash-BF16
  config_path: examples/deepseek_v41/model_config/config_full_bf16.json
  load_pretrained_weights: true
  pretrained_strict: true
  pretrained_engram_rows_per_chunk: 65536
```

`config_path` 是相对进程工作目录或绝对路径。加载完整官方权重必须使用
`config_full_bf16.json`；它保留发布模型的完整语言/视觉拓扑，同时
关闭当前训练不支持的 MTP/DSpark，并将参数类型设为 BF16。原始 `config.json`
保留发布模型的 FP8/FP4 推理设置，不能直接用于当前 BF16 训练入口。
`model_config/config_8p*.json` 和 `model_config/config_16p_A5.json` 会改变
compressor/indexer source 拓扑，
只用于随机初始化的裁剪模型。

```python
model = FSDPTurbo(config, model)
model.model.from_pretrained(
    config.model.model_name_or_path,
    strict=config.model.pretrained_strict,
    engram_rows_per_chunk=config.model.pretrained_engram_rows_per_chunk,
)
```

加载器通过 DCP 直接读取当前 rank 所需的 FSDP/TP/EP shard，并逐层合并 HF 的
`w1/w3/w2` 专家权重到本仓的 `gate_up_proj/down_proj`。Engram 表按 EP 行范围和
固定行块流式读入 Host 权重，不会在单个 rank 上构造完整表。权重目录需包含
`model.safetensors.index.json`（或单个/多个无索引 safetensors 文件）；当前入口
只接受已反量化权重，不在加载阶段处理 FP8 scale。残留的 `.scale` 以及训练模型
明确禁用的 `mtp.*` 权重会被忽略；`strict: true` 要求其余权重结构与当前仓内模型
JSON 完全一致。训练入口仅在 `model.load_pretrained_weights: true` 时加载 HF
初始权重；该开关不影响 DCP 断点恢复。当前 HF 初始加载要求
`distributed.expert_fully_shard_parallel_size: 1`；EP 和普通 FSDP 可正常切分。

训练适配器在 Block 内执行 Engram，确保投影参数已被 FSDP materialize。
图像和 padding 会截断 n-gram，并屏蔽对应位置的 Engram 注入。没有 tokenizer、
压缩词表不匹配或哈希桶超出表大小时构造会失败。
目前验证覆盖 CPU/Gloo 查询、稀疏梯度、优化器、梯度范数及小模型图文反向；
尚未验证 GPU/NPU 多机性能或发布尺寸。

## Engram Host offload

Engram 只暴露存储后端配置。默认使用不依赖额外 CANN 扩展的 PyTorch Host offload：

```yaml
model:
  engram_storage_backend: host_offload
```

使用 CANN `ElasticBuffer` 加速 Host offload 时改为：

```yaml
model:
  engram_storage_backend: elastic_buffer
```

训练、数据生成和 TND 打包应使用同一份模型结构 JSON。训练 YAML 设置 `model.config_path`，
数据生成脚本使用 `--config_path`，打包脚本使用 `--config-path`；未设置时分别从其
`model_name_or_path` / `--model_name_or_path` / `--model-name-or-path` 目录读取 `config.json`。

两个 Host 后端都会根据模型的 batch、序列长度和 N-gram 列数自动计算每 rank 的最大 token
容量，并固定使用 pinned host memory。物理行数向上对齐到 2048，以便沿 EP 等分；补齐行
不参与 hash。表的权威 BF16 分片和两个 FP32 Adam moment 位于 CPU，FSDP 仅管理 gate 参数；
ActiveRowAdam 与普通优化器使用相同的学习率和调度参数，其更新公式与 SparseAdam 一致。
稀疏梯度在归并、累积、裁剪和优化器更新期间保持 FP32。优化器仅通过
`index_select`/`index_copy_` 读写本步活跃行，最后将更新后的权重行转回 BF16，避免原生
SparseAdam 在 BF16 权重与 FP32 moments 组合下触发随整表大小增长的混合精度稀疏加。
两个 moment 仍是与 EP-local 完整表分片同形状的稠密张量；当前单表配置在 8 个
EP rank 同机时，BF16 权重与 FP32 moments 合计约 915.55 GiB。

两个 Host 后端均要求 `EP > 1`。`host_offload` 使用 PyTorch `all_to_all_single` 与显式
CPU/NPU 搬运，不依赖 `cann_ops_transformer`，当前仅面向 eager 正确性路径；
`elastic_buffer` 要求安装提供 `cann_ops_transformer.ElasticBuffer` 训练接口的 CANN 算子包。
算子包不支持直接注册 Host storage 时，eager 会在每次 forward 重新提交；
`torch.compile` 会显式拒绝该路径，避免产生表未更新的静默错误。Host 表不参与 TP，
可以与当前仅切分 shared expert 的 TP plan 共存；当前仍不支持对 Engram 模块本身应用 TP、
PP 或跨 EP degree checkpoint 转换；小算子 KV-AllGather CP 已支持，其他 CP 组合仍不支持。

token 压缩、特殊 token 处理和 hash multiplier 已与发布参考实现保持同一 ABI。压缩词表
不是 99,092 时会直接失败，不再通过告警后静默改写配置。
