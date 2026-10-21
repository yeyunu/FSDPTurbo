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

训练 head 的计算精度由 YAML 中的 `model.head_compute_dtype` 控制，默认 `fp32`。
如需启用 BF16 head + FP32 CE，可在 `config.yaml` 或 `config_tnd.yaml` 中设置：

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
该入口不读取官方发布的 FP8/FP4 权重，模型参数仍以 BF16 随机初始化。

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
| V4.1 attention tensor/context parallel | 暂不支持 |
| CSA2 跨层压缩状态与稀疏索引结构 | 支持构造与结构验证 |
| CANN Compressor/CompressorGrad 融合压缩算子 | 可选；设置 `model.use_fused_compressor: true` |
| Lightning Indexer 融合算子 | 可选；支持 Atlas A3（Ascend 910_93）、Ascend 910B/910C 和 Ascend950DT |
| MhcPreSinkhornPremix/MhcPost 融合算子 | 可选；设置 `model.use_fused_mhc: true` |
| Clipped SwiGLU 前反向融合 | 可选；设置 `model.use_fused_clipped_swiglu: true`，模型的 `swiglu_limit` 必须大于 0 |
| Inplace-RoPE Triton算子 | 可选；设置 `model.use_triton_inplace_rope: true` |
| 视觉 FlashAttentionScore/Grad 融合算子 | 可选；设置 `model.use_fused_vision_flash_attn: true`；仅支持 BF16 |
| CSA2 activation recompute | 暂不支持 |
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

Clipped SwiGLU 融合开关开启后，`swiglu_limit > 0` 的共享专家和原生路由专家
调用 CANN `ClippedSwiglu`，反向调用配套的 `ClippedSwigluGrad`，融合裁剪与激活计算。
该路径要求 `cann_ops_nn` 同时提供前反向接口及配套 CANN 内核；使用 EP 时，
dispatcher 须为 `custom_native_eager_forward`。激活保持 FP32 计算，路由权重仍在
down projection 前乘入，不改变参数布局。默认保持关闭，关闭时继续使用原裁剪与
SwiGLU 实现；`swiglu_limit <= 0` 时保持普通 SwiGLU。当前融合路径仅支持 eager
autograd 一阶梯度，不包含 TorchAir 图模式或高阶梯度。

默认 8 卡拓扑为 `FSDP=4, TP=2, EP=4, expert DP=2`。注意力包含 attention
sink 与 indexed sparse-attention 的特殊头布局，因此当前保持复制，不能直接套用通用
Colwise/Rowwise 规则；`cp_plan` 也保持为空。

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
其他参数和运行时 buffer 仍正常创建。
`prepare_deepseek_v41_model_for_fsdp` 转换 dtype、搬运其他张量并保留 meta 表。
随后 `FSDPTurbo` 根据 EP/FSDP mesh 分片参数并分配本地 Engram 权重，最后
`initialize_deepseek_v41_model` 只初始化各 rank 的本地参数 shard。
示例的 `build_model` 已采用此流程：

```python
model = build_deepseek_v41_model(
    tokenizer=tokenizer,
)
model = prepare_deepseek_v41_model_for_fsdp(
    model, device=device, parameter_dtype=torch.bfloat16,
)
model = FSDPTurbo(config, model)
initialize_deepseek_v41_model(model.model)
```

不要在切分前对 meta 表调用普通 `model.to()` 或整模型 `to_empty()`；请使用上面的
prepare 函数。优化器应在 `FSDPTurbo` 构造后创建。模型构造入口不接收 mesh，
切分统一由 FSDPTurbo 管理。延后初始化按全局种子、层号和行起点派生随机种子，
在固定拓扑、相同初始种子下可复现；
不保证与先初始化完整表再切分的随机值逐元素一致。

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
PP、CP 或跨 EP degree checkpoint 转换。

token 压缩、特殊 token 处理和 hash multiplier 已与发布参考实现保持同一 ABI。压缩词表
不是 99,092 时会直接失败，不再通过告警后静默改写配置。
