# 特性开发指南

本文档介绍如何开发一个新的FSDPTurbo特性，包含架构说明、开发教程与开发实践指南。

## 概述

FSDPTurbo采用**配置驱动**的特性管理架构：特性以YAML配置为入口，在模型构建阶段对`nn.Module`树执行一次性变换（切分 / 包装 / 转换 / 替换），训练循环本身不感知特性。核心组件：

| 组件 | 位置 | 作用 |
|---|---|---|
| `FSDPTurboConfig` | `fsdp_turbo/fsdp_turbo_config.py` | 特性配置定义、YAML解析与校验 |
| `FSDPTurbo` | `fsdp_turbo/fsdp_turbo.py` | 主入口，按固定顺序对模型应用各特性 |
| `apply_module_patches` / `patch_namespace_members` | `fsdp_turbo/utils/patch.py` | 函数/方法级替换，幂等安装 |
| `module_name_match` | `fsdp_turbo/utils/str_match.py` | 模块名通配符匹配 |
| `register_op` / `dispatch_op` | `fsdp_turbo/ops/registry.py` | 多后端（CPU/NPU）算子注册与分发 |

开发者只需新增一个配置项 + 一个模块变换函数，并在`FSDPTurbo`中接入，即可添加新特性。

### 特性应用顺序

`FSDPTurbo(config, model)`构造时按以下顺序对模型应用变换：

| 顺序 | 阶段 | 说明 |
|---|---|---|
| 1 | `init_parallel_state` | 初始化进程组与设备网格 |
| 2 | 量化转换 | MXFP8 / W8A16等权重格式转换 |
| 3 | 张量并行（TP） | 列/行并行切分Linear |
| 4 | 上下文并行（CP） | Ulysses / KV-AllGather注意力变换与loss patch |
| 5 | `module_patches` | 用户配置的函数替换 |
| 6 | 专家并行（EP / EFSDP） | MoE专家分片与dispatcher |
| 7 | 激活重计算 | checkpoint包装forward |
| 8 | 微批次切分 | chunk_batch包装forward |
| 9 | FSDP2分片 | `fully_shard`包装，必须最后执行 |

> [!NOTE]
> 新增特性时需确认插入位置：所有依赖原始模块结构的变换必须在FSDP2分片之前；`module_patches`在FSDP之前应用，目标不匹配会直接报错。

## 开发流程

以ChunkBatch（微批次切分）特性为例。

### 定义配置

在`fsdp_turbo/fsdp_turbo_config.py`中新增特性开关与计划配置：

```python
# fsdp_turbo/fsdp_turbo_config.py
@dataclass
class ChunkBatchPlanConfig:
    chunk_mbs: int = 1                 # 每个 micro-batch 的样本数
    apply_modules: List[str] = None    # 模块名模式，如 "model.layers.{*}"
    batch_dim: int = 0                 # batch 所在维度
    chunk_arg_indexes: Optional[List[int]] = None   # 需要切分的位置参数下标
    chunk_kwarg_names: Optional[List[str]] = None  # 需要切分的关键字参数名


@dataclass
class MemoryConfig:
    recompute: bool = False
    recompute_plan: List[str] = None
    chunk_batch: bool = False                                  # 特性开关
    chunk_batch_plan: Optional[ChunkBatchPlanConfig] = None    # 特性计划
```

说明如下：

- 开关（`chunk_batch`）与计划（`chunk_batch_plan`）分离，未开启时零开销。
- 配置按领域分组挂载（`distributed` / `memory` / `quantization`）。
- 模块名统一使用通配符：`{*}`匹配单层下标（如`model.layers.{*}`），`{0-3}`匹配数字范围，`*`匹配任意前缀（如`*.q_proj`）。

### 增加配置校验

实现`validate_xxx_config`校验方法，并在`__post_init__`中注册调用：

```python
# fsdp_turbo/fsdp_turbo_config.py
def validate_chunk_batch_config(self):
    if not self.memory.chunk_batch:
        return
    self.memory.chunk_batch_plan = (
        ChunkBatchPlanConfig() if self.memory.chunk_batch_plan is None else self.memory.chunk_batch_plan
    )
    if self.memory.chunk_batch_plan.chunk_mbs <= 0:
        raise ValueError("chunk_mbs must be positive.")
    if not self.memory.chunk_batch_plan.apply_modules:
        raise ValueError("chunk_batch_plan.apply_modules must not be empty when chunk_batch is enabled.")
    if not self.memory.chunk_batch_plan.chunk_arg_indexes and not self.memory.chunk_batch_plan.chunk_kwarg_names:
        raise ValueError(
            "chunk_batch_plan must specify chunk_arg_indexes or chunk_kwarg_names when chunk_batch is enabled."
        )


# 在 __post_init__ 中注册
def __post_init__(self):
    ...
    self.validate_chunk_batch_config()
```

说明如下：

- 特性未开启时快速返回，不影响其他配置。
- 校验错误信息包含完整字段路径（如`chunk_batch_plan.apply_modules`），便于定位YAML问题。
- 若新增顶层分组，还需在`load_config_from_yaml`中接入该分组的解析。

### 实现模块变换函数

在特性目录下实现模块变换函数，匹配目标模块并应用变换：

```python
# fsdp_turbo/memory/chunk_batch/chunk_batch.py
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match


def chunk_batch_modules(model, plan: ChunkBatchPlanConfig):
    chunk_mbs_modules = get_chunkmbs_modules(model, plan.apply_modules)
    apply_chunkmbs_module(chunk_mbs_modules, plan)
    return model


def get_chunkmbs_modules(modules, plan):
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[ChunkMBS] No module named {plan}.')
    return matched_modules
```

说明如下：

- 变换函数只包装模块的`forward`，**不替换模块对象本身**，以保持参数、buffer与模块限定名不变，避免破坏权重加载与FSDP遍历。
- 模块匹配不到时抛出`RuntimeError`，避免配置静默失效。
- 使用`print_rank(logger.info, ...)`打印应用日志，仅主进程输出。

### 在FSDPTurbo中接入

在`FSDPTurbo`中新增接入方法，并在`__init__`中按应用顺序调用：

```python
# fsdp_turbo/fsdp_turbo.py
from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules


class FSDPTurbo(torch.nn.Module):
    def __init__(self, config: FSDPTurboConfig, model: torch.nn.Module):
        ...
        self.apply_recompute_modules()
        self.apply_chunk_batch_modules()   # 插入到 FSDP 分片之前
        self.apply_fsdp_modules()

    def apply_chunk_batch_modules(self):
        if not self.config.memory.chunk_batch:
            return
        self.model = chunk_batch_modules(self.model, self.config.memory.chunk_batch_plan)
```

### 使用特性

在YAML配置中开启该特性，配置示例如下：

```yaml
# examples/xxx/config.yaml
memory:
  chunk_batch: true
  chunk_batch_plan:
    apply_modules: ["model.layers.{*}"]
    chunk_mbs: 1
    batch_dim: 0
    chunk_kwarg_names: ["hidden_states"]
```

## 替换第三方函数（module_patches）

当特性需要替换HuggingFace / PyTorch的函数（如注意力实现、loss计算、模型forward）时，可通过`module_patches`配置完成，无需修改特性代码：

```yaml
module_patches:
  - target: transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward
    replacement: fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5.qwen3_5_model_forward
```

说明如下：

- `target`支持两种形式：`module.member`（模块级函数）与`module.Class.member`（类方法/属性，对该类所有实例生效）。
- `replacement`必须是dotted path（不支持`module:name`冒号语法）。
- `module_patches`在FSDP2分片之前应用；target不匹配任何模块/命名空间成员时抛出`RuntimeError`，防止拼写错误静默失效。
- 特性代码内部需要patch时，使用`fsdp_turbo.utils.patch.patch_namespace_members`：安装是幂等的（同一replacement不会重复安装），且已加载的`from module import member`别名会被同步刷新。

## 多后端算子开发

涉及设备相关算子的特性（如MoE grouped matmul），使用算子注册表实现CPU/NPU多后端：

```python
# fsdp_turbo/ops/npu/grouped_matmul.py
from fsdp_turbo.ops.registry import register_op

@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights):
    ...  # NPU 加速实现
```

```python
# 调用侧与设备解耦
from fsdp_turbo.ops.registry import dispatch_op

output = dispatch_op('grouped_matmul', inputs, m_split, weights)
```

说明如下：

- 算子按设备组织在`fsdp_turbo/ops/{cpu,npu,triton}/`目录下，同名算子各目录一份实现。
- **CPU实现必须提供**：既是正确性参考实现（单测基线），也是无加速后端时的fallback——`get_op`在目标设备未注册时自动回退CPU实现并打印一次告警。
- 自定义autograd算子（如`fsdp_turbo/ops/npu/grouped_matmul_mc2.py`的`AllToAllGroupedMatmul`）应继承`torch.autograd.Function`，再通过注册表暴露统一的dispatch入口。

## 开发实践指南

### 开发实践建议

- 配置默认关闭：新特性开关默认`False` / `None`，未启用时不得影响基础功能。
- 校验完整性：在`__post_init__`的`validate_xxx_config`中校验模块名模式、取值范围与特性组合约束，错误信息包含字段路径。
- 快速失败：模块匹配、patch目标匹配失败时立即抛错，禁止静默跳过。
- 保持模块实例：变换只包装`forward`或修改属性，不替换模块对象，保证权重加载与FSDP遍历兼容。
- 应用顺序：所有依赖原始模块结构的变换必须在`apply_fsdp_modules`之前；FSDP2分片必须最后执行。
- 延迟导入：仅在特性分支内部import重量级依赖（如NPU扩展），避免未启用特性时引入副作用。

### 创建新特性的Checklist

开发完成后，可按下述清单逐项自查：

```text
特性创建 Checklist
├── 配置定义
│   ├── [ ] 新增 plan dataclass（开关与计划分离）
│   ├── [ ] 挂载到对应分组（distributed/memory/quantization）
│   └── [ ] load_config_from_yaml 接入解析（新增分组时）
├── 参数校验
│   ├── [ ] 实现 validate_xxx_config 并在 __post_init__ 注册
│   ├── [ ] 校验模块名模式与取值范围，错误信息含字段路径
│   └── [ ] 未启用时快速返回
├── 模块变换
│   ├── [ ] 实现 xxx_modules(model, plan)
│   ├── [ ] 匹配不到模块时抛 RuntimeError
│   └── [ ] 不替换模块对象，仅包装 forward/修改属性
├── 应用接入
│   ├── [ ] FSDPTurbo 新增 apply_xxx_modules，未启用直接返回
│   └── [ ] 确认插入位置在 FSDP2 分片之前
├── 设备算子（如涉及）
│   ├── [ ] 注册 CPU 参考实现
│   └── [ ] 注册 NPU 加速实现
└── 测试验证
    ├── [ ] 配置解析与校验单测
    ├── [ ] 模块匹配/变换单测（CPU 实现作为基线）
    └── [ ] 特性开/关对比：loss 一致性 + 显存/性能收益
```

### 常见问题

- **如何排查特性未生效？**

  1. 检查YAML key是否拼写正确——未知配置项会被忽略并打印`Ignoring unknown ... config key`告警。
  2. 查看日志中的`Applying ... to module: xxx`输出（仅主进程打印）。
  3. 确认特性开关字段为`true`，且plan中模块名模式与`model.named_modules()`输出匹配。

- **模块名匹配不到怎么办？**

  打印`model.named_modules()`查看真实模块路径；检查通配符语法：`{*}`匹配单层下标（`model.layers.{*}`），`{0-3}`匹配数字范围，`*`匹配任意前缀（`*.q_proj`）。

- **module_patches与特性内部patch如何选择？**

  | 场景 | 推荐方式 |
  |---|---|
  | 用户按模型替换第三方函数（注意力/loss/forward） | YAML `module_patches` |
  | 特性自身逻辑必须替换的函数 | 特性代码内`patch_namespace_members` |

- **patch是否幂等？**

  是。`fsdp_turbo.utils.patch`通过`_fsdp_turbo_patch_ids`记录已安装的替换，同一replacement重复安装会被跳过；模块级替换还会同步刷新已加载的`from module import member`别名，保证`from ... import`用法也能看到patch。

- **如何选择patch目标形式？**

  - `module.member`：模块级函数，如`transformers.loss.loss_utils.ForCausalLMLoss`。
  - `module.Class.member`：类方法或属性，对该类所有实例生效，如`transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward`。
