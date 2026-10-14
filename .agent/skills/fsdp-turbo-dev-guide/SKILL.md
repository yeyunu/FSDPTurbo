---
name: "fsdp-turbo-dev-guide"
description: "FSDPTurbo 仓库开发守则：device 抽象、ops 隔离、config 校验、日志、命名、models 叶子隔离等 8 条规则。在编写或修改 fsdp_turbo 代码时必须遵循。"
---

# FSDPTurbo 开发守则

本守则是 FSDPTurbo 仓库的强制开发规范。当 Agent 在 `fsdp_turbo/` 目录下新增、修改、审查代码，或回答该仓库相关实现问题时，**必须**遵循以下 8 条规则，并在生成代码后逐条自检。

仓库顶层模块结构（用于理解规则的作用范围）：

```
fsdp_turbo/
├── distributed/      # 分布式并行（FSDP/TP/CP/EP）
├── memory/           # 显存优化（recompute/chunk_batch/swap）
├── models/           # 各模型适配（叶子模块：包内其他模块禁止引用，模型目录之间相互独立）
├── modules/          # 通用 module 组件
├── ops/              # 算子抽象层（cpu/npu/triton + registry）
├── optimizer/        # 优化器
├── quantization/     # 量化
├── training/         # 训练入口（唯一可依赖其他模块的模块）
├── utils/            # 工具（log/device/dtype 等）
├── fsdp_turbo.py
└── fsdp_turbo_config.py
```

---

## 规则 1：Device 调用统一走 `torch.accelerator`

禁止直接调用 `torch.cuda.*` 或 `torch.npu.*`。所有与设备相关的调用都必须通过 `torch.accelerator` 访问，以屏蔽具体后端差异。

`torch.accelerator` 由 [device.py](fsdp_turbo/utils/device.py) 中的 `set_accelerator_compatible()` 在启动时注入：它会检测 NPU/CUDA，并把 `torch.npu` 或 `torch.cuda` 的属性代理到 `torch.accelerator` 上。

正确：
```python
device_type = torch.accelerator.current_accelerator().type
x = x.to(torch.accelerator.current_accelerator())
```

错误：
```python
x = x.to("npu")            # 禁止：硬编码设备
x = x.cuda()               # 禁止：直接调用 torch.cuda
device = torch.npu.current_device()  # 禁止：直接调用 torch.npu
```

---

## 规则 2：`torch_npu` 只允许在 `ops/npu/` 内导入

除 `fsdp_turbo/ops/npu/` 目录之外，任何模块都不得 `import torch_npu`。`torch_npu` 是 NPU 专用依赖，集中隔离在 `ops/npu/` 内可保证仓库在其他设备上可导入、可运行（CPU 兜底）。

注意：[device.py](fsdp_turbo/utils/device.py) 中 `import torch_npu` 是该规则的**唯一例外**，因为它负责在启动期建立 `torch.accelerator` 代理；除此之外一律禁止。

正确：在 `ops/npu/grouped_matmul.py` 内：
```python
import torch_npu  # 允许：位于 ops/npu/ 目录内
```

错误：在 `distributed/`、`memory/`、`training/` 等任意非 `ops/npu/` 文件内：
```python
import torch_npu                       # 禁止
from torch_npu.contrib import transfer_to_npu  # 禁止
```

---

## 规则 3：融合算子必须在 `ops/` 抽象并提供 CPU 兜底

所有自定义融合算子的调用（包括 `torch_npu` 函数和第三方融合算子库）都必须在 `fsdp_turbo/ops/` 下抽象封装，并通过 [registry.py](fsdp_turbo/ops/registry.py) 的 `@register_op` 注册。每个算子必须：

1. 在 `ops/cpu/` 提供等价的 CPU 实现，作为其他设备的兜底方案；
2. 在 `ops/npu/`（或 `ops/cuda/`）提供加速实现；
3. 调用方通过 `ops/` 层的**语义化包装函数**调用（如 `ops/moe.py` 的 `grouped_matmul`、`ops/quantize.py` 的 `dynamic_mx_quant`，包装函数内部走 `get_op` / `dispatch_op` 按设备分发），**禁止**在业务代码中直接散落 `dispatch_op('op_name', ...)` 调用，也**禁止**直接调用底层库函数。包装函数按领域组织：MoE 相关放 `ops/moe.py`，量化相关放 `ops/quantize.py`，新增量化算子包装一律写入 `ops/quantize.py`。

`get_op` 在目标设备实现缺失时会自动回落到 CPU 实现并告警一次（见 [registry.py](fsdp_turbo/ops/registry.py) 的 fallback 逻辑）。

注册示例：
```python
# ops/cpu/rms_norm.py
from fsdp_turbo.ops.registry import register_op

@register_op('rms_norm', 'cpu')
def rms_norm_cpu(x, weight, eps):
    ...  # 纯 torch 的 CPU 实现，作为兜底

# ops/npu/rms_norm.py
from fsdp_turbo.ops.registry import register_op
import torch_npu  # 允许：位于 ops/npu/

@register_op('rms_norm', 'npu')
def rms_norm_npu(x, weight, eps):
    ...  # NPU 融合算子实现
```

调用示例（语义化包装，见 `ops/moe.py` / `ops/quantize.py`）：
```python
from fsdp_turbo.ops.quantize import dynamic_mx_quant
from fsdp_turbo.ops.moe import grouped_matmul

data_fp8, scale_bytes = dynamic_mx_quant(x, axis=-1, dst_type=torch.float8_e4m3fn)
out = grouped_matmul(x, split_list, weights)
```

参考已有实现：[ops/__init__.py](fsdp_turbo/ops/__init__.py) 会按需导入 `ops.npu`，并在导入失败时静默跳过。

---

## 规则 4：除 `training/` 外模块须高内聚、可独立移植

除 `fsdp_turbo/training/` 之外的所有模块必须符合内聚设计：每个模块只负责自身功能，**不得**依赖 `fsdp_turbo` 内其他业务模块（`ops/`、`utils/` 除外）。目标是任意一个文件夹可以单独复制到其他训练框架中独立使用。

依赖关系约束：

- `ops/`、`utils/` 是公共底层依赖，任何模块可依赖；
- `distributed/`、`memory/`、`modules/`、`optimizer/`、`quantization/` 之间**不得**相互依赖；
- `training/` 是唯一允许编排（依赖）其他模块的顶层入口，但**不得**依赖 `models/`（见规则 8）；
- `models/` 是叶子模块，任何模块都不得依赖它（见规则 8）。

正确：`memory/recompute/` 只 `import` 标准库、`torch`、`fsdp_turbo.ops`、`fsdp_turbo.utils`。

错误：
```python
# 在 memory/chunk_batch/chunk_batch.py 中
from fsdp_turbo.distributed import parallel_state  # 禁止：跨业务模块依赖
from fsdp_turbo.training.trainer import Trainer     # 禁止：反向依赖 training
```

新增模块时，先自问：把该文件夹拷出去还能跑吗？若不能，说明依赖过界。

---

## 规则 5：Config 校验统一在 `FSDPTurboConfig.__post_init__` 完成

所有配置的校验、规范化、派生字段计算都必须在 `FSDPTurboConfig.__post_init__`（及其调用的 `validate_*` 方法）中完成。`FSDPTurboConfig` 对象创建完成后即为最终配置，程序运行期间**不得**再修改配置字段。

参考 [fsdp_turbo_config.py](fsdp_turbo/fsdp_turbo_config.py) 的现有实现：`__post_init__` 依次调用 `validate_optimizer_config` / `validate_tp_config` / `validate_cp_config` / `validate_ep_config` / `validate_recompute_config` / `validate_chunk_batch_config` / `validate_quantization_config` / `validate_fsdp_config`。

新增配置字段时：

1. 在对应子 dataclass 中声明字段；
2. 在 `FSDPTurboConfig` 中新增 `validate_xxx_config` 方法，完成默认值填充、类型检查、合法性校验、派生字段计算；
3. 在 `__post_init__` 中调用该方法；
4. 运行时只读取，不再写入。

错误：在 `trainer.py` 或算子内部对 `config.xxx` 做二次赋值或延迟校验。

---

## 规则 6：日志统一使用 `utils.log.print_rank`

禁止直接使用内置 `print`。所有日志输出必须通过 [log.py](fsdp_turbo/utils/log.py) 的 `print_rank` 函数，确保分布式下只在指定 rank 输出，避免多卡重复刷屏。

`print_rank(log, message, ranks=0)` 签名说明：

- `log`：一个 callable，如 `logger.info` / `logger.warning`；
- `message`：日志内容；
- `ranks`：只在哪些 rank 输出，默认 `0`（仅 rank 0）。

正确：
```python
import logging
from fsdp_turbo.utils.log import print_rank, log_warning_once

logger = logging.getLogger(__name__)
print_rank(logger.info, f"step={step} loss={loss.item()}", ranks=0)
print_rank(logger.warning, "fallback to cpu", ranks=[0, 1])  # 多 rank
log_warning_once(logger, "只告警一次的提示")  # 仅告警一次
```

错误：
```python
print(f"loss={loss}")          # 禁止
logger.info("...")             # 禁止：未走 print_rank，多卡会重复
```

---

## 规则 7：命名直观准确，禁止模糊缩写与无意义数字

所有变量、函数、模块、配置项的命名必须直观、准确、自解释，避免：

- 非常见缩写（如 `mbs` 单独使用、`t`、`x2`、`tmp`）；
- 数字后缀（如 `tensor1`、`func2`、`layer_3`）；
- 与用途无关的泛化命名（如 `data`、`handler`、`processor` 用在具体语义场景）。

仓库中已有约定俗成的缩写**允许**保留，例如：

- `mbs` = micro batch size（`micro_batch_size` 的缩写，出现在 `TrainRunConfig`）；
- `tp`/`cp`/`ep`/`fsdp` = tensor parallel / context parallel / expert parallel / fully sharded data parallel（仓库核心术语，见 [fsdp_turbo_config.py](fsdp_turbo/fsdp_turbo_config.py)）；
- `gmm` = grouped matmul（算子名，见 `ops/cpu/grouped_matmul.py`）。

命名建议：

- 布尔字段用 `is_/has_/enable_` 前缀，如 `enable_fsdp_low_precision_all_gather`；
- 数量字段用 `_size`/`_count`/`_num` 后缀，如 `tensor_parallel_size`；
- 配置 plan 类用 `XxxPlanConfig`，如 `FSDPPlanConfig`、`ChunkBatchPlanConfig`；
- 校验方法用 `validate_xxx_config`。

正确：
```python
gradient_accumulation_steps = global_batch_size // micro_batch_size
def validate_chunk_batch_config(self): ...
```

错误：
```python
ga = gb // mbs                 # 模糊缩写
def check2(self): ...          # 无意义数字
tmp1 = compute()               # 无意义命名
```

---

## 规则 8：`fsdp_turbo.models` 是叶子模块，且模型目录之间相互独立

`fsdp_turbo/models/` 是模型适配代码的**叶子节点**：它只能被外部使用方（`examples/`、`tests/`、配置中的 `replacement` 等）引用，仓库内部不允许出现反向依赖。

**8.1 包内禁止引用 `fsdp_turbo.models`**

`fsdp_turbo/` 包内除 `models/` 自身以外的任何模块——`training/`、`distributed/`、`memory/`、`modules/`、`ops/`、`optimizer/`、`quantization/`、`utils/`、`fsdp_turbo.py`、`fsdp_turbo_config.py`——都**不得**导入 `fsdp_turbo.models`，包括绝对导入、相对导入和延迟到函数内的局部导入。需要模型参与的能力（如 CP 前向替换、算子替换）必须通过配置/注册机制或由使用方（`examples/`、`tests/`）注入，而不是在包内直接 import 模型。

`examples/`、`tests/`、`docs/` 作为使用方**允许**引用 `fsdp_turbo.models`。

**8.2 各模型目录相互独立**

`fsdp_turbo/models/` 下的每个模型目录（当前为 `deepseek_v41/`、`kimi/`、`qwen/`）必须彼此独立：

- 禁止任何形式的跨模型导入：`from fsdp_turbo.models.qwen...`、`from ..qwen...`、`from .. import ...` 均禁止；
- 禁止依赖 `models/` 根目录下的共享代码（共享能力应下沉到 `ops/`、`utils/` 等公共底层）；
- 模型目录内部统一使用相对导入（`from .module import Xxx`），并由目录自身的 `__init__.py` 对外暴露公共接口；
- 新增或删除一个模型目录，不得影响其他模型目录的导入与运行。

**8.3 边界**

本规则只约束 `models/` 与包内其他模块、以及模型目录之间的关系；模型目录自身对其他公共模块（`ops/`、`utils/` 等）的依赖不在本规则范围内。

正确：在 `examples/qwen3_5/train.py`（使用方）中：
```python
from fsdp_turbo.models.qwen.qwen3_5 import qwen3_5_model_forward  # 允许：examples 是使用方
```

正确：在 `models/kimi/modeling_kimi_k3.py` 内：
```python
from .configuration_kimi_k3 import KimiK3Config   # 允许：模型目录内部相对导入
```

错误：在 `training/`、`distributed/`、`memory/` 等包内非 `models/` 文件中：
```python
from fsdp_turbo.models.kimi import register_kimi_k3   # 禁止：包内反向依赖 models
from fsdp_turbo.models.qwen.qwen3_5 import xxx        # 禁止
```

错误：在模型目录之间交叉引用：
```python
# 在 models/kimi/modeling_kimi_k3.py 中
from fsdp_turbo.models.deepseek_v41 import DeepseekV41ForCausalLMAdapter  # 禁止：跨模型依赖
from ..deepseek_v41.model import ModelArgs                                # 禁止
```

---

## 自检清单

提交或输出代码前，逐条确认：

1. [ ] 没有任何 `torch.cuda.*` / `torch.npu.*` 直接调用，统一 `torch.accelerator`；
2. [ ] `torch_npu` 仅出现在 `ops/npu/`（及 `utils/device.py` 的启动注入）；
3. [ ] 新算子已 `@register_op` 注册，且有 `ops/cpu/` 兜底实现，调用方走 `ops/` 语义化包装（无散落 `dispatch_op` 直调）；
4. [ ] 非 `training/` 模块未跨业务模块依赖（仅可依赖 `ops/`、`utils/`、标准库、`torch`）；
5. [ ] 新配置字段已在 `FSDPTurboConfig.__post_init__` 校验，运行期只读；
6. [ ] 无 `print`，日志均走 `print_rank` / `log_warning_once`；
7. [ ] 命名直观自解释，无非常见缩写、无数字后缀；
8. [ ] `fsdp_turbo/` 包内未引用 `fsdp_turbo.models`（仅 `examples/`、`tests/`、`docs/` 可引用），且 `models/` 下各模型目录之间无交叉导入、无 `models/` 根共享代码依赖。
