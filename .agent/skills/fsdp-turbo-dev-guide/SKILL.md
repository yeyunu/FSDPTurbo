---
name: "fsdp-turbo-dev-guide"
description: "FSDPTurbo 仓库开发守则：device 抽象、ops 隔离、config 校验、日志、命名、models 叶子隔离、modeling 禁止 patch、代码精简、工具函数复用、models 只放结构代码、models 两级目录、环境变量集中、文档同步、公共 API 注解、安全反序列化等 17 条规则。在编写或修改 fsdp_turbo 代码时必须遵循。"
---

# FSDPTurbo 开发守则

本守则是 FSDPTurbo 仓库的强制开发规范。当 Agent 在 `fsdp_turbo/` 目录下新增、修改、审查代码，或回答该仓库相关实现问题时，**必须**遵循以下 17 条规则，并在生成代码后逐条自检。

仓库顶层模块结构（用于理解规则的作用范围）：

```
fsdp_turbo/
├── distributed/      # 分布式并行（FSDP/TP/CP/EP）
├── memory/           # 显存优化（recompute/chunk_batch/swap）
├── models/           # 各模型适配（叶子模块：包内其他模块禁止引用；两级目录：系列/结构；只放结构代码）
├── modules/          # 通用 module 组件
├── ops/              # 算子抽象层（cpu/npu/triton + registry）
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

**存量待整改**（按改动代价分类）：

- 有 `torch.accelerator` 等价 API，可直接替换：
  - `memory/swap_optimizer/swap_optimizer.py:88-89`：`torch.npu.is_available()` / `torch.npu.memory_allocated()`；
  - `training/monitor.py:71,73`：`torch.npu.is_available()` / `torch.cuda.is_available()`；
  - `training/memory_snapshot.py:108`：`torch.cuda.is_available()`。
- 需先确认 `torch.accelerator` 是否有等价 API：`ops/npu/sparse_attention.py` 中的 `torch.npu.device` / `torch.npu.current_device()` / `torch.npu.current_stream()`（位于 `ops/npu/`，是否按规则 2 的口径豁免尚未确定，确认后再整改）。
- 仅为类型注解或注释，改为设备无关写法：`distributed/fine_grained_fully_shard/param_group.py:212`、`distributed/expert_parallel/domino_dispatcher.py:302-305`（`torch.cuda.Stream/Event` 注解）、`modules/kcp.py:38`（注释）。

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

**存量待整改**：

- `models/kimi/modeling_kimi_k3.py:82`：模块顶层 `if get_accelerator_name() == "npu": import torch_npu`；
- `training/profiler.py:40`：函数内 `import torch_npu`（`try` 包裹）；
- `training/memory_snapshot.py:103`：函数内 `import torch_npu`（`try` 包裹）。

三处都应下沉到 `ops/npu/` 的封装，或改用 `ops/` 提供的语义化接口后删除。

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
- `distributed/`、`memory/`、`modules/`、`quantization/` 之间**不得**相互依赖；
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

`fsdp_turbo/` 包内除 `models/` 自身以外的任何模块——`training/`、`distributed/`、`memory/`、`modules/`、`ops/`、`quantization/`、`utils/`、`fsdp_turbo.py`、`fsdp_turbo_config.py`——都**不得**导入 `fsdp_turbo.models`，包括绝对导入、相对导入和延迟到函数内的局部导入。需要模型参与的能力（如 CP 前向替换、算子替换）必须通过配置/注册机制或由使用方（`examples/`、`tests/`）注入，而不是在包内直接 import 模型。

`examples/`、`tests/`、`docs/` 作为使用方**允许**引用 `fsdp_turbo.models`。

**8.2 各模型目录相互独立**

`fsdp_turbo/models/` 下的每个模型结构目录（两级目录中的第二级，见规则 13）必须彼此独立：

- 禁止任何形式的跨模型导入：`from fsdp_turbo.models.qwen...`、`from ..qwen...`、`from .. import ...` 均禁止；
- 禁止依赖 `models/` 根目录下的共享代码（共享能力应下沉到 `ops/`、`utils/` 等公共底层）；
- 模型目录内部统一使用相对导入（`from .module import Xxx`），并由目录自身的 `__init__.py` 对外暴露公共接口；
- 新增或删除一个模型目录，不得影响其他模型目录的导入与运行。

**8.3 边界**

本规则只约束 `models/` 与包内其他模块、以及模型目录之间的关系；模型目录自身对其他公共模块（`ops/`、`utils/` 等）的依赖不在本规则范围内。

正确：在 `examples/qwen3_5/train.py`（使用方）中：
```python
from fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5 import qwen3_5_model_forward  # 允许：examples 是使用方
```

正确：在 `models/kimi/modeling_kimi_k3.py` 内：
```python
from .configuration_kimi_k3 import KimiK3Config   # 允许：模型目录内部相对导入
```

错误：在 `training/`、`distributed/`、`memory/` 等包内非 `models/` 文件中：
```python
from fsdp_turbo.models.kimi import register_kimi_k3   # 禁止：包内反向依赖 models
from fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5 import xxx        # 禁止
```

错误：在模型目录之间交叉引用：
```python
# 在 models/kimi/modeling_kimi_k3.py 中
from fsdp_turbo.models.deepseek_v41 import DeepseekV41ForCausalLMAdapter  # 禁止：跨模型依赖
from ..deepseek_v41.model import ModelArgs                                # 禁止
```

---

## 规则 9：`fsdp_turbo.models` 下的 modeling 禁止 patch，必须继承 transformers 类覆写

`fsdp_turbo/models/` 中的 modeling 适配代码（新增或修改）**不得**以任何运行期改写（patch）的方式改变模型行为，必须通过**继承 transformers 对应类并覆写**来表达。

**9.1 禁止的 patch 方式**

- 禁止 `*_patch.py` + `apply_*_patch()` 式「运行时替换类引用」：如 `modeling_kimi_linear.KimiSparseMoeBlock = PatchKimiSparseMoeBlock`；
- 禁止对类/实例做 `setattr` 挂载或替换成员来改变行为：如 `setattr(module, "forward", new_forward)`、改写 `Class.method`、改动 `ALL_ATTENTION_FUNCTIONS` 等库内注册表来偷换实现；
- 禁止在 `models/` 下提供「自由函数 forward」交由配置 `module_patches.replacement`（或 `utils/patch.py` 的 `patch_namespace_members` 等）在运行期替换 transformers 成员；
- 禁止用 `PatchXxx` 命名定义**非继承**的替代类去顶替 transformers 同名类（如 `class PatchKimiSparseMoeBlock(nn.Module)`）。

**9.2 正确做法：继承 + 覆写**

```python
# models/qwen/modeling_qwen3_5.py（示意）
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
)


class FSDPTurboQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    """只覆写差异部分，其余逻辑复用父类。"""

    def forward(self, hidden_states, *args, **kwargs):
        # 适配点：NPU 融合算子 / CP 序列切分
        return super().forward(hidden_states, *args, **kwargs)


class FSDPTurboQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 组件级替换：在子类中装配子类化的 layer / MoE block
        self.model.layers = nn.ModuleList(
            [FSDPTurboQwen3_5DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
```

要点：

1. 覆写优先 `super()` 复用父类逻辑，只改真正需要变化的代码路径；
2. 组件级替换（attention / MoE block / decoder layer）用「子类 + 在子类 `__init__` 中装配」，**不得**回头改写父类或原始模块的类对象；
3. 需要按配置开关时，通过构造参数或 config 字段**选择使用哪个子类**（如工厂函数返回不同子类），而不是运行时打补丁；
4. 仓库内 vendored 的参考实现（如 `models/kimi/modeling_kimi_linear.py` 中的类）同样适用：继承其参考类覆写，而不是替换类引用。

**9.3 边界与存量待整改**

- 本规则只约束 `fsdp_turbo/models/`。`distributed/context_parallel/patch.py`、`utils/patch.py` 提供的框架级 `module_patches` 能力本身不受限，但**不得**再被 `models/` 当作建模适配手段；
- 存量待整改（不要求本次立即重构，但改动到相关文件时应一并整改）：
  - `models/kimi/kimi_moe_patch.py`：`apply_kimi_k3_moe_patch()` 换类引用，以及 `PatchKimiSparseMoeBlock` / `PatchKimiMoeExperts` 非继承式替代类；
  - `models/kimi/modeling_kimi_k3.py`：对 `apply_kimi_k3_moe_patch()` 的调用；
  - `models/qwen/qwen3_5_moe/qwen3_5.py`、`models/qwen/qwen3_5_moe/qwen3_5_moe_cp.py`：自由函数 forward（`qwen3_5_model_forward` / `qwen3_5_moe_model_forward`）依赖配置 `replacement` 的运行期替换。

---

## 规则 10：代码简单干净，只做必要的检查

写代码的目标是**能用最少的代码把事做对**，不是把每个位置都包上防御逻辑。

- 边界校验只做一次：外部输入与配置在入口校验（配置统一收敛到 `FSDPTurboConfig.__post_init__`，见规则 5）；内部调用链**不得**对已校验过的同一条件重复校验；
- 只做必要的检查，不要为了「看起来稳健」堆叠 `assert` / `if not isinstance(...)` / `try-except`；异常处理只包住真正会抛且需要兜底的点，能直接调用就直接调用；
- 不实现当前用不到的能力、参数、开关、配置项（YAGNI），不为「以后可能要扩展」预留空壳；
- 不加与本次改动无关的代码：不做顺手重构、不留注释掉的死代码与调试残留；
- 优先复用标准库、`torch`、`transformers` 的既有能力，能一行表达就不写十行；
- 改动中发现可直接删除的冗余逻辑，一并清理（但不得越过规则 4、规则 8 的模块边界）。

错误：
```python
# 内部调用链重复校验已在入口校验过的条件（validate_chunk_batch_config 已校验）
def chunk_batch(tensor, chunk_size):
    if chunk_size <= 0:                               # 禁止：重复校验
        raise ValueError("chunk_size must be positive")
    if not isinstance(tensor, torch.Tensor):          # 禁止：调用方已保证类型
        raise TypeError("tensor must be a torch.Tensor")
```

---

## 规则 11：工具函数复用优先，其次才考虑抽象

写代码的目标是把事做对，不是把每段逻辑都抽成函数。动手写工具函数之前，按顺序确认：

1. **项目内是否已有**：先搜 [fsdp_turbo/utils/](fsdp_turbo/utils/)（`log.py`、`device.py`、`dtype.py`、`model.py`、`random.py`、`str_match.py`、`gmm_layout.py`、`patch.py` 等），再搜同模块或同领域的既有实现；
2. **标准库 / `torch` / `transformers` 是否已提供**；
3. 以上都不满足时，再判断复用范围：
   - 会被多个模块复用 → 抽到 `fsdp_turbo/utils/`；领域专用的放对应领域的公共文件（如算子相关的放 `ops/`）；
   - 只服务一处 → 就近定义，**不要**为「以后可能用到」提前抽象；
   - 新增公共函数需保证：命名与签名一眼可懂（见规则 7）、只依赖标准库/`torch`/公共底层、可被独立复用。

**禁止**：重复实现已有工具（如自己解析环境变量推断设备，而不用 `get_accelerator_name()`）；把只在一处使用的一次性逻辑硬抽成公共函数（过度抽象）。

正确：先用已有工具，不重复造轮子。
```python
from fsdp_turbo.utils.device import get_accelerator_name
from fsdp_turbo.utils.log import print_rank

print_rank(logger.info, f"running on {get_accelerator_name()}")
```

错误：
```python
# 1) 重复实现既有工具：utils/device.py 已有 get_accelerator_name()
def detect_device():
    return os.environ.get("DEVICE_TYPE", "npu")        # 禁止

# 2) 过度抽象：全仓只有一处调用的两行逻辑，硬抽成公共函数
def default_expert_scale(num_experts: int) -> float:
    return 1.0 / max(num_experts, 1)                   # 禁止：就近写在调用处即可
```

---

## 规则 12：`models/` 只放模型结构代码，参数配置 JSON 放 `examples/`

`fsdp_turbo/models/` 只允许存放**模型结构代码**（modeling、adapter、processor、结构级组件等 `.py` 实现）。模型参数配置文件（`config.json` 及其多机多卡变体，如 `config_8p_A3.json`、`config_16p_A5.json`）属于**示例配置**，必须放在 `examples/<示例目录>/` 下，不得放进 `models/`。

理由：`models/` 需要能被整体拷走复用（见规则 4、规则 8）；参数配置随硬件形态与并行度变化，属于使用侧数据，混入 `models/` 会让模型结构库带上环境相关参数，目录也会不断膨胀。

正确：
```
examples/deepseek_v41/config.json
examples/deepseek_v41/config_8p_A3.json
fsdp_turbo/models/deepseek/deepseek_v41/          # 只有 .py
```

错误：
```
fsdp_turbo/models/deepseek_v41/config.json        # 禁止：配置混入模型结构目录
fsdp_turbo/models/deepseek_v41/config_8p_A5.json  # 禁止
```

存量待整改：`fsdp_turbo/models/deepseek_v41/` 下现有 `config.json`、`config_8p_A3.json`、`config_8p_A5.json`、`config_16p_A5.json`，需迁移到 `examples/deepseek_v41/`，并同步修改读取它们的代码（如 `adapter.py` 的 `_load_model_args()` 从 `_REFERENCE_DIRECTORY / "config_8p_A5.json"` 读取）。不要求本次立即整改，但改动到该模型时应一并整改。

---

## 规则 13：`models/` 采用两级目录——模型系列 / 模型结构

`fsdp_turbo/models/` 下固定为两级目录：第一级是**模型系列**，第二级是该系列下的**模型结构**。

```
fsdp_turbo/models/
├── <模型系列>/            # 第一级：系列，如 deepseek、qwen、kimi
│   ├── __init__.py        # 系列级导出（可选，仅做转发）
│   └── <模型结构>/        # 第二级：具体结构，如 deepseek_v41、qwen3_5、kimi_k3
│       ├── __init__.py    # 该结构的对外接口
│       └── *.py
```

约束：

- 一级目录只按**模型系列**划分，二级目录才是具体**模型结构**；导入路径固定为 `fsdp_turbo.models.<系列>.<结构>`；
- 不得把结构文件直接放在系列目录下（`models/qwen/qwen3_5.py` 应改为 `models/qwen/qwen3_5/`）；
- 不得省略系列层（如 `models/qwen3_5/`），也不得出现三级及更深的结构划分；
- 规则 8 要求的相互独立、`__init__.py` 对外接口、目录内相对导入，均以**二级结构目录**为最小单元。

正确（目录形态）：
```
fsdp_turbo/models/qwen/qwen3_5/__init__.py
fsdp_turbo/models/qwen/qwen3_5/modeling_qwen3_5.py
fsdp_turbo/models/kimi/kimi_k3/modeling_kimi_k3.py
```

错误（目录形态）：
```
fsdp_turbo/models/qwen3_5/...                        # 禁止：缺少系列层
fsdp_turbo/models/qwen/qwen3_5.py                    # 禁止：结构文件直接放在系列目录下
fsdp_turbo/models/qwen/qwen3_5/modeling/layers/...   # 禁止：超过两级
```

存量待整改：现有 `models/deepseek_v41/`、`models/kimi/`、`models/qwen/` 均只有一级，需重组为两级，例如：

- `models/qwen/qwen3_5.py` → `models/qwen/qwen3_5/`
- `models/qwen/qwen3_5_moe.py` → `models/qwen/qwen3_5_moe/`
- `models/qwen/modeling_qwen4_exp.py` → `models/qwen/qwen4_exp/`
- `models/deepseek_v41/` → `models/deepseek/deepseek_v41/`（系列名按家族命名）
- `models/kimi/` 下的 `modeling_kimi_k3.py`、`modeling_kimi_linear.py` 按实际结构归属拆入对应二级目录

具体系列命名与拆分粒度在整改时确定；存量不要求本次立即整改，但**新增模型必须直接按两级结构创建**。

---

## 规则 14：环境变量集中定义、集中读取并登记文档

- 环境变量的**名字、默认值、解析逻辑**必须集中定义，统一收敛到 `fsdp_turbo/utils/env.py`（新建）；业务代码只调用该模块的读取函数或常量，**不得**再出现 `os.environ.get("XXX")` 字面量；
- 新增环境变量必须同步在 [environment_variable.md](docs/zh/features/environment_variable.md) 登记：名称、取值、默认值、生效时机、影响范围；
- 删除或重命名环境变量需保留旧名并告警一段时间，不得静默失效。

理由：当前全仓 27 处 `os.environ` 分散在 7 个文件（`training/trainer.py` 10 处、`training/memory_snapshot.py` 4 处、`models/qwen/modeling_qwen4_exp.py` 4 处等），同一变量（如 `LOCAL_RANK`）在多处重复解析，默认值容易不一致。

正确：
```python
from fsdp_turbo.utils.env import get_local_rank

local_rank = get_local_rank()
```

错误：
```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))   # 禁止：业务代码散落字面量
```

**存量待整改**：现有 27 处读取（`LOCAL_RANK`、`LOCAL_WORLD_SIZE`、`RANK`、`WORLD_SIZE`、`FLA_USE_FAST_OPS`、`FLA_NO_USE_TMA`、`CPU_AFFINITY_CONF`、`ASCEND_DEVICE_ID`）逐步迁移到 `utils/env.py`；其中 `FLA_USE_FAST_OPS`、`FLA_NO_USE_TMA` 目前尚未在环境变量文档中登记。

---

## 规则 15：文档必须随代码同步更新

改动代码时必须同步对应文档，不允许「代码先行、文档后补」：

| 改动类型 | 必须更新 |
| --- | --- |
| 新增/变更特性 | `docs/zh/features/<feature>.md`（无则新建） |
| 新增/变更配置项（`FSDPTurboConfig` 字段） | `docs/zh/training/user_guide.md` 及对应 feature 文档 |
| 新增/变更环境变量 | [environment_variable.md](docs/zh/features/environment_variable.md)（见规则 14） |
| 新增/变更模型支持 | `docs/zh/models/supported_models.md`、`docs/zh/develop/model_adaptation.md` |

- 新增文档必须在 [docs/zh/_menu.md](docs/zh/_menu.md) 登记，否则视为未交付；
- 文档中的路径、模块名、函数名、配置键必须能在仓库中检索到；文档与代码不一致时优先修正文档，不得为迁就过时文档而改动已发布的接口。

**存量待整改**：`docs/zh/_menu.md` 目前未收录 13 个文档——`appendixes.md`、`docs_guide.md`、`models/supported_models.md`、`training/user_guide.md`、`training/finetune/finetune.md`、`training/pretrain/pretrain.md`、`training/quick_start_fsdpturbo.md`、`training/geo3k_finetune_guide.md`、`project_guide.md`、`SECURITYNOTE.md`、`tools/profiling.md`、`develop/model_adaptation.md`、`release_notes_fsdpturbo.md`，需补齐或确认有意不收录。

---

## 规则 16：公共 API 必须有类型注解与 docstring

「公共 API」指跨模块调用或对外暴露的函数、类与公开方法；模块内部的一次性辅助函数不受此约束（避免与规则 10 的「减少代码量」冲突）。

- 参数与返回值必须标注类型，不得用 `Any` 兜底；无法精确表达时用 `Union` / `Protocol` / 泛型；
- docstring 至少说明用途；涉及参数语义、返回值、异常、副作用或并行/分布式约束时必须写清楚；
- 仅供类型检查使用的导入放在 `if TYPE_CHECKING:` 下，避免运行期开销；
- 类型注解不得引入额外的运行期依赖或副作用。

正确：
```python
from typing import Sequence

def all_to_all(tokens: torch.Tensor, splits: Sequence[int]) -> torch.Tensor:
    """Dispatch tokens to their expert owners.

    Args:
        tokens: Token features, shape ``[num_tokens, hidden]``.
        splits: Token count per rank, length equals world size.

    Returns:
        Tokens received from all ranks, concatenated.
    """
```

错误：
```python
def all_to_all(tokens, splits):   # 禁止：公共 API 缺类型注解与 docstring
    ...
```

**存量待整改**：全仓 1203 个 `def` 中仅 208 个带返回注解（约 17%）；新增/修改代码按本规则执行，存量随改动逐步补齐。

---

## 规则 17：禁止不安全反序列化与命令执行

- 禁止用 `torch.load(..., weights_only=False)` 加载来源不可信的文件；确需加载自产 checkpoint 时，必须在文档或 docstring 中写明信任边界，并保证文件路径来源受控；
- 禁止 `eval` / `exec`、`os.system`、`os.popen`、`subprocess(..., shell=True)`；
- 禁止用 `pickle.load`、未指定 `SafeLoader` 的 `yaml.load` 处理外部数据；
- pre-commit 已挂 bandit、gitleaks，但工具只是兜底，不得以「工具没报」代替安全判断。

错误：
```python
state = torch.load(user_supplied_path, weights_only=False)   # 禁止：任意 pickle 反序列化
os.system(f"cp {src} {dst}")                                  # 禁止：命令注入风险
```

**存量待整改**：`training/checkpoint.py:396` 使用 `weights_only=False` 加载 checkpoint（自产文件），整改时评估切换 `weights_only=True` 或改为显式白名单类型。

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
8. [ ] `fsdp_turbo/` 包内未引用 `fsdp_turbo.models`（仅 `examples/`、`tests/`、`docs/` 可引用），且 `models/` 下各模型目录之间无交叉导入、无 `models/` 根共享代码依赖；
9. [ ] `models/` 下的 modeling 未使用 patch（无 `*_patch.py`/`apply_*_patch()`、无 `setattr` 换成员、无靠配置 `replacement` 替换的自由函数 forward、无 `PatchXxx` 非继承替代类），适配一律继承 transformers 类覆写；
10. [ ] 代码量已最小化且只做必要检查（未重复校验已在入口校验的条件、无 YAGNI 参数/空壳、无死代码、无与改动无关的代码）；
11. [ ] 工具函数优先复用 `fsdp_turbo/utils/` 及标准库/`torch`/`transformers` 既有实现，未重复造轮子；新增抽象仅限会被多处复用的逻辑，未过度抽象；
12. [ ] `fsdp_turbo/models/` 下未新增 `config.json` 等参数配置文件，配置均放在 `examples/<示例目录>/`；
13. [ ] 新增模型按两级目录创建：`models/<模型系列>/<模型结构>/`，未出现只有一级、结构文件直放系列目录或超过两级的情况；
14. [ ] 未新增散落的 `os.environ.get("XXX")` 字面量，环境变量已集中到 `utils/env.py` 并在环境变量文档中登记；
15. [ ] 代码改动已同步对应文档（特性/配置项/环境变量/模型支持），新增文档已在 `docs/zh/_menu.md` 登记；
16. [ ] 新增/修改的公共 API 有完整类型注解与 docstring，内部辅助函数未被迫补注解；
17. [ ] 未引入不安全反序列化（`weights_only=False` 加载不可信文件、`pickle.load`、非 `SafeLoader` 的 `yaml.load`）与命令执行（`eval`/`exec`/`os.system`/`shell=True`）。
