---
name: "fsdp-turbo-dev-guide"
description: "FSDPTurbo 仓库开发守则：按设备与后端、模块设计、模型适配、配置日志与环境、代码风格、接口与文档、资料守则、协作与提交、安全共 9 类 23 条规则。在编写或修改 fsdp_turbo 代码、或编写资料文档时必须遵循。"
---

# FSDPTurbo 开发守则

本守则是 FSDPTurbo 仓库的强制开发规范。当 Agent 在 `fsdp_turbo/` 目录下新增、修改、审查代码，或在 `docs/` 等资料文档中编写内容时，**必须**遵循以下 9 类共 23 条规则，并在产出后逐条自检。

## 规则分类索引

| 类别 | 规则 | 关注点 |
| --- | --- | --- |
| 一、设备与后端 | 1–3 | 设备抽象、`torch_npu` 隔离、算子注册与调用 |
| 二、模块设计 | 4–6 | 结构设计、内聚与接口、解耦与可移植 |
| 三、模型适配 | 7–9 | `models/` 的依赖边界、实现方式、目录规范 |
| 四、配置、日志与环境 | 10–12 | 配置校验、日志、环境变量 |
| 五、代码风格 | 13–16 | 命名、精简、复用、注释 |
| 六、接口与文档 | 17–18 | 公共 API 契约、文档同步 |
| 七、资料守则 | 19–21 | 文档排版、设备术语、标题规范 |
| 八、协作与提交 | 22 | PR 前的 AI 检视 |
| 九、安全 | 23 | 反序列化与命令执行 |

仓库顶层模块结构（用于理解规则的作用范围）：

```text
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

## 一、设备与后端

### 规则 1：Device 调用统一走 `torch.accelerator`

禁止直接调用 `torch.cuda.*` 或 `torch.npu.*`。所有与设备相关的调用都必须通过 `torch.accelerator` 访问，以屏蔽具体后端差异。

`torch.accelerator` 由 `fsdp_turbo/utils/device.py` 中的 `set_accelerator_compatible()` 在启动时注入：它会检测 NPU/CUDA，并把 `torch.npu` 或 `torch.cuda` 的属性代理到 `torch.accelerator` 上。

正确：

```python
device_type = torch.accelerator.current_accelerator().type
x = x.to(torch.accelerator.current_accelerator())
```

错误：

```python
x = x.to("npu")            # forbidden: hardcoded device
x = x.cuda()               # forbidden: direct torch.cuda call
device = torch.npu.current_device()  # forbidden: direct torch.npu call
```

**存量待整改**（按改动代价分类）：

- 有 `torch.accelerator` 等价 API，可直接替换：
  - `memory/swap_optimizer/swap_optimizer.py:88-89`：`torch.npu.is_available()` / `torch.npu.memory_allocated()`；
  - `training/monitor.py:71,73`：`torch.npu.is_available()` / `torch.cuda.is_available()`；
  - `training/memory_snapshot.py:108`：`torch.cuda.is_available()`。
- 需先确认 `torch.accelerator` 是否有等价 API：`ops/npu/sparse_attention.py` 中的 `torch.npu.device` / `torch.npu.current_device()` / `torch.npu.current_stream()`（位于 `ops/npu/`，是否按规则 2 的口径豁免尚未确定，确认后再整改）。
- 仅为类型注解或注释，改为设备无关写法：`distributed/fine_grained_fully_shard/param_group.py:212`、`distributed/expert_parallel/domino_dispatcher.py:302-305`（`torch.cuda.Stream/Event` 注解）、`modules/kcp.py:38`（注释）。

### 规则 2：`torch_npu` 只允许在 `ops/npu/` 内导入

除 `fsdp_turbo/ops/npu/` 目录之外，任何模块都不得 `import torch_npu`。`torch_npu` 是 NPU 专用依赖，集中隔离在 `ops/npu/` 内可保证仓库在其他设备上可导入、可运行（CPU 兜底）。

注意：`fsdp_turbo/utils/device.py` 中 `import torch_npu` 是该规则的**唯一例外**，因为它负责在启动期建立 `torch.accelerator` 代理；除此之外一律禁止。

正确：在 `ops/npu/grouped_matmul.py` 内：

```python
import torch_npu  # allowed: inside ops/npu/
```

错误：在 `distributed/`、`memory/`、`training/` 等任意非 `ops/npu/` 文件内：

```python
import torch_npu                       # forbidden
from torch_npu.contrib import transfer_to_npu  # forbidden
```

**存量待整改**：

- `models/kimi/modeling_kimi_k3.py:82`：模块顶层 `if get_accelerator_name() == "npu": import torch_npu`；
- `training/profiler.py:40`：函数内 `import torch_npu`（`try` 包裹）；
- `training/memory_snapshot.py:103`：函数内 `import torch_npu`（`try` 包裹）。

三处都应下沉到 `ops/npu/` 的封装，或改用 `ops/` 提供的语义化接口后删除。

### 规则 3：融合算子必须在 `ops/` 抽象，调用只走语义化包装

所有自定义融合算子的调用（包括 `torch_npu` 函数和第三方融合算子库）都必须在 `fsdp_turbo/ops/` 下抽象封装，并通过 `fsdp_turbo/ops/registry.py` 的 `@register_op` 注册。每个算子必须：

1. 在 `ops/cpu/` 提供等价的 CPU 实现，作为其他设备的兜底方案；
2. 在 `ops/npu/`（或 `ops/cuda/`）提供加速实现；
3. 调用方通过 `ops/` 层的**语义化包装函数**调用（如 `ops/moe.py` 的 `grouped_matmul`、`ops/quantize.py` 的 `dynamic_mx_quant`，包装函数内部走 `get_op` / `dispatch_op` 按设备分发），**禁止**在业务代码中直接散落 `dispatch_op('op_name', ...)` 调用，也**禁止**直接调用底层库函数。包装函数按领域组织：MoE 相关放 `ops/moe.py`，量化相关放 `ops/quantize.py`，新增量化算子包装一律写入 `ops/quantize.py`。

`get_op` 在目标设备实现缺失时会自动回落到 CPU 实现并告警一次（见 `fsdp_turbo/ops/registry.py` 的 fallback 逻辑）。

注册示例：

```python
# ops/cpu/rms_norm.py
from fsdp_turbo.ops.registry import register_op

@register_op('rms_norm', 'cpu')
def rms_norm_cpu(x, weight, eps):
    ...  # pure torch CPU implementation, used as fallback

# ops/npu/rms_norm.py
from fsdp_turbo.ops.registry import register_op
import torch_npu  # allowed: inside ops/npu/

@register_op('rms_norm', 'npu')
def rms_norm_npu(x, weight, eps):
    ...  # NPU fused kernel implementation
```

调用示例（语义化包装，见 `ops/moe.py` / `ops/quantize.py`）：

```python
from fsdp_turbo.ops.quantize import dynamic_mx_quant
from fsdp_turbo.ops.moe import grouped_matmul

data_fp8, scale_bytes = dynamic_mx_quant(x, axis=-1, dst_type=torch.float8_e4m3fn)
out = grouped_matmul(x, split_list, weights)
```

参考已有实现：`fsdp_turbo/ops/__init__.py` 会按需导入 `ops.npu`，并在导入失败时静默跳过。

---

## 二、模块设计

### 规则 4：新增加速模块须先完成结构设计

新增加速特性（新的并行策略、显存优化、量化方案、训练组件等）时**先设计后编码**，不得写完再补结构。设计阶段必须回答：

1. **架构契合**：模块落在哪一层（`ops/` 算子层、`distributed/`、`memory/`、`modules/`、`quantization/`、引擎层），与现有分层和职责划分是否一致；现有架构放不下时先讨论调整方案，不得硬塞或绕过分层；
2. **特性设计合理**：数据流、生命周期、失败与回退路径（算子缺失、设备不支持、配置不合法）先想清楚，避免实现中反复推翻；
3. **职责与边界**：每个类/函数只承担单一职责，输入输出边界明确；不把多个阶段（切分、通信、计算、日志）塞进同一个类；
4. **扩展点预留**：可预见的变体（新设备后端、新调度策略、新量化格式、新并行维度）必须预留**接缝**——注册点、策略基类/`Protocol`、工厂函数、配置开关；
5. **预留接口 ≠ 空实现**：预留指可插入新实现的抽象点，不是提前写空的 `if` 分支、未使用的参数或占位类（与规则 14 的 YAGNI 一致）。

正确：以注册/协议留出扩展点，新增策略只需加实现。

```python
class ExpertDispatcher(Protocol):
    def dispatch(self, tokens: torch.Tensor, splits: list[int]) -> torch.Tensor: ...
    def combine(self, outputs: torch.Tensor) -> torch.Tensor: ...

DISPATCHERS: dict[str, type[ExpertDispatcher]] = {}

def register_dispatcher(name: str, cls: type[ExpertDispatcher]) -> None:
    DISPATCHERS[name] = cls
```

错误：把多个阶段塞进一个类，新增变体必须改这个类。

```python
class EverythingManager:
    def shard(self): ...
    def all_reduce(self): ...
    def compute_loss(self): ...
    def log(self): ...
```

### 规则 5：加速模块高内聚，对外接口精简且可移植

- 模块对外暴露的接口控制在**几个的数量级**（一个入口类/函数 + 少量公共函数与常量）；内部类、辅助函数、中间结构一律不导出；
- 判定标准：把该模块文件夹复制到第三方训练框架，配合 `ops/`、`utils/` 公共底层，**少量修改**（改导入路径、接一条配置）即可使能对应加速特性；
- 接口数量膨胀、或调用方必须自己拼装内部组件才能用起来，说明内聚不足，应重新划分模块边界；
- 模块的 `__init__.py` 只做导出，不写实现、不产生副作用。

正确：

```python
# memory/recompute/__init__.py
from .recompute import apply_recompute, RecomputePlanConfig

__all__ = ["apply_recompute", "RecomputePlanConfig"]  # small, stable surface
```

错误：

```python
# memory/recompute/__init__.py
from .recompute import (  # forbidden: leaks 10+ internals
    RecomputePlanConfig, _build_hook, _LayerScanner, _StateDictGuard,
    _skip_condition, _wrap_forward, ...
)
```

### 规则 6：模块之间充分解耦，单文件夹移出不报错

- 移出单个模块文件夹后，不得出现「因为依赖其他业务模块而导入失败或功能报错」；模块只能依赖公共底层；
- 依赖关系约束：

  - `ops/`、`utils/` 是公共底层依赖，任何模块可依赖；
  - `distributed/`、`memory/`、`modules/`、`quantization/` 之间**不得**相互依赖；
  - `training/` 是唯一允许编排（依赖）其他模块的顶层入口，但**不得**依赖 `models/`（见规则 7）；
  - `models/` 是叶子模块，任何模块都不得依赖它（见规则 7）。

- 新增模块时先自问：把该文件夹拷出去还能跑吗？若不能，说明依赖过界。

正确：`memory/recompute/` 只 `import` 标准库、`torch`、`fsdp_turbo.ops`、`fsdp_turbo.utils`。

错误：

```python
# in memory/chunk_batch/chunk_batch.py
from fsdp_turbo.distributed import parallel_state  # forbidden: cross-module dependency
from fsdp_turbo.training.trainer import Trainer     # forbidden: reverse dependency on training
```

---

## 三、模型适配

### 规则 7：`fsdp_turbo.models` 是叶子模块，模型结构目录之间相互独立

`fsdp_turbo/models/` 是模型适配代码的**叶子节点**：它只能被外部使用方（`examples/`、`tests/`、配置中的 `replacement` 等）引用，仓库内部不允许出现反向依赖。

#### 7.1 包内禁止引用 `fsdp_turbo.models`

`fsdp_turbo/` 包内除 `models/` 自身以外的任何模块——`training/`、`distributed/`、`memory/`、`modules/`、`ops/`、`quantization/`、`utils/`、`fsdp_turbo.py`、`fsdp_turbo_config.py`——都**不得**导入 `fsdp_turbo.models`，包括绝对导入、相对导入和延迟到函数内的局部导入。需要模型参与的能力（如 CP 前向替换、算子替换）必须通过配置/注册机制或由使用方（`examples/`、`tests/`）注入，而不是在包内直接 import 模型。

`examples/`、`tests/`、`docs/` 作为使用方**允许**引用 `fsdp_turbo.models`。

#### 7.2 各模型结构目录相互独立

`fsdp_turbo/models/` 下的每个模型结构目录（两级目录中的第二级，见规则 9.2）必须彼此独立：

- 禁止任何形式的跨模型导入：`from fsdp_turbo.models.qwen...`、`from ..qwen...`、`from .. import ...` 均禁止；
- 禁止依赖 `models/` 根目录下的共享代码（共享能力应下沉到 `ops/`、`utils/` 等公共底层）；`models/__init__.py` 的包级模型注册表（`_BUILTIN_MODELS` / `get_model_by_name`）属于包入口，模型结构目录同样不得依赖它；
- 模型目录内部统一使用相对导入（`from .module import Xxx`），并由目录自身的 `__init__.py` 对外暴露公共接口；
- 新增或删除一个模型目录，不得影响其他模型目录的导入与运行。

#### 7.3 边界

本规则只约束 `models/` 与包内其他模块、以及模型目录之间的关系；模型目录自身对其他公共模块（`ops/`、`utils/` 等）的依赖不在本规则范围内。

正确：在 `examples/qwen3_5/train.py`（使用方）中：

```python
from fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5 import qwen3_5_model_forward  # allowed: examples is a consumer
```

正确：在 `models/kimi/modeling_kimi_k3.py` 内：

```python
from .configuration_kimi_k3 import KimiK3Config   # allowed: relative import within the model directory
```

错误：在 `training/`、`distributed/`、`memory/` 等包内非 `models/` 文件中：

```python
from fsdp_turbo.models.kimi import register_kimi_k3   # forbidden: reverse dependency from the package
from fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5 import xxx        # forbidden
```

错误：在模型目录之间交叉引用：

```python
# in models/kimi/modeling_kimi_k3.py
from fsdp_turbo.models.deepseek.deepseek_v41 import DeepseekV41ForCausalLMAdapter  # forbidden: cross-model dependency
from ...deepseek.deepseek_v41.model import ModelArgs                               # forbidden
```

### 规则 8：`models/` 下的 modeling 禁止 patch，必须继承 transformers 类覆写

`fsdp_turbo/models/` 中的 modeling 适配代码（新增或修改）**不得**以任何运行期改写（patch）的方式改变模型行为，必须通过**继承 transformers 对应类并覆写**来表达。

#### 8.1 禁止的 patch 方式

- 禁止 `*_patch.py` + `apply_*_patch()` 式「运行时替换类引用」：如 `modeling_kimi_linear.KimiSparseMoeBlock = PatchKimiSparseMoeBlock`；
- 禁止对类/实例做 `setattr` 挂载或替换成员来改变行为：如 `setattr(module, "forward", new_forward)`、改写 `Class.method`、改动 `ALL_ATTENTION_FUNCTIONS` 等库内注册表来偷换实现；
- 禁止在 `models/` 下提供「自由函数 forward」交由配置 `module_patches.replacement`（或 `utils/patch.py` 的 `patch_namespace_members` 等）在运行期替换 transformers 成员；
- 禁止用 `PatchXxx` 命名定义**非继承**的替代类去顶替 transformers 同名类（如 `class PatchKimiSparseMoeBlock(nn.Module)`）。

#### 8.2 正确做法：继承 + 覆写

```python
# illustrative path: models/qwen/qwen3_5/modeling_qwen3_5.py
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
)

class FSDPTurboQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    """Override only the divergent parts; reuse the rest via super()."""

    def forward(self, hidden_states, *args, **kwargs):
        # Adapt: NPU fused ops and CP sequence sharding.
        return super().forward(hidden_states, *args, **kwargs)

class FSDPTurboQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Assemble subclassed layers instead of patching the parent class.
        self.model.layers = nn.ModuleList(
            [FSDPTurboQwen3_5DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
```

要点：

1. 覆写优先 `super()` 复用父类逻辑，只改真正需要变化的代码路径；
2. 组件级替换（attention / MoE block / decoder layer）用「子类 + 在子类 `__init__` 中装配」，**不得**回头改写父类或原始模块的类对象；
3. 需要按配置开关时，通过构造参数或 config 字段**选择使用哪个子类**（如工厂函数返回不同子类），而不是运行时打补丁；
4. 仓库内 vendored 的参考实现（如 `models/kimi/modeling_kimi_linear.py` 中的类）同样适用：继承其参考类覆写，而不是替换类引用。

#### 8.3 边界与存量待整改

- 本规则只约束 `fsdp_turbo/models/`。`distributed/context_parallel/patch.py`、`utils/patch.py` 提供的框架级 `module_patches` 能力本身不受限，但**不得**再被 `models/` 当作建模适配手段；
- 存量待整改（不要求本次立即重构，但改动到相关文件时应一并整改）：
  - `models/kimi/kimi_moe_patch.py`：`apply_kimi_k3_moe_patch()` 换类引用，以及 `PatchKimiSparseMoeBlock` / `PatchKimiMoeExperts` 非继承式替代类；
  - `models/kimi/modeling_kimi_k3.py`：对 `apply_kimi_k3_moe_patch()` 的调用；
  - `models/qwen/qwen3_5_moe/qwen3_5.py`、`models/qwen/qwen3_5_moe/qwen3_5_moe_cp.py`：自由函数 forward（`qwen3_5_model_forward` / `qwen3_5_moe_model_forward`）依赖配置 `replacement` 的运行期替换。

### 规则 9：`models/` 目录规范——只放结构代码，两级分层

`fsdp_turbo/models/` 既是模型结构代码的存放地，也是可整体拷走复用的独立单元，因此对**放什么**和**怎么分层**各有一条硬约束。

#### 9.1 只放模型结构代码

`fsdp_turbo/models/` 只允许存放**模型结构代码**（modeling、adapter、processor、结构级组件等 `.py` 实现）。模型参数配置文件（`config.json` 及其多机多卡变体，如 `config_8p_A3.json`、`config_16p_A5.json`）属于**示例配置**，必须放在 `examples/<示例目录>/` 下，不得放进 `models/`。

理由：`models/` 需要能被整体拷走复用（见规则 5、规则 7）；参数配置随硬件形态与并行度变化，属于使用侧数据，混入 `models/` 会让模型结构库带上环境相关参数，目录也会不断膨胀。

正确：

```text
examples/deepseek_v41/config.json
examples/deepseek_v41/config_8p_A3.json
fsdp_turbo/models/deepseek/deepseek_v41/          # 只有 .py
```

错误：

```text
fsdp_turbo/models/deepseek/deepseek_v41/config.json        # 禁止：配置混入模型结构目录
fsdp_turbo/models/deepseek/deepseek_v41/config_8p_A5.json  # 禁止
```

现状：`models/` 下已无 `config.json` 等参数配置（原 `models/deepseek_v41/config*.json` 已迁出），无需整改；新增配置一律放 `examples/<示例目录>/`。

#### 9.2 两级目录：模型系列 / 模型结构

`fsdp_turbo/models/` 下固定为两级目录：第一级是**模型系列**，第二级是该系列下的**模型结构**。

```text
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
- 规则 7 要求的相互独立、`__init__.py` 对外接口、目录内相对导入，均以**二级结构目录**为最小单元。

正确（目录形态）：

```text
fsdp_turbo/models/qwen/qwen3_5/__init__.py
fsdp_turbo/models/qwen/qwen3_5/modeling_qwen3_5.py
fsdp_turbo/models/deepseek/deepseek_v41/adapter.py
```

错误（目录形态）：

```text
fsdp_turbo/models/qwen3_5/...                        # 禁止：缺少系列层
fsdp_turbo/models/qwen/qwen3_5.py                    # 禁止：结构文件直接放在系列目录下
fsdp_turbo/models/qwen/qwen3_5/modeling/layers/...   # 禁止：超过两级
```

存量待整改：`models/deepseek/deepseek_v41/`、`models/qwen/qwen3_5/`、`models/qwen/qwen3_5_moe/` 已完成两级化；仍有以下一级形态需整改：

- `models/qwen/modeling_qwen4_exp.py` → `models/qwen/qwen4_exp/`
- `models/kimi/` 下的 `modeling_kimi_k3.py`、`modeling_kimi_linear.py`、`kimi_k3_cp.py` 等按实际结构归属拆入 `models/kimi/<结构>/`
- `models/glm5_next/` → 补一层系列目录后再放入结构目录

具体系列命名与拆分粒度在整改时确定；存量不要求本次立即整改，但**新增模型必须直接按两级结构创建**。

---

## 四、配置、日志与环境

### 规则 10：Config 校验统一在 `FSDPTurboConfig.__post_init__`，且只校验不改模型

所有配置的校验、规范化、派生字段计算都必须在 `FSDPTurboConfig.__post_init__`（及其调用的 `validate_*` 方法）中完成。`FSDPTurboConfig` 对象创建完成后即为最终配置，程序运行期间**不得**再修改配置字段。

**校验的职责边界**：`__post_init__` 只判断**配置本身是否合法**（取值范围、整除关系、维度匹配、互斥项、依赖项是否同时开启）并计算派生字段；**不得**在校验期间修改模型——不替换模块、不改 dtype/device、不注册 hook、不做量化转换、不修改 `torch` 全局状态。模型改动统一由 `FSDPTurbo` 的 `apply_*` 流水线完成。

理由：校验应当是纯函数式的、可单独测试的；把模型修改塞进校验会让「构造配置」产生难以追踪的副作用，也使配置无法脱离模型独立验证。

参考 `fsdp_turbo/fsdp_turbo_config.py` 的现有实现：`__post_init__` 依次调用 `validate_optimizer_config` / `validate_tp_config` / `validate_cp_config` / `validate_ep_config` / `validate_recompute_config` / `validate_chunk_batch_config` / `validate_quantization_config` / `validate_fsdp_config`。

新增配置字段时：

1. 在对应子 dataclass 中声明字段；
2. 在 `FSDPTurboConfig` 中新增 `validate_xxx_config` 方法，完成默认值填充、类型检查、合法性校验、派生字段计算；
3. 在 `__post_init__` 中调用该方法；
4. 运行时只读取，不再写入。

正确：

```python
def validate_fsdp_config(self):
    plan = self.distributed.fsdp_plan
    if plan.param_dtype not in SUPPORTED_DTYPES:      # legality only
        raise ValueError(f"unsupported param_dtype: {plan.param_dtype}")
```

错误：

```python
def __post_init__(self):
    self.validate_fsdp_config()
    torch.set_default_dtype(self.model.torch_dtype)   # forbidden: global side effect in validation
    convert_model_to_mxfp8(self._model)               # forbidden: model mutation in validation
```

### 规则 11：日志统一使用 `utils.log.print_rank`

禁止直接使用内置 `print`。所有日志输出必须通过 `fsdp_turbo/utils/log.py` 的 `print_rank` 函数，确保分布式下只在指定 rank 输出，避免多卡重复刷屏。

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
print_rank(logger.warning, "fallback to cpu", ranks=[0, 1])  # multi-rank
log_warning_once(logger, "warn only once")
```

错误：

```python
print(f"loss={loss}")          # forbidden
logger.info("...")             # forbidden: bypasses print_rank, duplicated on every rank
```

### 规则 12：环境变量集中定义、集中读取并登记文档

- 环境变量的**名字、默认值、解析逻辑**必须集中定义，统一收敛到 `fsdp_turbo/utils/env.py`（新建）；业务代码只调用该模块的读取函数或常量，**不得**再出现 `os.environ.get("XXX")` 字面量；
- 新增环境变量必须同步在 `docs/zh/features/environment_variable.md` 登记：名称、取值、默认值、生效时机、影响范围；
- 删除或重命名环境变量需保留旧名并告警一段时间，不得静默失效。

理由：当前全仓 27 处 `os.environ` 分散在 7 个文件（`training/trainer.py` 10 处、`training/memory_snapshot.py` 4 处、`models/qwen/modeling_qwen4_exp.py` 4 处等），同一变量（如 `LOCAL_RANK`）在多处重复解析，默认值容易不一致。

正确：

```python
from fsdp_turbo.utils.env import get_local_rank

local_rank = get_local_rank()
```

错误：

```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))   # forbidden: hardcoded env read in business code
```

**存量待整改**：现有 27 处读取（`LOCAL_RANK`、`LOCAL_WORLD_SIZE`、`RANK`、`WORLD_SIZE`、`FLA_USE_FAST_OPS`、`FLA_NO_USE_TMA`、`CPU_AFFINITY_CONF`、`ASCEND_DEVICE_ID`）逐步迁移到 `utils/env.py`；其中 `FLA_USE_FAST_OPS`、`FLA_NO_USE_TMA` 目前尚未在环境变量文档中登记。

---

## 五、代码风格

### 规则 13：命名直观准确，禁止模糊缩写与无意义数字

所有变量、函数、模块、配置项的命名必须直观、准确、自解释，避免：

- 非常见缩写（如 `mbs` 单独使用、`t`、`x2`、`tmp`）；
- 数字后缀（如 `tensor1`、`func2`、`layer_3`）；
- 与用途无关的泛化命名（如 `data`、`handler`、`processor` 用在具体语义场景）。

仓库中已有约定俗成的缩写**允许**保留，例如：

- `mbs` = micro batch size（`micro_batch_size` 的缩写，出现在 `TrainRunConfig`）；
- `tp`/`cp`/`ep`/`fsdp` = tensor parallel / context parallel / expert parallel / fully sharded data parallel（仓库核心术语，见 `fsdp_turbo/fsdp_turbo_config.py`）；
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
ga = gb // mbs                 # vague abbreviation
def check2(self): ...          # meaningless number
tmp1 = compute()               # meaningless name
```

### 规则 14：代码简单干净，只做必要的检查

写代码的目标是**能用最少的代码把事做对**，不是把每个位置都包上防御逻辑。

- 边界校验只做一次：外部输入与配置在入口校验（配置统一收敛到 `FSDPTurboConfig.__post_init__`，见规则 10）；内部调用链**不得**对已校验过的同一条件重复校验；
- 只做必要的检查，不要为了「看起来稳健」堆叠 `assert` / `if not isinstance(...)` / `try-except`；异常处理只包住真正会抛且需要兜底的点，能直接调用就直接调用；
- 不实现当前用不到的能力、参数、开关、配置项（YAGNI），不为「以后可能要扩展」预留空壳（预留抽象接缝的做法见规则 4）；
- 不加与本次改动无关的代码：不做顺手重构、不留注释掉的死代码与调试残留；
- 优先复用标准库、`torch`、`transformers` 的既有能力，能一行表达就不写十行；
- 改动中发现可直接删除的冗余逻辑，一并清理（但不得越过规则 6、规则 7 的模块边界）。

错误：

```python
# validate_chunk_batch_config already checks this condition.
def chunk_batch(tensor, chunk_size):
    if chunk_size <= 0:                               # forbidden: duplicate validation
        raise ValueError("chunk_size must be positive")
    if not isinstance(tensor, torch.Tensor):          # forbidden: caller already guarantees the type
        raise TypeError("tensor must be a torch.Tensor")
```

### 规则 15：工具函数复用优先，其次才考虑抽象

写代码的目标是把事做对，不是把每段逻辑都抽成函数。动手写工具函数之前，按顺序确认：

1. **项目内是否已有**：先搜 `fsdp_turbo/utils/`（`log.py`、`device.py`、`dtype.py`、`model.py`、`random.py`、`str_match.py`、`gmm_layout.py`、`patch.py` 等），再搜同模块或同领域的既有实现；
2. **标准库 / `torch` / `transformers` 是否已提供**；
3. 以上都不满足时，再判断复用范围：
   - 会被多个模块复用 → 抽到 `fsdp_turbo/utils/`；领域专用的放对应领域的公共文件（如算子相关的放 `ops/`）；
   - 只服务一处 → 就近定义，**不要**为「以后可能用到」提前抽象；
   - 新增公共函数需保证：命名与签名一眼可懂（见规则 13）、只依赖标准库/`torch`/公共底层、可被独立复用。

**禁止**：重复实现已有工具（如自己解析环境变量推断设备，而不用 `get_accelerator_name()`）；把只在一处使用的一次性逻辑硬抽成公共函数（过度抽象）。

正确：先用已有工具，不重复造轮子。

```python
from fsdp_turbo.utils.device import get_accelerator_name
from fsdp_turbo.utils.log import print_rank

print_rank(logger.info, f"running on {get_accelerator_name()}")
```

错误：

```python
# 1) Duplicated utility: utils/device.py already has get_accelerator_name().
def detect_device():
    return os.environ.get("DEVICE_TYPE", "npu")        # forbidden

# 2) Over-abstraction: two lines used at a single call site.
def default_expert_scale(num_experts: int) -> float:
    return 1.0 / max(num_experts, 1)                   # forbidden: inline it at the call site
```

### 规则 16：注释使用英文，且克制而精简

**语言**：注释一律使用英文，包括行内注释、块注释、`TODO`/`FIXME` 说明；docstring 同样使用英文（与规则 17 的示例一致）。中文只出现在面向用户的资料文档中（见规则 19）。

**克制**：只在代码本身无法表达意图的地方写注释——解释「为什么这样写」、隐含约束、非直觉的边界条件；不解释「代码在做什么」。

**精简**：一条注释讲清一个点；禁止「一行代码配一大段注释」，超过 3 行的解释性注释应改为文档或压缩为一句。

**范围说明**：本条约束代码中的注释与 docstring；目录树、表格、正文中的中文标注属于资料文档内容，按规则 19 处理。

**禁止**：

- 注释掉的死代码、调试残留（`# print(x)`、`# TODO: remove`）；
- 复述代码的废话注释（`i += 1  # increment i`）；
- 与代码不一致的过时注释；
- 无主的 `TODO`/`FIXME`（必须带负责人或 issue 编号）。

错误：

```python
# 首先我们需要判断 chunk_size 是否大于 0，如果大于 0 就继续，
# 否则抛出异常；这里之所以这样判断是因为上游可能传入非法值，
# 然后遍历每个 chunk 累加损失，最后返回总损失 ...
loss = compute_chunked_loss(logits, labels, chunk_size)   # 禁止：大段注释配一行代码
```

正确：

```python
# Cap logits memory at roughly 1 GB per rank.
loss = compute_chunked_loss(logits, labels, chunk_size)
```

---

## 六、接口与文档

### 规则 17：公共 API 必须有类型注解与 docstring

「公共 API」指跨模块调用或对外暴露的函数、类与公开方法；模块内部的一次性辅助函数不受此约束（避免与规则 14 的「减少代码量」冲突）。

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
def all_to_all(tokens, splits):   # forbidden: public API without type annotations or docstring
    ...
```

**存量待整改**：全仓 1203 个 `def` 中仅 208 个带返回注解（约 17%）；新增/修改代码按本规则执行，存量随改动逐步补齐。

### 规则 18：文档必须随代码同步更新

改动代码时必须同步对应文档，不允许「代码先行、文档后补」：

| 改动类型 | 必须更新 |
| --- | --- |
| 新增/变更特性 | `docs/zh/features/<feature>.md`（无则新建） |
| 新增/变更配置项（`FSDPTurboConfig` 字段） | `docs/zh/training/user_guide.md` 及对应 feature 文档 |
| 新增/变更环境变量 | `docs/zh/features/environment_variable.md`（见规则 12） |
| 新增/变更模型支持 | `docs/zh/models/supported_models.md`、`docs/zh/develop/model_adaptation.md` |

- 新增文档必须在 `docs/zh/_menu.md` 登记，否则视为未交付；
- 文档中的路径、模块名、函数名、配置键必须能在仓库中检索到；文档与代码不一致时优先修正文档，不得为迁就过时文档而改动已发布的接口。

**存量待整改**：`docs/zh/_menu.md` 目前未收录 13 个文档——`appendixes.md`、`docs_guide.md`、`models/supported_models.md`、`training/user_guide.md`、`training/finetune/finetune.md`、`training/pretrain/pretrain.md`、`training/quick_start_fsdpturbo.md`、`training/geo3k_finetune_guide.md`、`project_guide.md`、`SECURITYNOTE.md`、`tools/profiling.md`、`develop/model_adaptation.md`、`release_notes_fsdpturbo.md`，需补齐或确认有意不收录。

---

## 七、资料守则

### 规则 19：资料文档中中文与英文之间不加空格

适用于 `docs/` 下的资料文档、`README.md`、RFC/issue 正文、设计文档等面向人的文字；代码注释与 docstring 不适用本条（它们必须用英文，见规则 16）。

- 中文与英文单词、数字相邻时**不加空格**：`使用FSDP2进行8卡训练`、`见第3章`；
- 专有名词保留原文大小写，不翻译不拆词：`torch.accelerator`、`FSDP2`、`MXFP8`；
- 行内代码、代码块、链接、表格中的技术标识保持原样，不受本条影响；
- 中文语境使用中文标点，中英文与标点之间同样不加空格。

正确：

```text
使用FSDP2进行8卡训练，并行配置见config.yaml；量化支持MXFP8格式。
```

错误：

```text
使用 FSDP2 进行 8 卡训练，并行配置见 config.yaml；量化支持 MXFP8 格式。
```

**存量待整改**：现有 `docs/zh/`、`README.md` 以及本守则自身普遍带中英文间空格，不要求一次性全量重排；改动到相关段落时一并修正，新增文档直接按本条执行。

### 规则 20：设备型号必须使用术语规范全称

资料文档中描述产品/训练系列时，必须使用术语规范全称；不得只写 `A2`/`A3`/`A5` 等简称。正文中全称需用 `<term>` 包裹（与现有文档一致）。

| 简称 | 术语规范全称 |
| --- | --- |
| A5 | `<term>Ascend 950PR&950DT系列产品</term>` |
| A3 | `<term>Atlas A3训练系列产品</term>` |
| A2 | `<term>Atlas A2训练系列产品</term>` |

- 正文首次或正式引用时使用全称并保留 `<term>` 标记；表格、标题、文件名等不便使用标记的位置可省略 `<term>`，但**全称文本不得简写**（如 `docs/zh/training/install_guide.md` 表格中的 `Ascend 950PR&950DT系列产品`）；
- 代码、配置文件名（如 `config_8p_A5.json`）、日志、变量名不属于资料文档，保持原样，不强制替换；
- 同一文档内对同一产品的写法必须一致，不得混用简称与全称。

正确：

```text
仓库支持<term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3训练系列产品</term>和<term>Atlas A2训练系列产品</term>。
```

错误：

```text
仓库支持 A5、A3、A2 系列产品。
```

现状：`docs/zh/models/supported_models.md:18`、`docs/zh/training/quick_start_fsdpturbo.md:13`、`docs/zh/release_notes_fsdpturbo.md:53` 已使用上述规范全称（含 `<term>` 标记），无需整改；新增文档保持一致即可。

### 规则 21：特性文档标题不加序号

`docs/zh/features/*.md` 的一级标题只写特性名，**不加** `1.`、`一、`、`（一）` 等序号：

- 正确：`# 断点续训`、`# 上下文并行`
- 错误：`# 1. 断点续训`、`# 一、上下文并行`

补充约束：

- 文内小节标题同样不编号，层级由 `##` / `###` 表达；
- 确需编号的流程步骤放在标题后的正文或有序列表里，不写进标题；
- 理由：序号在章节增删后必然失效，且各文档编号口径不一，不利于检索与交叉引用。

现状：`docs/zh/features/` 下 15 个特性文档标题均已无序号，新增文档保持一致即可。

---

## 八、协作与提交

### 规则 22：提交 PR 前用新会话做 AI 代码检视

提交 PR 前，**新开一个会话**（不复用开发会话），让 AI 以本守则为唯一标准检视本次改动：

- 新会话必须**重新读取守则**，不能依赖上一个会话的上下文，以规避「自己写的自己认可」的确认偏差；
- 检视范围覆盖代码与资料文档两类产物；
- 可直接使用下面这段提示词：

```text
阅读 .agent/skills/fsdp-turbo-dev-guide/SKILL.md，按其中 23 条规则逐条检查本次改动（git diff），
只输出违规项与修改建议，不要修改代码。
```

- 将检视结论（或「无违规」）附在 PR 描述中；涉及安全、依赖越界、patch 建模的高危项必须先修再提；
- 检视发现存量问题但不在本次改动范围内的，按各规则的「存量待整改」处理，不在本 PR 扩大范围。

---

## 九、安全

### 规则 23：禁止不安全反序列化与命令执行

- 禁止用 `torch.load(..., weights_only=False)` 加载来源不可信的文件；确需加载自产 checkpoint 时，必须在文档或 docstring 中写明信任边界，并保证文件路径来源受控；
- 禁止 `eval` / `exec`、`os.system`、`os.popen`、`subprocess(..., shell=True)`；
- 禁止用 `pickle.load`、未指定 `SafeLoader` 的 `yaml.load` 处理外部数据；
- pre-commit 已挂 bandit、gitleaks，但工具只是兜底，不得以「工具没报」代替安全判断。

错误：

```python
state = torch.load(user_supplied_path, weights_only=False)   # forbidden: arbitrary pickle deserialization
os.system(f"cp {src} {dst}")                                  # forbidden: command injection risk
```

**存量待整改**：`training/checkpoint.py:396` 使用 `weights_only=False` 加载 checkpoint（自产文件），整改时评估切换 `weights_only=True` 或改为显式白名单类型。

---

## 自检清单

提交或输出前，按类别逐条确认：

### 设备与后端

- [ ] 规则 1：没有任何 `torch.cuda.*` / `torch.npu.*` 直接调用，统一 `torch.accelerator`；
- [ ] 规则 2：`torch_npu` 仅出现在 `ops/npu/`（及 `utils/device.py` 的启动注入）；
- [ ] 规则 3：新算子已 `@register_op` 注册，且有 `ops/cpu/` 兜底实现，调用方走 `ops/` 语义化包装（无散落 `dispatch_op` 直调）；

### 模块设计

- [ ] 规则 4：新模块先完成结构设计：架构契合、职责与边界清晰、可预见的变体留有扩展接缝（无空壳占位）；
- [ ] 规则 5：模块对外接口精简（几个的数量级），`__init__.py` 只导出不实现；文件夹复制到第三方框架后少量修改即可用；
- [ ] 规则 6：移出单个模块文件夹后不因依赖其他业务模块而报错（仅依赖 `ops/`、`utils/`、标准库、`torch`）；

### 模型适配

- [ ] 规则 7：`fsdp_turbo/` 包内未引用 `fsdp_turbo.models`（仅 `examples/`、`tests/`、`docs/` 可引用），且各模型结构目录之间无交叉导入、无 `models/` 根共享代码依赖；
- [ ] 规则 8：`models/` 下的 modeling 未使用 patch（无 `*_patch.py`/`apply_*_patch()`、无 `setattr` 换成员、无靠配置 `replacement` 替换的自由函数 forward、无 `PatchXxx` 非继承替代类），适配一律继承 transformers 类覆写；
- [ ] 规则 9：`models/` 下未新增 `config.json` 等参数配置文件（配置在 `examples/`），新增模型按 `models/<系列>/<结构>/` 两级目录创建；

### 配置、日志与环境

- [ ] 规则 10：新配置字段已在 `FSDPTurboConfig.__post_init__` 校验，运行期只读；校验只判断合法性，未修改模型或全局状态；
- [ ] 规则 11：无 `print`，日志均走 `print_rank` / `log_warning_once`；
- [ ] 规则 12：未新增散落的 `os.environ.get("XXX")` 字面量，环境变量已集中到 `utils/env.py` 并在环境变量文档中登记；

### 代码风格

- [ ] 规则 13：命名直观自解释，无非常见缩写、无数字后缀；
- [ ] 规则 14：代码量已最小化且只做必要检查（未重复校验已在入口校验的条件、无 YAGNI 参数/空壳、无死代码、无与改动无关的代码）；
- [ ] 规则 15：工具函数优先复用 `fsdp_turbo/utils/` 及标准库/`torch`/`transformers` 既有实现，未重复造轮子；新增抽象仅限会被多处复用的逻辑，未过度抽象；
- [ ] 规则 16：注释为英文、克制且精简，无大段注释配少量代码、无复述代码的废话注释、无注释掉的死代码；

### 接口与文档

- [ ] 规则 17：新增/修改的公共 API 有完整类型注解与 docstring，内部辅助函数未被迫补注解；
- [ ] 规则 18：代码改动已同步对应文档（特性/配置项/环境变量/模型支持），新增文档已在 `docs/zh/_menu.md` 登记；

### 资料守则

- [ ] 规则 19：资料文档中中文与英文、数字之间未加空格，专有名词保留原文大小写；
- [ ] 规则 20：设备型号使用 `<term>` 规范全称（`Ascend 950PR&950DT系列产品` / `Atlas A3训练系列产品` / `Atlas A2训练系列产品`），未使用 A2/A3/A5 简称；
- [ ] 规则 21：特性文档标题未加序号，标题只写特性名；

### 协作与提交

- [ ] 规则 22：已在新会话中用 AI 按本守则检视本轮改动，结论已附在 PR 描述中；

### 安全

- [ ] 规则 23：未引入不安全反序列化（`weights_only=False` 加载不可信文件、`pickle.load`、非 `SafeLoader` 的 `yaml.load`）与命令执行（`eval`/`exec`/`os.system`/`shell=True`）。
