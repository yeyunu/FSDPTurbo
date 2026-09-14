# 专家并行

FSDPTurbo面向MoE（Mixture of Experts）模型提供专家并行（Expert Parallel，EP），将专家按专家维切分到多个设备，token通过All-to-All通信分发到持有对应专家的设备上计算。

## 配置方法

在`distributed`段配置专家并行度与专家计划：

```yaml
distributed:
  expert_parallel_size: 8
  expert_fully_shard_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: "fused"
    apply_efsdp_modules:
      - model.language_model.layers.{*}.mlp.experts
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `expert_parallel_size` | 专家并行度，将experts模块按专家维切分到EP组 |
| `expert_fully_shard_parallel_size` | 专家FSDP2（EFSDP）并行度，对专家参数进一步全分片 |
| `ep_plan.apply_modules` | 应用EP的模块列表，支持`model.language_model.layers.{*}.mlp.experts`通配符 |
| `ep_plan.dispatcher` | 专家计算dispatcher，见下表 |
| `ep_plan.fixed_router` | 是否使用固定路由 |
| `ep_plan.apply_efsdp_modules` | 应用EFSDP的模块列表 |
| `ep_plan.gradient_divide_factor` | 专家梯度除法因子（特殊并行组合下需手动指定） |

## Dispatcher类型

| dispatcher | 说明 |
|---|---|
| `eager` | 逐专家调度，通用性最好 |
| `fused` | 分组GEMM（grouped_matmul），token聚合后批量计算，性能更优 |
| `mc2` | 昇腾MC2加速的通信计算融合实现 |
| `domino` | Domino调度实现 |
| `custom_ep_forward_eager` | 自定义EP forward（eager语义），模型需实现`ep_forward_eager`接口 |
| `custom_ep_forward_fused` | 自定义EP forward（fused语义），模型需实现`ep_forward_fused`接口 |

## 自定义EP forward

模型专家计算语义非标准时，可使用自定义dispatcher，并在专家模块上实现：

```python
def ep_forward_eager(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...

def ep_forward_fused(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...
```

普通GLU/SwiGLU结构的MoE模型应优先使用内置`eager` / `fused` / `mc2` / `domino`dispatcher。

## 模型结构要求

FSDPTurbo的EP路径要求experts模块持有**3D张量参数**，而非ModuleList形式的独立专家子模块。具体要求：

- experts模块需暴露`gate_up_proj`和`down_proj`两个3D张量参数，形状为`[num_experts, out_dim, in_dim]`。
- 2D堆叠格式的参数`[num_experts * out_dim, in_dim]`需通过`view(num_experts, -1, in_dim)`重塑为3D后才能用于grouped_matmul。
- **ModuleList形式专家不被支持**：逐个专家的Linear子模块无法直接参与EP的All-to-All分发与分组GEMM，需先转换为3D张量聚合结构。

## 注意事项

- **并行度组合约束**：`expert_parallel_size`需能整除每层专家总数，且`expert_parallel_size × expert_fully_shard_parallel_size ≤ 总设备数`。
- **专家负载均衡**：训练中可观察路由分布（top-k选择的专家频率），若出现严重不均，可考虑调整模型侧路由温度，或使用`fixed_router`（调试手段：以固定均匀路由替代实际路由，避免负载不均带来的性能与精度问题，仅在`expert_parallel_size > 1`时生效）。
- **EFSDP**：当专家参数量较大时，开启`apply_efsdp_modules`对专家参数进一步做FSDP2全分片，降低单卡显存占用。
