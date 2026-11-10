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
    dispatcher: "all2all"
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
| `ep_plan.apply_efsdp_modules` | 应用EFSDP的模块列表，要求`expert_parallel_size > 1`；未显式配置时自动取`apply_modules`中以`.experts`结尾的项 |
| `ep_plan.gradient_divide_factor` | 专家梯度除法因子（特殊并行组合下需手动指定） |
| `ep_plan.moonep_config` | `dispatcher: moonep` 时的通信、显存与静态形状调优（`MoonEPConfig`），见 [MoonEP dispatcher](#moonep-dispatcher) |
| `ep_plan.use_fused_ops` | 使能昇腾融合算子：GEMM、Permute、Unpermute、Swiglu |
| `ep_plan.use_use_swiglu` | 激活函数是否使用swiglu|
| `ep_plan.moe_chunk_size` | 使能chunkmoe时，token所需要chunk成的size大小，和moe_num_chunks二选一 |
| `ep_plan.moe_num_chunks` | 使能chunkmoe时，token所需要chunk成的份数大小，和moe_chunk_size二选一 |
| `ep_plan.chunk_moe_recompute` | 使能chunkmoe时，是否使能experts部分的重计算 |
| `ep_plan.enable_ep_balance` | 是否使能EP负载均衡，当前只适用于all2all和allgather dispatcher |
| `ep_plan.max_dup_experts_num` | 使能EP负载均衡时，每个ep rank上的冗余专家数 |

## Dispatcher类型

| dispatcher | 说明 |
|---|---|
| `all2all` | 逐专家调度，通用性最好 |
| `mc2` | 昇腾MC2加速的通信计算融合实现 |
| `domino` | Domino调度实现 |
| `allgather` | allgather调度实现 |
| `chunkmoe` | chunkmoe调度实现 |
| `custom_ep_forward_eager` | 自定义EP forward（eager语义），模型需实现`ep_forward_eager`接口 |
| `custom_ep_forward_fused` | 自定义EP forward（fused语义），模型需实现`ep_forward_fused`接口 |
| `moonep` | 基于对称内存的EP后端（BF16），要求`expert_parallel_size > 1`且`expert_fully_shard_parallel_size: 1`，详见 [MoonEP dispatcher](#moonep-dispatcher) |

## 自定义EP forward

模型专家计算语义非标准时，可使用自定义dispatcher，并在专家模块上实现：

```python
def ep_forward_eager(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...

def ep_forward_fused(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...
```

普通GLU/SwiGLU结构的MoE模型使用内置 `all2all` / `allgather` / `chunkmoe` / `mc2` / `domino` dispatcher。
通信调度方式与融合算子开关分开配置：`dispatcher` 默认是 `all2all`，`use_fused_ops` 默认是 `true`。
旧的 `dispatcher: eager` 对应 `dispatcher: all2all` 加 `use_fused_ops: false`；
旧的 `dispatcher: fused` 对应 `dispatcher: all2all` 加 `use_fused_ops: true`。
旧名称或其他不支持的值会在 `FSDPTurboConfig` 初始化时抛出 `ValueError`，
直接调用 EP forward 工厂时，不支持的名称也会立即抛出 `RuntimeError`。
`custom_<method>` 和 callable dispatcher 保持现有调用约定。

## MoonEP dispatcher

`dispatcher: moonep` 使用 MoonEP 对称内存后端执行专家计算与通信：每卡固定接收 `S*K` 行 dispatched 激活（形状全程静态，关键路径上无设备标量回读主机），冗余专家按步规划并预取到对称内存槽位，权重梯度以 FP32 暂存并经 stage → rendezvous → reduce 三段式归约送回归属卡。

**约束（`FSDPTurboConfig` 校验期强制）：**

- `expert_parallel_size > 1`，且专家总数能被 EP 并行度整除；
- `expert_fully_shard_parallel_size: 1`（MoonEP 自己持有专家权重，不能与 EFSDP 叠加），`ep_plan.apply_efsdp_modules` 须留空；
- 专家权重为 BF16 稠密 `nn.Parameter`，两个矩阵维度均为 128 的倍数；与 MoE 量化（`quantize.moe.*` 转换器、`quant_gmm`）互斥；
- CUDA 后端要求 EP 组同机且支持 NVLink/NVSwitch 多播，依赖 `moonep==0.0.1`；昇腾后端依赖 `npu_patch` 包（同一个 `moonep` 发行包按昇腾后端构建时提供）。

**静态形状派生：** `tokens_per_rank = data.batch_size * data.max_seq_length / ulysses_parallel_size`，`top_k` 从模型配置（`num_experts_per_token` / `num_experts_per_tok`）自动读取，二者一般不手填。昇腾的对称堆初始化前必须锁定 S/K，派生失败会在 `MoonEPRuntime.finalize` 时报错；CUDA 在首次专家前向推导后锁定。

**`MoonEPConfig` 调优项：** 下表是速查卡。`fsdp_turbo/distributed/expert_parallel/moonep/README.md` 把同一批旋钮按影响面分组，并给出每个默认值背后的实测数据。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `num_sms` | `32` | 集合通信 kernel 使用的 SM 数 |
| `token_padding` | `128` | token 对齐粒度 |
| `comm_stream_priority` | `-1` | 通信侧流优先级（仅 CUDA 生效） |
| `async_finish` | `false` | 通信在侧流异步收尾。CUDA 上直接生效；昇腾上默认被屏蔽，需 `npu_async_comm: true` 才打开 |
| `tokens_per_rank` / `top_k` | 派生 | 静态形状覆写，一般不手填 |
| `recompute_dispatch` | `false` | 反向重放一次 dispatch，换取第一 GEMM 少存 `K` 倍激活；整层重计算已覆盖专家模块时冗余（会有告警） |
| `validate_plan_invariants` | `false` | 每步执行设备侧规划校验（默认仅首步） |
| `shmem_ip_port` | 派生 | 多机 EP 组的 CANN SHMEM bootstrap 端点，默认取 `MASTER_ADDR:MASTER_PORT+1` |
| `shmem_stream_barrier` | `true` | 昇腾用 CANN SHMEM 流序全局屏障替代主机同步 + `dist.barrier` |
| `trace_enabled` | `false` | 逐卡打印 MoonEP 阶段标记，用来定位无报错的集合通信挂死 |
| `trace_sync_enabled` | `false` | 每个标记处同步设备；会串行化流水线，仅调试用 |
| `vmm_probe_enabled` | `false` | 昇腾 VMM 分配失败时在当前进程内按尺寸扫描，区分单映射上限与进程总上限 |
| `npu_async_comm` | `false` | 昇腾才读取。为 true 时后端报告支持异步通信，`async_finish` 才会在侧流上跑。该路径尚未在真机验证，失步表现为挂死；步进停住就改回 false |

Kimi 路由诊断在 `ep_plan.router_debug`，服务所有 EP dispatcher，默认不设置即关闭：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `apply_custom_router` | `false` | 用合成路由替换 Kimi 真实打分；打开时 `target_max_vio` 须为有限且 ≥ 0 |
| `target_max_vio` | `-1` | 合成路由的目标最大违规度 |
| `router_mode` | `lognormal` | `exact` 或 `lognormal` |
| `record_load_balance` | `false` | 打印每层 expert 计数和 EP 组上的 global maxvio |

**EDP 复制维梯度同步：** `edp > 1` 时，`MoonEPRuntime.sync_replica_gradients` 在 EP 组归约之后、除数之前，对 FP32 home 梯度沿 1-D `edp` 组做 SUM，与 fused HSDP 在复制维上的 all-reduce 对应；`gradient_divide_factor` 仍按 `ep × efsdp × edp` 计算，保持不变。`edp = 1` 时不增加任何集合通信。

**验证入口：** 单测见 `tests/unit_tests/distributed/expert_parallel/moonep/`；系统测试见 `tests/system_tests/model/moonep/`（含 `edp > 1` 的绝对参考检查 `test_moonep_edp_ref.py`）。

## 模型结构要求

FSDPTurbo的EP路径要求experts模块持有**3D张量参数**，而非ModuleList形式的独立专家子模块。具体要求：

- experts模块需暴露`gate_up_proj`和`down_proj`两个3D张量参数，形状为`[num_experts, out_dim, in_dim]`。
- 2D堆叠格式的参数`[num_experts * out_dim, in_dim]`需通过`view(num_experts, -1, in_dim)`重塑为3D后才能用于grouped_matmul。
- **ModuleList形式专家不被支持**：逐个专家的Linear子模块无法直接参与EP的All-to-All分发与分组GEMM，需先转换为3D张量聚合结构。

## 注意事项

- **并行度组合约束**：`expert_parallel_size`需能整除每层专家总数，且`expert_parallel_size × expert_fully_shard_parallel_size ≤ 总设备数`。
- **专家负载均衡**：训练中可观察路由分布（top-k选择的专家频率），若出现严重不均，可考虑调整模型侧路由温度，或使用`fixed_router`（调试手段：以固定均匀路由替代实际路由，避免负载不均带来的性能与精度问题，仅在`expert_parallel_size > 1`时生效）。
- **EFSDP**：当专家参数量较大时，开启`apply_efsdp_modules`对专家参数进一步做FSDP2全分片，降低单卡显存占用。
- **a2a同步点缓存**：`all2all`与`mc2` dispatcher在前向缓存a2a split sizes。激活重计算的重放路径直接复用缓存（跳过计数、EP all-gather与D2H同步），`mc2`的backward权重梯度GMM复用device侧每专家token总数作为`group_list`（免去host求和与pageable H2D同步）。由`distributed.ep_plan.reuse_dispatch_splits`控制（默认关闭），详见[激活重计算](recompute.md)。`ep_balance`开启时`all2all`路径的preprocess走负载均衡接口，不经过该缓存。

## 负载均衡接口

负载均衡的初始化、路由重排、重复专家权重通信和梯度累加统一位于
`fsdp_turbo/distributed/expert_parallel/ep_balance/`。配置方式保持不变，
`enable_ep_balance` 当前支持 `all2all` 和 `allgather`。

- `initialize_ep_balance(module, ep_group, enabled, max_dup_experts_num)`：初始化模块的均衡策略，保留已有策略。
- `strategy.prepare_all2all(top_k_index)`：返回重排后的路由、permute 元数据和通信 splits。
- `strategy.prepare_allgather(top_k_index)`：返回已收集的路由、本地专家数（含重复槽位）和各专家 token 数。
- `EPBalanceWeights(strategy, gate_up_weights)`：保存本次 forward 的重复权重缓冲区，并启动 gate/up 权重通信；
  `strategy=None` 时直接返回原权重，不进行通信。

Dispatcher 只需要接收接口返回的变量，不直接访问 planner、executor 或注册 hook：

```python
balance = EPBalanceWeights(ep_balance_strategy, gate_up_weights)
# 原有 token dispatch / permute
# 等待 gate/up 权重，同时启动 down 权重通信
gate_up_weights = balance.prepare_gate_up(gate_up_weights, down_weights)
# 原有第一个 GEMM 和激活
down_weights = balance.prepare_down(down_weights)
# 原有第二个 GEMM
```

权重准备调用的位置决定通信与计算的重叠，不应将它们提前合并成一次同步调用。
重复专家梯度使用原有 executor 通信与累加接口，在每次 forward 的临时权重视图上回收，
然后传给原参数；不会反复向持久参数注册 hook。无 token 的 rank 也参与这一步。
重计算使用 forward 缓存的均衡方案与 AllGather 路由，不重新统计或通信。
