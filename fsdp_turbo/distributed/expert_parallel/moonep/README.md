# MoonEP（昇腾专家并行后端）

MoonEP 是可选的专家并行后端，面向固定 shape 的 BF16 训练。它把专家权重与 token 交换放进
CANN SHMEM 的对称内存，让 MoE 层的通信与计算重叠，并按 home-rank 归约专家权重梯度。

## 启用

安装算子库 `npu_patch`（由 MoonEP 仓库的 `moonep` 发行包同时提供 `moonep` 与 `npu_patch`
两个 Python 包）。它目前**不在**公共或华为 PyPI 索引上，FSDPTurbo 也没有 `moonep` 安装 extra。请在 MoonEP 仓库执行
`bash npu_patch/scripts/build_base.sh -soc_type <SOC_TYPE>` 构建 kernel 库，再
`MOONEP_BUILD_BACKEND=npu pip install -e npu_patch`，或把预构建包放进运行镜像。

然后显式选择 dispatcher：

```python
config.distributed.ep_plan = EPPlanConfig(
    apply_modules=["model.layers.{*}.mlp.experts"],
    dispatcher="moonep",
)
config.distributed.expert_fully_shard_parallel_size = 1
```

NPU 上通信 buffer 的 `S`、`K` 必须在符号堆初始化前锁定，但**不要手写进 EP plan**。
`S` 由 `data.batch_size * data.max_seq_length / ulysses_size` 推出，`K` 由模型的
`num_experts_per_tok` / `num_experts_per_token` 推出。只有推导失败，或要覆盖公式时，
才写 `moonep_config.tokens_per_rank` / `top_k`。短于 `S` 的 microbatch 由 dispatcher
在专家入口补零；长于 `S` 会报错。理由见下一节。`fill_moonep_static_shape`
就地写回 `plan.moonep_config`，所以事后 dump 配置会看到推导出来的
`tokens_per_rank`；那是推导值，不要照抄回 yaml。

## 对称内存带来的三条约束

这三条都源于 CANN SHMEM 符号堆的生命周期：**先注册全部对称内存，再一次性初始化，用完在
释放底层显存之前 finalize**。

**一、静态形状必须提前确定。** 每一块被 kernel 寻址的显存都要先用
`buffer_c.create_vmm_physical` 注册，然后才能执行那一次集合式的
`ShmemRuntime.init_with_buffer`。通信 buffer 的尺寸由每 rank 的 token 容量
`S` 和 `K` 决定，所以它们不能等到第一次专家前向时才知道。`S`、`K` 在包装模型时从
data / 模型配置推出；实际 token 数短于 `S` 时 dispatcher 补到容量，超过则报错。

**二、符号堆必须由 MoonEP 独占。** 初始化前如果进程里已经有人初始化过 CANN SHMEM 堆，
MoonEP 会拒绝启动——否则先注册的那批对称内存不会包含专家权重。

**三、用完必须 finalize，且要在释放底层显存之前。** `aclshmem_finalize` 要求所有已注册的
VMM handle 仍然存活，所以销毁分布式进程组前需调用 `model.close()`，它会按正确顺序先
finalize 符号堆再释放 buffer 与专家权重。

关于 **EP 组跨节点**：并不要求 EP 组落在单节点内。单节点时符号堆走 UID 广播自动完成
bootstrap；EP 组跨越多个节点时改走 TCP bootstrap，此时需要 `moonep_config.shmem_ip_port`
（形如 `tcp://10.0.0.1:29501`），缺省则由环境里的 `MASTER_ADDR` 与 `MASTER_PORT + 1` 推导。
两者都没有会明确报错。节点数由 `LOCAL_WORLD_SIZE` 推断（torchrun 按节点连续分配 rank），
该变量缺失时退化为一次 hostname 哈希的 allgather。

> 需要说明的是，当前的验证都在**单节点（4 卡与 8 卡）**上完成，跨节点路径尚未实测。

## 其他前提

专家模块需提供融合的 `gate_up_proj`、`down_proj`、`act_fn` 和 `hidden_dim`。
权重朝向接受两种：`[E, 2*Hp, H]` + `[E, H, Hp]`（Qwen），或其转置（Kimi packed 布局）。
朝向只改变三个分组 GEMM 的操作数视图，不改参数形状，也不改 state-dict ABI。
SiTU 一类吃完整 `gate_up` 的激活走模块自己的 `_apply_activation` / `SituAndMul`，
不会被拆成 `act_fn(gate) * up`。这两种路径都还没有在真机上对 Kimi 做过数值验收，
不要据此写成「支持 Kimi」。

下列组合会在配置校验阶段**明确报错**，不会静默回退：

- 动态 token shape（MoonEP 的形状是静态的，换 shape 请补齐或丢弃）
- 专家 FSDP（`expert_fully_shard_parallel_size > 1` 或非空 `apply_efsdp_modules`）——
  专家权重由 MoonEP 持有，不能同时交给 FSDP
- MoE 量化（任何 `quantize.moe.*` converter 或 `quant_gmm`）—— MoonEP 要求 BF16 专家权重；
  量化 dense linear 不受影响
- `expert_parallel_size <= 1`

## 调优旋钮

`MoonEPConfig` 的旋钮按影响面分三类。**默认值是唯一在真机上完整验证过的组合**，改动前请
先用 `tests/system_tests/model/moonep/run_test_moonep.sh` 确认基线。

改变数值结果或运行时行为：

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `recompute_dispatch` | `False` | 反向重算 dispatch，多一次 all-to-all 换第一个专家 GEMM 上 K 倍的激活显存。整层重算已开启时不必再开，dispatch 已被重放。实测省 0.25 GiB 峰值（−2.0%） |
| `validate_plan_invariants` | `False` | 每步都跑 plan 与路由的主机侧检查，而不只是第一步。关闭时**第一次调用仍然检查**。调试期有用，每个 MoE 层多几个逐元素 kernel |

只影响性能：

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `num_sms` | `32` | 通信 kernel 占用的 SM 数 |
| `token_padding` | `128` | token 维度的对齐粒度（昇腾上 `npu_patch` 忽略它，NvS 布局不做 token padding） |

昇腾专有：

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `shmem_stream_barrier` | `True` | 用 CANN SHMEM 的流序全局栅栏替代主机 `synchronize` + `dist.barrier`，省掉反向里每个 MoE 层一次的流水线排空。实测快 2.3–3.8%、loss 逐位不变，且在 3 步与 20 步的逐张量正确性验证下都通过。保留为开关是因为它动的正是历史上出现过数据竞态的那处同步，而 20 步不构成证明 |
| `shmem_ip_port` | `None` | 跨节点 EP 组的符号堆 bootstrap 端点，见上文 |

设了但在昇腾上**默认无效**的旋钮（运行时会 warning）：

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `async_finish` | `False` | 昇腾的 kernel 内含设备侧全 rank 栅栏，其代次计数在侧流上会失步，因此后端默认报告 `supports_async_comm=False`，请求在到达 MoonEP 之前就被屏蔽，通信恒在计算流上。默认关闭；若显式打开，初始化时会以 INFO 报告实际生效值 |
| `comm_stream_priority` | `-1` | 侧流默认不承载任何工作，优先级没有作用对象 |

这两项由 `moonep_config.npu_async_comm: true` 一起解锁：新版 `npu_patch` 已把 `async_finish`
做成真异步（独立 comm stream + 流序栅栏），打开后后端改报
`supports_async_comm=True`，`async_finish` 与 `comm_stream_priority` 同时生效，上面的
warning 也不再出现。**该路径尚未在真机上验证**——栅栏代次失步的表现是挂死而不是数值错误，
所以只用于真机 A/B；步进停住就改回 `false`。

调试开关同样在 `moonep_config` 上，默认关闭：`trace_enabled`、`trace_sync_enabled`、`vmm_probe_enabled`。

## 验证

改了配置或换了环境之后，先确认算得对：

```bash
bash tests/system_tests/model/moonep/run_test_moonep.sh
```

它把前向输出与全部梯度同一个“每个 rank 持有全部专家、朴素计算 MoE”的参考逐张量对比，因此
能分别验证 dispatch、权重预取、combine 与 home-rank 梯度归约。这比只看 loss 强——loss 一致
只能说明整体没跑偏。

不需要真机的部分由单元测试覆盖：

```bash
pytest tests/unit_tests/distributed/expert_parallel/moonep -q
```

## 模块结构

| 模块 | 职责 |
|---|---|
| `dispatcher` | 替换专家前向的 module patch，也是对外的唯一入口之一 |
| `runtime` | 资源归属：对称专家权重、预取槽位、逐步状态 |
| `autograd` | 把 MoonEP 的集合通信放进反向图，并固定它们与专家 GEMM 的先后 |
| `plan` | 静态 shape 的分组布局与 plan 不变量校验 |
| `backend` / `backend_npu` | 全部设备相关工作，藏在同一个协议后面 |
| `settings` / `streams` / `trace` | 配置归一、流与事件辅助、定位挂起用的逐 rank 标记 |

包外只允许使用 `MoonEPRuntime` 与 `parallelize_moonep_module` 两个名字。
