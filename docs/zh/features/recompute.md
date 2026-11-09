# 激活重计算

激活重计算（梯度检查点）在前向时不保存部分模块的激活值，反向时重新计算，以时间换显存。

## 配置方法

在`memory`段开启重计算并指定作用模块：

```yaml
memory:
  recompute: true
  recompute_plan:
    - model.language_model.layers.{*}
    - model.visual.blocks.{*}
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `recompute` | 激活重计算总开关 |
| `recompute_plan` | 重计算作用模块列表，支持`model.layers.{*}`层级通配符，可只对部分层开启以平衡显存与耗时 |
| `distributed.ep_plan.reuse_dispatch_splits` | MoE a2a同步点缓存开关，默认`false`；详见[MoE a2a同步点缓存](#moe-a2a同步点缓存) |
| `distributed.ep_plan.dispatch_split_cache_cap` | 缓存栈每模块深度上限，默认`8`；梯度累积步数大于该值时建议调大以保持缓存全命中 |

YAML 的`recompute_plan`只支持模块 pattern 列表。更细的控制可由代码直接传入
`RecomputePlanConfig`：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `apply_modules` | `[]` | 模块 pattern 列表；条目也可写成`module.pattern.method_name`，此时只重计算匹配模块上的指定方法 |
| `flatten_inputs` | `False` | 是否把`(args, kwargs)`整体展平为位置参数后再交给 checkpoint |

## 实现机制

重计算基于`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`实现：

- **KV Cache处理**：若被包装的 forward 形参包含`past_key_values`，调用时通过签名 bind 将其置为
  `None`（位置传参与关键字传参都覆盖）；经`**kwargs`透传的同名实参也会置为`None`，避免 KV cache
  在重算路径中导致 dtype 不一致或 CheckpointError。
- **context_fn由调用方注入**：memory 包不内置任何低精度或随机数上下文。`recompute_modules`
  接受`context_fn_factory`，接线时对每个被包装的模块调用一次并以该模块为参数；返回的
  context_fn 由 checkpoint 每次执行时调用，返回`None`则不向 checkpoint 传 context_fn。
  多个上下文可用`compose_checkpoint_contexts`按序组合。
- **mxfp8组合**：`FSDPTurbo.apply_recompute_modules`在开启 mxfp8 时负责注入相位切换上下文，
  并在包装前完成模块打标；memory 包不依赖 quantization。
- **模型侧 replay 上下文注册**：模型若在进程级保存跨层共享状态，可调`register_replay_context(fn)`
  注册一个「只在反向 replay 窗口内生效」的上下文。注册工厂在**原始前向**接线时被调用一次，
  它返回的对象必须是可重复进入的（每次重算重新进入），并与 checkpoint 自身的 context_fn 按序嵌套
  （不替换它，故 mxfp8 相位机与方法级条目仍生效）；`unregister_replay_context(fn)`可注销。
  DeepSeek-V4.1 的`adapter.py`即以此注册 shared_attn 的重算隔离：工厂在前向时抓取
  `SharedAttentionRuntime.snapshot()`，replay 期间在私有副本上读写（`publish` 是唯一写入口），
  退出时恢复 live 状态，见`runtime.py`的`replay_window`。
- **方法级条目**：`module.pattern.method_name`只包装匹配模块上的同名方法，向 factory 传`None`。
- **去重与嵌套**：多个 pattern 命中同一模块或方法时只包装一次；`reject_nested=True`时在接线期拒绝
  模块间的祖先/后代嵌套。方法相关的动态嵌套仅告警，需由调用方排除。

## MoE a2a同步点缓存

重计算会整段重跑MoE forward，`dispatch_preprocess`中的专家计数、EP
all-gather与D2H拷贝（`tolist` / `.to("cpu")`）会全部重新执行，
其中D2H是host阻塞同步点。重算重跑router与前向同输入同算子，
split计数必然相同，因此前向侧缓存、重算侧直接复用：

- **重算复用**：正常前向把a2a split sizes缓存在专家模块上
  （按模块配对的栈，深度上限默认8、由`dispatch_split_cache_cap`控制，
  兜底未backward的前向残留）；重算路径命中缓存后直接跳过计数、
  EP all-gather与D2H同步，device侧indices由缓存的EP级专家计数重建。
  所有EP rank对称跳过集合通信，无死锁风险。
- **反向wgrad复用**（`mc2` dispatcher）：backward权重梯度GMM的`group_list`直接复用前向算好的
  device侧每专家token总数（`[E_local]` int64，KB级），免去host求和
  与pageable H2D同步；该优化不依赖重计算，每步backward均生效。
- **重算检测**：与FSDP2 `_fsdp_common.is_bw`同源，
  使用`torch._C._current_graph_task_id() != -1`判断当前forward是否运行在backward
  graph task内（checkpoint重算的forward在saved-tensor unpack中执行）。
- **开关**：`distributed.ep_plan.reuse_dispatch_splits`默认关闭；设为`true`后重算与反向均走缓存优化，
  关闭时回退原有路径，语义不变。

## 注意事项

- **与上下文并行（CP）的组合**：重计算与Ulysses CP可同时开启，重计算作用于CP切分后的本rank序列片段，两者互不冲突。
- **与CPU Offload的组合**：重计算降低激活显存，CPU Offload降低参数/梯度显存，两者叠加可进一步压缩显存峰值，但会增加计算与数据搬运开销。
- **粒度选择**：建议以Transformer层（`model.layers.{*}`）为重计算单位，逐层开启可在显存与耗时之间取得平衡。全部开启显存最省但耗时增加最多。
- **跨层共享状态**：模型若在进程级共享跨层状态（如DeepSeek-V4.1的 shared_attn 槽位），开启重计算前必须注册 replay 上下文，且必须在**前向接线时捕获**需要的状态：只暂停 replay 期间的写入并不够——前向结束后槽位已被最后一批 source 覆盖，早先分组的重算仍会读到错版本；同时应隔离 replay 期间的写入并在退出时恢复 live 状态。
