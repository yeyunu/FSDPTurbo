# 优化器状态交换

优化器状态交换（Swap Optimizer）将优化器状态卸载到CPU内存，更新时按显存余量分批回装，降低NPU显存占用。

## 功能原理

AdamW等优化器的状态（一阶动量`exp_avg`、二阶动量`exp_avg_sq`）通常占用与参数相当甚至更多的显存。Swap Optimizer通过`AdamWSwap`优化器实现：

1. 初始化时，将优化器状态从device复制到pinned CPU内存（半驻留参数除外，见[半驻留](#半驻留)）。
2. `step()`执行时，按当前可用显存将state升序分块回装到device，块内按参数组和dtype分桶调用fused adamw更新，更新后在专用swap流上**异步**交换回host。
3. 通过分批回装，将峰值显存从“全部state”降低为“单批state”；通过批量拷贝压平FSDP切分碎参数场景的host下发开销。

## 配置方法

在`config.yaml`的`optimizer`段将优化器类型设为`AdamWSwap`：

```yaml
optimizer:
  optimizer_type: "AdamWSwap"
  lr: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  mem_fraction_static: 0.9                    # 可选：批量H2D回装使用的空闲显存比例，默认0.8
  state_keys: ["exp_avg", "exp_avg_sq"]       # 可选：参与交换的state键，缺省使用AdamWSwap内部默认
  param_prefetch: false                        # 可选：参与交换的参数是否提前加载一个，减少带宽阻塞，缺省使用AdamWSwap内部默认（关闭）
  prefetch_states_at_backward: null            # 可选：states H2D提前到指定layer的backward触发，缺省在step()内回装
  clip_grad_keep_on_device: false              # 可选：cpu_offload打开时clip_grad对梯度更新后保留梯度在device侧，减少一次h2d和d2h的搬运，仅在cpu_offload和AdamWSwap同时开启时生效，缺省使用AdamWSwap内部默认（关闭）
  resident_states_ratio: null                  # 可选：半驻留比例，最小参数的states恒驻device不参与交换，0关闭（见[半驻留](#半驻留)）
  swap_prefetch_budget_ratio: null             # 可选：backward预取预算占触发点空闲显存的比例，默认0.7
  swap_prefetch_batches: null                  # 可选：backward预取切分的子批数，默认8
```

| 参数 | 说明 |
|---|---|
| `optimizer_type` | 设为`AdamWSwap`即启用优化器状态交换 |
| `mem_fraction_static` | 批量回装时允许使用的空闲显存比例，默认`0.8` |
| `state_keys` | 参与交换的优化器状态键列表，缺省使用AdamWSwap内部默认（`exp_avg`、`exp_avg_sq`，amsgrad时含`max_exp_avg_sq`） |
| `param_prefetch` | 参与交换的参数预加载配置，缺省使用AdamWSwap内部默认（不提前加载） |
| `prefetch_states_at_backward` | 触发states提前回装的module名模式（backward pre-hook），缺省为`null`（仅在`step()`内回装） |
| `swap_chunk_numel` | 块预算上限（numel），实际取`min(swap_chunk_numel, swap_numel // 2)`（ping-pong双缓冲）；碎参数聚成多参数块、状态拷贝合并下发，大参数独占一块；缺省`null`（4M numel） |
| `resident_states_ratio` | 半驻留比例：按numel升序收编最小参数，其states恒驻device、不建host副本、不参与交换与预取，降低碎参数场景的host调用次数；缺省`null`（关闭） |
| `swap_prefetch_budget_ratio` | backward预取预算占触发点空闲显存的比例，其余留给后继backward的梯度/中间张量分配；触发点通常在视觉塔附近、空闲显存已显著低于峰值，缺省`null`（0.7） |
| `swap_prefetch_batches` | backward预取切分的子批数，每子批各记一个事件，`step()`只等待当前块消费到的子批事件；缺省`null`（8） |
| `clip_grad_keep_on_device` | 在cpu_offload和AdamWSwap同时开启时，clip_grad对梯度更新后保留梯度在device侧，缺省使用AdamWSwap内部默认（梯度在host侧） |

`step()`内的默认回装曾与计算**串行**：H2D完成→fused adamw→D2H，逐param流水。由于优化器状态字节数约为参数的2~3倍，这段搬运在大模型上相当可观。现在H2D可通过backward预取提前（见[backward期states预取](#backward期states预取)），D2H则在专用host swap流上**异步**执行（见[异步D2H](#异步d2hrecord_stream模式)），两者都不再串行阻塞step。

## backward期states预取

`prefetch_states_at_backward`把整段states H2D提前到**backward过程中某个layer的反向时刻**，在专用swap流上异步执行，用剩余backward计算完全隐藏传输：

```yaml
optimizer:
  optimizer_type: "AdamWSwap"
  prefetch_states_at_backward: "vision_tower.encoder.blocks.{26-26}"   # 视觉塔反向首块（前向最后一层）
```

其依据是多模态模型的显存曲线：**峰值在语言塔反向**（梯度、激活与fp32 loss块叠加），而backward按语言塔→视觉塔的顺序执行——语言塔反向结束时显存出现**断崖式回落**（语言层激活/中间张量释放，grads保留至optimizer.step()）。触发层必须取**语言塔之后的反向模块**（推荐视觉塔反向首块，即其前向最后一个block）：此时峰值段已过，预算按回落后的空闲显存计算而充足，预取的states也只在与峰值无关的视觉塔反向窗口内驻留。

选层原则（模式语法：`*`/`?`为通配符，`{N-M}`校验层号区间，`{*}`匹配任意层号；`{26-26}`即视觉塔反向首块，层数按实际模型调整）：

- **推荐**：视觉塔反向首块（其前向最后一个block），或反向顺序位于语言塔之后的其它模块——触发点已越过语言塔反向的峰值段与grads分配段，预算按回落后的空闲显存计算，剩余视觉塔反向覆盖H2D；
- **太早**（语言塔任意层，含其反向首层即最后一层）：预取在峰值段触发，预算按接近峰值的空闲显存计算而偏小，且states要在剩余语言塔反向的峰值窗口内驻留、与后继grads分配竞争，装不下的大部分仍走step内串行路径；
- **太晚**（视觉塔反向末尾或无视觉塔的单塔模型末端）：重叠窗口不足以覆盖整段H2D，传输尾部仍会阻塞step；
- 经验起点：视觉塔反向首块。

> [!NOTE]
>
> - 预取按`requires_grad`过滤（冻结参数本就不参与step的D2H）；预算取触发点空闲显存的`swap_prefetch_budget_ratio`比例（默认0.7，后继backward仍需为梯度/中间张量分配显存，满额预取会将其推向OOM）；装不下的部分由`step()`内原有流水线兜底，无正确性风险。
> - **排空闸门**：触发时先在计算流上记录事件，swap流等待该事件后才发首个H2D——host领先device时，若不设闸门，第一次H2D会与触发点之前尚未完成的backward kernel抢带宽。
> - **子批事件**：预取按numel升序准入（与step内分块同一排序），装下的前缀均分为至多`swap_prefetch_batches`个子批，每子批拷贝完成后各记录一个事件；`step()`更新某块前只等待该块参数涉及的最深子批事件（事件为栅栏，隐含覆盖更早子批）——最先更新的小块不必等预取尾部。
> - `step()`内回装预算若连首块states都装不下、或某个param的states始终未装载，会显式抛出OOM断言（提示增大`mem_fraction_static`）而非静默跳过——静默跳过会让fused adamw读写空storage，导致数值污染或非法访存。
> - **梯度累积（G>1）**：trainer自动将预取延后到累积组**最后一个micro-batch**的backward触发，states驻留窗口不再跨越其余micro-batch的forward/backward；触发层pattern按最后一个micro-batch的backward选层（选层原则不变）。
> - 每步只触发一次（fire-once），`step()`开始时复位；若模式在模型中未匹配到，会打印告警并回退到step内回装。

## 异步D2H（record_stream模式）

更新完成后的states D2H采用MindSpeed `swap_optimizer`的成熟模式，在专用`swap_to_host_stream`上执行，CPU侧零阻塞，与default流上后续param的fused adamw重叠（step串行尾巴从Σadamw + ΣD2H压缩为≈max(Σadamw, ΣD2H)）：

1. default流上fused adamw完成后record更新事件；
2. host流`wait_event`该事件（read-after-write，保证拷回的是更新后的值），随后`non_blocking`拷回pinned host buffer；
3. `record_stream(swap_to_host_stream)` + `storage().resize_(0)`：立即释放显存占位，但allocator保证释放的block在host流拷贝完成之前不会被任何流（含refill的H2D重分配）复用——这是免CPU同步的跨流安全关键；
4. 跨步安全由**事件**保证：`step()`末尾记录本步全部D2H的完成事件（`swap_to_host_stream.record_event()`），下一步的H2D（`step()`开头与backward预取入口）在swap流上`wait_event`等待它，确保读同一块pinned host buffer的H2D晚于上一步的D2H写入，无须CPU硬同步。

参数本体（`cpu_offload`场景）的D2H保持default流blocking：FSDP2管理的`_local_tensor`是pageable host内存，`non_blocking`实际仍会同步；states（字节大头，约为param的2~3倍）的pinned buffer才能真正异步。

## step内调度（升序分块+批量拷贝）

调度的成员集合在构造期固定，块预算随每步空闲显存在`step()`内重建：

1. 全部交换参数按`param numel`**升序**排序后按预算切块：首块最小，唯一必然暴露的首块H2D尺寸最小；越往后块的计算与传输同步爬升，大参数不再前移挤占计算窗口；
2. 块内累计不超过预算，实际块预算取`min(swap_chunk_numel, swap_numel // 2)`——ping-pong下相邻两块states同时驻留device，预算须容纳两倍。碎参数聚成多参数块，numel超过预算的大参数独占一块；
3. 每块执行`wait本块H2D→发射下一块H2D→本块adamw→异步D2H`：下一块H2D的传输被本块adamw计算掩盖（ping-pong双缓冲），D2H无消费者等待，与本块/下一块计算天然并发；H2D完成事件仅在确有拷贝入队时记录——空载块（全被backward预取覆盖或被跳过）若仍记事件，栅栏会把整个预取尾部一并阻塞，使子批等待失效；backward预取覆盖的参数改由子批事件排序（见[backward期states预取](#backward期states预取)），`step()`只等当前块消费的最深子批；
4. 碎参数块的状态拷贝按dtype分组合并为一次`torch._foreach_copy_`（H2D、D2H各一次）；大参数块天然只有少量拷贝，走普通`copy_`。

第4步针对FSDP切分后的host-bound：切分让param数不变而numel变小，per-param的`resize_`/`copy_`/`record_stream`调用成为主要CPU开销，合并拷贝直接减少下发次数。相应地，碎参数不再需要靠大参数的计算窗口来掩盖，因此调度回归为单一升序，不再做中点两段切分。

### 分块粒度

`swap_chunk_numel`（默认4194304 = 4M numel）是块预算上限，实际取`min(swap_chunk_numel, swap_numel // 2)`：调小则块多、下一块H2D更早发射但主机下发次数回升；调大则块内计算长、下单少，但ping-pong双缓冲占用的显存同步上升。

### 半驻留

`resident_states_ratio`（缺省关闭）在构造期按`numel`升序收编最小参数，其optimizer states恒驻device：不建host副本、不释放device storage、不参与任何H2D/D2H搬运与backward预取。step内它们与交换参数走**同一条分块流水线**，但排在交换参数之后（交换参数按numel升序在前，驻留参数按numel升序在后，再统一切块）——因states常驻而天然被H2D准入跳过、不占交换预算，更新合入最后一个块的分桶kernel，不单列调度路径。放在末尾的依据是：驻留参数没有可被计算掩盖的H2D，混在升序前段只会挤占交换块的块预算与计算窗口；置于队尾则末尾块通常由纯驻留参数收尾，无H2D、无D2H。适用于FSDP切分后碎参数多、per-param host调用（`copy_`/`resize_`/`record_stream`）成为瓶颈的场景——碎参数的host开销随驻留比例直接下降，大参数仍走升序分块流水线。

> [!NOTE]
>
> **与`fsdp_plan.cpu_offload`互斥**（配置校验直接拒绝）——cpu_offload下参数本体在host，驻留的前提（states恒驻device、param随取随用）不成立。

代价是这部分states显存常驻（计入step预算测量的已用显存，自动压低交换预算），因此前提是显存峰值不在optimizer states侧。取值建议从0.1~0.3起步，结合profiling观察碎参数块的host占比调整。

`AdamWSwap`已注册到优化器工厂（OptimizerFactory），其余优化器参数（lr、betas、weight_decay等）与标准AdamW一致。参考示例：`examples/deepseekv4/config.yaml`。

## 与FSDP2 CPU Offload的区别

| 特性 | Swap Optimizer | FSDP2 `cpu_offload` |
|---|---|---|
| 卸载对象 | 仅优化器状态 | 参数+梯度+优化器状态 |
| 卸载粒度 | 按batch分批回装 | 整体卸载/回装 |
| 通信开销 | 较低（仅state） | 较高（参数+梯度） |
| 适用场景 | 优化器状态占显存主导 | 整体显存紧张 |

两者可组合使用，但需注意叠加后的数据搬运开销。

## 适用场景

- 优化器状态占用显存比例高（大模型+AdamW）
- 参数已通过FSDP2分片，但优化器状态仍占大量显存
