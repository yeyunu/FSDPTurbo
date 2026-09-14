# 张量并行

FSDPTurbo支持基于DTensor的张量并行（TP），将单层内的权重矩阵切分到多个设备，可与其他并行策略组合使用。

## 配置方法

在`distributed`段配置张量并行度与切分模块：

```yaml
distributed:
  tensor_parallel_size: 4
  tp_plan:
    colwise_parallel: ['*.q_proj', '*.k_proj', '*.v_proj']   # 列并行模块
    rowwise_parallel: ['*.o_proj']                            # 行并行模块
    sequence_parallel: []                                     # 序列并行模块
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `tensor_parallel_size` | 张量并行度 |
| `colwise_parallel` | 列并行模块列表，权重按输出维切分，支持`*.q_proj`通配符 |
| `rowwise_parallel` | 行并行模块列表，权重按输入维切分 |
| `sequence_parallel` | 序列并行模块列表，配合TP降低激活显存 |

## 与其他并行的组合

- TP与FSDP2可自由组合：`tensor_parallel_size × fully_shard_parallel_size ≤ 单节点卡数`。
- TP在FSDP2之前应用：先对模块做TP切分，再对整体做FSDP2全分片。

## 注意事项

- **梯度同步**：TP切分参数由FSDP2统一管理梯度规约（TP列并行/行并行模块不再排除在FSDP2分片之外），开启TP与关闭TP精度可对齐；梯度范数计算同时识别`Shard`与`_StridedShard`放置方式。
- **并行度约束**：`tensor_parallel_size`需能整除注意力头数与隐藏维度，否则列并行/行并行切分会失败。
- **推荐配置**：单节点内优先使用FSDP2 + TP组合（如8卡：`fully_shard_parallel_size=4, tensor_parallel_size=2`），TP不宜过大，建议 ≤ 8。
