# 断点续训

FSDPTurbo支持训练过程中定期保存权重与优化器状态，并从指定断点恢复训练。

## 配置方法

在`checkpoint`段配置权重保存与断点恢复：

```yaml
checkpoint:
  output_dir: "./output/qwen3_5_27b"
  save_steps: 500            # 每 500 步保存一次
  strict: true
  save_optim: true           # 保存优化器状态
  load_optim: true           # 恢复时加载优化器状态
  resume_from_checkpoint: null   # 断点续训时填入 checkpoint 路径
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `output_dir` | 权重保存目录 |
| `save_steps` | 保存间隔（步） |
| `strict` | 加载权重时是否严格匹配参数名 |
| `save_optim` / `load_optim` | 是否保存/加载优化器状态 |
| `resume_from_checkpoint` | 断点续训路径，设置为`output_dir`下对应checkpoint目录即可恢复训练 |

## 保存格式

FSDPTurbo使用PyTorch分布式Checkpoint（`torch.distributed.checkpoint`，简称DCP）格式保存权重，每个rank写入自己的参数分片。当DCP不可用时，自动回退到rank-0的`torch.save`/`torch.load`。

保存的状态包括：

- 模型参数（分片）
- 优化器状态（分片，由`save_optim`控制）
- 学习率调度器状态
- 训练步数（step / epoch）

## 断点续训流程

1. 将`resume_from_checkpoint`指向`output_dir`下的某个checkpoint子目录。
2. 训练启动时，`BaseTrainer`自动调用DCP加载，每个rank读取自己的分片。
3. 加载完成后从保存的step继续训练，学习率与优化器状态同步恢复。

## 注意事项

- **跨并行配置加载**：DCP支持不同并行度下的checkpoint加载（参数会重新分片），但需确保模型结构一致。建议保持并行配置相同以避免兼容性问题。
- **多机共享存储**：多机训练时，`output_dir`必须位于所有节点均可访问的共享存储上（如NFS、NAS），否则各rank无法协同读写。
- **存储空间**：DCP每个rank保存一个分片文件，总大小约等于模型 + 优化器状态，需预留足够磁盘空间。
