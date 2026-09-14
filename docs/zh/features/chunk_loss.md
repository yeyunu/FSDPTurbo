# 分块损失计算

分块损失（Chunk Loss）将损失计算分块流水处理，避免大词表场景下logits全量物化带来的显存峰值。

## 功能原理

大词表模型（如词表150k+）中，`lm_head`的输出logits形状为`[batch, seq_len, vocab_size]`，全量物化会占用大量显存。Chunk Loss将hidden states沿序列维按`chunk_size`切分，逐块通过`lm_head`计算logits与损失，利用`torch.func.grad_and_value`逐块计算梯度并累加，最后聚合为总损失。

## 配置方法

当前Chunk Loss通过模型属性`enable_chunk_loss`控制开启，尚未纳入`config.yaml`统一配置。开启方式：

```python
# 在模型定义中设置
model.enable_chunk_loss = True
```

开启后，模型forward内部会强制`output_hidden_states=True`，并将最后一层hidden states、`lm_head`权重、损失计算函数与`chunk_size`传入`chunk_loss()`完成分块计算。

## 与上下文并行（CP）的关系

Chunk Loss与CP**不兼容，不能同时开启**。开启Ulysses CP后若同时设置`enable_chunk_loss`，模型会直接报错（见`fsdp_turbo/models/kimi/kimi_k3_cp.py`）：

```text
ValueError: Kimi K3 chunk loss is not supported together with Ulysses CP.
```

两者虽然都在大词表 / 长序列场景下缓解显存压力，但路径相互冲突：

- CP的`loss_function_patches`负责全局label shift对齐与跨rank loss规约，仍会在本rank物化完整的local logits。
- Chunk Loss通过逐块计算规避全量logits物化，需要接管完整的hidden states → lm_head → loss计算流程。
- 二者接管损失计算的入口互斥，当前实现不支持组合使用。

## 适用场景

- 大词表模型（vocab_size > 100k）
- 超长序列训练（seq_len > 32k）
- 显存紧张、无法承受全量logits物化的场景
