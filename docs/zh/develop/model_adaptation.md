# 模型适配指南

本文档说明如何将一个新的HuggingFace Transformers模型接入FSDPTurbo进行分布式训练。可参考`examples/`下的现有示例（如`qwen3_5`）。

## 适配流程总览

1. 准备示例目录：`config.yaml` + `train.py` + `run.sh`
2. 编写训练入口`train.py`：构建模型 → 应用FSDPTurbo → 启动Trainer
3. 编写训练配置`config.yaml`：按需配置并行计划（fsdp_plan/tp_plan/cp_plan/ep_plan）
4. 编写启动脚本`run.sh`：torchrun拉起

## 编写训练入口

使用Transformers原生方式加载模型，随后通过`FSDPTurbo(config, model)`应用并行化：

```python
import torch
from transformers import AutoConfig, AutoModel

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer


class MyTrainer(BaseTrainer):
    def build_model(self):
        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
        )
        # 一行代码应用并行化
        model = FSDPTurbo(self.config, model)
        return model


def main():
    config = load_config_from_yaml("config.yaml")
    trainer = MyTrainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
```

## 配置并行计划

### 确定模块名

通过`model.named_modules()`查看模型的模块路径，FSDPTurbo的各类plan均基于模块名匹配，支持通配符：

- `{*}`：匹配单层下标，如`model.layers.{*}`
- `*`：匹配任意前缀，如`*.q_proj`

### FSDP2分片计划（必配）

在`distributed`段配置FSDP2分片计划，指定参与分片的模块与精度：

```yaml
distributed:
  fully_shard_parallel_size: 8   # 建议等于单节点卡数
  fsdp_plan:
    apply_modules:               # 需要 FSDP2 分片的模块
      model.layers.{*}: {}
      model.embed_tokens: {}
      lm_head: {}
    param_dtype: "bf16"
    reduce_dtype: "fp32"
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
```

### 张量并行 / 上下文并行（可选）

按需在`distributed`段配置张量并行与上下文并行：

```yaml
distributed:
  tensor_parallel_size: 2
  tp_plan:
    colwise_parallel: ['*.q_proj', '*.k_proj', '*.v_proj']
    rowwise_parallel: ['*.o_proj']

  ulysses_parallel_size: 2
  cp_plan:
    ulysses_function_patches:
      - target_functions:
          - transformers.models.xxx.modeling_xxx.eager_attention_forward
        type: full_attention
    loss_function_patches:
      - target_functions:
          - transformers.loss.loss_utils.ForCausalLMLoss
        type: causal_lm_loss
```

> [!NOTE]
>
> Ulysses CP可能需要通过`module_patches`替换模型forward（参考`examples/qwen3_5/config.yaml`）：

```yaml
module_patches:
  - target: transformers.models.xxx.modeling_xxx.XxxModel.forward
    replacement: fsdp_turbo.models.qwen.qwen3_5.qwen3_5_model_forward
```

### 专家并行（MoE模型，可选）

MoE模型可按需开启专家并行：

```yaml
distributed:
  expert_parallel_size: 8
  ep_plan:
    apply_modules:
      - model.layers.{*}.mlp.experts
    dispatcher: "fused"        # eager / fused / mc2 / domino
    apply_efsdp_modules:
      - model.layers.{*}.mlp.experts
```

> [!NOTE]
> 
> EP路径要求experts模块持有3D张量参数（`gate_up_proj`、`down_proj`），而非ModuleList形式的独立专家子模块。若模型专家计算语义非标准，可使用`dispatcher='custom_ep_forward_eager'` / `'custom_ep_forward_fused'`，并在专家模块上自行实现：
>
> ```python
> ep_forward_eager(hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router)
> ep_forward_fused(hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router)
> ```

## 编写启动脚本

使用torchrun拉起分布式训练：

```bash
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config config.yaml
```

## 验证建议

- 先用`--num-hidden-layers`（若模型支持）裁剪层数，在小规模并行配置下验证前反向打通。
- 对比单卡/单机与多机配置的loss曲线，确认并行引入无精度异常。
