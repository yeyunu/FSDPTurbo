# 量化训练

FSDPTurbo支持MXFP8量化训练，通过低精度计算与通信降低显存占用、提升训练吞吐。

## 配置方法

在`quantization`段配置量化计划：

```yaml
quantization:
  quantization_plan:
    quant_format: "E4M3"        # E4M3 / E5M2 / HIF8
    block_size: 32
    quant_apply_modules:        # 应用量化的模块
      - "*.linear"
```

对应Python API：

```python
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, QuantizationConfig, QuantizeConfig

config = FSDPTurboConfig(
    quantization=QuantizationConfig(
        quantization_plan=QuantizeConfig(
            quant_format='E4M3',   # 或 'E5M2'、'HIF8'
            block_size=32,
        ),
    ),
)
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `quant_format` | 量化格式：`E4M3` / `E5M2` / `HIF8`（华为hifloat8） |
| `quant_recipe` | 量化策略 |
| `block_size` | 分块量化块大小，实现细粒度缩放控制 |
| `quant_apply_modules` / `quant_ignored_modules` | 量化作用/排除的模块 |
| `converters` | 模型转换器列表，用于在模型中部署量化 |
| `enable_fsdp_low_precision_all_gather` | FSDP2通信低精度化开关 |
| `fsdp_low_precision_all_gather_mode` | 低精度AllGather模式（默认`on-demand`） |
| `quant_gmm` | 量化分组GEMM（EP路径下与专家并行配合） |
| `gemm_gradient_accumulation_fusion` | GEMM梯度累积融合 |

## 注意事项

- **硬件要求**：`E4M3` / `E5M2`需硬件支持FP8运算；`HIF8`（华为hifloat8）为昇腾NPU特有的FP8变体，仅在昇腾NPU上可用。不支持对应格式的硬件上使用会报错或回退。
- **与重计算的组合**：量化与重计算可同时开启，重计算区域内的算子仍以量化精度执行，需注意数值溢出风险。
- **与EP的组合**：MoE模型中可通过`quant_gmm: true`启用量化分组GEMM，使专家计算在EP路径下也以量化精度执行。
- **精度对齐建议**：首次启用量化时，建议先在小数据集上对比bf16基线的loss曲线，确认精度下降在可接受范围内再大规模训练。
