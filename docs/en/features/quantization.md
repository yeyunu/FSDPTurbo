# Quantization Training

FSDPTurbo supports MXFP8 quantization training, reducing memory usage and improving training throughput through low-precision computation and communication.

## Configuration

Configure the quantization plan in the `quantization` section:

```yaml
quantization:
  quantization_plan:
    quant_format: "E4M3"        # E4M3/E5M2/HIF8
    block_size: 32
    quant_apply_modules:        # Modules to which quantization applies
      - "*.linear"
```

Corresponding Python API:

```python
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, QuantizationConfig, QuantizeConfig

config = FSDPTurboConfig(
    quantization=QuantizationConfig(
        quantization_plan=QuantizeConfig(
            quant_format='E4M3',   # or 'E5M2'/'HIF8'
            block_size=32,
        ),
    ),
)
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `quant_format` | Quantization format: `E4M3`/`E5M2`/`HIF8` (Huawei hifloat8) |
| `quant_recipe` | Quantization strategy |
| `block_size` | Block size of block-wise quantization, enabling fine-grained scaling control |
| `quant_apply_modules`/`quant_ignored_modules` | Modules to which quantization applies/modules excluded from quantization |
| `converters` | List of model converters, used to deploy quantization in the model |
| `enable_fsdp_low_precision_all_gather` | Switch for low-precision FSDP2 communication |
| `fsdp_low_precision_all_gather_mode` | Low-precision AllGather mode (default `on-demand`) |
| `quant_gmm` | Quantized grouped GEMM (works with expert parallelism on the EP path) |
| `gemm_gradient_accumulation_fusion` | GEMM gradient accumulation fusion |

## Notes

- **Hardware requirements**: `E4M3`/`E5M2` require hardware support for FP8 computation; `HIF8` (Huawei hifloat8) is an FP8 variant unique to the Ascend NPU and is available only on Ascend NPUs. Using them on hardware that does not support the corresponding format raises an error or falls back.
- **Combining with recomputation**: Quantization and recomputation can be enabled at the same time; operators in the recomputation region still execute at quantized precision, so pay attention to the risk of numerical overflow.
- **Combining with EP**: In MoE models, `quant_gmm: true` can be used to enable quantized grouped GEMM so that expert computation also executes at quantized precision on the EP path.
- **Precision alignment suggestion**: When enabling quantization for the first time, it is recommended to first compare the loss curve with a bf16 baseline on a small dataset, and start large-scale training only after confirming that the precision degradation is within an acceptable range.
