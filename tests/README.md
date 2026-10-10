# FSDPTurbo 测试用例贡献说明

FSDPTurbo 测试用例用于看护仓库的多维并行（FSDP/TP/EP/CP）、显存优化与量化加速等核心能力。测试体系包含 UT（单元测试）与 ST（系统测试）两级：UT 在 CPU 上以 pytest 运行，看护模块级功能正确性；ST 在 8 NPU 上执行真实训练循环，通过「跑训练 → 解析日志为 JSON → 与基线比对」的机制对精度（loss / grad norm）、性能（单步时间）、显存做回归看护。两级用例均由 CI 门禁脚本 `ci/MindSpeed_FSDPTurbo_ut.sh` 统一拉起：先执行 UT，再扫描执行 `tests/system_tests/model/**/run_test_*.sh`。

## CI 门禁看护列表

CI 门禁用例看护仓库重点模型和基础特性，覆盖冒烟测试场景，PR 合入前须通过相关 CI 门禁用例测试。

| Tests | Module | Features | Scripts | Acc. | Throu. | Mem. |
|---|---|---|---|---|---|---|
| ST | Qwen3 | Qwen3-30B-A3B 全层真实权重, FSP=4, TP=2, EP=8, EFP=1 | [run_test_qwen3.sh](system_tests/model/qwen3/run_test_qwen3.sh) | Y | Y | Y |
| ST | Qwen3.5 | Qwen3.5-27B 减层冒烟, FSP=4, Ulysses CP=2 | [run_test_qwen3_5.sh](system_tests/model/qwen3.5/run_test_qwen3_5.sh) | Y | Y | Y |
| ST | Qwen3.5-MoE | Qwen3.5-35B-A3B 减层冒烟, FSP=4, Ulysses CP=2, EP=8, EFP=1 | [run_test_qwen3_5_moe.sh](system_tests/model/qwen3.5_moe/run_test_qwen3_5_moe.sh) | Y | Y | Y |
| ST | Kimi-K3 | Kimi-K3 减层冒烟, FSP=8, EP=8, EFP=1 | [run_test_kimi_k3.sh](system_tests/model/kimi_k3/run_test_kimi_k3.sh) | Y | Y | Y |
| ST | DeepSeekV4 | DeepSeekV4-4L 减层冒烟, FSP=8, recompute / chunk_batch / chunk_loss / swap_activation / swap_optimizer 特性组合 | [run_test_deepseekv4.sh](system_tests/model/deepseekv4/run_test_deepseekv4.sh) | Y | Y | Y |
| UT | distributed | 上下文并行（Ulysses / KCP / KDA）、EP dispatcher 量化 | [test_context_parallel.py](unit_tests/distributed/context_parallel/test_context_parallel.py) [test_kda_context_parallel.py](unit_tests/distributed/context_parallel/test_kda_context_parallel.py) [test_dispatcher_quant.py](unit_tests/distributed/expert_parallel/test_dispatcher_quant.py) | Y | | |
| UT | memory | 激活重计算 | [test_recompute.py](unit_tests/memory/test_recompute.py) | Y | | |
| UT | ops | causal_conv1d / 反量化 / FLA / KDA / permute 等算子 | [test_causal_conv1d.py](unit_tests/ops/test_causal_conv1d.py) [test_dequant.py](unit_tests/ops/test_dequant.py) [test_fla.py](unit_tests/ops/test_fla.py) [test_kda.py](unit_tests/ops/test_kda.py) [test_npu_permute.py](unit_tests/ops/test_npu_permute.py) | Y | | |
| UT | quantization | MXFP8 量化转换 / GMM 量化 / 重计算相位 | [test_mx_convert.py](unit_tests/quantization/test_mx_convert.py) [test_mx_gmm_quant.py](unit_tests/quantization/test_mx_gmm_quant.py) [test_mx_grouped_matmul.py](unit_tests/quantization/test_mx_grouped_matmul.py) [test_mx_quant.py](unit_tests/quantization/test_mx_quant.py) [test_mxfp8_recompute.py](unit_tests/quantization/test_mxfp8_recompute.py) [test_recompute_phase.py](unit_tests/quantization/test_recompute_phase.py) | Y | | |
| UT | training | dataloader / trainer | [test_dataloader.py](unit_tests/training/test_dataloader.py) [test_trainer.py](unit_tests/training/test_trainer.py) | Y | | |
| UT | utils | module patch | [test_patch.py](unit_tests/utils/test_patch.py) | Y | | |

## 开发流程

### 1. 本地验证

- UT（CPU）：`python -m pytest tests/unit_tests -v -x -ra --timeout=300`
- ST（8 NPU）：`bash tests/system_tests/model/<name>/run_test_<name>.sh`；生成/更新基线推荐多次运行后用 `test_tools/acquire_json.py` 合成中位数基线（抗 run-to-run 抖动）。

用例编写后，先确保用例在本地或 CI 备用环境运行无误，再生成基线数据。ST 用例需将脚本和同名基线文件一同上仓。

### 2. 用例登记

测试用例信息登记。为了方便后续维护，需要对用例的作者、上仓日期、简要描述以及其他信息进行标注。

ST 用例需在运行脚本开始时标注以下信息：

```bash
#=============================================
# Author: xxx
# Date: xxxx-xx-xx
# Description: Model or feature covered by the testcase
# Remarks: Instructions for the checkpoint, datasets and tokenizer or other more information
#=============================================
```

UT 用例在用例执行函数内标注以下信息：

```python
def test_feature_a():
    '''
    Author: xxx
    Date: xxxx-xx-xx
    Description: Model or feature covered by the testcase
    Remarks: Instructions for the checkpoint, datasets and tokenizer or other more information
    '''
    ...
```

同时在本文件「CI 门禁看护列表」中登记测试用例所看护的模型和特性信息，并结合最终校验目标，在精度（Acc.）、性能（Throu.）、显存（Mem.）列中填写 `Y`，无校验项保留空白即可。

### 3. 资源登记

如新增用例依赖权重、词表或数据集（如 Qwen3 全层真实权重用例依赖 `MODEL_PATH` / `DATASET_PATH`），请同步登记资源路径和用途，确保 CI 环境可复现。注意：

- 请尽量复用已有权重和数据；如需新增，请优先使用最小可验证规模（减层、随机初始化冒烟用例优先）。
- 模型名称需与 HuggingFace 或仓库 `examples/` 命名保持一致，严禁省略或自定义导致后续维护困难。
- 临时缓存、日志和中间文件应写入 `ST_LOG_DIR`（默认 `run_logs`）约定目录，并在用例执行结束后及时清理。

### 4. 用例上仓

新增模型、并行特性、显存优化或公共工具能力时，应同步补充看护用例。只有业务代码而无对应看护用例的 PR，需要在 PR 说明中解释原因。

## 开发规则

测试用例全部放置在 `tests` 目录下，当前层级如下：

- `tests/unit_tests/`：CI 门禁会拉起的 UT 用例（CPU，pytest），按功能模块分目录。
- `tests/system_tests/model/`：ST 训练脚本与用例，按模型分目录。
- `tests/system_tests/baseline_results/`：ST 基线 JSON，与脚本同名。
- `tests/system_tests/test_tools/`：测试公共工具（日志解析 `acquire_json.py`、基线比对 `compare_baseline.py`）。

### ST

① 贡献脚本用例请放置于 `tests/system_tests/model/{模型系列}/` 目录下，命名规则为 **run_test_{模型名}_{任务类型}_{关键并行策略}**，如 `run_test_qwen3_5_moe.sh`，请保持命名清晰且可追溯。

② ST 脚本由 CI 门禁统一扫描执行（内部 `torchrun` 起 8 NPU），脚本中不需要单独重定向日志；训练每步须在 rank0 打印汇总行，供 `acquire_json.py` 解析：

```
[Rank 0] step N step time: X.XXXXs loss: X.XXXX grad norm: X.XXXX memory: X.XXXX GB max memory: X.XXXX GB
```

③ 基线数据请放置于 `tests/system_tests/baseline_results/` 目录下，**命名需与 shell 脚本文件名完全对齐**，仅后缀由 `.sh` 替换为 `.json`，否则自动化脚本无法匹配。无基线的用例仅做功能看护（训练跑通即 PASS），提交基线 JSON（哪怕空列表）即激活精度/性能/显存门禁。

④ ST 基线比对当前支持 `lm loss`、`grad norm`、`time info`、`memo info` 等指标，默认容差为 loss rel 2%、grad norm rel 10%、time 中位数 rel 5%、显存 rel 10%，并支持 per-case 覆盖（`<metric> margin` / `<metric> abs tol`）。

⑤ 详细执行流程、判定规则、工具 CLI 与环境变量（`NUM_STEPS`、`ST_LOG_DIR`、`CI_CONTINUE_ON_FAILURE`、`START_COVERAGE` 等）参见 [system_tests/README.md](system_tests/README.md)。

### UT

① UT 在 CPU 上运行，不依赖 NPU 环境与真实权重；分布式、硬件相关逻辑通过 `unittest.mock` / `mock.patch` 构造。

② 按照功能模块进行目录组织，至多不超过两层目录，所有用例文件和测试函数均以 `test` 作为命名前缀。

③ 新增用例可以在原有用例基础上补充 `test_xxx` 方法，尽量保证测试目标聚焦、依赖清晰、执行时间可控。

④ 对于需要参数化配置的用例，建议通过 `@pytest.mark.parametrize` 传入参数并构造用例；如配套 JSON 配置文件，请保证 JSON 中的 key 值命名与测试函数或参数含义保持一致。
