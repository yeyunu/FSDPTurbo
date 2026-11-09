# MoonEP 系统测试

本目录是 MoonEP 的端到端数值门禁，不是整网训练看护。CI 用 globstar 扫描
`tests/system_tests/model/**/run_test_*.sh`，放在子目录后会自动收录。
**不提交** `tests/system_tests/baseline_results/run_test_moonep.json`：
main 的 CI 在无基线 JSON 时只做功能测试并 PASS，本用例比的是逐张量误差，
不走 loss / grad norm / 时间 / 显存那套基线比对。

## 比的是什么

`test_moonep.py` 把 MoonEP 的前向输出和全部梯度，同一个「每个 rank 持有全部
专家、朴素计算 MoE」的参考逐张量对比（相对 L2）。五项检查：

| 名称 | 验证的环节 |
|---|---|
| `output` | dispatch、预取、专家 GEMM、combine |
| `grad(hidden)` | 反向 dgrad |
| `grad(router probs)` | 路由权重反向 |
| `grad(gate_up_proj)` / `grad(down_proj)` | 本 rank 专家切片；参考梯度先在 EP 组 `all_reduce` 再比，专门验 home-rank 归约 |

默认形状与步数由环境变量给出，见 `run_test_moonep.sh`。

## 为什么不比 loss

home-rank 归约曾经读了对端还没写的槽位：当时 `output`、hidden 梯度、router
梯度都仍在 4.5e-4 / 4.9e-3 / 3.6e-3，只有两个专家权重梯度到 4.7e-1
（`test_moonep.py` 文件头有这组数字）。只看 loss 过不了这类静默错误。
Adam 也会吸收专家梯度上的常数缩放（漏 `gradient_divide_factor` 时 loss
曲线仍可看起来正常），所以本用例必须直接比张量。

## 容差从哪来

默认值在 `test_moonep.py` 里，可用 `MOONEP_OUTPUT_TOLERANCE` / `MOONEP_GRADIENT_TOLERANCE` 覆盖：

- `OUTPUT_TOLERANCE = 4e-3`
- `GRADIENT_TOLERANCE = 2e-2`

容差只需把「正确」和「损坏」分开，而这两者相距很远：正确时梯度误差约 6e-3，
归约读到对端未写入的槽位那次是 4.7e-1（见 `test_moonep.py` 文件头），相距约 77×。
当前取值落在两者之间、留约 4× 余量，不是贴着某一次测量定的。

**不要放宽这两个常数。** 放宽到 4.7e-1 一侧就等于把那类静默归约错误放回门禁之外。

## 怎么跑

需要 NPU/GPU 与已安装的 `npu_patch`，安装方式见
[`fsdp_turbo/distributed/expert_parallel/moonep/README.md`](../../../../fsdp_turbo/distributed/expert_parallel/moonep/README.md)。
没有算子库时 `run_test_moonep.sh` SKIP 并 `exit 0`。

```bash
# 昇腾
NUM_NPUS_PER_NODE=4 bash tests/system_tests/model/moonep/run_test_moonep.sh
# 或直接
torchrun --nproc_per_node=4 tests/system_tests/model/moonep/test_moonep.py
```

不需要真机的部分：

```bash
pytest tests/unit_tests/distributed/expert_parallel/moonep -q
pytest tests/unit_tests/ops/test_moonep_grouped_matmul.py -q
```
