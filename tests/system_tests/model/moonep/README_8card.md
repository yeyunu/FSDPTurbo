# MoonEP 单机 8 卡验证用例（精度 / 显存 / 性能）

本文件汇总 MoonEP 在**单机 8 卡**上的三类验证用例：精度、显存、性能。
全部为真机（NPU）用例，本地只准备脚本与配置，不推送远端。先在装好
`npu_patch` 的昇腾机器上 `source` CANN 环境再跑。

缺少 `npu_patch` 时，所有脚本都会打印 `SKIP` 并以 0 退出，不会误判失败。

## 1. 精度：逐张量 / 绝对参考

入口：[`run_test_moonep_8card.sh`](run_test_moonep_8card.sh)，串起本目录已有的两支用例，钉死三种 8 卡专家布局：

| 臂 | 布局 | 脚本 | 验证点 |
|---|---|---|---|
| A | EP=8（4 专家/卡，E=32） | `test_moonep.py` | dispatch / 预取 / 专家 GEMM / combine / home-rank 归约，逐张量对「每卡持全部专家的朴素 MoE」 |
| B | EDP=2 × EP=4 | `test_moonep_edp_ref.py` | 在 A 的基础上额外验证 EDP>1 的跨副本梯度 SUM，对世界求和的绝对参考 |
| C | EDP=4 × EP=2 | `test_moonep_edp_ref.py` | 同 B，另一种 EDP/EP 切分 |

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash tests/system_tests/model/moonep/run_test_moonep_8card.sh
```

容差来自实测（见 [`README.md`](README.md)）：`OUTPUT_TOLERANCE=4e-3`、`GRADIENT_TOLERANCE=2e-2`。
任一臂数值超差即整体非零退出。单独跑某一臂可直接调用对应 `run_test_moonep*.sh`，
用其自身的环境变量（`NUM_NPUS_PER_NODE`、`MOONEP_EDP`、`MOONEP_EP`、`MOONEP_NUM_EXPERTS` 等）。

## 2. 显存 + 性能：moonep vs all2all 的 Qwen3 A/B

[`../qwen3/test_qwen3.py`](../qwen3/test_qwen3.py) 已内建可切换 dispatcher：默认
`EP_DISPATCHER=all2all`（复现 main 基线），置为 `moonep` 即切到 MoonEP，其余一字不改。
它在 8 卡上跑 Qwen3-30B-A3B（FSP=4, TP=2, EP=8, EFSDP=1），每步打印
`step time / loss / grad norm / memory / max memory`，天然就是一组 moonep-vs-all2all 的 A/B：

```bash
# 基线（all2all）
EP_DISPATCHER=all2all SHUFFLE=0 bash tests/system_tests/model/qwen3/run_test_qwen3.sh
# MoonEP，其它一致
EP_DISPATCHER=moonep  SHUFFLE=0 bash tests/system_tests/model/qwen3/run_test_qwen3.sh
```

比较两次日志的 `step time` 中位数与 `max memory` 即可。`SHUFFLE=0` 固定取数顺序，
让两次的 loss 可比。模型与数据走 CI 的标准挂载（`/home/dataset/Qwen3-30B-A3B`、
`/home/dataset/wikitext`，见 [`../../README.md`](../../README.md)），不依赖任何个人产物。

Kimi 整网不适合做这组对比：它的 SiTU 要对完整的 gate/up 投影做变换，通用 `all2all`
的 GLU 路径吃不下这种融合激活，Kimi 的标准 EP 路径本就是 `custom_ep_forward_eager`，
两臂差的就不只是 dispatcher 了。Qwen3 的通用 SwiGLU 专家没有这个问题。

## 前置条件与注意

- 先 `source` CANN：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- `npu_patch` 需本地构建或打进镜像，安装方式见
  [`fsdp_turbo/distributed/expert_parallel/moonep/README.md`](../../../../fsdp_turbo/distributed/expert_parallel/moonep/README.md)。
- MoonEP 要求 BF16 专家权重，A/B 两臂的量化都保持关闭。
- 以上均为真机用例；本地只做了代码与脚本准备，**不推送远端**。
