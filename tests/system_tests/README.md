# FSDPTurbo 精度/性能基线看护用例说明

本目录维护 FSDPTurbo 的系统测试（ST）基线用例，覆盖 Qwen3.5 dense / MoE 和 Kimi-K3，对**精度（loss / grad norm）、性能（单步时间）、显存**做回归看护。机制对齐 MindSpeed-Bridge 的 ST 模式：跑训练脚本 → 解析日志为 JSON → 与提交的基线 JSON 比对，超差则用例失败。

## 目录结构

```
tests/system_tests/
├── README.md                          # 本文档
├── model/                             # 训练脚本与用例
│   ├── qwen3/                          # Qwen3 全层真实权重用例
│   │   ├── run_test_qwen3.sh
│   │   └── test_qwen3.py
│   ├── qwen3.5/                        # Qwen3.5 dense 减层冒烟用例
│   │   ├── run_test_qwen3_5.sh
│   │   └── test_qwen3_5.py
│   ├── qwen3.5_moe/                    # Qwen3.5-MoE 减层冒烟用例
│   │   ├── run_test_qwen3_5_moe.sh
│   │   └── test_qwen3_5_moe.py
│   ├── kimi_k3/                        # Kimi-K3 减层冒烟用例
│   │   ├── run_test_kimi_k3.sh
│   │   └── test_kimi_k3.py
│   └── deepseekv4/                     # DeepSeekV4 CI 基线（5 个可共存特性同时开启）
│       ├── run_test_deepseekv4.sh      # 启动脚本（torchrun）
│       ├── test_deepseekv4.py          # 训练循环（含 SwapActivation / SwapOptimizer / ChunkLoss）
│       ├── config.yaml                 # 合并所有可共存特性的配置
│       ├── modeling_deepseek_v4.py     # 自定义模型
│       └── tokenizer -> ...            # 分词器（软链接到 examples）
├── test_tools/                        # 看护工具
│   ├── acquire_json.py                # 日志 → JSON 解析器（可命令行直接用）
│   └── compare_baseline.py            # 解析日志 + 与基线比对（CI 调用）
└── baseline_results/                  # 提交的基线 JSON（与脚本同名，.json 后缀）
    ├── run_test_qwen3.json
    ├── run_test_qwen3_5.json
    ├── run_test_qwen3_5_moe.json
    ├── run_test_kimi_k3.json
    └── run_test_deepseekv4.json
```

ST 编排逻辑内联在 `ci/MindSpeed_FSDPTurbo_ut.sh` 的 `run_testcase` 函数里（扫描 `model/**/run_test_*.sh`，逐个跑训练 → 有基线则解析+比对，无基线则只测功能跳过比对）。

## 执行流程

对每个 `run_test_*.sh`：

1. **训练（总是跑）**：`bash -e run_test_*.sh`（内部 `torchrun` 起 8 NPU），stdout+stderr 写入 `run_logs/<name>.log`（当前目录下）。`-e` 保证脚本中间命令失败即向上传播退出码，不会被末尾 `torchrun` 掩盖而误判 PASSED。训练每步在 rank0 打印一行汇总：
   ```
   [Rank 0] step N step time: X.XXXXs loss: X.XXXX grad norm: X.XXXX memory: X.XXXX GB max memory: X.XXXX GB
   ```
   （`acquire_json` 只匹配该行；`[Rank 0] Loss: ...` 等其它行被忽略。loss/grad norm/显存用 `(\S+)` 捕获，可匹配 `nan`/`inf`/`-inf`，避免非有限值被静默丢弃导致各指标列表长度错位。）
2. **功能判定**：训练退出码非 0 → `failed_script_cases`，走 fail-fast（功能失败照常暴露）。
3. **基线存在性检查**：若 `baseline_results/<name>.json` 不存在 → 该用例判 `[PASS-functional]`（训练已过、跳过比对），计入 `skipped_cases`（标 `no baseline (functional only)`），返回 0。即**无基线也跑训练做功能看护**，只是不量精度/性能；提交一个基线 JSON（哪怕空列表）即可激活精度/性能看护。
4. **解析**：有基线时 `compare_baseline.py` 调 `acquire_json.transfer_logs_as_json` 把日志解析为 `<name>.json`（路径由 log 路径派生：`run.log` → `run.json`，落在 log 同目录）。JSON schema：
   ```json
   {"lm loss": [...], "grad norm": [...], "time info": [ms...], "memo info": [{"rank","allocated memory","max allocated memory"}]}
   ```
   （`time info` 由秒×1000 转为 ms；`memo info` 取每个 rank 最后一步的内存快照。）
5. **比对**：与 `baseline_results/<name>.json` 逐指标比对，任一指标超差 → 退出码 1。
6. **汇总**：`run_testcase` 把比对退出码非 0 的用例计入 `failed_compare_cases`，按 `CI_CONTINUE_ON_FAILURE` 决定 fail-fast 或继续；功能-only 的用例计入 `skipped_cases`（不计为失败）。

## 看护指标与判定

| 指标 | 比对方式 | 相对容差（默认） | 绝对容差地板（默认） | 失败条件 |
|---|---|---|---|---|
| `lm loss` | 逐 step 相对偏差 | rel 2% | abs_tol 0.1 | 任一步：`|expected|>0.1` 时偏差 >2%，否则绝对差 >0.1 |
| `grad norm` | 逐 step 相对偏差 | rel 10% | abs_tol 1.0 | 任一步：`|expected|>1.0` 时偏差 >10%，否则绝对差 >1.0 |
| `time info` | 丢前 `WARM_UP=2` 步后取**中位数**，相对偏差 | rel 5% | —（禁用） | 比基线中位数慢 >5%（更快或 ≤5% 慢均过） |
| `memo info` | per-rank `allocated`/`max allocated`，相对偏差 | rel 10% | —（禁用） | 任一 rank 显存超基线 >10% |

> **绝对容差地板（abs_tol）**：相对误差在近零值处无意义（如 grad norm 0.0007↔0.2316 相对差 330×、绝对差仅 0.23）。当 `|expected| <= abs_tol` 时改用绝对差 `|actual - expected| <= abs_tol` 判定，避免近零噪声误判 FAIL；`abs_tol=0` 表示禁用、纯相对。time/显存值不会近零，故默认禁用。真回归（如 loss 10→15、grad 100→200）仍照常 FAIL，不被掩盖。

补充规则：
- **基线列表为空** → 该指标 `SKIP`（不失败）。所以基线未填时门禁不生效，填了才激活。
- **actual 列表为空**（训练没产出该指标）→ `FAIL`。
- **长度不匹配**（baseline 与 actual 步数/rank 数不一致）→ `FAIL`。
- `START_COVERAGE=true` 时 `time info` 整项 `SKIP`（覆盖统计时不量性能）。
- 时间用中位数（而非均值）以抗单步抖动（如某一步因编译/调度异常变慢不会拖垮判定）。

### per-case 容差覆盖

在某个用例的 baseline JSON 里加键即可单独收紧/放宽该用例的容差，无需改代码。相对容差用 `"<metric> margin"`，绝对容差地板用 `"<metric> abs tol"`：

```json
{
    "time info": [12345.6, 7890.1, ...],
    "time info margin": 0.03,
    "memo info margin": 0.05,
    "lm loss margin": 0.01,
    "grad norm margin": 0.05,
    "grad norm abs tol": 2.0,
    "lm loss abs tol": 0.0
}
```

- `<metric> margin`：相对容差（小数，0.05=5%）。
- `<metric> abs tol`：绝对容差地板（`0` 表示禁用、纯相对）。设大可放宽近零判定，设 0 则近零值也走相对误差（更严，易误判）。
- 未写则用上表默认值。

## 现有用例

| 用例 | 脚本 | 模型 | 减层 | 并行配置（8 NPU） | NUM_STEPS |
|---|---|---|---|---|---|
| Qwen3 | `qwen3/run_test_qwen3.sh` / `test_qwen3.py` | Qwen3-30B-A3B | 否（全层真实权重） | FSP=4, TP=2, EP=8, EFP=1 | 10 |
| Qwen3.5 dense | `qwen3.5/run_test_qwen3_5.sh` / `test_qwen3_5.py` | Qwen3.5-27B | 4 | FSP=4, TP=1, ULYSSES=2 | 10 |
| Qwen3.5-MoE | `qwen3.5_moe/run_test_qwen3_5_moe.sh` / `test_qwen3_5_moe.py` | Qwen3.5-35B-A3B | 4 | FSP=4, TP=1, ULYSSES=2, EP=8, EFP=1 | 10 |
| Kimi-K3 | `kimi_k3/run_test_kimi_k3.sh` / `test_kimi_k3.py` | Kimi-K3 | 4 | FSP=8, TP=1, EP=8, EFP=1 | 10 |
| DeepSeekV4 | `deepseekv4/run_test_deepseekv4.sh` / `test_deepseekv4.py` | DeepSeekV4-4L | 4 | FSP=8, TP=1, EP=1 | 10 |

### DeepSeekV4 CI 基线用例

`model/deepseekv4/` 使用一份合并配置同时开启所有可共存的加速特性，作为 CI 冒烟看护基线。无法共存的并行策略（TP、EP、CP、fine_grained_fsdp）因互斥而未纳入。

| 已开启特性 | 配置位置 | 说明 |
|-----------|----------|------|
| recompute | `memory.recompute` | 激活重计算 |
| chunk_batch | `memory.chunk_batch` | 分块微批次 |
| chunk_loss | `features.chunk_loss` | 分块 loss 计算 |
| swap_activation | `features.swap_activation` | 激活换出到 CPU |
| swap_optimizer | `features.swap_optimizer` | 优化器状态换入换出 |

**用法**：

```bash
# 默认 10 步
bash run_test_deepseekv4.sh

# 自定义步数
NUM_STEPS=5 bash run_test_deepseekv4.sh
```

**特性开关**：通过 `config.yaml` 的 `features` 段控制，`test_deepseekv4.py` 在模型构建 / 优化器 / 训练步中按需激活，不修改 FSDPTurbo 核心代码。

**输出格式**：每步 rank0 打印 `[Rank 0] step N step time: X.XXXXs loss: X.XXXX grad norm: X.XXXX memory: X.XXXX GB max memory: X.XXXX GB`，与 `acquire_json.py` 兼容。

**模型规格**：DeepSeekV4-4L（4 层、16 专家、topk=16），meta 设备构建 → FSDPTurbo 包裹 → CPU 物化初始化，`max_seq_length=1024`，`batch_size=4`。

Qwen3.5 dense / MoE 两者均为 **CI 冒烟用例**：仅需 `config.json + tokenizer/processor`（不下载真实权重），用 `from_config` 在 `meta` 上建模型 → `to_empty` 物化 → 通过 `fsdp_turbo.utils.model.init_submodule_weights` 调用 text/visual 子模型的 `_init_weights` 随机初始化 → `tie_weights` → `FSDPTurbo` 包裹。其它默认：`BATCH_SIZE=1`、`MAX_LENGTH=512`、`LR=1e-4`、`recompute=True`、混合精度 `bf16` 参数 / `fp32` 约减。

Qwen3 为**全层真实权重用例**：`AutoModelForCausalLM.from_pretrained` 加载完整 Qwen3-30B-A3B 权重 + wikitext 数据集，不做减层/随机初始化。CI 需在 `MODEL_PATH=/home/dataset/Qwen3-30B-A3B`、`DATASET_PATH=/home/dataset/wikitext` 处提供真实权重与数据，否则该用例在训练阶段即失败（与基线比对无关）。每步同样在 rank0 打印 `[Rank 0] step N step time: ... loss: ... grad norm: ... memory: ... max memory: ... GB` 汇总行，由 `acquire_json` 解析后参与 loss/grad norm/时间/显存看护。

> 随机初始化说明：多模态顶层 `Qwen3_5MoeConfig`（`@strict`）不声明 `initializer_range`，直接 `model._init_weights` 会在专家模块上报错；故通过 `init_submodule_weights` 分别对 `model.model.language_model` 和 `model.model.visual` 调它们各自的 `_init_weights`（其 config 有 `initializer_range`），再 `tie_weights()`。`init_submodule_weights` 会在子模块缺少 `_init_weights`（某些 transformers 版本的多模态子模型未暴露该方法）时回退到按层类型的通用初始化（`Linear`/`Embedding` 用 `initializer_range` 默认 0.02 的正态、`LayerNorm` 用 1/0），避免 `AttributeError`。物化与初始化必须在 `FSDPTurbo` 包裹之前，否则 `to_empty` 会使 FSDP 捕获的参数引用脱钩，导致梯度不回流（`grad_norm=0`）。

## 新增用例

1. 在 `model/<name>/` 下加 `run_test_<name>.sh`（参照现有脚本，命名 `{模型}_{任务}_{关键并行策略}.sh`）与 `test_<name>.py`（训练循环里保证每步 rank0 打印上述汇总行，并 `compute_grad_norm` 算全局 grad norm）。
2. 在 `baseline_results/` 下加**同名** `run_test_<name>.json`（先放空列表，见下）。
3. 基线留空时用例跑通即 PASS（门禁未激活）；从真实 CI 运行填充基线后自动激活。

## 补充/更新基线

用例的基线 JSON 已从真实 8-NPU CI 运行填充（loss/grad norm/时间/显存均有值，门禁已激活）。需要重新生成或更新时，推荐用**多次运行取中位数**，抗 run-to-run 抖动（尤其 step time）：

1. 在 CI/备用环境多次跑：`bash tests/system_tests/model/<name>/run_test_<name>.sh`（每次日志落到 `run_logs/run_test_<name>.log`；多跑时改名保留，如 `.run0/.run1/.run2`）。
2. 用 `acquire_json.py` 把多条日志合成**中位数基线**（loss/grad/time 逐步取中位数、memo per-rank 取中位数）：
   ```bash
   python tests/system_tests/test_tools/acquire_json.py \
       --log run_logs/run_test_qwen3_5.run0.log \
             run_logs/run_test_qwen3_5.run1.log \
             run_logs/run_test_qwen3_5.run2.log \
       --json tests/system_tests/baseline_results/run_test_qwen3_5.json
   ```
   单条日志则逐字写入（等同该次运行结果）。
3. 提交。下次 CI 即对该用例逐 step 比对。

> 基线列表为空的指标会被 `compare_baseline` 跳过（`[SKIP]`，不失败）；填了才激活。所以也可先提交空基线让用例进功能看护，再填充激活精度/性能门禁。

## 工具 CLI

### `compare_baseline.py`（解析 + 比对，CI 调用）
```bash
python tests/system_tests/test_tools/compare_baseline.py \
    --generate-log <run.log> --baseline-json <baseline.json>
```
- 两个参数均必填。JSON 路径由 log 路径自动派生（`run.log` → `run.json`，同目录）。
- 退出码 0 = 通过，1 = 有指标超差。

### `acquire_json.py`（解析日志为 JSON / 合成中位数基线）
```bash
# 单条日志 → 逐字写入
python tests/system_tests/test_tools/acquire_json.py --log <run.log> --json <out.json>

# 多条日志 → 中位数基线（loss/grad/time 逐步中位数、memo per-rank 中位数）
python tests/system_tests/test_tools/acquire_json.py \
    --log run1.log run2.log run3.log --json tests/system_tests/baseline_results/run_test_<name>.json
```
- `--log` 可传一个或多个。单条逐字写；多条取中位数（抗单次抖动，推荐用于基线生成）。
- 不做比对，只生成 JSON。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `NUM_STEPS` | 10 | 训练步数（≥10 才能让时间中位数有意义） |
| `BATCH_SIZE` / `MAX_LENGTH` | 1 / 512 | 数据批大小 / 序列长度 |
| `NUM_HIDDEN_LAYERS` | 4 | 语言模型层数（减层冒烟） |
| `TENSOR_PARALLEL_SIZE` 等 | 见各脚本 | 并行配置 |
| `ST_LOG_DIR` | `run_logs` | 日志与解析 JSON 的输出目录（当前目录下，可覆盖） |
| `CI_CONTINUE_ON_FAILURE` | false | ST 模式：false=fail-fast，true=跑完全部再汇总 |
| `START_COVERAGE` | false | true 时跳过 `time info` 比对（覆盖统计不量性能） |
