# TileKernels

## 基本介绍

TileKernels是DeepSeek开源的高性能算子库，使用TileLang等技术实现计算内核，并提供Python调用接口。FSDPTurbo将其作为可选算子后端，用于替换模型中对应的计算，同时保留原有实现。

FSDPTurbo使用 `model` 下的 `tilekernels` 配置组，组内直接放算子布尔开关，当前只有 `tilekernels_mhc_post`，默认关闭。没有额外总开关，不依赖量化配置，也不自动修改未接入的模型。

<br />

## 环境配置

使用TileKernels需要额外安装TileKernels和tilelang：

- 安装tilelang
  ```
  git clone --recursive https://github.com/tile-ai/tilelang.git
  cd tilelang
  USE_CUDA=OFF USE_ROCM=OFF USE_ASCEND=ON python -m pip install -v .
  ```
- 安装tilekernels
  ```
  git clone --recursive https://github.com/tile-ai/tilekernels.git
  cd tilekernels
  # python>=3.12
  python -m pip install -v -e .

  # python<3.12
  python -m pip install --ignore-requires-python -v -e .
  ```

## 通用接入规范

新增算子按下表接入。`<算子名>` 仅为命名示意，不提前添加尚未实现的开关。

| 步骤      | 修改位置                                                                                   | 要求                                                                                                                                                |
| ------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 声明参数    | `fsdp_turbo/fsdp_turbo_config.py` 的 `ModelConfig.tilekernels` 字典 | 增加默认false的 `tilekernels_<算子名>` 键，保持配置组 → 算子开关两层，不增加enable或独立配置类。 |
| 校验和加载   | `FSDPTurboConfig.validate_model_config` | 将新键加入允许列表，严格校验bool，由 `__post_init__` 调用。现有YAML加载器直接保留字典；未知键报错，缺失键在组件构造时读取为false。 |
| 示例配置    | 使用该算子的 `examples/<示例>/config*.yaml`                                                    | 在 `model.tilekernels` 下增加默认关闭的算子开关，保持原训练参数和原后端开关。                                                                                                 |
| 参数传递    | 示例 `train.py` → 模型builder → `ModelArgs.tilekernels` | 与 `use_fused_mhc` 等已有参数共用构造链，只传配置组；不在FSDPTurbo中增加专用应用方法，也不为每个算子扩展Transformer签名。 |
| 算子调用点接线 | 对应模型的算子所属组件 | 组件构造时读取算子布尔值，调用时直接选择ops语义函数。关闭时保留原分支，不增加配置绑定子模块、模型遍历或运行期方法替换。 |
| 算子封装    | `ops/` 语义入口、`ops/npu/` 实现、`ops/cpu/` 参考                                                | 使用独立registry名称并在两端 `__init__.py` 注册；调用方只调用语义包装。普通算子可集中于 `ops/tilekernels.py`，量化入口遵循仓库约定放在 `ops/quantize.py`。第三方内核调用隔离在设备实现中并延迟导入。               |
| 自动反向    | 设备侧adapter                                                                            | 核对上游是否已提供autograd。若只提供独立前后向kernel，用 `torch.autograd.Function` 关联，保存所需状态并返回全部可微输入梯度；不要重复包装已有autograd。                                           |
| 验证和文档   | 对应测试、实现说明及本页第3节                                                                      | 检查开关组合、输出和全部梯度、dtype/布局、完整前反向性能及模型调用链。补充适用模型、执行契约、已测batch/seqlength和收益；新实现说明登记中文菜单。                                                             |

后端选择只依据开关，不将历史测试shape写成分流规则。设备和输入合法性检查可以报错，但不得把新后端的失败静默换成原实现。拷贝、精度转换、保存状态和反向都应计入对应测量范围；单算子通过不等于整模型训练已验收。

各算子独立控制，组件构造时保存布尔值，执行期间保持不变。不使用进程级可变开关或临时forward上下文；修改配置后重新构造模型，使forward、backward和重计算使用同一选择。

## 算子使用指导

后续新增算子在本节末尾追加独立小节，沿用“配置、执行契约、性能建议”的结构。

### mHC post

#### 配置

在DeepSeek-V4.1的 `examples/deepseek_v41/config.yaml`、`config_A5.yaml` 或使用中的 `config_tnd.yaml` 中设置：

```yaml
model:
  tilekernels:
    tilekernels_mhc_post: true
```

开启时所有mHC post调用使用TileKernels，不按shape分流。关闭时由 `model.use_fused_mhc` 决定原CANN/torch post分支；pre始终由 `use_fused_mhc` 控制。

`ModelConfig.tilekernels` 直接保存字典，每个配置实例拥有独立默认值。省略节点或填写 `{}` 时默认关闭；null、非字典节点、未知键和非布尔开关值均报错。校验统一在 `FSDPTurboConfig.__post_init__` 调用的 `validate_model_config` 中完成，无专用配置类或规范化流程。

Python中使用 `ModelConfig(tilekernels={"tilekernels_mhc_post": True})`，交由 `FSDPTurboConfig(model=...)` 校验；示例向 `build_deepseek_v41_model(tilekernels=config.model.tilekernels, ...)` 传入配置组。builder拷贝字典到ModelArgs，Block保存读取出的布尔值，单独构造模型也能生效，无需FSDPTurbo后处理。builder未提供tilekernels时保留传入ModelArgs的设置，显式 `{}` 则全部关闭。

旧的 `enable` / `use_tilekernels_mhc_post` 双开关写法不再支持，迁移为上述单开关：新值取旧两项的逻辑与。旧键会明确报错，避免静默改变选择。

当前DeepSeek-V4.1原生KV-AllGather CP要求融合算子关闭，TileKernels post同样受该校验约束。TND的post按token独立计算，接口语义兼容，但packed训练未验证；原有fused compressor/indexer的packed限制保持不变。

#### 算子输入

NPU路径要求Ascend 950，输入如下：

| 输入          | 形状                                   | 精度       |
| ----------- | ------------------------------------ | -------- |
| 子层输出x      | `[batch, seqlength, hidden_size]`    | BF16     |
| 残差residual | `[batch, seqlength, 4, hidden_size]` | BF16     |
| 扩展权重post   | `[batch, seqlength, 4]`              | 入口转FP32 |
| 混合权重comb   | `[batch, seqlength, 4, 4]`           | 入口转FP32 |

输入在同一设备，adapter负责连续化并保留转换的梯度。反向由autograd自动触发，返回四个输入的梯度，仅支持一阶。CPU注册项为torch参考，不代表运行TK kernel。

#### 性能建议

以下是A5上的历史完整前向加反向结果。实测使用上述4路残差、连续BF16激活和FP32权重；加速比为“CANN耗时 / TK耗时”。

| batch | seqlength | hidden\_size | 加速比    | 建议               |
| ----- | --------- | ------------ | ------ | ---------------- |
| 1     | 8192      | 4096         | 1.818× | 有收益，建议在目标环境验证后开启 |
| 1     | 8192      | 5120         | 1.819× | 有收益，建议在目标环境验证后开启 |
| 1     | 1024      | 4096         | 0.899× | 更慢，建议关闭          |
| 1     | 1024      | 5120         | 0.821× | 更慢，建议关闭          |
| 1     | 4096      | 4096 / 5120  | 1.6x   | 建议开启             |
