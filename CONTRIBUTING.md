# 贡献指南

感谢您关注 FSDPTurbo！我们欢迎并感激每一份贡献。本指南将帮助您了解参与项目共建的方式与流程。

## 1. 报告问题

- 如果您发现任何问题，请先查看仓库的 [issues 列表](https://gitcode.com/Ascend/FSDPTurbo/issues)，尝试寻找类似问题或解决方案。

- 如果现有 [issues 列表](https://gitcode.com/Ascend/FSDPTurbo/issues)中没有您遇到的问题，可以[提交一个新的 issue](https://gitcode.com/Ascend/FSDPTurbo/issues/create/choose)，并尽量提供清晰的问题描述、复现步骤与环境信息（如 CANN / PyTorch / torch_npu 版本、并行策略配置、最小可复现脚本等）。

## 2. 性能优化与新功能

- 有关性能优化的提案，请在提交 issue 时使用 `Performance` 标签，并描述性能优化特性、使用场景与预期收益（如吞吐、显存、精度指标）。

- 有关新功能的建议或讨论，请在提交 issue 时使用 `Feature` 标签，并描述背景、预期和方案。

## 3. 贡献代码流程

若您希望提交代码改动，请遵循以下简要步骤：

- 在您的个人分支上开发并提交，然后向本项目仓库发起 Pull Request（PR）。常见场景下的 PR 类型：`feat`（功能/脚本）/ `fix`（Bug 修复）/ `docs`（文档修改）/ `refactor`（重构）/ `adaptor`（模型适配）/ `test`（测试用例）/ `style`（代码风格）/ `chore`（工程杂项）。PR 描述请遵循 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.md)，说明变更背景、用户可见影响与测试情况。

- 发起 PR 后，请同步创建 issue 并关联该 PR，issue 标签请与 PR 类型对应：`Feature`（功能讨论）/ `Bug`（Bug 反馈）/ `Doc`（文档问题）。我们将在 issue 中与您讨论方案是否采纳，提出意见或修改要求，您也可以通过该 issue 跟进后续流程进展。

- 根据讨论进行修改，并更新 PR。

- 在评论区输入 `compile` 以触发门禁流水线（CI）。CI 会统一拉起 UT 与 ST 用例，对精度（loss / grad norm）、性能（单步时间）与显存做回归看护，PR 合入前须通过相关 CI 门禁用例测试。

- 当 PR 的 CI 通过且获得足够的标签后，仓库 Committer 将进行最终审核，并合入在研分支。

## 4. 文档与示例贡献

- 文档位于 `docs/zh/` 目录，按 `features`（特性）、`training`（训练）、`models`（模型）、`tools`（工具）等分类组织。修改代码引入用户可见变更（如新增配置项、修改默认行为）时，请同步更新对应文档。

- 训练示例位于 `examples/` 目录，按模型组织。新增示例请提供 `README.md`、`config.yaml`、`run.sh` 与 `train.py`，命名与 HuggingFace 模型名保持一致。

## 5. 行为准则

- 请保持友善、尊重的沟通方式，聚焦技术讨论。

- 您提交的贡献将遵循 Apache 2.0 许可证授权给本项目。

感谢您的参与与贡献！我们期待与您共同推动项目发展。
