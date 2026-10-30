# Contribution Guide

Thank you for your interest in FSDPTurbo! We welcome and appreciate every contribution. This guide will help you understand the ways and process of participating in the project.

## 1. Reporting Issues

- If you find any issues, first check the repository's [issues list](https://gitcode.com/Ascend/FSDPTurbo/issues) to look for similar issues or solutions.

- If the issue you encountered is not in the existing [issues list](https://gitcode.com/Ascend/FSDPTurbo/issues), you can [submit a new issue](https://gitcode.com/Ascend/FSDPTurbo/issues/create/choose), providing a clear description of the problem, reproduction steps, and environment information (such as CANN/PyTorch/torch_npu versions, parallelism strategy configuration, and a minimal reproducible script).

## 2. Performance Optimization and New Features

- For performance optimization proposals, use the `Performance` label when submitting the issue, and describe the performance optimization feature, usage scenarios, and expected benefits (such as throughput, memory, and accuracy metrics).

- For suggestions or discussions about new features, use the `Feature` label when submitting the issue, and describe the background, expectations, and approach.

## 3. Code Contribution Process

If you wish to submit code changes, please follow the brief steps below:

- Develop and commit on your personal branch, then submit a Pull Request (PR) to this repository. Common PR types: `feat` (feature/script)/`fix` (bug fix)/`docs` (documentation changes)/`refactor` (refactoring)/`adaptor` (model adaptation)/`test` (test cases)/`style` (code style)/`chore` (engineering chores). Follow the [PR Template](.gitcode/PULL_REQUEST_TEMPLATE.md) when describing the PR, including the background of the change, user-visible impact, and test status.

- After submitting the PR, create an issue and link it to the PR. The issue labels should correspond to the PR type: `Feature` (feature discussion)/`Bug` (bug feedback)/`Doc` (documentation issues). We will discuss in the issue whether to adopt the proposal and provide feedback or modification requests; you can also follow the subsequent progress through this issue.

- Make changes based on the discussion and update the PR.

- Enter `compile` in the comment section to trigger the CI gate pipeline. CI uniformly runs UT and ST test cases to provide regression protection for accuracy (loss/grad norm), performance (per-step time), and memory usage. The PR must pass the relevant CI gate tests before it can be merged.

- Once the PR passes CI and receives sufficient labels, the repository Committer will perform the final review and merge it into the development branch.

## 4. Documentation and Example Contributions

- Documentation is located in the `docs/zh/` directory and organized by categories such as `features` (features), `training` (training), `models` (models), and `tools` (tools). When code changes introduce user-visible changes (such as new configuration items or modified default behavior), please update the corresponding documentation accordingly.

- Training examples are located in the `examples/` directory, organized by model. New examples should provide `README.md`, `config.yaml`, `run.sh`, and `train.py`, with naming consistent with the Hugging Face model names.

## 5. Code of Conduct

- Please keep communication friendly and respectful, and focus on technical discussions.

- Your contributions will be licensed to this project under the Apache 2.0 license.

Thank you for your participation and contributions! We look forward to advancing the project together with you.
