# 安全声明

本声明面向FSDPTurbo的部署与运行场景，说明使用过程中的安全加固建议、数据安全注意事项与问题反馈渠道。FSDPTurbo基于Apache License 2.0开源，详见仓库根目录[LICENSE](../../LICENSE)。

## 系统安全加固

建议用户在系统中配置开启ASLR（Address Space Layout Randomization，级别2），又称**地址空间布局随机化**，可参考以下方式进行配置：

```bash
echo 2 > /proc/sys/kernel/randomize_va_space
```

## 运行用户建议

从安全性及权限最小化角度考虑，不建议使用root等管理员类型账户运行FSDPTurbo训练任务。

## 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。

2. 建议用户对个人数据、模型权重、训练配置、断点文件等敏感内容做好权限管控。涉及场景如FSDPTurbo安装目录权限管控、多用户共享数据集与权重目录的权限管控，管控权限可参考下表进行设置。

3. FSDPTurbo在训练过程中会生成分布式断点目录（`torch.distributed.checkpoint`分片文件）、日志文件与监控数据，请根据实际需求对生成文件权限进行进阶管控。

| 类型 | Linux权限参考最大值 |
| --- | --- |
| 用户主目录 | 750（rwxr-x---） |
| 程序文件（含脚本文件、库文件等） | 550（r-xr-x---） |
| 程序文件目录 | 550（r-xr-x---） |
| 配置文件 | 640（rw-r-----） |
| 配置文件目录 | 750（rwxr-x---） |
| 日志文件（正在记录） | 640（rw-r-----） |
| 日志文件（记录完毕或已归档） | 440（r--r-----） |
| 临时文件目录 | 750（rwxr-x---） |
| 业务数据文件（数据集、权重、断点等） | 640（rw-r-----） |
| 业务数据文件目录 | 750（rwxr-x---） |
| 密钥组件、私钥、证书、密文文件 | 600（rw-------） |

## 数据安全声明

1. **断点文件反序列化风险**：FSDPTurbo默认使用`torch.distributed.checkpoint`（DCP）保存与加载训练状态；当PyTorch版本低于2.2时，回退为rank 0上的`torch.save` / `torch.load`（见`fsdp_turbo/training/checkpoint.py`）。回退路径因需恢复调度器、DataLoader等非张量状态而使用`weights_only=False`，存在pickle反序列化导致的任意代码执行风险。请务必保障所加载断点来源可信，避免加载被篡改的断点文件使执行机/设备遭到攻击。

2. **模型权重与分词器加载**：FSDPTurbo通过HuggingFace Transformers的`from_pretrained`接口加载模型权重与分词器。若`model` / `tokenizer`配置指向远端仓库（HuggingFace Hub或镜像），请确保网络环境可信、权重来源可信；生产环境建议使用本地路径并事先完成校验（如哈希核对）。

3. **Host内存暂存数据**：内存优化特性`swap_activation`（激活值交换）与`swap_optimizer`（优化器状态交换）会将中间数据暂存于Host页锁定内存。多租户共用主机时，请注意进程与内存的隔离管控，避免敏感中间数据被越权访问。

## 运行安全声明

1. 建议用户结合运行资源状况编写训练配置。若配置与资源状况不匹配，如数据集加载内存大小超出内存容量限制、`swap_activation` / `swap_optimizer`的Host内存预留（`mem_fraction_static`）超出主机可用内存、断点写入超出磁盘空间大小等情况，可能引发错误并导致进程意外退出。

2. FSDPTurbo依赖PyTorch（>= 2.9.0）与torch_npu（>= 2.9.0，昇腾环境），可能因版本不匹配导致运行错误，具体版本要求见[安装指南](training/install_guide.md)。

3. 本软件在断点回退路径中使用PyTorch的`torch.load`做状态加载；对于PyTorch版本 <= 2.5.1，`weights_only`机制存在反序列化漏洞CVE-2025-32434，请用户保障所加载文件的安全性，避免恶意文件加载使执行机/设备遭到攻击。

4. 分布式训练依赖HCCL / NCCL集合通信，跨节点通信请在可信网络（内网或专线）环境中进行，避免训练数据与梯度在传输过程中被窃听或篡改。

## 漏洞与问题反馈

如您在使用FSDPTurbo过程中发现任何安全问题（包括但不限于漏洞、功能问题、合规问题），请通过[GitCode Issues](https://gitcode.com/Ascend/FSDPTurbo/issues)提交，我们将及时审视并处理。反馈安全漏洞时请勿在公开渠道披露漏洞细节。
