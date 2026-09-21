# FSDPTurbo软件安装

本文主要向用户介绍如何基于PyTorch框架快速完成FSDPTurbo（分布式训练加速库）的安装。

## 硬件配套和支持的操作系统

**表1**产品硬件支持列表

|产品|是否支持|
|--|:-:|
|Ascend 950PR&950DT系列产品|√|
|Atlas A3训练系列产品|√|
|Atlas A3推理系列产品|x|
|Atlas A2训练系列产品|√|
|Atlas A2推理系列产品|x|
|Atlas 200I/500 A2推理产品|x|
|Atlas推理系列产品|x|
|Atlas训练系列产品|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。
- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](../release_notes_fsdpturbo.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

请单击[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers)，并根据引导完成固件与驱动的安装。

> [!NOTICE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
>
> 更多安全相关内容请参见《[安全声明](../SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 安装FSDPTurbo

### 源码安装

FSDPTurbo当前仅支持源码安装。请参考如下操作获取对应源码及安装相关依赖，完成FSDPTurbo的安装。

1. （可选）创建虚拟环境

   建议使用Python 3.11及以上版本。若不希望影响全局Python环境，可使用venv、conda、uv等常用虚拟环境管理工具创建独立的虚拟环境。以conda为例，可参考如下命令：

   ```bash
   conda create -n fsdpturbo python=3.11
   conda activate fsdpturbo
   ```

2. 安装CANN

   安装配套版本的NPU驱动固件、CANN软件并配置CANN环境变量，具体请参考《[CANN快速安装](https://www.hiascend.com/cann/download?versionId=783&ids=d806%2Ch0501%2Ch0601%2Ch0703)》。

   CANN软件提供进程级环境变量设置脚本，训练场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   # 非root用户的默认安装路径
   source ${HOME}/Ascend/cann/set_env.sh
   ```

   ```shell
   # root用户默认安装路径
   source /usr/local/Ascend/cann/set_env.sh
   ```

   以上命令以CANN离线安装场景未指定安装路径为例，给出了不同用户的默认安装路径及对应的配置环境变量的命令。若用户指定安装路径或使用其他安装方式，请参考《[CANN软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html)》查阅对应的配置环境变量的命令。

3. 安装PyTorch及TorchNPU

   根据引导安装配套版本的PyTorch以及TorchNPU，具体请参考《[TorchNPU快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C98%2C107%2C1%2C6%2C177%2C)》。

   > [!NOTE]
   >
   > - 更多TorchNPU插件版本请单击[Link](https://gitcode.com/ascend/pytorch/releases)。
   > - TorchNPU相关文档请参见《[TorchNPU使用导读](https://www.hiascend.com/document/detail/zh/Pytorch/latest/index/index.html)》。

4. 安装Triton-Ascend

   安装配套版本的Triton-Ascend，请参考[Triton-Ascend安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#id1)，获取配套版本的Triton-Ascend安装指令。可参考如下安装命令：

   ```shell
   pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

   > [!NOTE]
   >
   > Triton-Ascend 3.2.0及以下版本，Triton-Ascend和Triton不能同时存在。需先卸载社区Triton，再安装Triton-Ascend。

5. 安装FSDPTurbo

   ```shell
   git clone https://gitcode.com/Ascend/FSDPTurbo.git
   cd FSDPTurbo
   pip3 install -e .
   ```

   基础依赖（torch、torchvision、transformers、datasets、triton等）将随上述命令自动安装。如需使用以下扩展能力，可安装对应的可选依赖：

   ```shell
   # 安装fla-core、flash-linear-attention等线性注意力相关依赖（训练Kimi K3等采用线性注意力结构的模型时需要）
   pip3 install -e ".[npu]"
   ```

   > [!NOTE]
   >
   > - `[npu]`可选依赖包含torch_npu。若已按步骤3安装配套版本的TorchNPU，pip将跳过已满足的依赖，不会重复安装或变更版本。
   > - torchvision为自动安装的基础依赖，仅多模态（图文）训练需要：纯文本训练不使用，安装后如无需要可执行`pip uninstall torchvision`移除。
   > - 若环境中已安装PyTorch，安装torchvision时需选择与该PyTorch配套的版本（如`torch==2.10.0`对应`torchvision==0.25.0`），否则pip可能因依赖解析变更已安装的PyTorch版本。

## 验证安装

执行以下命令，确认FSDPTurbo已正确安装：

```bash
python -c "import fsdp_turbo; print(fsdp_turbo.__file__)"
```

可进一步确认NPU与TorchNPU状态：

```bash
# 确认NPU是否可以正常使用
npu-smi info
# 确认torch_npu是否可用，输出True表示正常
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

> [!NOTE]
>
> 每个新终端执行训练任务前，需先加载CANN环境变量（参见[源码安装](#源码安装)步骤2）。更多训练相关环境变量说明请参考[环境变量说明](../features/environment_variable.md)。

安装成功后，可参考[快速入门](./quick_start_fsdpturbo.md)启动第一个训练任务。
