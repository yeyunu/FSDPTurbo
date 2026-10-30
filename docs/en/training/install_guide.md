# Installing FSDPTurbo

This document describes how to quickly install FSDPTurbo (a distributed training acceleration library) based on the PyTorch framework.

## Supported Hardware and Operating Systems

**Table 1** Supported hardware products

|Product|Supported|
|--|:-:|
|Ascend 950PR&950DT products|√|
|Atlas A3 training products|√|
|Atlas A3 inference products|x|
|Atlas A2 training products|√|
|Atlas A2 inference products|x|
|Atlas 200I/500 A2 inference products|x|
|Atlas inference products|x|
|Atlas training products|x|

> [!NOTE]
>
> In the table in this section, "√" indicates supported and "x" indicates not supported.

- For the operating systems supported by each hardware product in physical-machine deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility).
- For the operating systems supported by each hardware product in virtual-machine and container deployment scenarios, see the "[Operating System Compatibility](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)" section of the *CANN Software Installation* guide.

## Before You Start

Download and install the corresponding software versions by following the "[Matching Notes for Related Product Versions](../release_notes_fsdpturbo.md#related-product-version-compatibility-description)" section of the *Release Notes*.

Click [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers) and follow the instructions to install the firmware and drivers.

> [!NOTICE]
>
> It is recommended to use a non-root user to install and run programs, and to control the permissions of the files in the installer directory: set folder permissions to 750 and file permissions to 640. You can control the permissions of installed files by setting umask, for example, set umask to 0027.
>
> For more security-related content, see the "File Permission Control" notes for each component in the [Security Statement](../SECURITYNOTE.md).

## Installing FSDPTurbo

### Source Installation

FSDPTurbo currently supports source installation only. Follow the steps below to obtain the source code and install the related dependencies to complete the FSDPTurbo installation.

1. (Optional) Create a virtual environment.

   Python 3.12 is recommended; for details, see the [Release Notes](../release_notes_fsdpturbo.md). To avoid affecting the global Python environment, use a common virtual environment management tool such as venv, conda, or uv to create an isolated virtual environment. Using conda as an example:

   ```bash
   conda create -n fsdpturbo python=3.11
   conda activate fsdpturbo
   ```

2. Install CANN.

   Install the matching NPU driver and firmware, the CANN software, and configure the CANN environment variables. For details, see [CANN Quick Installation](https://www.hiascend.com/cann/download?versionId=783&ids=d806%2Ch0501%2Ch0601%2Ch0703).

   The CANN software provides a process-level environment variable setup script. In training scenarios, you must source this script before executing business code on the NPU; otherwise, the business code will not run.

   ```shell
   # Default installation path for non-root users
   source ${HOME}/Ascend/cann/set_env.sh
   ```

   ```shell
   # Default installation path for root users
   source /usr/local/Ascend/cann/set_env.sh
   ```

   The commands above use the CANN offline installation scenario without a specified installation path as an example, showing the default installation paths for different users and the corresponding commands for configuring environment variables. If you specify an installation path or use another installation method, see [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html) for the corresponding environment variable configuration commands.

3. Install PyTorch and TorchNPU.

   Follow the instructions to install the matching versions of PyTorch and TorchNPU. For details, see [TorchNPU Quick Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C98%2C107%2C1%2C6%2C177%2C).

   > [!NOTE]
   >
   > - For more TorchNPU plugin versions, click [Link](https://gitcode.com/ascend/pytorch/releases).
   > - For TorchNPU documentation, see the [TorchNPU Usage Guide](https://www.hiascend.com/document/detail/en/Pytorch/latest/index/index.html).

4. Install Triton-Ascend.

   To install the matching version of Triton-Ascend, see the [Triton-Ascend Installation Guide](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#id1) for the installation commands of the matching version. An example installation command is as follows:

   ```shell
   pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

   > [!NOTE]
   >
   > For Triton-Ascend 3.2.0 and earlier versions, Triton-Ascend and Triton cannot coexist. You must uninstall the community Triton first, and then install Triton-Ascend.

5. Install FSDPTurbo.

   ```shell
   git clone https://gitcode.com/Ascend/FSDPTurbo.git
   cd FSDPTurbo
   pip3 install -e .
   ```

   Basic dependencies (torch, torchvision, transformers, datasets, triton, and so on) are installed automatically with the above command. To use the following extended capabilities, install the corresponding optional dependencies:

   ```shell
   # Install linear-attention-related dependencies such as fla-core and flash-linear-attention (required when training models with linear attention architectures, such as Kimi K3)
   pip3 install -e ".[npu]"
   ```

   > [!NOTE]
   >
   > - The `[npu]` optional dependency includes torch_npu. If you installed the matching version of TorchNPU in step 3, pip will skip the already-satisfied dependencies and will not reinstall them or change their versions.
   > - torchvision is a basic dependency installed automatically and is needed only for multimodal (image-text) training: it is not used for pure-text training. If it is not needed after installation, you can remove it with `pip uninstall torchvision`.
   > - If PyTorch is already installed in the environment, choose a torchvision version that matches that PyTorch (for example, `torch==2.10.0` corresponds to `torchvision==0.25.0`); otherwise, pip may change the installed PyTorch version during dependency resolution.

## Verifying the Installation

Run the following command to confirm that FSDPTurbo has been installed correctly:

```bash
python -c "import fsdp_turbo; print(fsdp_turbo.__file__)"
```

You can further confirm the status of the NPU and TorchNPU:

```bash
# Check whether the NPU is working properly
npu-smi info
# Check whether torch_npu is available; True means it is working properly
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

> [!NOTE]
>
> Before running a training task in each new terminal, you must first load the CANN environment variables (see step 2 of [Source Installation](#source-installation)). For more information about training-related environment variables, see [Environment Variables](../features/environment_variable.md).

After a successful installation, see [Quick Start](./quick_start_fsdpturbo.md) to start your first training task.
