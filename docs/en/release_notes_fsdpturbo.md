# Release Notes

## Key Features

FSDPTurbo supports free combinations of multi-dimensional parallelism (FSDP, TP, EP, EFSDP, and CP), balancing training throughput and cluster scalability to easily support long-sequence training. It also provides memory optimization techniques such as activation recomputation, batch chunking, and activation and optimizer state swapping, as well as MXFP8 and W8A16 low-bit quantization acceleration, enabling training of larger models with less memory and lower cost.

## Version Mapping

### Product Version Information

<table>
  <tbody>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product Name</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product Version</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.2.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Release version</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Component Name</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>FSDPTurbo</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Release Date</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>October 2026</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Maintenance Period</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
    </tr>
  </tbody>
</table>

### Related Product Version Compatibility Description

**Table 1** FSDPTurbo software version compatibility table

| FSDPTurbo Version | Python Version | PyTorch Version | torchvision Version | TorchNPU Version | CANN Version | Triton-Ascend Version | transformers Version |
| --- | --- | --- | --- | --- | --- |--- | --- |
| main (in development) | 3.12 | 2.10.0 | 0.25.0 | In development | In development | In development | 5.15.0 |
| 26.2.0 | 3.12 | 2.10.0 | 0.25.0 | 26.2.0 | 9.2.0 | 3.2.2 | 5.15.0 |

> [!NOTE]
>
> - FSDPTurbo is built on PyTorch FSDP2 and **does not depend on MindSpeed Core or Megatron**; it only requires the accompanying PyTorch ecosystem software (CANN, PyTorch, TorchNPU, Triton-Ascend, transformers) to run.
> - torchvision is a basic dependency installed automatically with FSDPTurbo; it is only needed for multimodal (image-text) training and is not used for pure text training. If PyTorch is already installed in the environment, choose a torchvision version that matches the PyTorch version when installing it (e.g. `torch==2.10.0` corresponds to `torchvision==0.25.0`); otherwise, pip may change the installed PyTorch version due to dependency resolution.
> - Supported products: <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 training products</term>, and <term>Atlas A2 training products</term>. See [Software Installation](./training/install_guide.md) for hardware and software compatibility details.
> - The Triton-Ascend version is tightly coupled with the CANN version; the Triton-Ascend version must correspond one-to-one with the CANN version. See [Triton-Ascend Compatibility](https://triton-ascend.readthedocs.io/zh-cn/latest/release_note.html#id13) for details.

## Version Compatibility Description

> [!NOTE]
>
> In the tables in this section, "/" means not compatible (unverified or unsupported), and "Y" means compatible.

**Table 3** FSDPTurbo and TorchNPU version compatibility

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 250px">
    <col style="width: 250px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">FSDPTurbo</th>
      <th colspan="1">TorchNPU Version</th>
    </tr>
    <tr>
      <th>26.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**Table 4** FSDPTurbo and CANN version compatibility

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 250px">
    <col style="width: 250px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">FSDPTurbo</th>
      <th colspan="1">CANN Version</th>
    </tr>
    <tr>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## Update Description

### New Features

| Feature | Description | Purpose |
| --- | --- | --- |
| Multi-dimensional parallelism | FSDP2 fully sharded data parallelism, tensor parallelism, expert parallelism (eager/fused/mc2/domino dispatchers and EFSDP), and context parallelism (Ulysses and KV AllGather), with free combinations supported | Balancing training throughput and cluster scalability |
| Memory optimization | Activation recomputation, batch chunking, chunk loss computation, activation swapping, optimizer state swapping (AdamWSwap), and CPU Offload | Reducing memory usage to support large model training with limited memory |
| Quantization training | MXFP8 (E4M3/E5M2/HIF8) with block-wise quantization, W8A16 quantization-aware training, and low-precision AllGather | Reducing memory usage and improving training speed |
| Checkpoint-based resumable training | Save/restore of weights and optimizer states based on PyTorch Distributed Checkpoint (DCP) | Supporting failure recovery and long-duration training |
| Native Hugging Face compatibility | Parallelization completed with one line — `FSDPTurbo(config, model)` — with the model loaded through native Transformers interfaces | Reducing the integration cost of distributed training for large models |
| Model support | Newly supported dense model Qwen3.5-27B (multimodal), and MoE models including Qwen3.5-35B-A3B, Qwen3-30B-A3B, DeepSeek V4, and Kimi K3 | Covering mainstream dense and MoE models for out-of-the-box use |

### Removed Features

None

### Interface Change Description

None

## Resolved Issues

None

## Known Issues

None

## Upgrade Impact

This is the first commercial release; there are no upgrade scenarios from historical versions.

## Related Documentation

| Document Name | Description |
| --- | --- |
| [FSDPTurbo Software Installation](./training/install_guide.md) | Hardware and OS compatibility description, software dependency version requirements, and the source installation process |
| [FSDPTurbo Quick Start](./training/quick_start_fsdpturbo.md) | The complete getting-started workflow from environment preparation to launching the first training task |
| [Supported Features](./features/supported_features.md) | A summary of supported training features and entry points to each feature's documentation |

## Virus Scan Results

|Antivirus Software|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-10-07 08:00:00.0|2026-10-08|No virus, no malware|
|Kaspersky|12.0.0.6672|2026-10-08 10:11:00|2026-10-08|No virus, no malware|
|Bitdefender|7.5.1.200224|7.101645|2026-10-08|No virus, no malware|

## Vulnerability Patch List

None

## Revision History

| Document Version | Release Date | Modification Description |
| --- | --- | --- |
| 01 | 2026-10-15 | First official release. |
