# Security Statement

This statement addresses the deployment and operation scenarios of FSDPTurbo, describing security hardening recommendations, data security precautions, and issue feedback channels for its use. FSDPTurbo is open-sourced under the Apache License 2.0; see [LICENSE](../../LICENSE) in the repository root for details.

## System Security Hardening

Users are advised to enable ASLR (Address Space Layout Randomization, level 2) on the system, also known as **address space layout randomization**, which can be configured as follows:

```bash
echo 2 > /proc/sys/kernel/randomize_va_space
```

## Recommended Running User

From the perspective of security and least privilege, it is not recommended to run FSDPTurbo training tasks with administrator-type accounts such as root.

## File Permission Control

1. Users are advised to set the umask of the running system to 0027 or higher on hosts (including host machines) and in containers, ensuring that newly created folders have a maximum default permission of 750 and newly created files a maximum default permission of 640.

2. Users are advised to enforce permission control over sensitive content such as personal data, model weights, training configurations, and checkpoint files. This applies to scenarios such as permission control for the FSDPTurbo installation directory and for multi-user shared dataset and weight directories; refer to the table below to set the permissions.

3. During training, FSDPTurbo generates distributed checkpoint directories (`torch.distributed.checkpoint` shard files), log files, and monitoring data; users should apply advanced permission control to the generated files according to actual requirements.

| Type | Linux Permission Reference Maximum |
| --- | --- |
| User home directory | 750 (rwxr-x---) |
| Program files (including script files, library files, etc.) | 550 (r-xr-x---) |
| Program file directories | 550 (r-xr-x---) |
| Configuration files | 640 (rw-r-----) |
| Configuration file directories | 750 (rwxr-x---) |
| Log files (being written) | 640 (rw-r-----) |
| Log files (completed or archived) | 440 (r--r-----) |
| Temporary file directories | 750 (rwxr-x---) |
| Service data files (datasets, weights, checkpoints, etc.) | 640 (rw-r-----) |
| Service data file directories | 750 (rwxr-x---) |
| Key components, private keys, certificates, encrypted files | 600 (rw-------) |

## Data Security

1. **Checkpoint file deserialization risk**: FSDPTurbo uses `torch.distributed.checkpoint` (DCP) by default to save and load training state; when the PyTorch version is lower than 2.2, it falls back to `torch.save` / `torch.load` on rank 0 (see `fsdp_turbo/training/checkpoint.py`). Because the fallback path needs to restore non-tensor states such as schedulers and DataLoaders, it uses `weights_only=False`, which carries a risk of arbitrary code execution caused by pickle deserialization. Ensure that the source of the loaded checkpoints is trustworthy, and avoid loading tampered checkpoint files that could compromise the execution machine/device.

2. **Model weight and tokenizer loading**: FSDPTurbo loads model weights and tokenizers through the Hugging Face Transformers `from_pretrained` interface. If the `model` / `tokenizer` configuration points to a remote repository (Hugging Face Hub or a mirror), ensure that the network environment and the weight source are trustworthy; in production environments, it is recommended to use local paths and complete verification in advance (e.g. hash checking).

3. **Data temporarily stored in host memory**: The memory optimization features `swap_activation` (activation swapping) and `swap_optimizer` (optimizer state swapping) temporarily store intermediate data in host page-locked memory. When a host is shared among multiple tenants, pay attention to process and memory isolation control to prevent unauthorized access to sensitive intermediate data.

## Runtime Security

1. Users are advised to write training configurations based on the available runtime resources. If the configuration does not match the resource conditions — for example, the memory required to load the dataset exceeds the memory capacity limit, the host memory reserved by `swap_activation` / `swap_optimizer` (`mem_fraction_static`) exceeds the host's available memory, or checkpoint writing exceeds the available disk space — errors may be triggered and cause the process to exit unexpectedly.

2. FSDPTurbo depends on PyTorch (>= 2.9.0) and torch_npu (>= 2.9.0, Ascend environment); version mismatches may cause runtime errors. See [Installation Guide](training/install_guide.md) for the specific version requirements.

3. This software uses PyTorch's `torch.load` for state loading in the checkpoint fallback path; for PyTorch versions <= 2.5.1, the `weights_only` mechanism has the deserialization vulnerability CVE-2025-32434. Users should ensure the safety of the files being loaded and avoid loading malicious files that could compromise the execution machine/device.

4. Distributed training relies on HCCL collective communication. Cross-node communication should be performed in a trusted network (intranet or dedicated line) environment to prevent training data and gradients from being eavesdropped on or tampered with during transmission.

## Vulnerability and Issue Feedback

If you discover any security issues while using FSDPTurbo (including but not limited to vulnerabilities, functional issues, and compliance issues), please submit them via [GitCode Issues](https://gitcode.com/Ascend/FSDPTurbo/issues); we will review and handle them in a timely manner. When reporting security vulnerabilities, do not disclose vulnerability details through public channels.
