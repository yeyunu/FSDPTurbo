# AGENTS.md

Instructions for AI coding agents operating in this repository.
Human-facing documentation lives in [README.md](README.md).

## Mandatory first step: read the development guide skill

This repository enforces a development guide as a skill:

```
.agent/skills/fsdp-turbo-dev-guide/SKILL.md
```

Before you create, modify, review, or answer questions about any code under
`fsdp_turbo/`, read that file in full. Skill name for harnesses that address
skills by name: `fsdp-turbo-dev-guide`. If your harness has already loaded it
into context, treat the loaded content as authoritative; if you cannot confirm
that, read the file from disk. Do not rely on this file as a substitute — it is
only a pointer and a summary.

The guide currently defines 11 rules covering: device abstraction via
`torch.accelerator`, `torch_npu` import isolation, fused-operator registration
with a CPU fallback, module cohesion and independent portability, config
validation in `FSDPTurboConfig.__post_init__`, logging via `print_rank`, naming,
the `fsdp_turbo.models` leaf boundary, the ban on patching model code, minimal
code with only the necessary checks, and reuse of existing utility functions
over new abstractions.

## Required behavior

1. Obey every rule in the guide. After producing code, walk the guide's
   self-check list item by item and report the result of each item.
2. If a request conflicts with the guide, name the conflict and propose a
   compliant alternative. Implement only after the user confirms. Never deviate
   from the guide silently.
3. Do not introduce the patterns the guide forbids. The usual offenders:
   - `torch.cuda.*` / `torch.npu.*` calls instead of `torch.accelerator`;
   - `import torch_npu` outside `fsdp_turbo/ops/npu/` — the only exception is the
     startup injection in `fsdp_turbo/utils/device.py`;
   - any import of `fsdp_turbo.models` from inside the `fsdp_turbo` package;
   - cross-directory imports between model directories under `fsdp_turbo/models/`;
   - runtime patching of model classes or members under `fsdp_turbo/models/`
     (subclass the Transformers class and override instead);
   - `print` instead of `print_rank`.
4. The guide marks some existing code as legacy, pending refactor. Do not start
   unrelated refactors for it, but bring it into compliance when you touch it.
5. If the guide's applicability to the task at hand is unclear, ask before
   proceeding. Do not relax a constraint on your own authority.
