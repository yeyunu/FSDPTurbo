# AGENTS.md

Instructions for AI coding agents operating in this repository.
Human-facing documentation lives in [README.md](README.md).

## Mandatory first step: read the development guide skill

This repository enforces a development guide as a skill:

```
.agent/skills/fsdp-turbo-dev-guide/SKILL.md
```

Before you create, modify, review, or answer questions about any code under
`fsdp_turbo/` or any material under `docs/`/`README.md`, read that file in full.
Skill name for harnesses that address skills by name: `fsdp-turbo-dev-guide`. If
your harness has already loaded it into context, treat the loaded content as
authoritative; if you cannot confirm that, read the file from disk. Do not rely on
this file as a substitute — it is only a pointer and a summary.

The guide currently defines 23 rules grouped into 9 categories: device and backend
(`torch.accelerator` abstraction, `torch_npu` import isolation, fused-operator
registration with a CPU fallback), module design (structure-first design, cohesion
with a small public surface, decoupling so a single folder can be moved out),
model adaptation (the `fsdp_turbo.models` leaf boundary, no patching of model
code, structure code only with parameter JSON configs in `examples/`, and the
two-level `models/<model-series>/<model-structure>/` layout),
configuration/logging/environment (validation in `FSDPTurboConfig.__post_init__`
that checks legality without mutating models, logging via `print_rank`,
centralized environment variables), code style (naming, minimal code with only the
necessary checks, reuse of existing utilities, English and sparse comments),
interfaces and docs (type annotations and docstrings on public APIs, docs kept in
sync with code), documentation conventions (no spaces between Chinese and English,
standard product terms instead of A2/A3/A5, no numbered feature-doc titles),
collaboration (a fresh AI review session before opening a PR), and security (no
unsafe deserialization or command execution).

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
   - `config.json` or other parameter config files placed under
     `fsdp_turbo/models/` (they belong in `examples/<example>/`);
   - model code added without both directory levels, i.e. not
     `fsdp_turbo/models/<model-series>/<model-structure>/`;
   - hardcoded `os.environ.get("NAME")` reads in business code instead of
     `fsdp_turbo/utils/env.py`;
   - code changes without the matching documentation update (and, for new docs,
     a `docs/zh/_menu.md` entry);
   - `torch.load(..., weights_only=False)` on untrusted files, `pickle.load`,
     `eval`/`exec`, `os.system`, or `shell=True`;
   - comments or docstrings written in Chinese, or long comment blocks above a
     single line of code (comments must be English and sparse);
   - spaces between Chinese and English words or numbers in documentation;
   - `A2`/`A3`/`A5` abbreviations in docs instead of the standard product terms
     (`Ascend 950PR&950DT系列产品`, `Atlas A3训练系列产品`,
     `Atlas A2训练系列产品`);
   - numbered titles in feature docs (`# 1. 断点续训` instead of `# 断点续训`);
   - a new acceleration module added without a design pass (layer fit, class
     responsibilities, extension seams) or one that cannot be moved out on its own
     because it imports another business module;
   - `print` instead of `print_rank`.
4. The guide marks some existing code as legacy, pending refactor. Do not start
   unrelated refactors for it, but bring it into compliance when you touch it.
5. If the guide's applicability to the task at hand is unclear, ask before
   proceeding. Do not relax a constraint on your own authority.
