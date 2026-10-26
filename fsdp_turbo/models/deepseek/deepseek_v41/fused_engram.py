# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""DS4.1 Engram variant wired to the fused operator through the ops layer."""

from fsdp_turbo.ops.engram import fused_engram_wkv

from .engram import Engram


class FusedEngram(Engram):
    """Engram variant whose wkv gating runs through the fused NPU kernels."""

    def forward(self, x, hash_ids, token_mask=None):
        memory = self.embed(hash_ids).flatten(-2).to(dtype=x.dtype)
        return fused_engram_wkv(
            x,
            memory,
            self.wkv.weight,
            self.k_weight,
            self.q_weight,
            clamp_value=self.clamp_value,
            eps=self.eps,
            token_mask=token_mask,
        )
