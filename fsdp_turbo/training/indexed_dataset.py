# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Read existing Megatron token/index caches without importing its training framework."""

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedIndexedDataset(Dataset):
    """Reuse MMIDIDX token storage and exact document/sample/shuffle maps.

    ``cache_prefix`` is the common filename prefix before ``-sample_index.npy``.
    Labels are already next-token aligned; callers must not shift them again.
    The cache must use extra-token samples (sequence_length+1). Raw token data
    and .npy integer arrays are memory-mapped read-only with pickle disabled.
    """

    def __init__(self, prefix: str, cache_prefix: str, sequence_length: int, pad_token_id: int) -> None:
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        with Path(prefix + ".idx").open("rb") as stream:
            if stream.read(9) != b"MMIDIDX\x00\x00" or struct.unpack("<Q", stream.read(8))[0] != 1:
                raise ValueError("Expected a version1 MMIDIDX token dataset.")
            code = struct.unpack("<B", stream.read(1))[0]
            dtypes = {1: "u1", 2: "i1", 3: "<i2", 4: "<i4", 5: "<i8", 8: "<u2"}
            if code not in dtypes:
                raise ValueError("Indexed token storage must contain integer token IDs.")
            count, _documents = struct.unpack("<QQ", stream.read(16))
            offset = stream.tell()
        self.lengths = np.memmap(prefix + ".idx", dtype="<i4", mode="r", offset=offset, shape=(count,))
        self.pointers = np.memmap(prefix + ".idx", dtype="<i8", mode="r", offset=offset + count * 4, shape=(count,))
        self.tokens = np.memmap(prefix + ".bin", dtype=dtypes[code], mode="r")
        self.document_index = np.load(cache_prefix + "-document_index.npy", mmap_mode="r", allow_pickle=False)
        self.sample_index = np.load(cache_prefix + "-sample_index.npy", mmap_mode="r", allow_pickle=False)
        self.shuffle_index = np.load(cache_prefix + "-shuffle_index.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        """Return the number of cached samples."""
        return len(self.shuffle_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Reconstruct one packed sample and its next-token labels."""
        sample = int(self.shuffle_index[index])
        begin_document, begin_offset = map(int, self.sample_index[sample])
        end_document, end_offset = map(int, self.sample_index[sample + 1])
        parts = []
        for position in range(begin_document, end_document + 1):
            document = int(self.document_index[position])
            start = begin_offset if position == begin_document else 0
            stop = end_offset + 1 if position == end_document else int(self.lengths[document])
            pointer = int(self.pointers[document]) // self.tokens.dtype.itemsize
            parts.append(self.tokens[pointer + start : pointer + stop])
        tokens = np.concatenate(parts).astype(np.int64)
        if len(tokens) != self.sequence_length + 1:
            raise ValueError("Cache must contain full sequence_length+1 samples; use matching original indices.")
        inputs = torch.from_numpy(tokens[:-1].copy())
        labels = torch.from_numpy(tokens[1:].copy())
        labels[labels == self.pad_token_id] = -100
        inputs[inputs == self.pad_token_id] = 0
        return {"input_ids": inputs, "labels": labels, "position_ids": torch.arange(self.sequence_length)}
