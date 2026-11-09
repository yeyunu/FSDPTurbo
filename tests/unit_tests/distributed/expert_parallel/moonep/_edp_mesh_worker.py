# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""One rank of the CPU mesh check for world=4, EP=2, efsdp=1 → edp=2.

Launched as a subprocess by ``test_edp_mesh.py``. Does not construct
MoonEPRuntime (that needs the accelerator backend). It only proves the
``('edp', 'efsdp', 'ep')`` mesh that ``ParallelState._create_mesh`` would
build can come up on gloo, and that the replica groups can SUM.
"""

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main() -> int:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{os.environ['RENDEZVOUS_FILE']}",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    try:
        # Same dim order as ParallelState: leading edp is world/(efsdp*ep).
        mesh = init_device_mesh("cpu", (2, 1, 2), mesh_dim_names=("edp", "efsdp", "ep"))
        edp_group = mesh["edp"].get_group()
        ep_group = mesh["ep"].get_group()
        marker = torch.tensor([float(dist.get_rank())], dtype=torch.float32)
        dist.all_reduce(marker, op=dist.ReduceOp.SUM, group=edp_group)
        print(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "edp_size": dist.get_world_size(group=edp_group),
                    "ep_size": dist.get_world_size(group=ep_group),
                    "edp_ranks": dist.get_process_group_ranks(edp_group),
                    "ep_ranks": dist.get_process_group_ranks(ep_group),
                    "replica_sum": float(marker.item()),
                }
            )
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
