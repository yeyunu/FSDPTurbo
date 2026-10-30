# Checkpoint-based Resumable Training

FSDPTurbo supports periodically saving weights and optimizer states during training, and resuming training from a specified checkpoint.

## Configuration

Configure weight saving and checkpoint resumption in the `checkpoint` section:

```yaml
checkpoint:
  output_dir: "./output/qwen3_5_27b"
  save_steps: 500            # Save every 500 steps
  strict: true
  save_optim: true           # Save the optimizer states
  load_optim: true           # Load the optimizer states on resumption
  resume_from_checkpoint: null   # Fill in the checkpoint path for resumable training
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `output_dir` | Weight saving directory |
| `save_steps` | Saving interval (steps) |
| `strict` | Whether to strictly match parameter names when loading weights |
| `save_optim`/`load_optim` | Whether to save/load the optimizer states |
| `resume_from_checkpoint` | Path for resumable training; set it to the corresponding checkpoint directory under `output_dir` to resume training |

## Save Format

FSDPTurbo saves weights in the PyTorch distributed checkpoint (`torch.distributed.checkpoint`, DCP for short) format, with each rank writing its own parameter shards. When DCP is unavailable, it automatically falls back to rank-0 `torch.save`/`torch.load`.

The saved state includes:

- Model parameters (sharded)
- Optimizer states (sharded, controlled by `save_optim`)
- Learning rate scheduler state
- Training step count (step/epoch)

## Resumable Training Flow

1. Point `resume_from_checkpoint` to a checkpoint subdirectory under `output_dir`.
2. At training startup, `BaseTrainer` automatically invokes DCP loading, with each rank reading its own shards.
3. After loading, training resumes from the saved step, with the learning rate and optimizer states restored synchronously.

## Notes

- **Loading across parallel configurations**: DCP supports loading checkpoints under different parallel degrees (parameters will be re-sharded), but the model structure must remain consistent. It is recommended to keep the parallel configuration identical to avoid compatibility issues.
- **Multi-node shared storage**: during multi-node training, `output_dir` must be located on shared storage accessible to all nodes (such as NFS, NAS); otherwise the ranks cannot read and write cooperatively.
- **Storage space**: with DCP, each rank saves one shard file, and the total size is approximately the model plus the optimizer states; sufficient drive space must be reserved.
