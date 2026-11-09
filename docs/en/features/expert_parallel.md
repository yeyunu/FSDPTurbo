# Expert Parallelism

FSDPTurbo provides expert parallelism (EP) for MoE (Mixture of Experts) models, sharding the experts across multiple devices along the expert dimension. Tokens are dispatched through All-to-All communication to the devices holding the corresponding experts for computation.

## Configuration

Configure the expert parallel degree and the expert plan in the `distributed` section:

```yaml
distributed:
  expert_parallel_size: 8
  expert_fully_shard_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: "fused"
    apply_efsdp_modules:
      - model.language_model.layers.{*}.mlp.experts
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `expert_parallel_size` | Expert parallel degree; shards the experts module along the expert dimension across the EP group |
| `expert_fully_shard_parallel_size` | Expert FSDP2 (EFSDP) parallel degree; further fully shards the expert parameters |
| `ep_plan.apply_modules` | List of modules to which EP is applied, supporting the `model.language_model.layers.{*}.mlp.experts` wildcard |
| `ep_plan.dispatcher` | Dispatcher for expert computation; see the table below |
| `ep_plan.fixed_router` | Whether to use a fixed router |
| `ep_plan.apply_efsdp_modules` | List of modules to which EFSDP is applied |
| `ep_plan.gradient_divide_factor` | Gradient divide factor for the experts (must be specified manually under special parallelism combinations) |
| `ep_plan.moonep_config` | Communication, memory, and static-shape tuning for `dispatcher: moonep` (`MoonEPConfig`); see [MoonEP dispatcher](#moonep-dispatcher) |

## Dispatcher Types

| dispatcher | Description |
|---|---|
| `eager` | Per-expert dispatch; has the best generality |
| `fused` | Grouped GEMM (grouped_matmul); computes in batches after tokens are aggregated, with better performance |
| `mc2` | Communication-computation fusion implementation accelerated by Ascend MC2 |
| `domino` | Domino dispatch implementation |
| `custom_ep_forward_eager` | Custom EP forward (eager semantics); the model must implement the `ep_forward_eager` interface |
| `custom_ep_forward_fused` | Custom EP forward (fused semantics); the model must implement the `ep_forward_fused` interface |
| `moonep` | Symmetric-memory EP backend (BF16); requires `expert_parallel_size > 1` and `expert_fully_shard_parallel_size: 1`; see [MoonEP dispatcher](#moonep-dispatcher) |

## Custom EP Forward

When a model's expert computation semantics are non-standard, a custom dispatcher can be used, implemented on the experts module:

```python
def ep_forward_eager(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...

def ep_forward_fused(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...
```

MoE models with the standard GLU/SwiGLU structure should prefer the built-in `eager`/`fused`/`mc2`/`domino` dispatchers.

## MoonEP dispatcher

`dispatcher: moonep` runs expert compute and communication on the MoonEP symmetric-memory backend: every rank receives a fixed `S*K` dispatched activation rows (static shapes throughout, no device-scalar host reads on the critical path), redundant experts are planned per step and prefetched into symmetric-memory slots, and weight gradients are staged in FP32 and reduced back to their home ranks in a stage → rendezvous → reduce sequence.

**Constraints (enforced at `FSDPTurboConfig` validation):**

- `expert_parallel_size > 1`, and the expert count must be divisible by the EP degree;
- `expert_fully_shard_parallel_size: 1` (MoonEP owns the expert weights and cannot stack with EFSDP); `ep_plan.apply_efsdp_modules` must stay empty;
- Expert weights must be BF16 dense `nn.Parameter`s whose two matrix dimensions are multiples of 128; mutually exclusive with MoE quantization (`quantize.moe.*` converters, `quant_gmm`);
- The CUDA backend requires an intra-node EP group with NVLink/NVSwitch multicast and `moonep==0.0.1`; the Ascend backend requires the `npu_patch` package, built from the same `moonep` distribution with the Ascend backend selected.

**Static-shape derivation:** `tokens_per_rank = data.batch_size * data.max_seq_length / ulysses_parallel_size`; `top_k` is read from the model config (`num_experts_per_token` / `num_experts_per_tok`). Neither is normally typed in by hand. Ascend must lock S/K before its symmetric heap initializes — a failed derivation raises in `MoonEPRuntime.finalize`; CUDA derives them on the first expert forward and then locks them.

**`MoonEPConfig` tuning knobs:** the table below is the reference card. `fsdp_turbo/distributed/expert_parallel/moonep/README.md` groups the same knobs by what they affect and carries the measured figures behind each default.

| Field | Default | Description |
|---|---|---|
| `num_sms` | `32` | SM count used by the collective kernels |
| `token_padding` | `128` | Token alignment granularity |
| `comm_stream_priority` | `-1` | Communication side-stream priority (CUDA only) |
| `async_finish` | `false` | Asynchronous completion on a side stream. Live on CUDA; on Ascend it stays masked off unless `npu_async_comm: true` |
| `tokens_per_rank` / `top_k` | derived | Static-shape overrides; not normally set by hand |
| `recompute_dispatch` | `false` | Replay the dispatch in backward for `K` times less saved activation on the first GEMM; redundant when full-layer recompute already covers the experts (warned) |
| `validate_plan_invariants` | `false` | Run the device-side plan checks every step instead of only the first |
| `shmem_ip_port` | derived | CANN SHMEM bootstrap endpoint for a multi-node EP group; defaults to `MASTER_ADDR:MASTER_PORT+1` |
| `shmem_stream_barrier` | `true` | On Ascend, replace the host synchronize plus `dist.barrier` with CANN SHMEM's stream-ordered global barrier |
| `trace_enabled` | `false` | Per-rank MoonEP phase markers, for localising a collective hang that produces no traceback |
| `trace_sync_enabled` | `false` | Synchronize the device at every marker; serializes the pipeline, debug only |
| `vmm_probe_enabled` | `false` | On an Ascend VMM allocation failure, sweep sizes in-process to separate a per-mapping cap from a per-process cap |
| `npu_async_comm` | `false` | Read only on Ascend. When true, the backend reports asynchronous communication and `async_finish` runs on a side stream. Not yet validated on hardware: a desynchronised barrier hangs; set it back to false if a step stops |

Kimi routing diagnostics live on `ep_plan.router_debug` for every EP dispatcher. Leaving the block unset keeps the real router:

| Field | Default | Description |
|---|---|---|
| `apply_custom_router` | `false` | Replace Kimi router scores with a synthetic load; `target_max_vio` must be finite and ≥ 0 when this is on |
| `target_max_vio` | `-1` | Target max-violation of the synthetic routing |
| `router_mode` | `lognormal` | `exact` or `lognormal` |
| `record_load_balance` | `false` | Log per-layer expert counts and the EP-reduced global maxvio |

**EDP replica gradient sync:** with `edp > 1`, `MoonEPRuntime.sync_replica_gradients` SUMs the FP32 home gradients across the 1-D `edp` group after the EP-group reduction and before the divisor, matching fused HSDP's all-reduce on the replicate dim; `gradient_divide_factor` stays `ep × efsdp × edp` either way. With `edp = 1` no extra collective is added.

**Verification entry points:** unit tests under `tests/unit_tests/distributed/expert_parallel/moonep/`; system tests under `tests/system_tests/model/moonep/` (including the `edp > 1` absolute-reference check `test_moonep_edp_ref.py`).

## Model Structure Requirements

The EP path of FSDPTurbo requires the experts module to hold **3D tensor parameters**, rather than independent per-expert submodules in ModuleList form. Specific requirements:

- The experts module must expose two 3D tensor parameters, `gate_up_proj` and `down_proj`, with the shape `[num_experts, out_dim, in_dim]`.
- Parameters in the 2D stacked format `[num_experts * out_dim, in_dim]` must be reshaped to 3D via `view(num_experts, -1, in_dim)` before they can be used in grouped_matmul.
- **ModuleList-form experts are not supported**: per-expert Linear submodules cannot directly participate in EP's All-to-All dispatch and grouped GEMM; they must first be converted into a 3D tensor aggregated structure.

## Notes

- **Parallelism combination constraint**: `expert_parallel_size` must evenly divide the total number of experts per layer, and `expert_parallel_size × expert_fully_shard_parallel_size ≤ total number of devices`.
- **Expert load balancing**: during training, the routing distribution (the frequency with which experts are selected by top-k) can be observed. If severe imbalance occurs, consider adjusting the model-side routing temperature, or use `fixed_router` (a debugging measure: replaces the actual router with a fixed uniform router to avoid the performance and precision issues caused by imbalanced load; it only takes effect when `expert_parallel_size > 1`).
- **EFSDP**: when the expert parameter count is large, enable `apply_efsdp_modules` to further fully shard the expert parameters with FSDP2, reducing the per-card memory usage.
