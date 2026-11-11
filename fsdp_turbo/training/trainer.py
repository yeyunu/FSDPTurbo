import os
import logging
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, uses_hybrid_cp_zigzag_layout
from fsdp_turbo.ops.indexer_kl_metric import collect_indexer_kl_metrics
from fsdp_turbo.utils.dtype import get_dtype
from fsdp_turbo.utils.log import set_log_level, print_rank
from fsdp_turbo.training.checkpoint import Checkpointer
from fsdp_turbo.training.monitor import Monitor

logger = logging.getLogger(__name__)


def get_local_rank() -> int:
    """
    获取当前进程的 Local Rank（节点内编号）
    优先使用 torchrun 设置的 LOCAL_RANK 环境变量
    """
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        return int(local_rank)

    # 回退方案：全局 rank 对节点内卡数取模（假设每节点8卡）
    if dist.is_initialized():
        global_rank = dist.get_rank()
        # 从环境变量获取每节点卡数，如果没有则默认为8
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 8))
        return global_rank % local_world_size

    return 0


def parse_cpu_affinity_conf(conf_str: str) -> dict:
    """
    解析 CPU_AFFINITY_CONF 字符串
    格式: "1,npu0:144-155,npu1:156-167,..."
    返回: {'npu0': (144, 155), 'npu1': (156, 167), ...}
    """
    npu_map = {}
    if not conf_str:
        return npu_map

    # 按逗号分割
    parts = [p.strip() for p in conf_str.split(',')]

    # 跳过开头的开关标志（如 "1"）
    start_idx = 0
    if parts and not parts[0].startswith('npu'):
        start_idx = 1

    for part in parts[start_idx:]:
        if not part:
            continue
        if ':' in part:
            npu_name, range_str = part.split(':', 1)
            npu_name = npu_name.strip()
            range_str = range_str.strip()
            if '-' in range_str:
                try:
                    start, end = map(int, range_str.split('-'))
                    npu_map[npu_name] = (start, end)
                except ValueError:
                    logger.warning("无法解析CPU范围: %s", part)
    return npu_map


def apply_cpu_affinity_by_conf():
    """
    主入口：读取环境变量，解析，并根据 Local Rank 绑定 CPU
    必须在 dist.init_process_group 之后调用
    """
    conf_str = os.environ.get("CPU_AFFINITY_CONF", "")
    if not conf_str:
        logger.info("CPU_AFFINITY_CONF 未设置，跳过CPU绑核")
        return

    # 解析配置
    npu_map = parse_cpu_affinity_conf(conf_str)
    if not npu_map:
        logger.warning("CPU_AFFINITY_CONF 解析结果为空: %s", conf_str)
        return

    # 获取 Local Rank
    local_rank = get_local_rank()
    npu_name = f"npu{local_rank}"

    if npu_name not in npu_map:
        logger.warning("Local Rank %s 对应的 %s 未在配置中找到，跳过绑核", local_rank, npu_name)
        return

    start_cpu, end_cpu = npu_map[npu_name]

    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        logger.warning("当前平台不支持 CPU 绑核，跳过绑核")
        return
    set_affinity = getattr(os, "sched_setaffinity")
    get_affinity = getattr(os, "sched_getaffinity")

    # 执行绑核
    try:
        # 设置当前进程（或主线程）的 CPU 亲和性
        set_affinity(0, range(start_cpu, end_cpu + 1))

        # 验证绑核结果
        actual = sorted(get_affinity(0))
        expected = list(range(start_cpu, end_cpu + 1))
        if actual == expected:
            logger.info("[OK] Local Rank %s (%s) 已绑定到 CPU %s-%s", local_rank, npu_name, start_cpu, end_cpu)
        else:
            logger.warning("[WARN] 绑核结果不符，期望 %s，实际 %s", expected, actual)

    except Exception as e:
        logger.error("绑核失败: %s", e, exc_info=True)


class BaseTrainer:
    """Base trainer with common training lifecycle.

    Subclasses override builder methods to customize component creation.
    Default flow: init distributed -> build components -> train.

    Attributes:
        model: The training model (after parallelism wrapping).
        tokenizer: The tokenizer instance.
        processor: The multimodal processor, when configured.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler.
        data_manager: The data loader / manager.
        config: The training configuration.
    """

    def __init__(self, config: Optional[FSDPTurboConfig] = None):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        self._ckpt_manager = None
        self._monitor = None
        self._global_step = 0
        self._consumed_samples = 0
        self._accumulated_losses = {"loss": [], "aux_loss": []}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        """Build all components. Call before train()."""
        self._init_distributed()
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_scheduler()
        self.dataloader = self.build_dataloader()
        self._ckpt_manager = self.build_checkpointer()
        self._monitor = self.build_monitor()

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop."""
        print_rank(logger.info, "\n" + str(self.config))

        self._prepare_training(resume_from_checkpoint)
        memory_tracker = self._create_memory_tracker()
        memory_tracker.on_train_begin()
        max_steps = self._resolve_max_steps()

        print_rank(logger.info, "***** Running training *****")
        print_rank(logger.info, f"  Max steps = {max_steps}")
        print_rank(logger.info, f"  Gradient accumulation steps = {self.config.run.gradient_accumulation_steps}")

        with self._create_profiler() as prof:
            # Compute start epoch and skip batches from consumed_samples.
            dataset_size = len(self.dataloader.dataset) if self.dataloader is not None else 0
            samples_per_epoch = dataset_size if dataset_size > 0 else 0
            if self._consumed_samples > 0 and samples_per_epoch > 0:
                start_epoch = self._consumed_samples // samples_per_epoch
                skip_batches = (self._consumed_samples % samples_per_epoch) // self.config.data.batch_size
                print_rank(
                    logger.info,
                    f"  Resuming from consumed_samples={self._consumed_samples}: "
                    f"start_epoch={start_epoch}, skip_batches={skip_batches}",
                )
            else:
                start_epoch = 0
                skip_batches = 0

            for epoch in range(self.config.run.num_train_epochs):
                print_rank(logger.info, f"  Epoch {epoch + 1}/{self.config.run.num_train_epochs}")
                # Set epoch on DistributedSampler for proper shuffling across epochs.
                if (
                    self.dataloader is not None
                    and hasattr(self.dataloader, "sampler")
                    and hasattr(self.dataloader.sampler, "set_epoch")
                ):
                    self.dataloader.sampler.set_epoch(epoch)

                # Skip epochs and batches we already consumed.
                if epoch < start_epoch:
                    continue
                for step, batch in enumerate(self.dataloader):
                    if epoch == start_epoch and step < skip_batches:
                        continue
                    # Marks the last micro-batch of the accumulation group; the prefetch hook defers to it
                    self._is_last_micro_step = (step + 1) % self.config.run.gradient_accumulation_steps == 0
                    self._monitor.on_step_start()
                    memory_tracker.on_step_start()
                    try:
                        batch = self.prepare_batch(batch)
                        loss, aux_loss = self.train_step(batch)
                    except Exception as error:
                        memory_tracker.on_error(self._global_step, error)
                        raise
                    self._accumulated_losses["loss"].append(loss.detach())
                    if aux_loss is not None:
                        self._accumulated_losses["aux_loss"].append(aux_loss.detach())

                    # Track how many samples have been processed.
                    if isinstance(batch, dict):
                        batch_size = next(
                            (v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor)),
                            1,
                        )
                        if "sfa_layout" in batch and self.config.data.sfa_layout in (
                            "BSND",
                            "TND",
                        ):
                            # A packed JSONL row is one sampler item even when
                            # BSND puts its documents on the batch axis.
                            batch_size = self.config.data.batch_size
                    elif isinstance(batch, (list, tuple)):
                        batch_size = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else 1
                    else:
                        batch_size = 1
                    self._consumed_samples += batch_size

                    if (step + 1) % self.config.run.gradient_accumulation_steps == 0:
                        synced = self.sync_loss(self._accumulated_losses)
                        loss = synced["loss"]
                        aux_loss = synced.get("aux_loss")
                        try:
                            grad_norm = self._on_optimizer_step()
                        except Exception as error:
                            # The first optimizer step allocates optimizer
                            # states, a common OOM point; let the tracker see it.
                            memory_tracker.on_error(self._global_step, error)
                            raise
                        for v in self._accumulated_losses.values():
                            v.clear()
                        self._global_step += 1
                        self._on_log_step(loss, grad_norm, aux_loss, indexer_kl=synced.get("indexer_kl"))
                        self._on_checkpoint_step()
                        if prof is not None:
                            prof.step()

                    memory_tracker.on_step_end(self._global_step)

                    if 0 < max_steps <= self._global_step:
                        break
                if 0 < max_steps <= self._global_step:
                    break

        print_rank(logger.info, f"Training complete. Total steps: {self._global_step}")
        self._monitor.summary()
        self._release_model_resources()

    def _release_model_resources(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()  # pylint: disable=not-callable

    # ------------------------------------------------------------------
    # Training loop hooks
    # ------------------------------------------------------------------

    def _prepare_training(self, resume_from_checkpoint: Optional[str] = None):
        """Resume from checkpoint and set model to train mode."""
        self._global_step = 0
        checkpoint = resume_from_checkpoint or self.config.checkpoint.resume_from_checkpoint
        if checkpoint and self._ckpt_manager is not None:
            self._load_checkpoint(checkpoint)
        self.model.train()

        if self.dataloader is None:
            raise RuntimeError("dataloader is None. Override build_dataloader() to provide a dataloader.")

    def _on_optimizer_step(self) -> float:
        """Clip gradients, step optimizer and scheduler, zero grads.

        Returns the gradient norm before clipping (for monitoring).
        Override to inject custom logic between clipping and stepping.
        """
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        swap_manager = getattr(self.model, "swap_manager", None)
        if swap_manager is not None:
            swap_manager.step_end()
        return grad_norm

    def _on_log_step(
        self,
        loss: torch.Tensor,
        grad_norm: float,
        aux_loss: Optional[torch.Tensor] = None,
        indexer_kl: Optional[torch.Tensor] = None,
    ):
        """Log training metrics if the logging interval is reached."""
        if self._global_step % self.config.run.logging_steps != 0:
            return
        loss_val = loss.item()
        aux_loss_val = aux_loss.item() if aux_loss is not None else None
        indexer_kl_val = indexer_kl.item() if indexer_kl is not None else None
        if self._monitor is not None:
            self._monitor.log(self._global_step, loss_val, grad_norm, aux_loss=aux_loss_val, indexer_kl=indexer_kl_val)
        else:
            parts = [f"Step {self._global_step} | Loss: {loss_val:.6f}"]
            if aux_loss_val is not None:
                parts.append(f"Aux Loss: {aux_loss_val:.6f}")
            if indexer_kl_val is not None:
                parts.append(f"Indexer KL: {indexer_kl_val:.6f}")
            print_rank(logger.info, f"  {' | '.join(parts)}")

    def _on_checkpoint_step(self):
        """Save checkpoint if the save interval is reached."""
        if self._ckpt_manager is not None and self._global_step % self.config.checkpoint.save_steps == 0:
            self._save_checkpoint()

    def train_step(self, batch):
        """Execute a single training step: forward + backward."""
        track_kl = (
            self.config.model.track_indexer_kl_loss and (self._global_step + 1) % self.config.run.logging_steps == 0
        )
        # Capture only the original forward. Fused backward keeps its collector on ctx.
        collector = (
            collect_indexer_kl_metrics(use_fused=self.config.model.use_fused_indexer_kl_loss)
            if track_kl
            else nullcontext(None)
        )
        with collector as metrics:
            outputs = self.model(**batch)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss. Make sure the batch contains labels.")
        loss = loss / self.config.run.gradient_accumulation_steps
        loss.backward()
        if metrics is not None:
            value = metrics.value(loss.detach()) / self.config.run.gradient_accumulation_steps
            self._accumulated_losses.setdefault("indexer_kl", []).append(value)
        aux_loss = outputs.aux_loss if hasattr(outputs, 'aux_loss') else None
        return loss, aux_loss

    @classmethod
    def sync_loss(cls, accumulated_losses: dict) -> dict:
        """Return detached FP32 step-loss sums averaged over data and CP ranks.

        train_step already divides the main loss by gradient_accumulation_steps,
        so summing its micro-step losses gives the local optimizer-step mean.
        Do not divide by the micro-step count again. Other metrics must supply
        their own normalization before being passed here.

        Average across the data mesh (the same mesh used by DistributedSampler),
        including FSDP ranks that consume distinct samples. Also retain the CP
        average: a CP-normalized loss replicated on CP ranks stays unchanged.
        TP replicas are not additional data samples and need no reduction.

        This only reduces detached logging values; backward losses and gradients
        are not modified. Empty metric lists return None and must be consistent
        across ranks participating in a reduction.
        """
        reduction_groups = []
        if torch.distributed.is_initialized():
            from fsdp_turbo.distributed.parallel_state import get_parallel_state

            parallel_state = get_parallel_state()
            for mesh_name in ("data", "cp"):
                group_size = parallel_state.get_group_size(mesh_name)
                if group_size > 1:
                    reduction_groups.append((parallel_state.get_group(mesh_name), group_size))

        result = {}
        for name, losses in accumulated_losses.items():
            if not losses:
                result[name] = None
                continue
            # Convert before summation to avoid BF16 accumulation rounding.
            synced = torch.stack([loss.detach().float() for loss in losses]).sum()
            for group, group_size in reduction_groups:
                torch.distributed.all_reduce(synced, op=torch.distributed.ReduceOp.SUM, group=group)
                synced.div_(group_size)
            result[name] = synced
        return result

    # ------------------------------------------------------------------
    # Distributed initialization
    # ------------------------------------------------------------------

    def _init_distributed(self):
        """Initialize distributed backend, process group, seed, and logging."""
        from fsdp_turbo.utils.device import set_accelerator_compatible

        backend = set_accelerator_compatible()
        if torch.distributed.is_initialized():
            return

        set_log_level("INFO")

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(local_rank)
        torch.accelerator.set_device(local_rank)
        apply_cpu_affinity_by_conf()

        self._set_seed()

    def _set_seed(self):
        """Set random seed. Override to customize seed source."""
        from fsdp_turbo.utils.random import set_seed

        set_seed(self.config.run.seed, set_deterministic=self.config.run.deterministic)

    # ------------------------------------------------------------------
    # Component builders - override in subclasses
    # ------------------------------------------------------------------

    def build_tokenizer(self):
        """Build tokenizer using TokenizerFactory."""
        if getattr(self.config.data, "is_multimodal", False):
            if self.processor is None:
                self.processor = self.build_processor()
            if self.processor is not None and hasattr(self.processor, "tokenizer"):
                return self.processor.tokenizer
        if not self.config.model.tokenizer_name_or_path:
            print_rank(logger.warning, "tokenizer_name_or_path is empty, skipping tokenizer build.")
            return None
        from fsdp_turbo.training.factories import TokenizerFactory

        print_rank(logger.info, "> Building Tokenizer...")
        return TokenizerFactory.create(self.config.model.tokenizer_name_or_path)

    def build_processor(self):
        """Build a multimodal processor from the configured model path."""
        if not getattr(self.config.data, "is_multimodal", False):
            return None
        source = self.config.model.tokenizer_name_or_path or self.config.model.model_name_or_path
        if not source:
            print_rank(logger.warning, "No model path configured, skipping processor build.")
            return None
        from transformers import AutoProcessor

        print_rank(logger.info, "> Building Processor...")
        return AutoProcessor.from_pretrained(source)

    def build_model(self):
        """Build model using ModelFactory."""
        if not self.config.model.model_name_or_path:
            print_rank(logger.warning, "model_name_or_path is empty, skipping model build.")
            return None
        from fsdp_turbo.training.factories import ModelFactory

        print_rank(logger.info, "> Building Model...")
        return ModelFactory.create(
            self.config.model.model_name_or_path,
            torch_dtype=get_dtype(self.config.model.torch_dtype),
        )

    def build_optimizer(self):
        """Build optimizer using OptimizerFactory."""
        if self.model is None:
            print_rank(logger.warning, "Model is None, skipping optimizer build.")
            return None
        print_rank(logger.info, "> Building Optimizer...")
        from fsdp_turbo.quantization.cache import hook_optimizer_step
        from fsdp_turbo.training.factories import OptimizerFactory

        kwargs = dict(
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        # Swap optimizer specific kwargs: None means use AdamWSwap internal defaults.
        # Only forward them for swap optimizers -- plain torch.optim optimizers
        # (e.g. AdamW) do not accept these and would raise on unexpected kwargs.
        if self.config.optimizer.optimizer_type == "AdamWSwap":
            if self.config.optimizer.mem_fraction_static is not None:
                kwargs["mem_fraction_static"] = self.config.optimizer.mem_fraction_static
            if self.config.optimizer.state_keys is not None:
                kwargs["state_keys"] = self.config.optimizer.state_keys
            if self.config.optimizer.param_prefetch is not None:
                kwargs["param_prefetch"] = self.config.optimizer.param_prefetch
            if self.config.optimizer.swap_chunk_numel is not None:
                kwargs["chunk_numel"] = self.config.optimizer.swap_chunk_numel
            if self.config.optimizer.resident_states_ratio is not None:
                kwargs["resident_states_ratio"] = self.config.optimizer.resident_states_ratio
            if self.config.optimizer.swap_prefetch_budget_ratio is not None:
                kwargs["prefetch_budget_ratio"] = self.config.optimizer.swap_prefetch_budget_ratio
            if self.config.optimizer.swap_prefetch_batches is not None:
                kwargs["prefetch_batches"] = self.config.optimizer.swap_prefetch_batches

        # ignored_modules (e.g. vision_tower.encoder.blocks, mm_projector) keep
        # plain nn.Parameter while FSDP-sharded language layers hold DTensor.
        # In one param group AdamW's foreach path buckets them by (device, dtype)
        # into the same _foreach_mul_ batch and raises "mixed torch.Tensor and
        # DTensor". Split them so every foreach bucket is homogeneous; the
        # hyperparameters stay identical, so the two groups train the same.
        dtensor_params, plain_params = [], []
        for param in self.model.parameters():
            if type(param) is DTensor:  # pylint: disable=unidiomatic-typecheck
                dtensor_params.append(param)
            else:
                plain_params.append(param)
        param_groups = []
        if dtensor_params:
            param_groups.append({"params": dtensor_params})
        if plain_params:
            param_groups.append({"params": plain_params})

        optimizer = OptimizerFactory.create(
            self.config.optimizer.optimizer_type,
            param_groups,
            **kwargs,
        )
        hook_optimizer_step(self.model, optimizer)
        self._register_optimizer_states_prefetch(optimizer)
        return optimizer

    def _register_optimizer_states_prefetch(self, optimizer):
        """Trigger optimizer states prefetch on the backward pre-hook of a configured layer."""
        pattern = self.config.optimizer.prefetch_states_at_backward
        if not pattern:
            return
        if not hasattr(optimizer, "prefetch_states_to_device"):
            print_rank(
                logger.warning,
                "[Swap Optimizer] prefetch_states_at_backward ignored: "
                "optimizer %s does not support states prefetch" % type(optimizer).__name__,
            )
            return

        # Lazy import: a module-level import would be circular (fsdp_turbo top level depends back on training)
        from fsdp_turbo.fsdp_turbo import FSDPTurbo
        from fsdp_turbo.utils.str_match import module_name_match

        # Unwrap the FSDPTurbo wrapper so module names share the namespace of the pattern
        search_root = self.model
        if isinstance(search_root, FSDPTurbo):
            search_root = search_root.model
        for name, module in search_root.named_modules():
            if module_name_match(pattern, name):

                def _backward_pre_hook(_module, _grad_output, _optimizer=optimizer, _trainer=self):
                    # Gradient accumulation: defer to the last micro-batch's backward
                    if getattr(_trainer, "_is_last_micro_step", True):
                        _optimizer.prefetch_states_to_device()

                module.register_full_backward_pre_hook(_backward_pre_hook)
                print_rank(
                    logger.info,
                    "[Swap Optimizer] states prefetch trigger at <%s> (backward pre-hook%s)"
                    % (
                        name,
                        ", deferred to the last micro-batch under gradient accumulation"
                        if self.config.run.gradient_accumulation_steps > 1
                        else "",
                    ),
                )
                return
        print_rank(
            logger.warning,
            "[Swap Optimizer] WARNING: trigger module <%s> not found, states will still be H2D inside step()" % pattern,
        )

    def build_scheduler(self):
        """Build LR scheduler using SchedulerFactory."""
        if self.optimizer is None:
            print_rank(logger.warning, "Optimizer is None, skipping scheduler build.")
            return None
        print_rank(logger.info, "> Building LR Scheduler...")
        from fsdp_turbo.training.factories import SchedulerFactory

        max_steps = self._resolve_max_steps()
        return SchedulerFactory.create(
            self.config.optimizer.lr_scheduler_type,
            self.optimizer,
            T_max=max_steps,
            eta_min=self.config.optimizer.min_lr,
        )

    def build_dataloader(self):
        """Build dataloader from HuggingFace datasets."""
        from fsdp_turbo.training.dataloader import build_dataloader

        return build_dataloader(self.config.data, tokenizer=self.tokenizer, processor=self.processor)

    def build_checkpointer(self):
        """Build checkpoint manager."""
        print_rank(logger.info, "> Building Checkpointer...")
        return Checkpointer(
            self.config.checkpoint.output_dir,
            save_optim=self.config.checkpoint.save_optim,
            load_optim=self.config.checkpoint.load_optim,
        )

    def build_monitor(self):
        """Build training monitor."""
        print_rank(logger.info, "> Building Monitor...")
        return Monitor(logging_steps=self.config.run.logging_steps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_profiler(self):
        from fsdp_turbo.training.profiler import create_profiler

        return create_profiler(self.config.run.profile)

    def _create_memory_tracker(self):
        """Device memory snapshot tracker (config: run.memory_snapshot)."""
        from fsdp_turbo.training.memory_snapshot import build_memory_snapshot_tracker

        return build_memory_snapshot_tracker(self.config.run.memory_snapshot, self.config.checkpoint.output_dir)

    def _resolve_max_steps(self):
        if self.config.run.max_steps > 0:
            return self.config.run.max_steps
        if self.dataloader is not None:
            try:
                num_batches = len(self.dataloader)
                return num_batches * self.config.run.num_train_epochs // self.config.run.gradient_accumulation_steps
            except TypeError:
                pass
        return 100000

    def prepare_batch(self, batch):
        """Move a batch to device and shard model inputs over the full CP group."""
        batch = self._move_batch_to_device(batch)
        return self._split_batch_for_context_parallel(batch)

    def _move_batch_to_device(self, batch):
        device = torch.accelerator.current_accelerator()
        if isinstance(batch, Mapping):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
            }
        return batch

    def _is_vlm_model(self):
        """Return whether the model architecture contains a vision tower."""
        model = self.model
        visited = set()
        while model is not None and id(model) not in visited:
            visited.add(id(model))
            model_config = getattr(model, "config", None)
            if model_config is not None and hasattr(model_config, "vision_config"):
                return getattr(model_config, "vision_config", None) is not None
            model = getattr(model, "model", None) or getattr(model, "module", None)
        return False

    def _split_batch_for_context_parallel(self, batch):
        """Split once over flattened KV-AllGather x Ulysses ranks."""
        parallel_state = self.model.parallel_state
        if parallel_state.get_cp_group_size() <= 1:
            return batch
        if self._is_vlm_model():
            return batch

        from fsdp_turbo.distributed.context_parallel.batch import split_sequence_batch

        cp_plan = self.config.distributed.cp_plan if self.config is not None else None
        return split_sequence_batch(
            batch,
            group=parallel_state.get_cp_group(),
            sequence_keys=("input_ids", "attention_mask", "mm_token_type_ids"),
            build_position_ids=True,
            zigzag=uses_hybrid_cp_zigzag_layout(cp_plan),
        )

    def _clip_grad_norm(self):
        """Clip gradient norm and return the total norm before clipping.

        Uses clip_grad and clip_grad_norm_type from OptimizerConfig.
        Delegates to the FSDP-aware clip_grads module.
        Returns the gradient norm before clipping for monitoring.
        """
        from fsdp_turbo.training.clip_grads import clip_grad_norm

        keep_on_device = False
        if self.config.optimizer.clip_grad_keep_on_device:
            from fsdp_turbo.memory.swap_optimizer import AdamWSwap

            # Only AdamWSwap handles device-resident grads with host params
            # (it uploads both); any other optimizer would fail with params on
            # host and grads on device.
            if isinstance(self.optimizer, AdamWSwap):
                keep_on_device = True
            else:
                logger.warning(
                    "optimizer.clip_grad_keep_on_device=True is ignored because the "
                    "optimizer is not AdamWSwap; falling back to host gradient write-back."
                )

        return clip_grad_norm(
            self.model,
            max_norm=self.config.optimizer.clip_grad,
            norm_type=self.config.optimizer.clip_grad_norm_type,
            group=self._get_grad_sync_group(),
            keep_on_device=keep_on_device,
        )

    def _compute_grad_norm(self):
        """Compute the total gradient norm without clipping."""
        from fsdp_turbo.training.clip_grads import compute_grad_norm

        return compute_grad_norm(
            list(self.model.parameters()),
            norm_type=self.config.optimizer.clip_grad_norm_type,
            group=self._get_grad_sync_group(),
        )

    def _get_grad_sync_group(self):
        """Return the process group for gradient norm all-reduce.

        Override this in subclasses to provide a specific group (e.g.
        the data-parallel group).  Returns ``None`` by default, which
        lets ``clip_grads`` infer the group from DTensor gradients
        automatically.
        """
        return None

    def _save_checkpoint(self):
        if self._ckpt_manager is not None and hasattr(self._ckpt_manager, "save"):
            self._ckpt_manager.save(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self._global_step,
                data_cfg=self.config.data,
                dataloader=self.dataloader,
                consumed_samples=self._consumed_samples,
            )

    def _load_checkpoint(self, checkpoint_path):
        if self._ckpt_manager is not None and hasattr(self._ckpt_manager, "load"):
            ckpt_info = self._ckpt_manager.load(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                strict=self.config.checkpoint.strict,
            )
            # Restore global_step only if the dataset hasn't changed.
            saved_fp = ckpt_info.get("dataset_fingerprint", "")
            current_fp = None
            if self.config is not None and self.config.data is not None:
                from fsdp_turbo.training.checkpoint import dataset_fingerprint

                current_fp = dataset_fingerprint(self.config.data)

            if saved_fp and saved_fp == current_fp:
                self._global_step = ckpt_info.get("global_step", 0)
                self._consumed_samples = ckpt_info.get("consumed_samples", 0)
                print_rank(
                    logger.info,
                    f"Restored global_step to {self._global_step}, "
                    f"consumed_samples to {self._consumed_samples} (dataset unchanged)",
                )
            else:
                self._global_step = 0
                self._consumed_samples = 0
                print_rank(logger.warning, "Dataset changed since checkpoint; global_step and consumed_samples reset")

            # Restore dataloader sampler state if available.
            dl_state = ckpt_info.get("dataloader_state_dict")
            if dl_state is not None and self.dataloader is not None:
                if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "load_state_dict"):
                    self.dataloader.sampler.load_state_dict(dl_state)
