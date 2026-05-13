import os
import logging
from contextlib import nullcontext
from typing import Optional

import torch

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.utils.dtype import get_dtype
from fsdp_turbo.utils.log import set_log_level, print_rank
from fsdp_turbo.training.checkpoint import Checkpointer
from fsdp_turbo.training.monitor import Monitor

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer with common training lifecycle.

    Subclasses override builder methods to customize component creation.
    Default flow: init distributed -> build components -> train.

    Attributes:
        model: The training model (after parallelism wrapping).
        tokenizer: The tokenizer instance.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler.
        data_manager: The data loader / manager.
        config: The training configuration.
    """

    def __init__(self, config: Optional[FSDPTurboConfig] = None):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        self._ckpt_manager = None
        self._monitor = None
        self._global_step = 0

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
        checkpoint = resume_from_checkpoint or self.config.run.resume_from_checkpoint

        if checkpoint and self._ckpt_manager is not None:
            self._load_checkpoint(checkpoint)

        self.model.train()

        max_steps = self._resolve_max_steps()

        if self.dataloader is None:
            raise RuntimeError("dataloader is None. Override build_data_manager() to provide a dataloader.")

        print_rank(logger.info, "***** Running training *****")
        print_rank(logger.info, f"  Max steps = {max_steps}")
        print_rank(logger.info, f"  Gradient accumulation steps = {self.config.run.gradient_accumulation_steps}")

        self._global_step = 0
        with self._create_profiler() as prof:
            for epoch in range(self.config.run.num_train_epochs):
                print_rank(logger.info, f"  Epoch {epoch + 1}/{self.config.run.num_train_epochs}")
                for step, batch in enumerate(self.dataloader):
                    loss = self.train_step(batch)

                    if (step + 1) % self.config.run.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self._global_step += 1

                        if self._global_step % self.config.run.logging_steps == 0:
                            loss_val = loss.item()
                            print_rank(logger.info, f"  Step {self._global_step} | Loss: {loss_val:.6f}")
                            if self._monitor is not None:
                                self._monitor.log(self._global_step, loss_val)

                        if self._ckpt_manager is not None and self._global_step % self.config.run.save_steps == 0:
                            self._save_checkpoint()

                        if prof is not None:
                            prof.step()

                    if 0 < max_steps <= self._global_step:
                        break

                if 0 < max_steps <= self._global_step:
                    break

        print_rank(logger.info, f"Training complete. Total steps: {self._global_step}")
        if self._monitor is not None:
            self._monitor.summary()

    def train_step(self, batch):
        """Execute a single training step: forward + backward."""
        batch = self._move_batch_to_device(batch)
        outputs = self.model(**batch)
        loss = outputs.loss / self.config.run.gradient_accumulation_steps
        loss.backward()
        return loss

    # ------------------------------------------------------------------
    # Distributed initialization
    # ------------------------------------------------------------------

    def _init_distributed(self):
        """Initialize distributed backend, process group, seed, and logging."""
        if torch.distributed.is_initialized():
            return

        try:
            import torch_npu  # noqa: F401
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
            fallback, backend = torch.npu, "hccl"
        except ImportError:
            if torch.cuda.is_available():
                fallback, backend = torch.cuda, "nccl"
            else:
                raise RuntimeError("No available accelerator (NPU/CUDA).")

        from fsdp_turbo.utils.device import set_accelerator_compatible
        set_accelerator_compatible(fallback)
        set_log_level("INFO")

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(local_rank)
        torch.accelerator.set_device(local_rank)

        self._set_seed()

    def _set_seed(self):
        """Set random seed. Override to customize seed source."""
        from fsdp_turbo.utils.random import set_seed
        set_seed(self.config.run.seed, set_deterministic=True)

    # ------------------------------------------------------------------
    # Component builders - override in subclasses
    # ------------------------------------------------------------------

    def build_tokenizer(self):
        """Build tokenizer using TokenizerFactory."""
        if not self.config.model.tokenizer_name_or_path:
            print_rank(logger.warning, "tokenizer_name_or_path is empty, skipping tokenizer build.")
            return None
        from fsdp_turbo.training.factories import TokenizerFactory
        print_rank(logger.info, "> Building Tokenizer...")
        return TokenizerFactory.create(self.config.model.tokenizer_name_or_path)

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
        from fsdp_turbo.training.factories import OptimizerFactory
        print(self.config.optimizer.lr, type(self.config.optimizer.lr))
        return OptimizerFactory.create(
            self.config.optimizer.optimizer_type,
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
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
        data_cfg = self.config.data
        if data_cfg is None or not data_cfg.dataset_path:
            print_rank(logger.warning, "dataset_path is empty, skipping dataloader build.")
            return None

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("`datasets` library is required for build_dataloader. Install it with: pip install datasets")
        
        print_rank(logger.info, f"> Building DataLoader for {data_cfg.dataset_path} {data_cfg.dataset_config} {data_cfg.split}")
        dataset = load_dataset(data_cfg.dataset_path, data_cfg.dataset_config, split=data_cfg.split)

        if self.tokenizer is not None:
            def tokenize_function(examples):
                return self.tokenizer(
                    examples[data_cfg.text_column],
                    truncation=True,
                    max_length=data_cfg.max_seq_length,
                    padding="max_length",
                )

            dataset = dataset.map(tokenize_function, batched=True)
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

            def add_labels(examples):
                examples["labels"] = examples["input_ids"].clone()
                return examples

            dataset = dataset.map(add_labels)

        sampler = None
        if torch.distributed.is_initialized():
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=data_cfg.shuffle)
            shuffle = False
        else:
            shuffle = data_cfg.shuffle

        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )

    def build_checkpointer(self):
        """Build checkpoint manager."""
        print_rank(logger.info, "> Building Checkpointer...")
        return Checkpointer(self.config.run.output_dir)

    def build_monitor(self):
        """Build training monitor."""
        print_rank(logger.info, "> Building Monitor...")
        return Monitor(logging_steps=self.config.run.logging_steps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_profiler(self):
        profile_cfg = self.config.run.profile
        if not profile_cfg.enabled:
            return nullcontext()

        schedule = torch.profiler.schedule(
            wait=profile_cfg.wait_steps,
            warmup=profile_cfg.warmup_steps,
            active=profile_cfg.active_steps,
            repeat=profile_cfg.repeat,
            skip_first=profile_cfg.skip_first,
        )

        try:
            import torch_npu
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            return torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
                record_shapes=profile_cfg.record_shapes,
                profile_memory=profile_cfg.profile_memory,
                with_stack=profile_cfg.with_stack,
                experimental_config=experimental_config,
                schedule=schedule,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_cfg.output_dir),
            )
        except ImportError:
            return torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=profile_cfg.record_shapes,
                profile_memory=profile_cfg.profile_memory,
                with_stack=profile_cfg.with_stack,
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_cfg.output_dir),
            )

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

    def _move_batch_to_device(self, batch):
        device = torch.accelerator.current_accelerator()
        if isinstance(batch, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def _save_checkpoint(self):
        if self._ckpt_manager is not None and hasattr(self._ckpt_manager, "save"):
            self._ckpt_manager.save(self.model, self.optimizer, self.lr_scheduler, self._global_step)

    def _load_checkpoint(self, checkpoint_path):
        if self._ckpt_manager is not None and hasattr(self._ckpt_manager, "load"):
            self._ckpt_manager.load(checkpoint_path, self.model, self.optimizer, self.lr_scheduler)
