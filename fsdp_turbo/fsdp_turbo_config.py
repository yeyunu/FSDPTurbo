# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Callable, Literal, Union, Optional
from pathlib import Path

import torch
import yaml

from fsdp_turbo.utils.dtype import get_dtype

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    torch_dtype: torch.dtype = torch.bfloat16


@dataclass
class OptimizerConfig:
    optimizer_type: str = "AdamW"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr: float = 1e-4
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"
    min_lr: float = 0.0


@dataclass
class ProfileConfig:
    enabled: bool = False
    wait_steps: int = 0
    warmup_steps: int = 0
    active_steps: int = 1
    repeat: int = 1
    skip_first: int = 0
    output_dir: str = "./profile"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False


@dataclass
class DataConfig:
    dataset_path: str = ""
    dataset_config: Optional[str] = None
    text_column: str = "text"
    split: str = "train"
    batch_size: int = 1
    max_seq_length: int = 4096
    num_workers: int = 0
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class TrainRunConfig:
    seed: int = 42
    max_steps: int = -1
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    logging_steps: int = 1
    output_dir: str = "./output"
    resume_from_checkpoint: Optional[str] = None
    profile: ProfileConfig = field(default_factory=ProfileConfig)


@dataclass
class FSDPPlanConfig:
    ignored_modules: List[str] = None
    apply_modules: Dict[str, Any] = None

    # mp_policy settings
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    cast_forward_inputs: bool = True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0

    # fsdp2 hook manager
    hook_modules: Optional[List[str]] = None

    # FSDP implementation strategy
    # 'custom': Use FSDPTurbo custom FSDP implementation
    # 'native': Use PyTorch native FSDP implementation (default)
    fsdp_implementation: Literal['custom', 'native'] = 'native'


@dataclass
class TPPlanConfig:
    colwise_parallel: List[str] = None
    rowwise_parallel: List[str] = None
    sequence_parallel: List[str] = None


@dataclass
class CPPlanConfig:
    pass


@dataclass
class EPPlanConfig:
    apply_modules: List[str] = None
    dispatcher: Union[Literal["eager", "fused", "mc2"], Callable] = None
    apply_efsdp_modules: List[str] = None
    _gradient_divide_factor: float = None


@dataclass
class QuantizeConfig:
    quant_format: Optional[str] = None
    quant_recipe: Optional[str] = None
    block_size: int = 32
    quant_apply_modules: List[str] = None
    quant_ignored_modules: List[str] = None
    converters: List[str] = None
    quant_gmm: bool = False
    gemm_gradient_accumulation_fusion: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)  # for future extensibility


@dataclass
class DistributedConfig:
    data_parallel_size: int = 1

    fully_shard_parallel_size: int = 1
    fsdp_plan: FSDPPlanConfig = None

    tensor_parallel_size: int = 1
    tp_plan: TPPlanConfig = None

    context_parallel_size: int = 1
    ulysses_parallel_size: int = 1

    expert_parallel_size: int = 1
    expert_fully_shard_parallel_size: int = 1
    expert_data_parallel_size: int = 1
    ep_plan: EPPlanConfig = None


@dataclass
class MemoryConfig:
    recompute: bool = False
    recompute_plan: List[str] = None


@dataclass
class QuantizationConfig:
    quantization_plan: Optional[QuantizeConfig] = None


@dataclass
class FSDPTurboConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: TrainRunConfig = field(default_factory=TrainRunConfig)

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    def __post_init__(self):
        self.validate_tp_config()
        self.validate_ep_config()
        self.validate_recompute_config()
        self.validate_quantization_config()
        self.validate_fsdp_config()

    def validate_fsdp_config(self):
        ''' fully shard plan
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                fsdp_plan=FSDPPlanConfig(
                    'ignored_modules':['*mlp.experts*'],
                    'apply_modules': {
                        'model.layers.*': {reshard_after_forward=None, shard_placement_fn=None}
                    }
                )
            )
        )
        '''
        self.distributed.fsdp_plan = FSDPPlanConfig() if self.distributed.fsdp_plan is None else self.distributed.fsdp_plan
        if self.distributed.fully_shard_parallel_size > 1:
            if self.distributed.expert_parallel_size > 1:
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.ep_plan.apply_modules)
            if self.distributed.tensor_parallel_size > 1:
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.tp_plan.colwise_parallel)
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.tp_plan.rowwise_parallel)
            self.distributed.fsdp_plan.ignored_modules = list(
                set(self.distributed.fsdp_plan.ignored_modules))  # remove duplicates

    def validate_tp_config(self):
        ''' tensor parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                tp_plan=TPPlanConfig(
                    colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                    rowwise_parallel=['*.o_proj']
                )
            )
        )
        '''
        self.distributed.tp_plan = TPPlanConfig() if self.distributed.tp_plan is None else self.distributed.tp_plan
        self.distributed.tp_plan.colwise_parallel = [] if self.distributed.tp_plan.colwise_parallel is None else self.distributed.tp_plan.colwise_parallel
        self.distributed.tp_plan.rowwise_parallel = [] if self.distributed.tp_plan.rowwise_parallel is None else self.distributed.tp_plan.rowwise_parallel
        self.distributed.tp_plan.sequence_parallel = [] if self.distributed.tp_plan.sequence_parallel is None else self.distributed.tp_plan.sequence_parallel

    def validate_ep_config(self):
        ''' expert parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                ep_plan=EPPlanConfig(
                    apply_modules: ['*mlp.experts*'],
                    dispatcher: 'eager', 'fused', 'mc2'
                )
            )
        )
        '''
        self.distributed.ep_plan = EPPlanConfig(apply_modules=[],
                                                dispatcher='eager') if self.distributed.ep_plan is None else self.distributed.ep_plan
        self.distributed.ep_plan._gradient_divide_factor = self.distributed.expert_parallel_size * self.distributed.expert_fully_shard_parallel_size * self.distributed.expert_data_parallel_size
        if self.distributed.ep_plan.apply_efsdp_modules is None:
            self.distributed.ep_plan.apply_efsdp_modules = []
            for ep_module in self.distributed.ep_plan.apply_modules:
                if ep_module.endswith('.experts'):
                    self.distributed.ep_plan.apply_efsdp_modules.append(ep_module.removesuffix('.experts'))

    def validate_recompute_config(self):
        self.memory.recompute_plan = [] if self.memory.recompute_plan is None else self.memory.recompute_plan

    def validate_quantization_config(self):
        self.quantization.quantization_plan = QuantizeConfig() if self.quantization.quantization_plan is None else self.quantization.quantization_plan

    def __str__(self):
        import dataclasses as dc

        def _format_value(v):
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return "[]"
                if len(v) <= 3:
                    return str(v)
                return f"[{v[0]}, {v[1]}, ... ] (len={len(v)})"
            if isinstance(v, dict):
                if len(v) == 0:
                    return "{}"
                return str(v)
            if isinstance(v, torch.dtype):
                return str(v)
            return repr(v)

        def _format_dataclass(obj, indent=2):
            lines = []
            prefix = " " * indent
            for f in dc.fields(obj):
                val = getattr(obj, f.name)
                if val is None:
                    lines.append(f"{prefix}{f.name}: None")
                elif dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                    lines.append(f"{prefix}{f.name}:")
                    lines.extend(_format_dataclass(val, indent + 2))
                else:
                    lines.append(f"{prefix}{f.name}: {_format_value(val)}")
            return lines

        sep = "=" * 60
        sections = [
            ("Model", self.model),
            ("Optimizer", self.optimizer),
            ("Data", self.data),
            ("Run", self.run),
            ("Distributed", self.distributed),
            ("Memory", self.memory),
            ("Quantization", self.quantization),
        ]

        lines = [sep, f'{"FSDPTurboConfig":^60}', sep]
        for section_name, section_obj in sections:
            lines.append(f"  [{section_name}]")
            lines.extend(_format_dataclass(section_obj, indent=4))
            lines.append("")
        lines.append(sep)
        return "\n".join(lines)


def _coerce_value(value, field_type):
    if not isinstance(value, str):
        return value

    origin = getattr(field_type, "__origin__", None)
    if origin is Union:
        args = [a for a in field_type.__args__ if a is not type(None)]
        if args:
            field_type = args[0]

    if field_type is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert '{value}' to float, keeping as str")
            return value

    if field_type is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert '{value}' to int, keeping as str")
            return value

    if field_type is torch.dtype:
        try:
            return get_dtype(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert '{value}' to torch.dtype: {e}, keeping as str")
            return value

    return value


def _dict_to_dataclass(cls, data):
    if data is None:
        return None
    if not isinstance(data, dict):
        return data
    import dataclasses as dc
    field_types = {f.name: f.type for f in dc.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            logger.warning(f"Ignoring unknown config key '{key}' for {cls.__name__}")
            continue
        ft = field_types[key]
        if dc.is_dataclass(ft) and isinstance(value, dict):
            actual_cls = ft
            kwargs[key] = _dict_to_dataclass(actual_cls, value)
        else:
            kwargs[key] = _coerce_value(value, ft)
    return cls(**kwargs)


def load_config_from_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Extract top-level sub-configs (promoted from old TrainConfig)
    model_raw = raw.pop("model", None)
    optimizer_raw = raw.pop("optimizer", None)
    data_raw = raw.pop("data", None)
    run_raw = raw.pop("run", None)

    model_cfg = _dict_to_dataclass(ModelConfig, model_raw) if model_raw is not None else ModelConfig()
    optimizer_cfg = _dict_to_dataclass(OptimizerConfig,
                                       optimizer_raw) if optimizer_raw is not None else OptimizerConfig()
    data_cfg = _dict_to_dataclass(DataConfig, data_raw) if data_raw is not None else DataConfig()
    run_cfg = _dict_to_dataclass(TrainRunConfig, run_raw) if run_raw is not None else TrainRunConfig()

    # Extract distributed sub-config
    distributed_raw = raw.pop("distributed", None)
    if distributed_raw is not None:
        # Extract plan configs from distributed dict
        fsdp_plan_raw = distributed_raw.pop("fsdp_plan", None)
        tp_plan_raw = distributed_raw.pop("tp_plan", None)
        ep_plan_raw = distributed_raw.pop("ep_plan", None)

        fsdp_plan = _dict_to_dataclass(FSDPPlanConfig, fsdp_plan_raw) if fsdp_plan_raw is not None else None
        tp_plan = _dict_to_dataclass(TPPlanConfig, tp_plan_raw) if tp_plan_raw is not None else None
        ep_plan = _dict_to_dataclass(EPPlanConfig, ep_plan_raw) if ep_plan_raw is not None else None

        distributed_valid_fields = {f.name for f in fields(DistributedConfig)}
        extra_keys = set(distributed_raw.keys()) - distributed_valid_fields
        if extra_keys:
            logger.warning(f"Ignoring unknown distributed config keys: {extra_keys}")
        filtered_distributed_raw = {k: v for k, v in distributed_raw.items() if k in distributed_valid_fields}

        distributed_cfg = DistributedConfig(fsdp_plan=fsdp_plan, tp_plan=tp_plan, ep_plan=ep_plan,
                                            **filtered_distributed_raw)
    else:
        distributed_cfg = DistributedConfig()

    # Extract memory sub-config
    memory_raw = raw.pop("memory", None)
    if memory_raw is not None:
        memory_cfg = _dict_to_dataclass(MemoryConfig, memory_raw)
    else:
        memory_cfg = MemoryConfig()

    # Extract quantization sub-config
    quantization_raw = raw.pop("quantization", None)
    if quantization_raw is not None:
        quantization_plan_raw = quantization_raw.pop("quantization_plan", None)
        quantization_plan = _dict_to_dataclass(QuantizeConfig,
                                               quantization_plan_raw) if quantization_plan_raw is not None else None
        quantization_cfg = QuantizationConfig(quantization_plan=quantization_plan)
    else:
        quantization_cfg = QuantizationConfig()

    valid_fields = {f.name for f in fields(FSDPTurboConfig)}
    extra_keys = set(raw.keys()) - valid_fields
    if extra_keys:
        logger.warning(f"Ignoring unknown top-level config keys: {extra_keys}")
    filtered_raw = {k: v for k, v in raw.items() if k in valid_fields}

    return FSDPTurboConfig(
        model=model_cfg,
        optimizer=optimizer_cfg,
        data=data_cfg,
        run=run_cfg,
        distributed=distributed_cfg,
        memory=memory_cfg,
        quantization=quantization_cfg,
        **filtered_raw,
    )
