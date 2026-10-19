import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

import fsdp_turbo.distributed.parallel_state as parallel_state_module
import fsdp_turbo.training.trainer as trainer_module
from fsdp_turbo.ops.triton.utils import custom_device_ctx
from fsdp_turbo.utils import device as device_utils


@pytest.fixture(name="accelerator_backend", params=["npu", "cuda"])
def mock_accelerator_backend(monkeypatch, request):
    device_type = request.param
    accelerator = ModuleType("torch.accelerator")
    accelerator.current_accelerator = lambda: SimpleNamespace(type=device_type)
    accelerator.current_device_index = lambda: 2
    accelerator.set_device_index = Mock()
    context = MagicMock()
    backend = ModuleType(f"torch.{device_type}")
    backend.device = Mock(return_value=context)
    backend.set_device = Mock()
    backend.is_available = Mock(return_value=True)
    monkeypatch.setattr(torch, "accelerator", accelerator)
    monkeypatch.setitem(sys.modules, "torch.accelerator", accelerator)
    monkeypatch.setattr(torch, device_type, backend, raising=False)
    if device_type == "npu":
        npu_package = ModuleType("torch_npu")
        npu_package.__path__ = []
        contrib_package = ModuleType("torch_npu.contrib")
        contrib_package.transfer_to_npu = ModuleType("torch_npu.contrib.transfer_to_npu")
        monkeypatch.setitem(sys.modules, "torch_npu", npu_package)
        monkeypatch.setitem(sys.modules, "torch_npu.contrib", contrib_package)
    else:
        monkeypatch.setitem(sys.modules, "torch_npu", None)
    return SimpleNamespace(
        device_type=device_type,
        process_group_backend="hccl" if device_type == "npu" else "nccl",
        accelerator=accelerator,
        backend=backend,
        context=context,
    )


def _assert_device_context(backend):
    assert torch.accelerator.current_device_index() == 2
    with custom_device_ctx(SimpleNamespace(type=backend.device_type, index=2)):
        backend.context.__enter__.assert_called_once()
    backend.backend.device.assert_called_once_with(2)
    backend.context.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("process_group_initialized", [False, True])
def test_parallel_state_always_installs_device_context(monkeypatch, accelerator_backend, process_group_initialized):
    compatibility_setup = Mock(wraps=device_utils.set_accelerator_compatible)
    initialize_process_group = Mock()
    monkeypatch.setattr(device_utils, "set_accelerator_compatible", compatibility_setup)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: process_group_initialized)
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize_process_group)
    monkeypatch.setattr(parallel_state_module, "print_rank", Mock())
    monkeypatch.setattr(parallel_state_module.Singleton, "_instances", {})

    def create_mesh(state, mesh_dim_names, mesh_shape):
        assert hasattr(torch.accelerator, "device")
        return 1

    monkeypatch.setattr(parallel_state_module.ParallelState, "_create_mesh", create_mesh)
    monkeypatch.setattr(parallel_state_module.ParallelState, "add_sub_device_mesh", Mock())
    parallel_state_module.ParallelState()

    compatibility_setup.assert_called_once_with()
    _assert_device_context(accelerator_backend)
    if process_group_initialized:
        initialize_process_group.assert_not_called()
    else:
        initialize_process_group.assert_called_once_with(
            backend=f"cpu:gloo,{accelerator_backend.device_type}:{accelerator_backend.process_group_backend}"
        )


@pytest.mark.parametrize("process_group_initialized", [False, True])
def test_trainer_always_installs_device_context(monkeypatch, accelerator_backend, process_group_initialized):
    compatibility_setup = Mock(wraps=device_utils.set_accelerator_compatible)
    initialize_process_group = Mock()
    set_log_level = Mock()
    apply_cpu_affinity = Mock()
    monkeypatch.setattr(device_utils, "set_accelerator_compatible", compatibility_setup)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: process_group_initialized)
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize_process_group)
    monkeypatch.setattr(trainer_module, "set_log_level", set_log_level)
    monkeypatch.setattr(trainer_module, "apply_cpu_affinity_by_conf", apply_cpu_affinity)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    trainer = trainer_module.BaseTrainer()
    trainer._set_seed = Mock()

    trainer._init_distributed()

    compatibility_setup.assert_called_once_with()
    _assert_device_context(accelerator_backend)
    if process_group_initialized:
        initialize_process_group.assert_not_called()
        accelerator_backend.accelerator.set_device_index.assert_not_called()
        accelerator_backend.backend.set_device.assert_not_called()
        apply_cpu_affinity.assert_not_called()
        set_log_level.assert_not_called()
        trainer._set_seed.assert_not_called()
    else:
        initialize_process_group.assert_called_once_with(
            backend=accelerator_backend.process_group_backend, rank=1, world_size=4
        )
        accelerator_backend.accelerator.set_device_index.assert_called_once_with(2)
        accelerator_backend.backend.set_device.assert_called_once_with(2)
        apply_cpu_affinity.assert_called_once_with()
        set_log_level.assert_called_once_with("INFO")
        trainer._set_seed.assert_called_once_with()


def test_repeated_compatibility_setup_keeps_device_context(accelerator_backend):
    assert not hasattr(torch.accelerator, "device")
    assert device_utils.set_accelerator_compatible() == accelerator_backend.process_group_backend
    assert device_utils.set_accelerator_compatible() == accelerator_backend.process_group_backend
    _assert_device_context(accelerator_backend)
