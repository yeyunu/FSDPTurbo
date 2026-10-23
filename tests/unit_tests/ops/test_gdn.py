# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops.registry import get_op


def test_chunk_gdn_npu_registry_uses_ascendc_when_available():
    from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE, IS_MINDSPEED_OPS_AVAILABLE

    if not (IS_FLA_NPU_AVAILABLE and IS_MINDSPEED_OPS_AVAILABLE):
        import pytest

        pytest.skip('AscendC GDN backend is not available')

    registered = get_op('chunk_gated_delta_rule', 'npu')
    assert registered.__name__ == 'chunk_gdn_npu'


def test_fused_recurrent_gdn_npu_registry_available():
    from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE

    if not IS_FLA_NPU_AVAILABLE:
        import pytest

        pytest.skip('FLA NPU backend is not available')

    registered = get_op('fused_recurrent_gated_delta_rule', 'npu')
    assert registered.__name__ == 'fused_recurrent_gdn_npu'


def test_chunk_gdn_cuda_registry_available():
    try:
        import fla  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip('FLA (CUDA) backend is not available')

    registered = get_op('chunk_gated_delta_rule', 'cuda')
    assert registered.__name__ == 'chunk_gdn_cuda'


def test_fused_recurrent_gdn_cuda_registry_available():
    try:
        import fla  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip('FLA (CUDA) backend is not available')

    registered = get_op('fused_recurrent_gated_delta_rule', 'cuda')
    assert registered.__name__ == 'fused_recurrent_gdn_cuda'
