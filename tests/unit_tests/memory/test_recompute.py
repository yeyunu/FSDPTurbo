from unittest.mock import Mock, patch

from fsdp_turbo.memory.recompute.recompute import recompute_wrapper


def test_recompute_wrapper_installs_mxfp8_replay_context():
    module = Mock()
    function = Mock()
    wrapped = recompute_wrapper(module, function)

    with patch("fsdp_turbo.memory.recompute.recompute.checkpoint", return_value="output") as mock_checkpoint:
        assert wrapped("input") == "output"

    args, kwargs = mock_checkpoint.call_args
    assert args == (function, "input")
    assert kwargs["use_reentrant"] is False
    assert callable(kwargs["context_fn"])
    forward_context, replay_context = kwargs["context_fn"]()
    assert forward_context is not None
    assert replay_context is not None
