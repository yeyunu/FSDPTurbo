import unittest

import torch
import torch.nn.functional as F

from fsdp_turbo.modules.kda import _sliced_short_convolution


class _ShortConv(torch.nn.Conv1d):
    def __init__(self, channels, kernel_size, activation="silu", bias=False):
        super().__init__(
            channels,
            channels,
            kernel_size,
            groups=channels,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x,
        weight=None,
        cache=None,
        output_final_state=False,
        cu_seqlens=None,
    ):
        del cache, output_final_state, cu_seqlens
        weight = self.weight if weight is None else weight
        output = F.conv1d(
            x.transpose(1, 2),
            weight,
            bias=None,
            padding=self.padding[0],
            groups=x.size(-1),
        )[..., : x.size(1)]
        if self.activation in ("silu", "swish"):
            output = F.silu(output)
        elif self.activation is not None:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return output.transpose(1, 2), None


class KDAUlyssesTests(unittest.TestCase):
    def test_sliced_short_convolution_matches_full_channel_slice(self):
        torch.manual_seed(7)
        conv = _ShortConv(channels=4, kernel_size=3)
        complete = torch.randn(2, 6, 4)
        rank = 1
        local = complete[..., 2:]

        actual = _sliced_short_convolution(conv, local, rank=rank)
        expected = F.conv1d(
            complete.transpose(1, 2),
            conv.weight,
            bias=None,
            padding=2,
            groups=4,
        )[..., : complete.size(1)]
        expected = F.silu(expected).transpose(1, 2)[..., 2:]
        torch.testing.assert_close(actual, expected)

    def test_sliced_short_convolution_rejects_unknown_activation(self):
        conv = _ShortConv(channels=2, kernel_size=2, activation="gelu")
        with self.assertRaisesRegex(ValueError, "activation"):
            _sliced_short_convolution(conv, torch.randn(1, 3, 1), rank=0)

    def test_sliced_short_convolution_rejects_bias(self):
        conv = _ShortConv(channels=2, kernel_size=2, bias=True)
        with self.assertRaisesRegex(ValueError, "bias=False"):
            _sliced_short_convolution(conv, torch.randn(1, 3, 1), rank=0)


if __name__ == "__main__":
    unittest.main()
