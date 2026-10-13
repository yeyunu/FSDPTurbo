from unittest import mock

import torch

from fsdp_turbo.models.deepseek_v41.model import ModelArgs
from fsdp_turbo.models.deepseek_v41.vision import Aligner, ViT


def _model_args():
    return ModelArgs(
        dim=8,
        vision_n_layers=2,
        vision_dim=16,
        vision_n_heads=4,
        vision_inter_dim=12,
        vision_patch_size=2,
        vision_downsample_ratio=3,
    )


def test_packed_vision_matches_isolated_images_and_calls_each_module_once():
    torch.manual_seed(7)
    args = _model_args()
    vision = ViT(args)
    aligner = Aligner(args)
    patches = [
        torch.randn(9, 3, 2, 2),
        torch.randn(18, 3, 2, 2),
    ]
    grids = [(3, 3), (6, 3)]

    expected = [aligner(vision(image, n_h, n_w), n_h, n_w) for image, (n_h, n_w) in zip(patches, grids)]

    with (
        mock.patch.object(vision.patch_embed, "forward", wraps=vision.patch_embed.forward) as patch_embed_forward,
        mock.patch.object(vision.blocks[0], "forward", wraps=vision.blocks[0].forward) as block0_forward,
        mock.patch.object(vision.blocks[1], "forward", wraps=vision.blocks[1].forward) as block1_forward,
        mock.patch.object(aligner, "forward", wraps=aligner.forward) as aligner_forward,
    ):
        features = vision(patches, grids)
        actual = aligner(features, grids)

    assert patch_embed_forward.call_count == 1
    assert block0_forward.call_count == 1
    assert block1_forward.call_count == 1
    assert aligner_forward.call_count == 1
    for actual_image, expected_image in zip(actual, expected):
        torch.testing.assert_close(actual_image, expected_image, rtol=1e-5, atol=1e-6)


def test_packed_vision_backward_is_finite():
    args = _model_args()
    vision = ViT(args)
    aligner = Aligner(args)
    patches = [
        torch.randn(9, 3, 2, 2, requires_grad=True),
        torch.randn(18, 3, 2, 2, requires_grad=True),
    ]
    grids = [(3, 3), (6, 3)]

    output = aligner(vision(patches, grids), grids)
    sum(image.float().square().mean() for image in output).backward()

    assert all(image.grad is not None for image in patches)
    assert all(torch.isfinite(image.grad).all() for image in patches)


def test_packed_vision_runs_tower_hooks_before_patch_projection():
    vision = ViT(_model_args())
    events = []
    vision.register_forward_pre_hook(lambda *_: events.append("unshard"))
    vision.patch_embed.register_forward_pre_hook(lambda *_: events.append("projection"))
    vision.register_forward_hook(lambda *_: events.append("reshard"))
    patches = [torch.randn(9, 3, 2, 2), torch.randn(18, 3, 2, 2)]
    grids = [(3, 3), (6, 3)]

    output = vision(patches, grids)
    assert events == ["unshard", "projection", "reshard"]
    assert [image.shape[0] for image in output] == [9, 18]
