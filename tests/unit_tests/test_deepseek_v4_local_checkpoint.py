"""Exercise packed expert source mapping with small safe CPU checkpoints."""

import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/deepseekv4'))
from load_4layer_shards import load_local_checkpoint


class StateModel:
    def __init__(self):
        self.state = {
            'model.layers.0.mlp.experts.gate_up_proj': torch.empty(256, 4, 3),
            'model.layers.0.mlp.experts.down_proj': torch.empty(256, 3, 2),
            'model.embed_tokens.weight': torch.empty(6, 3),
            'model.layers.0.mlp.gate.tid2eid': torch.empty(6, 1, dtype=torch.int64),
        }

    def state_dict(self):
        return self.state


def checkpoint(directory):
    tensors = {
        'embed.weight': torch.arange(18, dtype=torch.float32).reshape(6, 3),
        'layers.0.ffn.gate.tid2eid': torch.arange(6).reshape(6, 1),
    }
    for expert in range(256):
        for projection, shape in (('w1', (2, 3)), ('w3', (2, 3)), ('w2', (3, 2))):
            tensors[f'layers.0.ffn.experts.{expert}.{projection}.weight'] = torch.full(
                shape, float(expert * 10 + int(projection[1]))
            )
    save_file(tensors, str(directory / 'model.safetensors'))
    (directory / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {k: 'model.safetensors' for k in tensors}})
    )


def test_all_original_experts_loaded(tmp_path):
    checkpoint(tmp_path)
    model = StateModel()
    result = load_local_checkpoint(model, str(tmp_path))
    assert result['local_expert_projections'] == 768
    for expert in range(256):
        packed = model.state['model.layers.0.mlp.experts.gate_up_proj'][expert]
        assert torch.all(packed[:2] == expert * 10 + 1)
        assert torch.all(packed[2:] == expert * 10 + 3)
        assert torch.all(model.state['model.layers.0.mlp.experts.down_proj'][expert] == expert * 10 + 2)


def test_missing_tensor_fails(tmp_path):
    checkpoint(tmp_path)
    model = StateModel()
    model.state['unmapped.weight'] = torch.empty(2)
    with pytest.raises(KeyError):
        load_local_checkpoint(model, str(tmp_path))


def test_shape_mismatch_fails(tmp_path):
    checkpoint(tmp_path)
    model = StateModel()
    model.state['model.embed_tokens.weight'] = torch.empty(5, 3)
    with pytest.raises(ValueError, match='Shape mismatch'):
        load_local_checkpoint(model, str(tmp_path))
