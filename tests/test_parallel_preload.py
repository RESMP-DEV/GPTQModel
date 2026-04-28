import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.utils.structure import LazyTurtle

@pytest.fixture
def temp_checkpoint_dir():
    temp_dir = tempfile.mkdtemp()
    
    # Create shard 1
    tensors_1 = {
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 16),
    }
    save_file(tensors_1, os.path.join(temp_dir, "model-00001-of-00002.safetensors"))
    
    # Create shard 2
    tensors_2 = {
        "model.layers.1.mlp.up_proj.weight": torch.randn(16, 16),
        "model.layers.1.mlp.down_proj.weight": torch.randn(16, 16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(16, 16),
    }
    save_file(tensors_2, os.path.join(temp_dir, "model-00002-of-00002.safetensors"))
    
    # Create index
    index = {
        "weight_map": {
            "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.up_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.self_attn.k_proj.weight": "model-00002-of-00002.safetensors",
        }
    }
    with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
        
    # Write a minimal config.json
    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump({"model_type": "llama"}, f)
        
    yield temp_dir
    
    shutil.rmtree(temp_dir)

def test_preload_no_device_map(temp_checkpoint_dir):
    turtle = LazyTurtle(
        model_local_path=temp_checkpoint_dir,
        config={"model_type": "llama"},
    )
    preloaded = turtle._preloaded_tensors
    
    assert len(preloaded) == 8
    for name, tensor in preloaded.items():
        assert tensor.device.type == "cpu"

def test_preload_cpu_device_map(temp_checkpoint_dir):
    device_map = {name: "cpu" for name in [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
    ]}
    
    turtle = LazyTurtle(
        model_local_path=temp_checkpoint_dir,
        config={"model_type": "llama"},
        device_map=device_map,
    )
    preloaded = turtle._preloaded_tensors
    
    assert len(preloaded) == 8
    for name, tensor in preloaded.items():
        assert tensor.device.type == "cpu"

@pytest.mark.cuda
def test_preload_mixed_device_map(temp_checkpoint_dir):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device_map = {
        "model.layers.0.mlp.up_proj.weight": "cuda:0",
        "model.layers.0.mlp.down_proj.weight": "cuda:0",
        "model.layers.0.self_attn.q_proj.weight": "cpu",
        "model.layers.0.self_attn.k_proj.weight": "cpu",
        "model.layers.1.mlp.up_proj.weight": "cuda:0",
        "model.layers.1.mlp.down_proj.weight": "cpu",
        "model.layers.1.self_attn.q_proj.weight": "cuda:0",
        "model.layers.1.self_attn.k_proj.weight": "cpu",
    }
    
    turtle = LazyTurtle(
        model_local_path=temp_checkpoint_dir,
        config={"model_type": "llama"},
        device_map=device_map,
    )
    preloaded = turtle._preloaded_tensors
    
    assert len(preloaded) == 8
    for name, tensor in preloaded.items():
        expected_device = device_map[name]
        if expected_device == "cpu":
            assert tensor.device.type == "cpu"
        else:
            assert tensor.device.type == "cuda"
            assert tensor.device.index == 0

class DummyModule(torch.nn.Module):
    pass

def test_checkpoint_tensors_for_submodule_uses_preloaded(temp_checkpoint_dir, mocker):
    turtle = LazyTurtle(
        model_local_path=temp_checkpoint_dir,
        config={"model_type": "llama"},
    )
    preloaded = turtle._preloaded_tensors
    
    # Verify everything was preloaded
    assert len(preloaded) == 8
    
    # Mock safe_open so we can assert it is not called
    mock_safe_open = mocker.patch("gptqmodel.utils.structure.safe_open")
    
    # Create dummy models to satisfy the API
    target_model = DummyModule()
    # Mocking _get_qualified_name instead of creating a real nested module
    mocker.patch("gptqmodel.utils.structure._get_qualified_name", return_value="model.layers.0.mlp")
    
    # In LazyTurtle, _load_checkpoint_tensors_for_module_path iterates over the weight map
    # and pulls from preloaded_tensors if available. Let's see if it gets called correctly.
    
    # We also need to patch _runtime_to_checkpoint_renamings logic if any, but since we 
    # didn't pass hf_conversion_map_reversed, it should be empty
    
    result = turtle.checkpoint_tensors_for_submodule(
        target_model=target_model,
        target_submodule=target_model, # value doesn't matter because of _get_qualified_name mock
    )
    
    assert len(result) == 2
    assert "up_proj.weight" in result
    assert "down_proj.weight" in result
    
    # Verify we got exactly the tensors from the preload cache
    assert result["up_proj.weight"] is preloaded["model.layers.0.mlp.up_proj.weight"]
    assert result["down_proj.weight"] is preloaded["model.layers.0.mlp.down_proj.weight"]
    
    # Verify we didn't read from disk again
    mock_safe_open.assert_not_called()

