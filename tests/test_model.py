from pathlib import Path

import pytest
import torch

from src.model import DEFAULT_CHECKPOINT, BaseModel


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    net = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
    net.eval()
    return net


def test_forward_output_shape_and_range(model):
    with torch.no_grad():
        out = model.forward_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert out.shape == (1,)
    assert 0.0 <= out.item() <= 1.0


def test_invalid_smiles_raises(model):
    with pytest.raises(ValueError):
        model.forward_smiles("not_a_smiles")


def test_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "ckpt.pth"
    model.save(path)
    clone = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
    clone.load(path)
    clone.eval()
    with torch.no_grad():
        assert torch.allclose(model.forward_smiles("CCO"), clone.forward_smiles("CCO"))


def test_load_missing_checkpoint_gives_clear_error(model, tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        model.load(tmp_path / "missing.pth")


def test_train_step_returns_finite_loss():
    net = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
    loss = net.train_smiles("CCO", [1.0], verbose=False)
    assert torch.isfinite(loss)


def test_labels_to_target():
    net = BaseModel(max_atoms=10, num_outputs=2, labels=["a", "b"])
    assert net.labels_to_target(["b"]).tolist() == [0.0, 1.0]


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).is_file(), reason="no trained checkpoint at models/model.pth")
def test_real_checkpoint_inference():
    net = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
    net.load()
    net.eval()
    with torch.no_grad():
        out = net.forward_smiles("c1ccccc1")
    assert out.shape == (1,) and 0.0 <= out.item() <= 1.0
