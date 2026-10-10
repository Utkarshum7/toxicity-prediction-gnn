import pytest
import torch

from src.process import parse_smiles, smiles_to_edge, smiles_to_onehot


def test_onehot_shape_and_atom_columns():
    onehot = smiles_to_onehot("CCO", max_atoms=300)
    assert onehot.shape == (300, 8)
    assert onehot[0, 1] == 1 and onehot[1, 1] == 1  # two carbons
    assert onehot[2, 2] == 1                        # oxygen
    assert onehot.sum() == 3                        # padding rows stay zero


def test_unsupported_element_is_all_zero_row():
    onehot = smiles_to_onehot("CBr")  # Br is not in the atom vocabulary
    assert onehot[0].sum() == 1 and onehot[1].sum() == 0


@pytest.mark.parametrize("bad", ["not_a_smiles", "", "   ", "C(C"])
def test_invalid_smiles_rejected(bad):
    with pytest.raises(ValueError):
        smiles_to_onehot(bad)
    with pytest.raises(ValueError):
        smiles_to_edge(bad)


def test_oversized_molecule_rejected():
    with pytest.raises(ValueError, match="at most"):
        smiles_to_onehot("C" * 301, max_atoms=300)
    with pytest.raises(ValueError, match="at most"):
        parse_smiles("C" * 11, max_atoms=10)


@pytest.mark.parametrize("smiles,num_edges", [
    ("CCO", 2),     # two single bonds
    ("C=C", 2),     # double bond repeated twice
    ("C#C", 3),     # triple bond repeated three times
    ("[Na+]", 0),   # no bonds
])
def test_edge_counts(smiles, num_edges):
    edge = smiles_to_edge(smiles)
    assert edge.dtype == torch.int64
    assert edge.shape == (2, num_edges)


def test_aromatic_ring_is_kekulized():
    # benzene: 6 ring bonds, 3 of them double after Kekulization -> 3 + 3*2 = 9 edges
    assert smiles_to_edge("c1ccccc1").shape == (2, 9)
