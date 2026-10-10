import pandas as pd
import pytest
import torch

import train


def _frame(n_pos=4, n_neg=40):
    rows = [{"smiles": "CCO", "SR-ATAD5": 1.0, "mol_id": f"P{i}"} for i in range(n_pos)]
    rows += [{"smiles": "CCN", "SR-ATAD5": 0.0, "mol_id": f"N{i}"} for i in range(n_neg)]
    rows += [{"smiles": "CCC", "SR-ATAD5": float("nan"), "mol_id": "U0"}]  # unlabeled
    return pd.DataFrame(rows)


def test_split_has_no_leakage_and_drops_unlabeled():
    tr, te = train.split_and_oversample(_frame(), seed=1)
    assert not set(tr["mol_id"]) & set(te["mol_id"])
    assert "U0" not in set(tr["mol_id"]) | set(te["mol_id"])
    assert te["SR-ATAD5"].notna().all()


def test_oversampling_only_touches_training_split():
    tr, te = train.split_and_oversample(_frame(), seed=1)
    # test split keeps its natural (imbalanced) distribution: no duplicated molecules
    assert te["mol_id"].is_unique
    # training positives were repeated to roughly match negatives
    assert (tr["SR-ATAD5"] == 1).sum() > (te["SR-ATAD5"] == 1).sum()


def test_smoke_training_run_writes_checkpoint(tmp_path):
    """Tiny end-to-end run on the bundled Tox21 CSV (a few samples, seconds on CPU)."""
    out = tmp_path / "smoke.pth"
    train.main(["--max-samples", "5", "--output", str(out)])
    assert out.is_file()
    state = torch.load(out, weights_only=True)
    assert "linear2.weight" in state
