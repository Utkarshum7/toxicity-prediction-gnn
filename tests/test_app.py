import pytest
import torch

import app as webapp
from src.model import BaseModel


@pytest.fixture()
def client(monkeypatch):
    torch.manual_seed(0)
    net = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
    net.eval()
    monkeypatch.setattr(webapp, "model", net)  # random weights: tests shape/flow, not accuracy
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Toxicity Prediction" in res.data
    assert b"escapeHtml" in res.data


def test_health(client):
    assert client.get("/health").get_json() == {"model_loaded": True}


def test_predict_requires_input(client):
    assert client.post("/predict", json={}).status_code == 400
    assert client.post("/predict", data="not json").status_code == 400


def test_predict_from_smiles_skips_pubchem(client, monkeypatch):
    def boom(_):
        raise AssertionError("PubChem must not be called when SMILES is supplied")
    monkeypatch.setattr(webapp, "convert_name_to_smiles", boom)
    res = client.post("/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    body = res.get_json()
    assert res.status_code == 200 and body["success"] is True
    assert 0.0 <= body["prediction"] <= 100.0
    assert body["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"


def test_predict_from_name_uses_pubchem(client, monkeypatch):
    monkeypatch.setattr(webapp, "convert_name_to_smiles", lambda name: ("CCO", {"toxicity": ["x"]}))
    body = client.post("/predict", json={"chemical_name": "ethanol"}).get_json()
    assert body["smiles"] == "CCO" and body["toxicity_data"] == {"toxicity": ["x"]}


def test_unknown_name_returns_400(client, monkeypatch):
    monkeypatch.setattr(webapp, "convert_name_to_smiles", lambda name: (None, None))
    assert client.post("/predict", json={"chemical_name": "zzz"}).status_code == 400


@pytest.mark.parametrize("smiles", ["not_a_smiles", "C" * 301])
def test_invalid_or_oversized_smiles_returns_400(client, smiles):
    res = client.post("/predict", json={"smiles": smiles})
    assert res.status_code == 400 and "error" in res.get_json()


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.setattr(webapp, "model", None)
    res = webapp.app.test_client().post("/predict", json={"smiles": "CCO"})
    assert res.status_code == 503
