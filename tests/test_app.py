from pathlib import Path

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
    assert b"MoleculeIQ" in res.data
    assert b"Research use only" in res.data
    assert b'href="/static/css/app.css"' in res.data and b'src="/static/js/app.js"' in res.data
    assert b'id="analyze-form"' in res.data


def test_static_assets_served(client):
    css, js = client.get("/static/css/app.css"), client.get("/static/js/app.js")
    assert css.status_code == 200 and js.status_code == 200
    css.close(), js.close()


def test_frontend_never_uses_html_injection_sinks():
    """Server/PubChem text must only reach the DOM via textContent (XSS regression guard)."""
    source = (Path(webapp.app.static_folder) / "js" / "app.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("//", "*", "/*")))
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in code, f"unsafe DOM sink {sink!r} found in app.js"


def test_frontend_does_not_present_score_as_percentage():
    source = (Path(webapp.app.static_folder) / "js" / "app.js").read_text(encoding="utf-8")
    assert "probability" in source and "Uncalibrated model output" in source
    assert "% chance" not in source


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
    # raw sigmoid output is exposed unscaled; `prediction` stays raw * 100 for compatibility
    assert 0.0 <= body["raw_output"] <= 1.0
    assert body["prediction"] == pytest.approx(body["raw_output"] * 100)


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
