from src import pubchem


def _nested(*texts):
    """Mimic PUG-View: text sits three Section levels below the queried heading."""
    return {"Record": {"Section": [{"Section": [{"Section": [{"Information": [
        {"Value": {"StringWithMarkup": [{"String": t} for t in texts]}}
    ]}]}]}]}}


def test_nested_sections_are_walked_deduped_and_capped(monkeypatch):
    texts = ["H302 (95%): Harmful if swallowed", "H302 (95%): Harmful if swallowed", "  "] + [f"line {i}" for i in range(30)]
    monkeypatch.setattr(pubchem, "_fetch_json", lambda url: _nested(*texts))
    out = pubchem._heading_strings(1, "x")
    assert out[0].startswith("H302") and out.count(out[0]) == 1
    assert len(out) == pubchem.MAX_ITEMS


def test_long_strings_are_truncated(monkeypatch):
    monkeypatch.setattr(pubchem, "_fetch_json", lambda url: _nested("a" * 5000))
    assert len(pubchem._heading_strings(1, "x")[0]) == pubchem.MAX_CHARS


def test_hazard_codes_extracted_from_ghs(monkeypatch):
    def fake(url):
        return _nested("H302 (95%): Harmful if swallowed", "Warning") if "GHS" in url else None
    monkeypatch.setattr(pubchem, "_fetch_json", fake)
    info = pubchem.get_pubchem_data(1)
    assert info["hazard_codes"] == ["H302 (95%): Harmful if swallowed"]
    assert "Warning" in info["ghs_classification"]


def test_compound_profile_included(monkeypatch):
    props = {"PropertyTable": {"Properties": [{"CID": 2244, "MolecularFormula": "C9H8O4", "XLogP": 1.2, "TPSA": None}]}}
    monkeypatch.setattr(pubchem, "_fetch_json", lambda url: props if "/property/" in url else None)
    compound = pubchem.get_pubchem_data(2244)["compound"]
    assert compound == {"cid": 2244, "MolecularFormula": "C9H8O4", "XLogP": 1.2}  # None values dropped


def test_compound_is_cid_only_when_properties_unavailable(monkeypatch):
    monkeypatch.setattr(pubchem, "_fetch_json", lambda url: None)
    info = pubchem.get_pubchem_data(7)
    assert info["compound"] == {"cid": 7} and info["safety_summary"] is None


def test_property_fallback_when_nothing_found(monkeypatch):
    props = {"PropertyTable": {"Properties": [{"MolecularFormula": "C2H6O"}]}}
    monkeypatch.setattr(pubchem, "_fetch_json", lambda url: props if "/property/" in url else None)
    assert "C2H6O" in pubchem.get_pubchem_data(1)["safety_summary"]


def test_get_pubchem_data_never_raises(monkeypatch):
    def boom(url):
        raise RuntimeError("network down")
    monkeypatch.setattr(pubchem, "_fetch_json", boom)
    assert pubchem.get_pubchem_data(1)["toxicity"] == []
