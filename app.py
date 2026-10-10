"""Flask web app: chemical name (or SMILES) -> ATAD5 stress-response score."""
import logging
import os

import torch
from flask import Flask, jsonify, render_template, request

from src.model import BaseModel
from src.process import parse_smiles
from src.pubchem import convert_name_to_smiles

logger = logging.getLogger(__name__)

app = Flask(__name__)


def load_model():
    """Load the trained checkpoint; return None (and log why) if unavailable."""
    try:
        net = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
        net.load()
        net.eval()
        logger.info("Model initialized successfully")
        return net
    except Exception as exc:
        logger.warning("Model not loaded: %s", exc)
        return None


model = load_model()


def predict_smiles(smiles):
    """Return the raw model output (sigmoid, 0-1) for a SMILES string; raises ValueError if invalid."""
    parse_smiles(smiles, max_atoms=model.max_atoms)
    with torch.no_grad():
        return float(model.forward_smiles(smiles).item())


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model checkpoint not loaded. Train one with `python train.py` "
                                 "or place it at models/model.pth."}), 503

    data = request.get_json(silent=True) or {}
    chemical_name = str(data.get("chemical_name") or "").strip()
    smiles = str(data.get("smiles") or "").strip()
    if not chemical_name and not smiles:
        return jsonify({"error": "Provide 'chemical_name' or 'smiles'"}), 400

    toxicity_data = None
    if not smiles:
        smiles, toxicity_data = convert_name_to_smiles(chemical_name)
        if not smiles:
            return jsonify({"error": f'Could not convert "{chemical_name}" to SMILES notation'}), 400

    try:
        raw_output = predict_smiles(smiles)
        prediction = raw_output * 100
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Prediction failed for %s", smiles)
        return jsonify({"error": "Internal error while processing the molecule"}), 500

    return jsonify({
        "success": True,
        "prediction": prediction,   # raw_output * 100, kept for backward compatibility
        "raw_output": raw_output,   # unscaled sigmoid output of the network
        "smiles": smiles,
        "plot": {"x": ["ATAD5 Stress Response"], "y": [prediction], "type": "bar"},
        "chemical_name": chemical_name or smiles,
        "message": f"Model score for ATAD5 stress response: {prediction:.1f}% (uncalibrated, not a probability)",
        "toxicity_data": toxicity_data,
    })


if __name__ == "__main__":
    # Debug mode exposes an interactive debugger; opt in explicitly via FLASK_DEBUG=1.
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
