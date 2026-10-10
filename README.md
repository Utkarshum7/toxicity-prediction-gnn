# Toxicity Prediction with Graph Neural Networks

[![tests](https://github.com/Utkarshum7/toxicity-prediction-gnn/actions/workflows/tests.yml/badge.svg)](https://github.com/Utkarshum7/toxicity-prediction-gnn/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.5.3-3c2179)
![RDKit](https://img.shields.io/badge/RDKit-2023.09.5-green)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

A small graph-neural-network model that scores a molecule's likelihood of triggering the **ATAD5 DNA-damage stress response** (Tox21 assay `SR-ATAD5`), plus a Flask web app that accepts a chemical *name*, resolves it through PubChem, and shows the model score next to PubChem's hazard information.

**Live demo:** <https://moleculeiq.vercel.app> (Vercel; see [Deployment](#deployment-vercel) for how it runs and its limits)

> **Research / educational use only.** Outputs are uncalibrated model scores from a small, minimally evaluated model. They are **not** a substitute for laboratory validation, regulatory toxicology, or professional medical advice. See [Limitations](#limitations-and-responsible-use).

## Table of contents

1. [Overview](#overview)
2. [How it works](#how-it-works)
3. [Dataset](#dataset)
4. [Model and training](#model-and-training)
5. [Installation (Windows)](#installation-windows)
6. [Model checkpoint](#model-checkpoint)
7. [Usage](#usage)
8. [Deployment (Vercel)](#deployment-vercel)
9. [Testing](#testing)
10. [Project layout](#project-layout)
11. [Limitations and responsible use](#limitations-and-responsible-use)
12. [Known issues](#known-issues)
13. [Future improvements](#future-improvements)
14. [License and acknowledgments](#license-and-acknowledgments)

## Overview

**Problem.** Early toxicity screening is expensive; computational models can help prioritise which compounds deserve lab assays. ATAD5 is involved in DNA replication and repair, and its stress response is used in Tox21 as a read-out of genotoxic stress.

**What this project does**

- Converts a SMILES string into a molecular graph (atoms = nodes, bonds = edges) with RDKit.
- Runs a 2-layer GCN + 2-layer MLP (PyTorch Geometric) that outputs a single score in `[0, 1]` for `SR-ATAD5`.
- Provides a **CLI** (`predict.py`), a **training script** (`train.py`), and a **web app** (`app.py`) that takes a chemical name or SMILES and also fetches GHS / toxicity / health-hazard text from PubChem.
- Includes input validation (invalid SMILES, empty input, molecules over the 300-atom limit), a pytest suite, and a CI workflow.

## How it works

```
chemical name ──PubChem──▶ SMILES ──RDKit──▶ node features (300×8 one-hot)  ┐
                                         └─▶ edge index (bonds)             ├─▶ GCNConv(8→8) ─ReLU─▶ GCNConv(8→8) ─ReLU
                                                                            ┘          │
                         score in [0,1] ◀─ Sigmoid ◀─ Linear(2400→1) ◀─ ReLU ◀─ Linear(2400→2400) ◀─ Flatten(300×8)
```

**Molecular representation** ([src/process.py](src/process.py))

- **Nodes:** one row per atom in a zero-padded `300 × 8` matrix. The 8 columns are the one-hot atom type `H, C, O, N, P, S, Cl, F`. Any other element (e.g. Br, I, Na) gets an all-zero row.
- **Edges:** the molecule is Kekulized; each bond contributes one edge `begin → end`. Double bonds are repeated twice and triple bonds three times, so bond order acts like an edge multiplicity.
- **Limits:** molecules with more than 300 atoms are rejected with a clear error.

## Dataset

[Tox21](https://tripod.nih.gov/tox21/challenge/) as distributed by [MoleculeNet / DeepChem](https://github.com/deepchem/deepchem/blob/master/datasets/tox21.csv.gz), included at [`datasets/tox21.csv`](datasets/tox21.csv) (about 0.5 MB). Measured from the bundled file:

| | |
|---|---|
| Molecules | 8,013 |
| Columns | 12 assay labels + `mol_id` + `smiles` |
| Task used | `SR-ATAD5` only |
| `SR-ATAD5` labels | 267 positive, 6,964 negative, 782 missing |

Only the `SR-ATAD5` column is used. The other 11 assays are not modelled. Redistribution terms: Tox21 data were publicly released by the NIH/NCATS Tox21 program and are redistributed by DeepChem; this repo does not contain a separate data license file, so please check the upstream terms before reusing the data commercially.

## Model and training

| | |
|---|---|
| Architecture | `GCNConv(8→8)` → ReLU → `GCNConv(8→8)` → ReLU → flatten (300·8 = 2400) → `Linear(2400→2400)` → ReLU → `Linear(2400→1)` → Sigmoid |
| Parameters | 5,764,945 (measured) |
| Loss / optimiser | Binary cross-entropy, SGD, lr = 1e-5, one molecule per step |
| Default epochs | 1 |

**Training procedure** ([train.py](train.py))

1. Drop rows with a missing `SR-ATAD5` label, shuffle (seeded), split 80 % / 20 %.
2. **Oversample the positive class in the training split only** (repeat positives `int(neg/pos)` times). With the bundled data this gives 11,034 training samples (5,460 positive) and a held-out set of 1,447 molecules (57 positive).
3. Train with per-sample SGD, then report mean squared error on the held-out set and save the checkpoint.

> **No accuracy / ROC-AUC / F1 numbers are reported.** The only evaluation the script performs is held-out mean squared error, which is not informative on its own for a ~3 % positive-rate task. Because training oversamples positives, the output is **not a calibrated probability**.

## Installation (Windows)

Verified on **Windows 11, Python 3.12.1** (the full test suite and the web app were run in this environment). CI runs on Ubuntu with Python 3.11 and 3.12. Other platforms have not been tested by hand.

```powershell
git clone https://github.com/Utkarshum7/toxicity-prediction-gnn.git
cd toxicity-prediction-gnn

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes on special dependencies:

- **PyTorch 2.2.2 / PyG 2.5.3** are pinned together. PyG's `GCNConv` works without the optional `torch-scatter` / `torch-sparse` extras, so no special wheels are needed.
- On **Linux** (Python 3.11/3.12), `requirements.txt` references the CPU-only PyTorch 2.2.2 wheel directly, pinned by SHA-256 (the default Linux wheel pulls in gigabytes of CUDA libraries). On Windows/macOS the PyPI wheel is already CPU-only and is used as-is.
- `requirements.txt` holds only what the web app and inference need. Training additionally needs `pandas` and `matplotlib`: `pip install -r requirements-train.txt`.
- `numpy` is pinned to `1.26.4` because `torch 2.2.2` is not built against NumPy 2.
- RDKit is installed from PyPI (`rdkit==2023.9.5`); no conda is required.
- The web app needs internet access for PubChem lookups (the UI itself loads no external scripts, fonts or styles).

For development / tests (includes the training packages and pytest): `pip install -r requirements-dev.txt`.

## Model checkpoint

Trained checkpoints (`*.pth`) are **not committed** (23 MB binary; `models/.gitignore` and the root `.gitignore` exclude them). Create one by training:

```powershell
python train.py
```

This writes `models/model.pth`. Alternatively, point to a checkpoint elsewhere with the `TOX_MODEL_PATH` environment variable or `predict.py --model`.

The original upstream project ([Null-byte-00/toxicity-prediction-gnn](https://github.com/Null-byte-00/toxicity-prediction-gnn/releases)) publishes a pretrained `model.pth` with the same layer sizes. Its compatibility with this repository has **not** been verified here.

If no checkpoint is present, the CLI exits with an explanatory message and the web app starts but answers `/predict` with HTTP 503.

## Usage

### Train

```powershell
python train.py                                  # full run, 1 epoch, saves models/model.pth
python train.py --epochs 3 --seed 7              # more epochs, different seed
python train.py --max-samples 200 --output tmp.pth --plot loss.png   # quick smoke run
```

A 200-sample smoke run (200 train + 200 test molecules) took about 8 seconds on the development machine; a full epoch processes 11,034 training molecules, so expect minutes rather than seconds (not measured end to end).

### Predict from the command line

```powershell
python predict.py "CC(=O)Oc1ccccc1C(=O)O" "c1ccccc1"
python predict.py                    # interactive prompt
```

Format of the output (the numbers depend on your trained checkpoint, so none are quoted here):

```
CC(=O)Oc1ccccc1C(=O)O   <score>% (ATAD5 stress-response model score)
```

Invalid input gives a message instead of a traceback, e.g. `not_a_smiles  ERROR: Invalid SMILES string: not_a_smiles` (exit code 2).

### Web app

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. The **MoleculeIQ** dashboard accepts either a chemical name (resolved through PubChem, e.g. `aspirin`) or a SMILES string. For a name lookup it shows:

- **Model output**: the raw network output for `SR-ATAD5` on a 0-1 scale (arc gauge + number), explicitly labelled *uncalibrated*. The gauge fill is proportional to the raw value; nothing is rescaled, clamped or colour-coded as "safe/unsafe".
- **PubChem information** (each card is tagged with its source): CID and SMILES with copy buttons, molecular profile (formula, weight, IUPAC name, XLogP, exact mass, TPSA, H-bond and rotatable-bond counts), toxicity summary, health hazards, GHS signal word(s) and hazard statements, and a hazards summary. Sections PubChem has no data for show an empty state.
- **Model details**: assay, architecture, input constraints and limitations.

SMILES input skips PubChem, so only the model output is shown. The UI is plain HTML/CSS/JS (`templates/index.html`, `public/static/`); no build step, no external fonts or chart libraries. All server- and PubChem-provided text is inserted with `textContent` (a test guards against HTML-injection sinks). Not shown: GHS pictogram images.

Debug mode is off by default; enable with `$env:FLASK_DEBUG = "1"`.

JSON API:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"smiles\": \"CCO\"}"
```

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `GET /health` | `{"model_loaded": true/false}` |
| `POST /predict` | Body `{"chemical_name": "..."}` or `{"smiles": "..."}`. Returns `raw_output` (unscaled sigmoid output, 0-1), `prediction` (`raw_output` x 100, kept for compatibility), `smiles`, `message`, and `toxicity_data` (PubChem, only for name lookups; includes a `compound` object with `cid` and computed properties). Errors: 400 invalid input, 503 no checkpoint |

## Deployment (Vercel)

Live at <https://moleculeiq.vercel.app> (Vercel project `moleculeiq`, region `iad1`).

**Architecture.** One Flask function on Vercel's Python 3.12 runtime (Fluid compute) serves the page, `/predict`, `/health` and `/ready`. Static assets in `public/static/` are served by Vercel's CDN. The function performs real GNN inference with the trained checkpoint and calls PubChem outbound. There is no separate inference service and no mock output.

**Checkpoint delivery.** `models/model.pth` is not in Git. It is bundled into the function when deploying with the Vercel CLI from a machine that has it (`.vercelignore` keeps it; it is never served as a static file). At startup `src/checkpoint.py` verifies its SHA-256 against `TOX_MODEL_SHA256` and refuses to load a mismatching file; the model is loaded once per instance, not per request. If the file is absent, an https `TOX_MODEL_URL` can be configured instead: it is downloaded once into `/tmp` (the only writable location), verified, and cached. A failed load is retried on later requests (rate-limited), and `/predict` answers `503` while no checkpoint is available.

**Environment variables**

| Variable | Required | Purpose |
|---|---|---|
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | yes, value `1` | The dependencies (CPU PyTorch + PyG + RDKit + SciPy/scikit-learn) unpack to about 1.07 GB, above the 500 MB Python limit; this enables Vercel's *Large Functions* (beta, up to 5 GB). **Store it as a plain (non-sensitive) variable**; as a sensitive variable the build did not honour it. |
| `TOX_MODEL_SHA256` | recommended | SHA-256 of the checkpoint; mismatching files are rejected. Not a secret. |
| `TOX_MODEL_URL` | optional | https URL to download the checkpoint when it is not bundled. Treat as a secret if it is signed. |
| `TOX_MODEL_PATH` | optional | Alternative checkpoint path. |
| `FLASK_DEBUG` | local only | `1` enables Flask debug mode. Never set on Vercel. |

**Deploying**

```powershell
npm i -g vercel
vercel login
vercel link --project moleculeiq
vercel env add VERCEL_SUPPORT_LARGE_FUNCTIONS production --no-sensitive --value 1
vercel env add TOX_MODEL_SHA256 production --no-sensitive --value <sha256-of-models/model.pth>
vercel deploy --prod          # run from a checkout that contains models/model.pth
```

Use `--value` (not a pipe) when setting variables from PowerShell: piping prepends a byte-order mark to the value (the code now tolerates that, but avoid it). The project is intentionally **not** connected to Git: a Git-triggered build would not contain the untracked checkpoint. To enable Git deployments, host the checkpoint at an https URL and set `TOX_MODEL_URL` as well.

**Limitations of this deployment**

- *Large Functions is a Vercel beta feature*; the deployment depends on it because of the PyTorch footprint.
- Hobby plan limits apply: 2 GB memory / 1 vCPU, 300 s maximum duration (the function is configured for 120 s; PubChem lookups normally take 1-2 s).
- A request that lands on a fresh instance (after idle time or scale-out) pays for importing PyTorch, PyG and RDKit and for verifying and loading the checkpoint. Observed on the live site: about 5.2-5.5 s (three observations), versus about 0.4 s for warm requests. These are spot measurements, not a benchmark.
- Memory: the process uses roughly 300 MB resident after loading the model (measured locally on Windows; not measured inside Vercel), well under the 2 GB Hobby limit.
- The public endpoints have no authentication or rate limiting. Anyone can call `/predict`, and every name lookup makes outbound PubChem requests.
- Results depend on PubChem availability; if PubChem fails or is slow, name lookups return an error or empty hazard sections while SMILES input keeps working.

## Testing

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

The suite (no network, no checkpoint required) covers SMILES preprocessing (valid/invalid/oversized), model forward pass shape and range, checkpoint save/load, a training step, leak-free train/test splitting, a tiny end-to-end training smoke run, PubChem response parsing (with mocked HTTP), and the Flask routes. One extra test runs inference against `models/model.pth` when it exists and is skipped otherwise. GitHub Actions ([`tests.yml`](.github/workflows/tests.yml)) runs the same suite.

## Project layout

```
app.py              Flask web app (entry point)
train.py            Training / evaluation CLI
predict.py          Inference CLI
src/
  model.py          BaseModel (GCN + MLP), save/load
  process.py        SMILES -> node features / edge index, validation
  pubchem.py        Name -> SMILES and PubChem hazard text
  checkpoint.py     Locate / verify (SHA-256) / optionally download the checkpoint
templates/          Web UI page (index.html)
public/static/      UI stylesheet and script (css/app.css, js/app.js); served by Vercel's CDN, and by Flask locally
vercel.json         Vercel configuration (Flask preset, function duration)
datasets/tox21.csv  Tox21 data (MoleculeNet copy)
models/             Checkpoints (git-ignored)
tests/              pytest suite
.github/workflows/  CI
```

## Limitations and responsible use

- Research/educational output only; not validated for any regulatory, clinical, or safety decision.
- Predicts a single assay (`SR-ATAD5`). It says nothing about other kinds of toxicity.
- Scores are uncalibrated (oversampled training) and no performance metrics are reported.
- Model design limits: fixed 300-atom input; atom vocabulary limited to H, C, O, N, P, S, Cl, F (other elements are invisible); the flatten + dense layers make the output depend on atom ordering in the SMILES; bonds are inserted as one directed edge each.
- Hazard text shown by the web app comes from PubChem and may be incomplete, truncated (at most 10 items of 800 characters per category), or absent.
- Name resolution takes the first PubChem match, which may not be the compound you meant.

## Known issues

- `BaseModel.train_smiles` calls `backward()` and `step()` without `optimizer.zero_grad()`, so gradients accumulate across samples. This is how the original implementation trains; it was left unchanged to avoid altering the training methodology without re-evaluating results. Fixing it is the first item below.

## Future improvements

- Add `zero_grad()` (or mini-batching) and re-tune the learning rate.
- Proper evaluation: ROC-AUC / PR-AUC on a scaffold split, calibration, comparison to a fingerprint + logistic-regression baseline.
- Symmetrise bond edges, add richer atom/bond features (degree, charge, aromaticity), and replace flatten with a permutation-invariant pooling (e.g. global mean/add pool) to lift the 300-atom limit.
- Multi-task learning over all 12 Tox21 assays.
- Publish a trained checkpoint as a GitHub Release asset with a model card.

## License and acknowledgments

Released under the **GNU General Public License v3.0** ([LICENSE](LICENSE)).

This project is derived from [**Null-byte-00/toxicity-prediction-gnn**](https://github.com/Null-byte-00/toxicity-prediction-gnn) (GPL-3.0), which provided the original model, SMILES preprocessing, training script and CLI. The Flask web application, PubChem integration, input validation, repository restructuring, tests and CI in this repository were added by Utkarsh Amaresh.

Thanks also to the [Tox21](https://tripod.nih.gov/tox21/) program, [MoleculeNet](https://moleculenet.org/) / [DeepChem](https://github.com/deepchem/deepchem), [RDKit](https://www.rdkit.org/), [PyTorch Geometric](https://pyg.org/) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
