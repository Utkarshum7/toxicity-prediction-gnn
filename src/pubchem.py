"""Name -> SMILES lookup and hazard text retrieval from PubChem."""
import logging
import re

import pubchempy as pcp
import requests

logger = logging.getLogger(__name__)

PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={heading}"
PUG_PROPERTY = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
    "/property/MolecularFormula,IUPACName,XLogP,ExactMass/JSON"
)
TIMEOUT = 10        # seconds per PubChem request
MAX_ITEMS = 10      # strings kept per category (PubChem records can be very large)
MAX_CHARS = 800     # characters kept per string
H_CODE = re.compile(r"^H\d{3}\b")


def _fetch_json(url):
    response = requests.get(url, timeout=TIMEOUT)
    return response.json() if response.status_code == 200 else None


def _collect_strings(section, out):
    """Depth-first walk: PUG-View nests text several 'Section' levels below the heading."""
    for info in section.get("Information", []):
        for item in info.get("Value", {}).get("StringWithMarkup", []):
            text = item.get("String", "").strip()
            if text:
                out.append(text)
    for child in section.get("Section", []):
        _collect_strings(child, out)


def _heading_strings(cid, heading):
    """Deduplicated, size-capped text strings PubChem returns for one record heading."""
    data = _fetch_json(PUG_VIEW.format(cid=cid, heading=heading))
    strings = []
    for section in (data or {}).get("Record", {}).get("Section", []):
        _collect_strings(section, strings)
    unique = list(dict.fromkeys(strings))
    return [s[:MAX_CHARS] for s in unique[:MAX_ITEMS]]


def get_pubchem_data(cid):
    """Collect GHS / hazard / toxicity text for a compound; never raises."""
    info = {
        "toxicity": [],
        "health_effects": [],
        "safety_summary": None,
        "ghs_classification": [],
        "hazard_codes": [],
    }
    try:
        info["toxicity"] = _heading_strings(cid, "Toxicity+Summary")
        info["health_effects"] = _heading_strings(cid, "Health+Hazards")
        info["ghs_classification"] = _heading_strings(cid, "GHS+Classification")
        info["hazard_codes"] = [s for s in info["ghs_classification"] if H_CODE.match(s)]
        summary = _heading_strings(cid, "Hazards+Summary")
        if summary:
            info["safety_summary"] = summary[0]

        if not any(info.values()):
            data = _fetch_json(PUG_PROPERTY.format(cid=cid))
            if data and data.get("PropertyTable", {}).get("Properties"):
                props = data["PropertyTable"]["Properties"][0]
                info["safety_summary"] = (
                    f"Chemical Formula: {props.get('MolecularFormula', 'N/A')}\n"
                    f"IUPAC Name: {props.get('IUPACName', 'N/A')}\n"
                    f"LogP: {props.get('XLogP', 'N/A')} (measure of lipophilicity)\n"
                    f"Exact Mass: {props.get('ExactMass', 'N/A')} g/mol"
                )
    except Exception:
        logger.exception("Error fetching PubChem data for CID %s", cid)
    return info


def convert_name_to_smiles(chemical_name):
    """Return (smiles, toxicity_data) for a chemical name, or (None, None)."""
    try:
        compounds = pcp.get_compounds(chemical_name, "name")
        if compounds:
            compound = compounds[0]
            return compound.canonical_smiles, get_pubchem_data(compound.cid)
        logger.info("No compounds found for %s", chemical_name)
    except Exception:
        logger.exception("PubChem name lookup failed for %s", chemical_name)
    return None, None
