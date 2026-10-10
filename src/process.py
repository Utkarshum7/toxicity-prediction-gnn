"""Convert SMILES strings into the tensors consumed by the GNN."""
import torch
from rdkit import Chem

# Atom types the one-hot encoding knows about. Any other element (Br, I, Na, ...)
# produces an all-zero row, i.e. it is invisible to the model.
ATOM_INDEX = {
    "H": 0,
    "C": 1,
    "O": 2,
    "N": 3,
    "P": 4,
    "S": 5,
    "Cl": 6,
    "F": 7,
}


def parse_smiles(smiles, max_atoms=None):
    """Parse a SMILES string into an RDKit molecule.

    Raises ValueError for empty or unparsable input, or for molecules with more
    than ``max_atoms`` atoms (when given).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES string is empty")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    if max_atoms is not None and molecule.GetNumAtoms() > max_atoms:
        raise ValueError(
            f"Molecule has {molecule.GetNumAtoms()} atoms; the model supports at most {max_atoms}"
        )
    return molecule


def smiles_to_onehot(smiles, max_atoms=300, add_hs=False):
    """Turn SMILES into a (max_atoms, 8) one-hot atom-type tensor (zero padded)."""
    molecule = parse_smiles(smiles)
    if add_hs:
        molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() > max_atoms:
        raise ValueError(
            f"Molecule has {molecule.GetNumAtoms()} atoms; the model supports at most {max_atoms}"
        )
    out_tensor = torch.zeros(max_atoms, len(ATOM_INDEX))
    for index, atom in enumerate(molecule.GetAtoms()):
        column = ATOM_INDEX.get(atom.GetSymbol())
        if column is not None:
            out_tensor[index, column] = 1
    return out_tensor


def smiles_to_edge(smiles, add_hs=False):
    """Turn SMILES into a (2, num_edges) edge-index tensor.

    Each bond is listed once (begin -> end atom); double and triple bonds are
    repeated 2x and 3x so bond order acts as an edge weight in message passing.
    """
    molecule = parse_smiles(smiles)
    Chem.rdmolops.Kekulize(molecule)
    if add_hs:
        molecule = Chem.AddHs(molecule)
    edge_index = [[], []]
    for bond in molecule.GetBonds():
        first_atom_index = bond.GetBeginAtom().GetIdx()
        second_atom_index = bond.GetEndAtom().GetIdx()
        repeats = {
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
        }.get(bond.GetBondType(), 1)
        for _ in range(repeats):
            edge_index[0].append(first_atom_index)
            edge_index[1].append(second_atom_index)
    return torch.tensor(edge_index, dtype=torch.int64)
