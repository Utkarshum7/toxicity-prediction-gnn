"""Command-line inference: print the ATAD5 stress-response score for SMILES strings.

    python predict.py "CC(=O)Oc1ccccc1C(=O)O"     # one or more SMILES as arguments
    python predict.py                              # interactive prompt (Ctrl+C / empty line to quit)
"""
import argparse
import sys

import torch

from src.model import DEFAULT_CHECKPOINT, BaseModel


def score(model, smiles):
    with torch.no_grad():
        return model.forward_smiles(smiles).item() * 100


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("smiles", nargs="*", help="SMILES string(s) to score")
    parser.add_argument("--model", default=DEFAULT_CHECKPOINT, help="path to a checkpoint")
    args = parser.parse_args(argv)

    model = BaseModel(num_outputs=1, labels=["SR-ATAD5"], lr=0.00001)
    try:
        model.load(args.model)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    model.eval()

    def run(smiles):
        try:
            print(f"{smiles}\t{score(model, smiles):.2f}% (ATAD5 stress-response model score)")
            return True
        except ValueError as exc:
            print(f"{smiles}\tERROR: {exc}", file=sys.stderr)
            return False

    if args.smiles:
        return 0 if all([run(s) for s in args.smiles]) else 2

    try:
        while True:
            smiles = input("Enter a SMILES string: ").strip()
            if not smiles:
                break
            run(smiles)
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
