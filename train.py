"""Train the GNN on the Tox21 SR-ATAD5 task.

Per-sample SGD, matching the original implementation. The train/test split is made
*before* positives are oversampled so held-out molecules never appear in training.
"""
import argparse
import random
from pathlib import Path

import pandas as pd
import torch

from src.model import DEFAULT_CHECKPOINT, BaseModel

TARGET = "SR-ATAD5"
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "tox21.csv"


def oversample(dataframe):
    """Repeat the positive class so it roughly matches the negative class size."""
    positive = dataframe[dataframe[TARGET] == 1]
    negative = dataframe[dataframe[TARGET] == 0]
    ratio = len(negative) / len(positive)
    print(f"Ratio: {ratio}")
    oversampled = pd.concat([positive] * int(ratio), ignore_index=True)
    return pd.concat([oversampled, negative], ignore_index=True)


def split_and_oversample(dataframe, train_fraction=0.8, seed=42):
    """Drop unlabeled rows, shuffle, split, then oversample the training split only."""
    labeled = dataframe.dropna(subset=[TARGET]).sample(frac=1, random_state=seed)
    cut = int(len(labeled) * train_fraction)
    train, test = labeled.iloc[:cut], labeled.iloc[cut:]
    train = oversample(train).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test.reset_index(drop=True)


def train_epochs(model, train, epochs=1, log_every=500):
    skipped = 0
    for epoch in range(epochs):
        for step, (smiles, label) in enumerate(zip(train["smiles"], train[TARGET])):
            try:
                model.train_smiles(smiles, [label], verbose=False)
            except ValueError as exc:
                skipped += 1
                print(f"Skipping {smiles}: {exc}")
            if (step + 1) % log_every == 0:
                print(f"epoch {epoch + 1}/{epochs}: {step + 1}/{len(train)} samples")
    return skipped


def evaluate(model, test):
    """Mean squared error between model output and label on the held-out split."""
    losses = []
    with torch.no_grad():
        for smiles, label in zip(test["smiles"], test[TARGET]):
            try:
                output = model.forward_smiles(smiles)
            except ValueError:
                continue
            losses.append(((output - label) ** 2).item())
    return sum(losses) / len(losses), losses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT,
                        help="where to save the checkpoint (default: models/model.pth)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="use only the first N training/test rows (for quick smoke runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=Path, default=None,
                        help="optional path to save a per-sample test-loss plot (PNG)")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, test = split_and_oversample(pd.read_csv(args.data), seed=args.seed)
    if args.max_samples:
        train, test = train.head(args.max_samples), test.head(args.max_samples)
    print(f"train samples: {len(train)}, test samples: {len(test)}")

    model = BaseModel(num_outputs=1, labels=[TARGET], lr=0.00001)
    skipped = train_epochs(model, train, args.epochs)
    if skipped:
        print(f"Skipped {skipped} unparsable/oversized training molecules")

    avg_loss, losses = evaluate(model, test)
    print(f"Average test MSE: {avg_loss}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.plot(losses)
        plt.xlabel("test sample")
        plt.ylabel("squared error")
        plt.savefig(args.plot)

    model.save(args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
