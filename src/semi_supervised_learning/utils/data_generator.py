from typing import List, Optional
import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import ray


def generate_synthetic_dataset(
    n_samples: int = 10_000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
    multi_label: bool = False,
    n_informative: int = 2,
    weights: Optional[List[float]] = None,
):
    X, y = (
        make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            weights=weights,
        )
        if not multi_label
        else make_multilabel_classification(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes
        )
    )
    # create a polars dataframe features  and label
    df = pl.DataFrame({"features": X, "label": y})
    # save as a parquet file
    df.write_parquet("./data/synthetic_dataset.parquet")


def generate_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_labeled_unlabeled(data, label_size: float = 0.1, random_state: int = 42):
    return train_test_split(
        data,
        train_size=1 - label_size,
        random_state=random_state,
    )


def generate_dataloader(data, n_batch: int = 32, shuffle: bool = True):
    dataset = TensorDataset(data[0], data[1])

    return DataLoader(dataset, batch_size=n_batch, shuffle=shuffle)


if __name__ == "__main__":
    if overwrite := True:
        generate_synthetic_dataset(
            n_samples=10_000, n_features=10, n_classes=1, weights=[0.9]
        )

    df = pl.read_parquet("./data/synthetic_dataset.parquet")
    print(df.head(5))
