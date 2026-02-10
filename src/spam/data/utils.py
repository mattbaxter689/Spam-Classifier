import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import set_config
from spam.data.dataset import SpamDataset

set_config(transform_output="pandas")


def create_datasets_from_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
):
    """
    Splits a DataFrame into train / val / test datasets.

    Fractions must sum to 1.0
    """

    assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6

    email: pd.DataFrame = df.copy()

    # Split off test set
    train_val_df, test_df = train_test_split(
        email,
        test_size=test_fraction,
        stratify=email[label_col],
        random_state=random_state,
    )

    # Split train vs val
    val_relative_fraction = val_fraction / (train_fraction + val_fraction)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_fraction,
        stratify=train_val_df[label_col],
        random_state=random_state,
    )

    # Create datasets
    train_ds = SpamDataset(
        data=train_df,
        label_col=label_col,
    )

    val_ds = SpamDataset(
        data=val_df,
        label_col=label_col,
    )

    test_ds = SpamDataset(
        data=test_df,
        label_col=label_col,
    )

    return train_ds, val_ds, test_ds
