import pandas as pd
from sklearn.model_selection import train_test_split
from spam.data.dataset import SpamDataset


def create_datasets_from_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    augment_fn=None,
    random_state: int = 42,
):
    """
    Splits a DataFrame into train / val / test datasets.

    Fractions must sum to 1.0
    """

    assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6

    email: pd.DataFrame = df.copy()
    email["text"] = email.apply(
        lambda row: f"Subject: {row['subject']} [SEP] Body: {row['body']}", axis=1
    ).drop(columns=["subject", "body"])

    # Split off test set
    train_val_df, test_df = train_test_split(
        email,
        test_size=test_fraction,
        stratify=df[label_col],
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
        train_df,
        text_col=text_col,
        label_col=label_col,
        augment_fn=None,
    )

    val_ds = SpamDataset(
        val_df,
        text_col=text_col,
        label_col=label_col,
        augment_fn=None,
    )

    test_ds = SpamDataset(
        test_df, text_col=text_col, label_col=label_col, augment_fn=None
    )

    return train_ds, val_ds, test_ds
