import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import AutoTokenizer


class SpamDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        label_col: str = "label",
    ) -> None:
        """
        data: pandas dataframe containing the data to pass
        text_col: The column to transform with AutoTokenizer
        label_col: The column representing spam or not
        max_length: Max length of the transformed token allowed
        augment_fn: Function to perform any processing to text data
        """
        super().__init__()
        self.data = data
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["label"], dtype=torch.float),
        }
