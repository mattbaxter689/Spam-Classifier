import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class SpamDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        max_length: int = 128,
        augment_fn=None,
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
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        text = row[self.text_col]
        label = row[self.label_col]

        if self.augment_fn:
            text = self.augment_fn(text)

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }
