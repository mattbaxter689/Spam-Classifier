from transformers import AutoModel
from spam.data.utils import create_datasets_from_dataframe
from spam.training.train_manager import TrainingManager
from spam.registry.register import ChampionChallengerManager
from spam.data_configs.data_config import ModelHyperParams
import pandas as pd
import torch.nn as nn
import argparse
import torch

torch.set_num_threads(2)
torch.serialization.add_safe_globals([ModelHyperParams])


def encoder_factory() -> nn.Module:
    return AutoModel.from_pretrained("distilbert-base-uncased")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    print("Data path:", args.data_path)

    # reference the already tokenized dataset on blob storage
    data = pd.read_parquet(f"{args.data_path}/Enron_tokenized.parquet")
    # small sample here just to test functionality
    data = data.sample(n=101)
    print(data["label"].value_counts())

    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train, val, test = create_datasets_from_dataframe(data)
    manager = TrainingManager(encoder_factory, train, val, test)
    manager.tune(n_trials=1)
    manager.tune_threshold()
    test_metrics = manager.train_final()
    post_fit_manager = ChampionChallengerManager(challenger_metrics=test_metrics)
    post_fit_manager.promote()


if __name__ == "__main__":
    main()
