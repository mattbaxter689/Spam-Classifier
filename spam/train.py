# TODO: This is the main file for testing at the moment. It will change as time goes on
import mlflow
from transformers import AutoTokenizer, AutoModel
from spam.data.utils import create_datasets_from_dataframe
from spam.training.train_manager import TrainingManager
from spam.registry.register import ChampionChallengerManager
import pandas as pd
import torch.nn as nn


def encoder_factory() -> nn.Module:
    return AutoModel.from_pretrained("distilbert-base-uncased")


def main():
    # Update the server to reference workspace mlflow.Set with env var and reference
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # reference the already tokenized dataset on blob storage
    data = pd.read_csv("dataset/Enron.csv")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train, val, test = create_datasets_from_dataframe(data, tokenizer)

    manager = TrainingManager(encoder_factory, train, val, test)
    manager.tune()
    manager.tune_threshold()
    test_metrics, run_id = manager.train_final()
    post_fit_manager = ChampionChallengerManager(challenger_metrics=test_metrics)
    post_fit_manager.promote()


if __name__ == "__main__":
    main()
