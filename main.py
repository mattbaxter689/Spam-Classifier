# TODO: This is the main file for testing at the moment. It will change as time goes on
import mlflow
from spam.data.utils import create_datasets_from_dataframe
from spam.training.trainer import TrainingManager
from spam.registry.register import ChampionChallengerManager
import pandas as pd
import os


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment("spam-classifier")
    data = pd.read_csv("dataset/Enron.csv")
    train, val, test = create_datasets_from_dataframe(data)

    manager = TrainingManager(train, val, test)
    manager.tune()
    recall = manager.train_final()
    post_fit_manager = ChampionChallengerManager()
    post_fit_manager.promote(recall)


if __name__ == "__main__":
    main()
