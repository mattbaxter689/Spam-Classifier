# TODO: This is the main file for testing at the moment. It will change as time goes on
import mlflow
from spam.data.utils import create_datasets_from_dataframe
from spam.training.trainer import TrainingManager
import pandas as pd
import os


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment("spam-classifier")
    data = pd.read_csv("dataset/Enron.csv")
    train, val, test = create_datasets_from_dataframe(data)

    manager = TrainingManager(
        train,
        val,
    )
    manager.tune()


if __name__ == "__main__":
    main()
