# TODO: This is the main file for testing at the moment. It will change as time goes on
from spam.data.utils import create_datasets_from_dataframe
import pandas as pd


def main():
    data = pd.read_csv("dataset/Enron.csv")
    train, val, test = create_datasets_from_dataframe(data)


if __name__ == "__main__":
    main()
