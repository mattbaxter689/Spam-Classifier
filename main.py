from spam.data.utils import create_datasets_from_dataframe
import pandas as pd
import os


def main():
    data = pd.read_csv("dataset/Enron.csv")
    print(data)


if __name__ == "__main__":
    main()
