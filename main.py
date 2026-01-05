from spam.data.utils import create_datasets_from_dataframe
import pandas as pd


def main():
    data = pd.read_csv("dataset/Enron.csv")
    train, val, test = create_datasets_from_dataframe(data)

    for val in train:
        print(val)


if __name__ == "__main__":
    main()
