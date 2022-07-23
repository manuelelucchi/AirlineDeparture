import pandas as pd
from pandas import DataFrame

from preprocess import split_data, split_labels


def preprocess() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    data = pd.read_csv("./data/diabete.csv")
    (train_data, test_data) = split_data(data)
    (train_data, train_labels) = split_labels(train_data, 'Outcome')
    (test_data, test_labels) = split_labels(test_data, 'Outcome')
    return (train_data, train_labels, test_data, test_labels)
