from cgi import test
import datetime as dt
import sys

import pandas
import download
import read
from numpy import ndarray
import numpy
from pandas import DataFrame, Series

default_values: dict = {
    'CANCELLED': 0,
    'DIVERTED': 0
}

columns_to_remove_for_canceled: list[str] = [
    'DIVERTED',  # the flight has been diverted to an unplanned airport
]

columns_to_remove_for_diverted: list[str] = [
    'CANCELLED',  # the flight has been cancelled
]

names_columns_to_convert: list[str] = [
    'OP_CARRIER',
    'ORIGIN',
    'DEST',
]

date_columns_to_convert: list[str] = [
    'FL_DATE'
]

time_columns_to_convert: list[str] = [
    'CRS_DEP_TIME',
    'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME'
]

numeric_columns_to_convert: list[str] = [
    'DISTANCE'
]


def preprocess() -> tuple[DataFrame, Series, DataFrame, Series]:

    if not read.check_preprocessed_data_exists():
        download.download_dataset()
        data = read.get_small()  # .get_first_frame()
        data = common_preprocess(data)
        read.save_preprocessed_data(data)
    else:
        data = read.get_preprocessed_data()

    if sys.argv[1] == "canceled":
        index = 'CANCELLED'
        data = preprocess_for_canceled(data)

    if sys.argv[1] == "diverted":
        index = 'DIVERTED'
        data = preprocess_for_diverted(data)

    train_data, test_data = split_data(data)
    (train_data, train_labels) = split_labels(train_data, index)
    (test_data, test_labels) = split_labels(test_data, index)
    return (train_data, train_labels, test_data, test_labels)


def split_labels(data: DataFrame, index: str) -> DataFrame:
    labels = data[index]
    data = data.drop(index, axis=1)
    return data, labels


def common_preprocess(data: DataFrame) -> DataFrame:

    # Replace Nan values with the correct default values
    data.fillna(default_values, inplace=True)

    # Remove rows with Nan key values
    data.dropna(how='any', axis='index', inplace=True)

    convert_names_into_numbers(data)
    convert_dates_into_numbers(data)
    convert_times_into_numbers(data)
    convert_numerics_into_numbers(data)

    return data


def preprocess_for_canceled(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data.drop(columns_to_remove_for_canceled, axis=1, inplace=True)

    return data


def preprocess_for_diverted(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data.drop(columns_to_remove_for_diverted, axis=1, inplace=True)

    return data


def convert_names_into_numbers(data: DataFrame) -> DataFrame:

    for c in names_columns_to_convert:
        unique_values: ndarray = []
        values_map: dict = {}
        counter: float = 0

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)
        adder: float = 1 / len(unique_values)

        for v in unique_values:
            values_map[v] = counter
            counter += adder

        data[c].replace(to_replace=values_map, inplace=True)

    return data


def convert_dates_into_numbers(data: DataFrame) -> DataFrame:

    multiplier: float = 1 / 365

    for c in date_columns_to_convert:
        unique_values: ndarray = []
        values_map: dict = {}

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)

        for v in unique_values:
            date = dt.datetime.strptime(v, "%Y-%m-%d")
            day = date.timetuple().tm_yday
            values_map[v] = day * multiplier

        data[c].replace(to_replace=values_map, inplace=True)

    return data


def convert_times_into_numbers(data: DataFrame) -> DataFrame:

    multiplier: float = 1 / 2359

    for c in time_columns_to_convert:
        unique_values: ndarray = []
        values_map: dict = {}

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)

        for v in unique_values:
            values_map[v] = v * multiplier

        data[c].replace(to_replace=values_map, inplace=True)

    return data


def convert_numerics_into_numbers(data: DataFrame) -> DataFrame:

    for c in numeric_columns_to_convert:
        unique_values: ndarray = []
        values_map: dict = {}

        unique_values = data[c].unique()
        unique_values = numpy.sort(unique_values)
        multiplier: float = 1 / unique_values.max()

        for v in unique_values:
            values_map[v] = v * multiplier

        data[c].replace(to_replace=values_map, inplace=True)

    return data


def split_data(data: DataFrame):
    # Take 25% of the data set as test set
    test_sample = data.sample(round(len(data.index) / 4))
    training_sample = data.drop(test_sample.index)
    return training_sample, test_sample


# Division between training set and testing set

# Chiedere cosa fare in caso di valori null su colonne possibilmente rilevanti
# Chiedere se i dati su delay causati da cose come aereo in ritardo o meteo sono disponibili al momento del calcolo
# Chiedere se la tempistica attuale è accettabile

# Trasformare tutto in numeri

# Separare giorni dai mesi
