from cgi import test
import datetime as dt
import sys

import pandas
import download
import read
from numpy import ndarray
import numpy
from pandas import DataFrame, Series
import pyspark.sql as sql
from pyspark.sql.functions import lower, col, udf
from zlib import crc32

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

max_distance = 4970


def preprocess() -> tuple[DataFrame, Series, DataFrame, Series]:

    if not read.check_preprocessed_data_exists():
        download.download_dataset()
        data = read.get_small()  # .get_first_frame()
        data = common_preprocess(data)
        # read.save_preprocessed_data(data)
    else:
        data = read.get_preprocessed_data()

    if sys.argv[1] == "canceled":
        index = 'CANCELLED'
        data = preprocess_for_canceled(data)

    if sys.argv[1] == "diverted":
        index = 'DIVERTED'
        data = preprocess_for_diverted(data)

    data_p, data_n = balance_dataframe(data, index, 0.05)

    train_data_p, test_data_p = split_data(data_p)
    train_data_n, test_data_n = split_data(data_n)
    train_data = pandas.concat(
        [train_data_p, train_data_n]).drop_duplicates().reset_index(drop=True)
    test_data = pandas.concat(
        [test_data_p, test_data_n]).drop_duplicates().reset_index(drop=True)
    (train_data, train_labels) = split_labels(train_data, index)
    (test_data, test_labels) = split_labels(test_data, index)
    return (train_data, train_labels, test_data, test_labels)


def balance_dataframe(data: DataFrame, index: str, fraction: float) -> DataFrame:
    positives = data[data[index] == 1]
    negatives = data[data[index] == 0]
    return positives.sample(fraction=fraction), negatives.sample(fraction=fraction)


def split_labels(data: DataFrame, index: str) -> DataFrame:
    labels = data[index]
    data = data.drop(index, axis=1)
    return data, labels


def common_preprocess(data: DataFrame) -> DataFrame:

    # Replace Nan values with the correct default values
    data = data.fillna(value=0)

    # Remove rows with Nan key values
    data = data.dropna(how='any')

    data = convert_names_into_numbers(data)
    data = convert_dates_into_numbers(data)
    data = convert_times_into_numbers(data)
    data = convert_distance_into_numbers(data)

    return data


def preprocess_for_canceled(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data = data.drop('DIVERTED')

    return data


def preprocess_for_diverted(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data = data.drop('CANCELLED')

    return data


def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2**32


def str_to_float(s, encoding="utf-8"):
    return bytes_to_float(s.encode(encoding))


def convert_names_into_numbers(data: DataFrame) -> DataFrame:

    udf_names_conversion = udf(lambda x: str_to_float(x))

    for c in names_columns_to_convert:
        data = data.withColumn(c, udf_names_conversion(col(c)))
    return data


def date_to_day_of_year(date_string) -> float:

    multiplier: float = 1 / 365

    date = dt.datetime.strptime(date_string, "%Y-%m-%d")
    day = date.timetuple().tm_yday - 1
    return day * multiplier


def convert_dates_into_numbers(data: DataFrame) -> DataFrame:

    udf_dates_conversion = udf(lambda x: date_to_day_of_year(x))

    for c in date_columns_to_convert:
        data = data.withColumn(c, udf_dates_conversion(col(c)))

    return data


def time_to_interval(time) -> float:
    multiplier: float = 1 / 2359
    return float(time) * multiplier


def convert_times_into_numbers(data: DataFrame) -> DataFrame:
    udf_time_conversion = udf(lambda x: time_to_interval(x))

    for c in time_columns_to_convert:
        data = data.withColumn(c, udf_time_conversion(col(c)))

    return data


def convert_distance_into_numbers(data: DataFrame) -> DataFrame:
    multiplier: float = float(1) / float(max_distance)
    udf_numeric_conversion = udf(lambda x: float(x) * multiplier)

    data = data.withColumn('DISTANCE', udf_numeric_conversion(col('DISTANCE')))

    return data


def split_data(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    # Take 25% of the data set as test set
    test_sample = data.sample(fraction=0.25)
    training_sample = data.filter(test_sample.index)
    return training_sample, test_sample

# Chiedere cosa fare in caso di valori null su colonne possibilmente rilevanti
# Chiedere se i dati su delay causati da cose come aereo in ritardo o meteo sono disponibili al momento del calcolo
# Aggiungere delay alla partenza


# Separare giorni dai mesi
preprocess()
