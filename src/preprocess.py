import datetime as dt
import sys
import download
import read
import numpy
from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from zlib import crc32
from numpy import ndarray

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

preprocess_columns_to_convert: list[str] = [
    'OP_CARRIER',
    'ORIGIN',
    'DEST',
    'FL_DATE',
    'CRS_DEP_TIME',
    'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME',
    'DISTANCE',
    'index'
]

max_distance = 4970


def preprocess() -> tuple[ndarray, ndarray, ndarray, ndarray]:

    if not read.check_preprocessed_data_exists():
        download.download_dataset()
        data = read.get_small()  # .get_first_frame()
        data = common_preprocess(data)
        read.save_preprocessed_data(data)
    else:
        data = read.get_preprocessed_data()
        udf_string_conversion = udf(lambda x: float(x), DoubleType())
        for c in preprocess_columns_to_convert:
            data = data.withColumn(c, udf_string_conversion(col(c)))

    if sys.argv[1] == "canceled":
        index = 'CANCELLED'
        data = preprocess_for_canceled(data)

    if sys.argv[1] == "diverted":
        index = 'DIVERTED'
        data = preprocess_for_diverted(data)

    print(data.schema)
    data_p, data_n = balance_dataframe(data, index, 0.05)

    train_data_p, test_data_p = split_data(data_p)
    train_data_n, test_data_n = split_data(data_n)

    train_data = train_data_p.union(train_data_n)
    test_data = test_data_p.union(test_data_n)

    (train_data, train_labels) = split_labels(train_data, index)
    (test_data, test_labels) = split_labels(test_data, index)

    return (numpy.array(train_data.collect()),
            numpy.array(train_labels.collect()),
            numpy.array(test_data.collect()),
            numpy.array(test_labels.collect()))


def balance_dataframe(data: DataFrame, index: str, fraction: float) -> DataFrame:
    positives = data[data[index] == 1]
    negatives = data[data[index] == 0]
    return positives.sample(fraction=fraction), negatives.sample(fraction=fraction)


def split_labels(data: DataFrame, index: str) -> tuple[DataFrame, Column]:
    labels = data.select(index)
    data = data.drop(index)
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

    udf_string_conversion = udf(lambda x: float(x), DoubleType())
    data = data.withColumn(
        'CANCELLED', udf_string_conversion(col('CANCELLED')))
    return data


def preprocess_for_diverted(data: DataFrame) -> DataFrame:

    # Remove useless columns
    data = data.drop('CANCELLED')

    udf_string_conversion = udf(lambda x: float(x), DoubleType())
    data = data.withColumn('DIVERTED', udf_string_conversion(col('DIVERTED')))
    return data


def convert_names_into_numbers(data: DataFrame) -> DataFrame:

    def str_to_float(s: str):
        encoding = "utf-8"
        b = s.encode(encoding)
        return float(crc32(b) & 0xffffffff) / 2**32

    udf_names_conversion = udf(lambda x: str_to_float(x), DoubleType())

    for c in names_columns_to_convert:
        data = data.withColumn(c, udf_names_conversion(col(c)))

    return data


def convert_dates_into_numbers(data: DataFrame) -> DataFrame:

    def date_to_day_of_year(date_string) -> float:
        multiplier: float = 1 / 365

        date = dt.datetime.strptime(date_string, "%Y-%m-%d")
        day = date.timetuple().tm_yday - 1
        return day * multiplier

    udf_dates_conversion = udf(lambda x: date_to_day_of_year(x), DoubleType())

    for c in date_columns_to_convert:
        data = data.withColumn(c, udf_dates_conversion(col(c)))

    return data


def convert_times_into_numbers(data: DataFrame) -> DataFrame:
    def time_to_interval(time) -> float:
        multiplier: float = 1 / 2359
        return float(time) * multiplier

    udf_time_conversion = udf(lambda x: time_to_interval(x), DoubleType())

    for c in time_columns_to_convert:
        data = data.withColumn(c, udf_time_conversion(col(c)))

    return data


def convert_distance_into_numbers(data: DataFrame) -> DataFrame:
    multiplier: float = float(1) / float(max_distance)
    udf_numeric_conversion = udf(lambda x: float(x) * multiplier, DoubleType())

    data = data.withColumn('DISTANCE', udf_numeric_conversion(col('DISTANCE')))

    return data


def split_data(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    # Take 25% of the data set as test set
    test_sample = data.sample(fraction=0.25)
    training_sample = data.subtract(test_sample)

    return training_sample, test_sample

# Chiedere cosa fare in caso di valori null su colonne possibilmente rilevanti
# Chiedere se i dati su delay causati da cose come aereo in ritardo o meteo sono disponibili al momento del calcolo
# Aggiungere delay alla partenza


# Separare giorni dai mesi
