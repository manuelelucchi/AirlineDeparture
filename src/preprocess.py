import datetime as dt
import sys
from datetime import datetime as dt
import download
import read
import numpy
import pyspark.sql as ps
from pyspark.sql.functions import col, udf, rand
from pyspark.sql.types import *
from zlib import crc32
from numpy import ndarray
import pandas as pd

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

string_columns_to_convert: list[str] = [
    'CANCELLED',
    'DIVERTED'
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
    'CANCELLED',
    'DIVERTED',
    'index'
]

max_distance = 4970

time_file = open("./data/times.txt", "a")


def print_and_save_time(s: str):
    time_file.write(s + '\n')
    print(s)


def preprocess(index: str, useAllFrames: bool, size: int, balance_size: int, usePyspark: bool) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    if not read.check_preprocessed_data_exists():
        download.download_dataset()

        start_time = dt.now()
        data = read.get_dataset(size, useAllFrames, usePyspark)
        finish_time = dt.now() - start_time
        print_and_save_time("Dataset reading concluded: " +
                            str(finish_time.total_seconds()) + " seconds")
        data = common_preprocess(data, usePyspark)
        read.save_dataset(data, usePyspark)
    else:
        data = read.load_dataset(usePyspark)
        if usePyspark:
            udf_string_conversion = udf(lambda x: float(x), DoubleType())
            for c in preprocess_columns_to_convert:
                data = data.withColumn(c, udf_string_conversion(col(c)))

    data = remove_extra_columns(index, data, usePyspark)

    data_p, data_n = balance_dataframe(data, index, balance_size, usePyspark)

    start_time = dt.now()
    train_data_p, test_data_p = split_data(data_p, usePyspark)
    train_data_n, test_data_n = split_data(data_n, usePyspark)

    if usePyspark:
        train_data = train_data_p.union(train_data_n)
        test_data = test_data_p.union(test_data_n)
    else:
        train_data = pd.concat(
            [train_data_p, train_data_n]).drop_duplicates().reset_index(drop=True)
        test_data = pd.concat(
            [test_data_p, test_data_n]).drop_duplicates().reset_index(drop=True)

    (train_data, train_labels) = split_labels(train_data, index, usePyspark)
    (test_data, test_labels) = split_labels(test_data, index, usePyspark)

    if usePyspark:
        result = (numpy.array(train_data.collect()),
                  numpy.array(train_labels.collect()),
                  numpy.array(test_data.collect()),
                  numpy.array(test_labels.collect()))

        finish_time = dt.now() - start_time
        print_and_save_time("Dataset splitting concluded: " +
                            str(finish_time.total_seconds()) + " seconds")
        return result
    else:
        result = (train_data.to_numpy(), train_labels.to_numpy(),
                  test_data.to_numpy(), test_labels.to_numpy())

        finish_time = dt.now() - start_time
        print_and_save_time("Dataset splitting concluded: " +
                            str(finish_time.total_seconds()) + " seconds")
        return result


def balance_dataframe(data: ps.DataFrame | pd.DataFrame, index: str, n: int, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:
    start_time = dt.now()
    positives = data[data[index] == 1]
    negatives = data[data[index] == 0]
    if usePyspark:
        result = positives.orderBy(rand()).limit(
            n), negatives.orderBy(rand()).limit(n)
        finish_time = dt.now() - start_time
        print_and_save_time("Dataset balancing concluded: " +
                            str(finish_time.total_seconds()) + " seconds")
        return result
    else:
        result = positives.sample(n), negatives.sample(n)
        finish_time = dt.now() - start_time
        print_and_save_time("Dataset balancing concluded: " +
                            str(finish_time.total_seconds()) + " seconds")
        return result


def split_labels(data: ps.DataFrame | pd.DataFrame, index: str, usePyspark: bool) -> tuple[ps.DataFrame | pd.DataFrame, ps.Column | pd.Series]:
    if usePyspark:
        labels = data.select(index)
        data = data.drop(index)
    else:
        labels = data[index]
        data = data.drop(index, axis=1)
    return data, labels


def common_preprocess(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    common_start_time = dt.now()
    # Replace Nan values with the correct default values
    data = data.fillna(value=0)

    # Remove rows with Nan key values
    if usePyspark:
        data = data.dropna(how='any')
    else:
        data = data.dropna(how='any', axis='index')

    null_removal_finish_time = dt.now() - common_start_time
    print_and_save_time("Null values removal concluded: " +
                        str(null_removal_finish_time.total_seconds()) + " seconds")

    names_start_time = dt.now()
    data = convert_names_into_numbers(data, usePyspark)
    names_finish_time = dt.now() - names_start_time
    print_and_save_time("Names conversion concluded: " +
                        str(names_finish_time.total_seconds()) + " seconds")

    dates_start_time = dt.now()
    data = convert_dates_into_numbers(data, usePyspark)
    dates_finish_time = dt.now() - dates_start_time
    print_and_save_time("Dates conversion concluded: " +
                        str(dates_finish_time.total_seconds()) + " seconds")

    times_start_time = dt.now()
    data = convert_times_into_numbers(data, usePyspark)
    times_finish_time = dt.now() - times_start_time
    print_and_save_time("Times conversion concluded: " +
                        str(times_finish_time.total_seconds()) + " seconds")

    distance_start_time = dt.now()
    data = convert_distance_into_numbers(data, usePyspark)
    distance_finish_time = dt.now() - distance_start_time
    print_and_save_time("Distance conversion concluded: " +
                        str(distance_finish_time.total_seconds()) + " seconds")

    if usePyspark:
        strings_start_time = dt.now()
        data = convert_strings_into_numbers(data, usePyspark)
        strings_finish_time = dt.now() - strings_start_time
        print_and_save_time("Strings conversion concluded: " +
                            str(strings_finish_time.total_seconds()) + " seconds")

    common_finish_time = dt.now() - common_start_time
    print_and_save_time("Common preprocessing concluded: " +
                        str(common_finish_time.total_seconds()) + " seconds")
    return data


def remove_extra_columns(index: str, data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    start_time = dt.now()
    oppositeIndex = 'DIVERTED' if index == 'CANCELLED' else 'CANCELLED'
    if usePyspark:
        data = data.drop(oppositeIndex)
        data = data.drop('index')
    else:
        data = data.drop(oppositeIndex, axis=1)

    finish_time = dt.now() - start_time
    print_and_save_time("Extra column removal concluded: " +
                        str(finish_time.total_seconds()) + " seconds")
    return data


def convert_strings_into_numbers(data: ps.DataFrame | pd.DataFrame) -> ps.DataFrame | pd.DataFrame:
    udf_string_conversion = udf(lambda x: float(x), DoubleType())
    for c in string_columns_to_convert:
        data = data.withColumn(c, udf_string_conversion(col(c)))
    return data


def convert_names_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    if usePyspark:
        def str_to_float(s: str):
            encoding = "utf-8"
            b = s.encode(encoding)
            return float(crc32(b) & 0xffffffff) / 2**32

        udf_names_conversion = udf(lambda x: str_to_float(x), DoubleType())

        for c in names_columns_to_convert:
            data = data.withColumn(c, udf_names_conversion(col(c)))
    else:
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


def convert_dates_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:
    multiplier: float = 1 / 365
    if usePyspark:
        def date_to_day_of_year(date_string) -> float:

            date = dt.datetime.strptime(date_string, "%Y-%m-%d")
            day = date.timetuple().tm_yday - 1
            return day * multiplier

        udf_dates_conversion = udf(
            lambda x: date_to_day_of_year(x), DoubleType())

        for c in date_columns_to_convert:
            data = data.withColumn(c, udf_dates_conversion(col(c)))
    else:
        for i in date_columns_to_convert:
            unique_values: ndarray = []
            values_map: dict = {}

            unique_values = data[i].unique()
            unique_values = numpy.sort(unique_values)

            for v in unique_values:
                date = dt.datetime.strptime(v, "%Y-%m-%d")
                day = date.timetuple().tm_yday - 1
                values_map[v] = day * multiplier

            data[i].replace(to_replace=values_map, inplace=True)

    return data


def convert_times_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    if usePyspark:
        def time_to_interval(time) -> float:
            t = int(float(time))
            h = t // 100
            m = t % 100
            t = h * 60 + m
            return float(t / 1140)

        udf_time_conversion = udf(lambda x: time_to_interval(x), DoubleType())

        for c in time_columns_to_convert:
            data = data.withColumn(c, udf_time_conversion(col(c)))
    else:
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


def convert_distance_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    if usePyspark:
        multiplier: float = float(1) / float(max_distance)
        udf_numeric_conversion = udf(
            lambda x: float(x) * multiplier, DoubleType())

        data = data.withColumn(
            'DISTANCE', udf_numeric_conversion(col('DISTANCE')))
    else:
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


def split_data(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> tuple[ps.DataFrame | pd.DataFrame, ps.DataFrame | pd.DataFrame]:
    # Take 25% of the data set as test set
    if usePyspark:
        #test_sample = data.sample(fraction=0.25)
        #training_sample = data.subtract(test_sample)
        test_sample, training_sample = data.randomSplit(
            [0.25, 0.75], seed=4000)
    else:
        test_sample = data.sample(round(len(data.index) / 4))
        training_sample = data.drop(test_sample.index)
    return training_sample, test_sample
