import sys
from datetime import datetime as dt
import time as tm
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


def preprocess(index: str, split_number: int, useAllFrames: bool, size: int, usePyspark: bool) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    if not read.check_preprocessed_data_exists():
        download.download_dataset()

        start_time = tm.time()
        data = read.get_dataset(size, useAllFrames, usePyspark)

        finish_time = tm.time() - start_time
        print_and_save_time("Dataset reading concluded: " +
                            str(finish_time) + " seconds")
        # if earlyBalance:
        #data = early_balance(data, index, balance_size, usePyspark)

        data = common_preprocess(data, usePyspark)
        read.save_dataset(data, usePyspark)
    else:
        data = read.load_dataset(usePyspark)
        if usePyspark:
            udf_string_conversion = udf(lambda x: float(x), DoubleType())
            for c in preprocess_columns_to_convert:
                data = data.withColumn(c, udf_string_conversion(col(c)))

    data = remove_extra_columns(index, data, usePyspark)

    #data_p, data_n = balance_dataframe(data, index, balance_size, usePyspark)

    start_time = tm.time()
    splits = split_data(data, usePyspark, index, split_number)

    if usePyspark:
        #    train_data = train_data_p.union(train_data_n)
        #    test_data = test_data_p.union(test_data_n)
        # else:
        #    train_data = pd.concat(
        #        [train_data_p, train_data_n]).drop_duplicates().reset_index(drop=True)
        #    test_data = pd.concat(
        #        [test_data_p, test_data_n]).drop_duplicates().reset_index(drop=True)

        #(train_data, train_labels) = split_labels(train_data, index, usePyspark)
        #(test_data, test_labels) = split_labels(test_data, index, usePyspark)

        # if usePyspark:
        #    result = (numpy.array(train_data.collect()),
        #              numpy.array(train_labels.collect()),
        #              numpy.array(test_data.collect()),
        #              numpy.array(test_labels.collect()))

        #    result[1].shape = [result[1].shape[0]]
        #    result[3].shape = [result[3].shape[0]]

        finish_time = tm.time() - start_time
        print_and_save_time("Dataset splitting concluded: " +
                            str(finish_time) + " seconds")
        return splits
    else:
        #    result = (train_data.to_numpy(), train_labels.to_numpy(),
        #              test_data.to_numpy(), test_labels.to_numpy())

        finish_time = tm.time() - start_time
        print_and_save_time("Dataset splitting concluded: " +
                            str(finish_time) + " seconds")
        return splits


def early_balance(data: ps.DataFrame | pd.DataFrame, index: str, n: int, usePyspark: bool):
    start_time = tm.time()
    positives = data[data[index] == 1]
    negatives = data[data[index] == 0]

    if usePyspark:
        result = positives.orderBy(rand()).limit(
            n).union(negatives.orderBy(rand()).limit(n))

        result.count()
        finish_time = tm.time() - start_time
        print_and_save_time("Early balancing concluded: " +
                            str(finish_time) + " seconds")
        return result
    else:
        finish_time = tm.time() - start_time
        print_and_save_time("Early balancing concluded: " +
                            str(finish_time) + " seconds")
        return pd.concat([positives.sample(n), negatives.sample(n)]).reset_index(drop=True)


def balance_dataframe(data: ps.DataFrame | pd.DataFrame, index: str, n: int, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:
    start_time = tm.time()
    positives = data[data[index] == 1]
    negatives = data[data[index] == 0]
    if usePyspark:

        result = positives.orderBy(rand()).limit(
            n), negatives.orderBy(rand()).limit(n)
        finish_time = tm.time() - start_time
        print_and_save_time("Dataset balancing concluded: " +
                            str(finish_time) + " seconds")
        return result
    else:
        result = positives.sample(n), negatives.sample(n)
        finish_time = tm.time() - start_time
        print_and_save_time("Dataset balancing concluded: " +
                            str(finish_time) + " seconds")
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

    common_start_time = tm.time()

    if usePyspark:
        # Replace Nan values with the correct default values
        data = data.fillna(value=0)
        # Remove rows with Nan key values
        data = data.dropna(how='any')
    else:
        data.fillna(value=0, inplace=True)
        data.dropna(how='any', axis='index', inplace=True)

    null_removal_finish_time = tm.time() - common_start_time
    print_and_save_time("Null values removal concluded: " +
                        str(null_removal_finish_time) + " seconds")

    names_start_time = tm.time()
    data = convert_names_into_numbers(data, usePyspark)
    names_finish_time = tm.time() - names_start_time
    print_and_save_time("Names conversion concluded: " +
                        str(names_finish_time) + " seconds")

    dates_start_time = tm.time()
    data = convert_dates_into_numbers(data, usePyspark)
    dates_finish_time = tm.time() - dates_start_time
    print_and_save_time("Dates conversion concluded: " +
                        str(dates_finish_time) + " seconds")

    times_start_time = tm.time()
    data = convert_times_into_numbers(data, usePyspark)
    times_finish_time = tm.time() - times_start_time
    print_and_save_time("Times conversion concluded: " +
                        str(times_finish_time) + " seconds")

    distance_start_time = tm.time()
    data = convert_distance_into_numbers(data, usePyspark)
    distance_finish_time = tm.time() - distance_start_time
    print_and_save_time("Distance conversion concluded: " +
                        str(distance_finish_time) + " seconds")

    if usePyspark:
        strings_start_time = tm.time()
        data = convert_strings_into_numbers(data)
        strings_finish_time = tm.time() - strings_start_time
        print_and_save_time("Strings conversion concluded: " +
                            str(strings_finish_time) + " seconds")

    common_finish_time = tm.time() - common_start_time
    print_and_save_time("Common preprocessing concluded: " +
                        str(common_finish_time) + " seconds")
    return data


def remove_extra_columns(index: str, data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    start_time = tm.time()
    oppositeIndex = 'DIVERTED' if index == 'CANCELLED' else 'CANCELLED'
    if usePyspark:
        data = data.drop(oppositeIndex)
        data = data.drop('index')
        data.count()
    else:
        data.drop(oppositeIndex, axis=1, inplace=True)

    finish_time = tm.time() - start_time
    print_and_save_time("Extra column removal concluded: " +
                        str(finish_time) + " seconds")
    return data


def convert_strings_into_numbers(data: ps.DataFrame | pd.DataFrame) -> ps.DataFrame | pd.DataFrame:
    udf_string_conversion = udf(lambda x: float(x), DoubleType())
    for c in string_columns_to_convert:
        data = data.withColumn(c, udf_string_conversion(col(c)))
    data.count()
    return data


def convert_names_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    def str_to_float(s: str):
        encoding = "utf-8"
        b = s.encode(encoding)
        return float(crc32(b) & 0xffffffff) / 2**32

    if usePyspark:
        udf_names_conversion = udf(lambda x: str_to_float(x), DoubleType())
        for c in names_columns_to_convert:
            data = data.withColumn(c, udf_names_conversion(col(c)))
        data.count()
    else:
        for c in names_columns_to_convert:
            data[c] = data[c].apply(str_to_float)
    return data


def convert_dates_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:
    multiplier: float = 1 / 365

    def date_to_day_of_year(date_string) -> float:

        date = dt.strptime(date_string, "%Y-%m-%d")
        day = date.timetuple().tm_yday - 1
        return day * multiplier

    if usePyspark:
        udf_dates_conversion = udf(
            lambda x: date_to_day_of_year(x), DoubleType())
        for c in date_columns_to_convert:
            data = data.withColumn(c, udf_dates_conversion(col(c)))
        data.count()
    else:
        for i in date_columns_to_convert:
            data[i] = data[i].apply(date_to_day_of_year)

    return data


def convert_times_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    def time_to_interval(time) -> float:
        t = int(float(time))
        h = t // 100
        m = t % 100
        t = h * 60 + m
        return float(t / 1140)

    if usePyspark:
        udf_time_conversion = udf(lambda x: time_to_interval(x), DoubleType())
        for c in time_columns_to_convert:
            data = data.withColumn(c, udf_time_conversion(col(c)))
        data.count()
    else:
        for c in time_columns_to_convert:
            data[c] = data[c].apply(time_to_interval)

    return data


def convert_distance_into_numbers(data: ps.DataFrame | pd.DataFrame, usePyspark: bool) -> ps.DataFrame | pd.DataFrame:

    multiplier: float = float(1) / float(max_distance)
    if usePyspark:
        udf_numeric_conversion = udf(
            lambda x: float(x) * multiplier, DoubleType())
        data = data.withColumn(
            'DISTANCE', udf_numeric_conversion(col('DISTANCE')))
        data.count()
    else:
        for c in numeric_columns_to_convert:
            data[c] = data[c].apply(lambda x: x * multiplier)
    return data


def split_data(data: ps.DataFrame | pd.DataFrame, usePyspark: bool, label: str, k: int) -> tuple[ps.DataFrame | pd.DataFrame, ps.DataFrame | pd.DataFrame]:
    split_list = []

    if usePyspark:

        total_positives = data.filter(col(label) == 1).count()
        total_negatives = data.filter(col(label) == 0).count()
        positives_negatives_ratio = total_positives/total_negatives
        k_elements_number = round(data.count() / k)

        k_positive_elements = round(
            k_elements_number * positives_negatives_ratio)
        k_negative_elements = round(
            k_elements_number * (1 - positives_negatives_ratio))

        i = 0
        while i < k:
            k_positive_sample = data.where(
                col(label) == 1).limit(k_positive_elements)
            print(k_positive_sample.count())
            k_negative_sample = data.where(
                col(label) == 0).limit(k_negative_elements)
            print(k_negative_sample.count())
            k_sample = k_positive_sample.union(k_negative_sample)
            print(k_sample.count())

            split_list.append(k_sample)
            print(data.count())
            data = data.subtract(k_sample)
            print(data.count())
            print("Concluso giro numero " + str(i))
            i += 1

    else:

        data_positives = data.query(label + ' == 1')
        data_negatives = data.query(label + ' == 0')

        total_positives = len(data_positives)
        total_negatives = len(data_negatives)
        positives_negatives_ratio = total_positives/total_negatives

        k_elements_number = round(len(data) / k)

        k_positive_elements = round(
            k_elements_number * positives_negatives_ratio)
        k_negative_elements = round(
            k_elements_number * (1 - positives_negatives_ratio))

        for i in range(1, k + 1):
            k_positive_sample = data_positives.head(k_positive_elements)
            k_negative_sample = data_negatives.head(k_negative_elements)
            k_sample = pd.concat([k_positive_sample, k_negative_sample])

            split_list.append(k_sample.to_numpy())
            data_positives = data_positives.drop(k_positive_sample.index)
            data_negatives = data_negatives.drop(k_negative_sample.index)

    return split_list
