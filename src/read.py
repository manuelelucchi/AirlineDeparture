import os
from xmlrpc.client import Boolean, boolean
import pandas as pd
from pandas import DataFrame
from constants import path
from pyspark.sql import SparkSession
from pyspark.sql.types import *

columns_to_get: list[str] = [
    'FL_DATE',
    'OP_CARRIER',
    'ORIGIN',
    'DEST',
    'CRS_DEP_TIME',
    'CRS_ARR_TIME',
    'CANCELLED',
    'DIVERTED',
    'CRS_ELAPSED_TIME',
    'DISTANCE'
]

dataframe_schema = StructType([
    StructField('FL_DATE', StringType(), True),
    StructField('OP_CARRIER', StringType(), True),
    StructField('ORIGIN', StringType(), True),
    StructField('DEST', StringType(), True),
    StructField('CRS_DEP_TIME', StringType(), True),
    StructField('CRS_ARR_TIME', StringType(), True),
    StructField('CANCELLED', StringType(), True),
    StructField('DIVERTED', StringType(), True),
    StructField('CRS_ELAPSED_TIME', StringType(), True),
    StructField('DISTANCE', StringType(), True)
])

spark = SparkSession.builder.appName(
    "Airline Departure").master('local[*]').getOrCreate()


def get_all_frames() -> DataFrame:
    files = os.listdir(path)
    big_frame = spark.createDataFrame(
        spark.sparkContext.emptyRDD(), schema=dataframe_schema)

    for f in files:
        if f.endswith('.csv'):
            # Reading only data at disposal before departure
            # frame = pd.read_csv(filepath_or_buffer=path +
            #                    '/' + f, usecols=columns_to_get)
            #big_frame = pd.concat([big_frame, frame])

            frame = spark.read.option("header", True).csv(path + '/' + f)
            frame = frame.select(columns_to_get)

            big_frame = frame.union(big_frame)
            print('Frame ' + f + ' loaded')
    return big_frame


def get_small() -> DataFrame:
    files: list = os.listdir(path)
    # big_frame = pd.read_csv(filepath_or_buffer=path +
    #                        '/' + files[0], usecols=columns_to_get, nrows=1000000)
    big_frame = spark.read.option("header", True).csv(path + '/' + files[0])
    big_frame = big_frame.select(columns_to_get).limit(1000000)
    print('Small frame loaded')
    return big_frame


def get_first_frame() -> DataFrame:
    files: list = os.listdir(path)
    # big_frame = pd.read_csv(filepath_or_buffer=path +
    #                        '/' + files[0], usecols=columns_to_get)
    big_frame = spark.read.option("header", True).csv(path + '/' + files[0])
    big_frame = big_frame.select(columns_to_get)
    big_frame.show()
    print('First frame loaded')
    return big_frame


def check_preprocessed_data_exists() -> bool:
    files = os.listdir('./data')
    for f in files:
        if f.startswith('preprocessed'):
            return True
    return False


def get_preprocessed_data() -> DataFrame:
    #data = pd.read_csv(filepath_or_buffer=path + '/' + 'preprocessed.csv')
    data = spark.read.option("header", True).csv(
        path + '/' + 'preprocessed.csv')
    print('Preprocessed frame loaded')
    return data


def save_preprocessed_data(data: DataFrame):
    data.to_csv(path_or_buf=path + '/' + 'preprocessed.csv', index=False)
    print('Preprocessed csv created')
