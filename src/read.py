from dataclasses import replace
import os
from constants import path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
import pyspark.sql as ps

# ======================================================

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
    "Airline Departure").master('local[1]').getOrCreate()

# =================================================================

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


def check_preprocessed_data_exists() -> bool:
    files = os.listdir('./data')
    for f in files:
        if f.startswith('preprocessed'):
            return True
    return False


def load_dataset(usePyspark: bool) -> pd.DataFrame:
    if usePyspark:
        data = spark.read.option("header", True).csv(
            path + '/preprocessed')
    else:
        data = pd.read_csv(filepath_or_buffer=path + '/' + 'preprocessed.csv')

    print('Preprocessed dataset loaded')
    return data


def save_dataset(data: ps.DataFrame, usePyspark: bool):
    if usePyspark:
        data.write.format('csv').option('header', True).mode('overwrite').option(
            'sep', ',').save(path + '/preprocessed')
    else:
        data.to_csv(path_or_buf=path + '/' + 'preprocessed.csv', index=False)
    print('Preprocessed dataset saved')


def get_dataset(limit: float = -1, allFrames: bool = True, usePyspark: bool = False) -> pd.DataFrame | ps.DataFrame:
    files = os.listdir(path)
    if usePyspark:
        big_frame = spark.createDataFrame(
            spark.sparkContext.emptyRDD(), schema=dataframe_schema)
    else:
        big_frame = pd.DataFrame()

    if not allFrames:
        files = [files[0]]

    for f in files:
        if f.endswith('.csv'):
            if usePyspark:
                frame = spark.read.option("header", True).csv(path + '/' + f)
                frame = frame.select(columns_to_get)
                frame = frame.sample(fraction=1.0, withReplacement=False)

                if limit != -1:
                    frame = frame.limit(limit)

                big_frame = frame.union(big_frame)
            else:
                frame = pd.read_csv(filepath_or_buffer=path +
                                    '/' + f, usecols=columns_to_get)
                if limit != -1:
                    frame = frame.sample(n=limit, replace=False)
                big_frame = pd.concat([big_frame, frame])

    if usePyspark:
        big_frame = big_frame.select(
            "*").withColumn("index", monotonically_increasing_id())
        big_frame.count()

    return big_frame
