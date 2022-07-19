import os
from xmlrpc.client import Boolean, boolean
import pandas as pd
from pandas import DataFrame

path = './data'
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

def get_all_frames() -> DataFrame:
    files = os.listdir('./data')
    big_frame = DataFrame()

    for f in files:
        if f.endswith('.csv'):
            # Reading only data at disposal before departure
            frame = pd.read_csv(filepath_or_buffer=path + '/' + f, usecols=columns_to_get)
            big_frame = pd.concat([big_frame, frame])
            print('Frame ' + f + ' loaded')
    return big_frame

def get_first_frame() -> DataFrame:
    files: list = os.listdir('./data')
    big_frame = pd.read_csv(filepath_or_buffer=path + '/' + files[0], usecols=columns_to_get)
    print('First frame loaded')
    return big_frame

def check_preprocessed_data_exists() -> bool:
    files = os.listdir('./data')
    for f in files:
        if f.startswith('preprocessed'):
            return True
    return False

def get_preprocessed_data() -> DataFrame:
    data = pd.read_csv(filepath_or_buffer=path + '/' + 'preprocessed.csv')
    print('Preprocessed frame loaded')
    return data

def save_preprocessed_data(data: DataFrame):
    data.to_csv(path_or_buf=path + '/' + 'preprocessed.csv', index=False)
    print('Preprocessed csv created')