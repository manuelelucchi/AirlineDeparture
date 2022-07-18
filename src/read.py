import os
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
