import os
import pandas as pd
from pandas import DataFrame

path = './data'


def get_all_frames() -> DataFrame:
    files = os.listdir('./data')
    big_frame = DataFrame()

    for f in files:
        if f.endswith('.csv'):
            frame = pd.read_csv(path + '/' + f)
            big_frame = pd.concat([big_frame, frame])
            print('Frame ' + f + ' loaded')
    return big_frame


def get_first_frame() -> DataFrame:
    files: list = os.listdir('./data')
    return pd.read_csv(path + '/' + files[0])
