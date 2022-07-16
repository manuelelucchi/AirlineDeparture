import os
import pandas as pd

path= './data'

def GetAllFrames():
    files:list = os.listdir('./data')
    big_frame:pd.DataFrame = pd.DataFrame()

    for f in files:
        if f.endswith('.csv'):
            frame:pd.DataFrame = pd.read_csv(path + '/' + f)
            big_frame = pd.concat([big_frame, frame])

            print(big_frame)
            print('frame ' + f + ' done')
    return big_frame

def GetFirstFrame():
    files:list = os.listdir('./data')
    return pd.read_csv(path + '/' + files[0])
