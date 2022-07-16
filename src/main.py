from pandas import DataFrame
from download import download_dataset
from preprocess import preprocess
from read import get_first_frame

download_dataset()
data = get_first_frame()
processed_data = preprocess(data)

print(processed_data)
