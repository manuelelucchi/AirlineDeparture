from multiprocessing.spawn import get_preparation_data
from pandas import DataFrame
from download import download_dataset
from preprocess import preprocess
from read import get_first_frame, get_preprocessed_data
from read import check_preprocessed_data_exists

if not check_preprocessed_data_exists():
    download_dataset()
    data = get_first_frame()
    processed_data = preprocess(data)
else:
    processed_data = get_preprocessed_data()

print(processed_data)
