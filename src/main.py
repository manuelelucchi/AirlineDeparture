from random import random
from multiprocessing.spawn import get_preparation_data
from pandas import DataFrame
from download import download_dataset
from model import Model
from preprocess import preprocess
from read import get_first_frame, get_preprocessed_data
from read import check_preprocessed_data_exists

train_data, train_labels, test_data, test_labels = preprocess()

# model = Model(batch_size=20, learning_rate=0.1)

# model.train(train_data, train_labels, iterations=10)

#res = model.eval()

# if res == label:
#    print("Correct prediction")
# else:
#    print("Wrong prediction")
