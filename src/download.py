import os
from kaggle.api.kaggle_api_extended import KaggleApi


# ==================================================================================

os.environ['KAGGLE_USERNAME'] = "davidetricella"
os.environ['KAGGLE_KEY'] = "5d0e079e57605185aeda37716bca4471"


dataset = 'yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018'
path = './data'


def download_dataset():
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.listdir(path):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path, unzip=True, quiet=False)

# =================================================================================
