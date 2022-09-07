import os
from kaggle.api.kaggle_api_extended import KaggleApi


# ==================================================================================

os.environ['KAGGLE_USERNAME'] = "davidetricella"
os.environ['KAGGLE_KEY'] = "e1ab3aae4a07f36b37a3a8bace74d9df"


dataset = 'yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018'
path = './data'


def download_dataset():
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.listdir(path):
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset, path, unzip=True, quiet=False)
        except:
            print("Error downloading the dataset")

# =================================================================================
