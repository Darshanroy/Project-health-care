from kaggle.api.kaggle_api_extended import KaggleApi
import os
from dotenv import load_dotenv
import pandas as pd
from zenml.steps import step, Output

# Load environment variables from .env file
load_dotenv()

# Get Kaggle username and API key from environment variables
kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

# Initialize the Kaggle API
api = KaggleApi()


@step
def kaggle_data_download() -> str:
    # Replace 'dataset-name' with the actual name of the dataset on Kaggle
    dataset_name = 'fedesoriano/stroke-prediction-dataset'
    # Replace 'destination_folder' with the folder where you want to download the dataset
    destination_folder = 'healthcare-dataset-stroke-data/'

    final_destination_folder = os.path.join(destination_folder,'healthcare-dataset-stroke-data.csv')
    # Download the dataset
    api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)

    return final_destination_folder


@step
def load_dataframe(destination_folder: str) -> pd.DataFrame:
    data = pd.read_csv(destination_folder)
    return data

