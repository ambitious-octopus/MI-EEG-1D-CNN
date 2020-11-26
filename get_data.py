import wget
import os

def get_data():
    """
    This create a new folder data and download the necessary files
    :return: the path
    """
    os.makedirs("data")
    url = "https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip"
    save_path = os.path.join(os.getcwd(), "data")
    wget.download(url, save_path)
    return save_path