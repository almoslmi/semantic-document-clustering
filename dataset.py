import os
import pandas as pd

from typing import List
from zipfile import ZipFile
from paths import list_files


def load_news_dataset(data_path: str, sample_size: int = 0) -> pd.DataFrame:
    files = extract_dataset(data_path)
    if sample_size > 0:
        files = files[:1]

    frames = []
    for file in files:
        frames.append(pd.read_csv(file, index_col=0))
    df = pd.concat(frames)

    return df.sample(sample_size) if sample_size > 0 else df


def extract_dataset(data_path: str) -> List[str]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    # Check if CSV files exist before attempting extract
    csv_files = list_files(data_path, '.csv')
    if csv_files:
        return csv_files

    zip_files = list_files(data_path, '.zip')
    if not zip_files:
        raise FileNotFoundError(_invalid_zip(data_path))

    zip_file = ZipFile(zip_files[0])
    zip_file.extractall(data_path)

    csv_files = list_files(data_path, '.csv')
    if not csv_files:
        raise FileNotFoundError(_invalid_zip(data_path))

    return csv_files


def _invalid_zip(data_path) -> str:
    return f'''Expected .zip data file downloaded from
    https://www.kaggle.com/snapcrack/all-the-news/version/4 in directory `{data_path}`'''
