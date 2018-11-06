import os

from typing import List


def list_files(data_path: str, ext: str) -> List[str]:
    files = [f'{data_path}/{f}' for f in os.listdir(data_path)]
    return list(filter(lambda f: is_file(f, ext), files))


def is_file(file_path: str, ext: str) -> bool:
    return os.path.isfile(file_path) and file_path.endswith(ext)
