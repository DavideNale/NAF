import os
import yaml


def load_params(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, "r") as file:
        params = yaml.safe_load(file)
    return params
