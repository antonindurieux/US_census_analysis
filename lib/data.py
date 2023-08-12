import os
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
import requests


def download_data() -> None:
    """
    Download zip file and extract data.
    """
    data_path = Path("data/")

    if data_path.is_dir():
        print("The data is already downloaded.")

    else:
        data_path.mkdir(parents=False, exist_ok=True)

        with open(data_path / "us_census_full.zip", "wb") as f:
            request = requests.get(
                "http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip"
            )
            print("Downloading data...")
            f.write(request.content)

        with zipfile.ZipFile(data_path / "us_census_full.zip", "r") as zip_ref:
            print("Unzipping data...")
            zip_ref.extractall(data_path)

        os.remove(data_path / "us_census_full.zip")


def get_data_header_and_dtypes() -> tuple[list[str], dict[str, Union[str, type]]]:
    """
    Get data header and a dictionary of column types.

    Returns:
        tuple[list[str], dict[str, Union[str, type]]]: Tuple of list of column names, dictionary of column types.
    """

    with open("data/us_census_full/census_income_metadata.txt") as file:
        lines = [line.rstrip() for line in file]
    header = [l[l.find("(") + 1 : l.find(")")] for l in lines[81:121]]
    header.append("target")

    dtype_list = [l.split(" ")[-1] for l in lines[81:121]]
    dtype_translation = {"continuous": int, "nominal": "category"}
    dtype_list = [dtype_translation[e] for e in dtype_list]
    dtype_col_dict = dict(zip(header, dtype_list))
    dtype_col_dict["target"] = "category"

    return header, dtype_col_dict
