import os
import zipfile
from pathlib import Path

import requests


def get_data() -> None:
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
