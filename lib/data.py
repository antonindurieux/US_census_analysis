import os
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd
import requests

DATA_PATH = Path("data/")


def download_data() -> None:
    """
    Download zip file and extract data.
    """
    if DATA_PATH.is_dir():
        print(f"The data is already downloaded in {DATA_PATH}/")

    else:
        DATA_PATH.mkdir(parents=False, exist_ok=True)
        data_link = "http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip"
        with open(DATA_PATH / "us_census_full.zip", "wb") as f:
            request = requests.get(data_link)
            print(f"Downloading data from {data_link}...")
            f.write(request.content)

        with zipfile.ZipFile(DATA_PATH / "us_census_full.zip", "r") as zip_ref:
            print("Unzipping data...")
            zip_ref.extractall(DATA_PATH)

        os.remove(DATA_PATH / "us_census_full.zip")


def get_data_header_and_dtypes() -> tuple[list[str], dict[str, Union[str, type]]]:
    """
    Get data header and a dictionary of column types.

    Returns:
        tuple[list[str], dict[str, Union[str, type]]]: Tuple of list of column names, dictionary of column types.
    """

    with open(DATA_PATH / "us_census_full/census_income_metadata.txt") as file:
        lines = [line.rstrip() for line in file]
    header = [l[l.find("(") + 1 : l.find(")")] for l in lines[81:121]]
    header.append(">50000")

    dtype_list = [l.split(" ")[-1] for l in lines[81:121]]
    dtype_translation = {"continuous": int, "nominal": "category"}
    dtype_list = [dtype_translation[e] for e in dtype_list]
    dtype_col_dict = dict(zip(header, dtype_list))
    dtype_col_dict[">50000"] = "category"

    return header, dtype_col_dict


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data files as DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Learn DataFrame, test DataFrame, and concatenated DataFrame.
    """
    header, dtype_dict = get_data_header_and_dtypes()

    print("Loading learn data...")
    learn_df = pd.read_csv(
        DATA_PATH / "us_census_full/census_income_learn.csv",
        names=header,
        usecols=[i for i in range(42) if i != 24],
        sep=", ",
        engine="python",
    )

    print("Loading test data...")
    test_df = pd.read_csv(
        DATA_PATH / "us_census_full/census_income_test.csv",
        names=header,
        usecols=[i for i in range(42) if i != 24],
        sep=", ",
        engine="python",
    )

    print("Process columns...")
    # Handle nans
    learn_df = learn_df.fillna("Missing value")
    test_df = test_df.fillna("Missing value")

    # Cast columns
    learn_df = learn_df.astype(dtype_dict)
    test_df = test_df.astype(dtype_dict)

    # Handle special cases to avoid plot errors
    for cat_var_num in [
        "own business or self employed",
        "veterans benefits",
        "detailed industry recode",
        "detailed occupation recode",
        "year",
    ]:
        learn_df[cat_var_num] = learn_df[cat_var_num].cat.as_ordered()
        test_df[cat_var_num] = test_df[cat_var_num].cat.as_ordered()

    # Convert target to boolean
    learn_df[">50000"] = learn_df[">50000"] == "50000+."
    test_df[">50000"] = test_df[">50000"] == "50000+."

    # Concatenation of both datasets
    learn_df["set"] = "learn"
    test_df["set"] = "test"
    learn_test_df = pd.concat([learn_df, test_df])

    print("Done")

    return learn_df, test_df, learn_test_df
