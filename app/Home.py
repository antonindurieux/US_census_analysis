import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib import data

st.set_page_config(layout="wide")

st.title("US Census Analysis")

st.markdown(
    """
    This app will present the analysis of a dataset from the United States Census Bureau, in order to predict wether a person is making more or less than $50,000 per year.  
    The studied dataset contains demographic and economic information about ~300,000 individuals.
    """
)

with st.spinner("Downloading data (should take a few seconds)..."):
    data.download_data()

if "learn_test_df" not in st.session_state:
    with st.spinner("Loading DataFrames (should take a few seconds)..."):
        learn_df, test_df, learn_test_df = data.load_data()

        st.session_state["learn_test_df"] = learn_test_df
        st.session_state["learn_df"] = learn_df
        st.session_state["test_df"] = test_df
        st.session_state["numerical_cols"] = list(
            learn_df.columns[learn_df.dtypes == int]
        )
        st.session_state["categorical_cols"] = list(
            learn_df.columns[learn_df.dtypes == "category"]
        )
        st.session_state["target"] = ">50000"

st.header("Learn data")
st.dataframe(st.session_state["learn_df"], hide_index=True)
st.text(
    f"Rows: {st.session_state['learn_df'].shape[0]}, Columns: {st.session_state['learn_df'].shape[1]}"
)

st.header("Test data")
st.dataframe(st.session_state["test_df"], hide_index=True)
st.text(
    f"Rows: {st.session_state['test_df'].shape[0]}, Columns: {st.session_state['test_df'].shape[1]}"
)
