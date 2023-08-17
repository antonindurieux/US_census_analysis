import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib import visualization

RANDOM_SEED = 2910

st.set_page_config(layout="centered")

st.title("Results & insights")
if "learn_test_df" not in st.session_state:
    st.text(
        "Data not loaded properly.\nPlease return to the home page and wait the data loading to finish."
    )
    st.stop()

if "pipelines" not in st.session_state:
    st.text(
        "Models are not trained.\nPlease go back to the modeling page and click on the 'Train models' button."
    )
    st.stop()

with st.spinner("Loading models and data..."):
    target = st.session_state["target"]
    pipelines = st.session_state["pipelines"]
    X_learn = st.session_state["X_learn"]
    learn_test_df = st.session_state["learn_test_df"]
    target = st.session_state["target"]

    hgb_pipeline = pipelines["histogram gradient boosting"]

st.markdown(
    """We will use the SHAP library to see what are the most important features for our gradient boosting model:"""
)

if "shap_values" not in st.session_state:
    with st.spinner("Computing SHAP values (should take less than a min)..."):
        X_learn_sample = X_learn.sample(
            20000, random_state=RANDOM_SEED
        )  # sample to reduce execution time
        transformed_data = hgb_pipeline["col_trans"].transform(X_learn_sample)
        st.session_state["X_shap_sample"] = pd.DataFrame(
            transformed_data, columns=X_learn_sample.columns
        )
        explainer = shap.TreeExplainer(
            hgb_pipeline["model"], st.session_state["X_shap_sample"]
        )
        st.session_state["shap_values"] = explainer(st.session_state["X_shap_sample"])

fig1 = shap.plots.bar(st.session_state["shap_values"])

st.set_option("deprecation.showPyplotGlobalUse", False)
st.pyplot(fig1)

st.markdown(
    """
        We see that the most important features for our model are sex, age, weeks worked in year and dividends from stocks, we can represent their distribution to see again how they relate to the target:
        """
)

most_important_features = st.session_state["X_shap_sample"].columns[
    np.argsort(np.mean(np.abs(st.session_state["shap_values"].values), axis=0))[::-1][
        :4
    ]
]
variable = st.selectbox(
    key="variable_tab",
    label="Variable to show",
    options=(most_important_features),
)
fig2 = visualization.plot_single_variable_histogram(learn_test_df, variable, target)  # type: ignore
st.pyplot(fig2)

st.markdown(
    """
                As noted before, these features are highly discriminative between the 2 classes.  
                They also refer to known cultural biases or logical factors.  
                - Women tend to get lower salaries than men in general (it was probably even more true in the 90's).  
                - Age is an important factor as children don't work yet, retirees don't get their full salary anymore, and working people get a salary related to their experience.  
                - The number of weeks worked in year is also logically directly connected to the level of income.  
                - We can make the assumption that for getting a high level of dividends, you need to be heavily invested, which is easier if you get a high income.
                """
)
