import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

plt.style.use("seaborn-v0_8")

st.set_page_config(layout="centered")

st.title("General observations on the dataset")

if "learn_test_df" not in st.session_state:
    st.text(
        "Data not loaded properly.\nPlease return to the home page and wait the data loading to finish."
    )
    st.stop()

learn_df = st.session_state["learn_df"]
test_df = st.session_state["test_df"]
learn_test_df = st.session_state["learn_test_df"]
target = st.session_state["target"]

col1, col2 = st.columns(2)

with col1:
    st.header("Proportions of missing values")
    missing_values = pd.concat(
        [
            (learn_df == "?").sum() / len(learn_df),
            (test_df == "?").sum() / len(test_df),
        ],
        axis=1,
    ).rename(columns={0: "learn", 1: "test"})
    st.dataframe(missing_values[missing_values.sum(axis=1) > 0])
    st.markdown(
        "The migration-related columns contain around half of unknown values and thus are not very informative."
    )

with col2:
    st.header("Proportions of undefined values")
    unknown_values = pd.concat(
        [
            learn_df.isin(["Not in universe", "Do not know", "NA", "All other"]).sum()
            / len(learn_df),
            test_df.isin(["Not in universe", "Do not know", "NA", "All other"]).sum()
            / len(test_df),
        ],
        axis=1,
    ).rename(columns={0: "learn", 1: "test"})
    st.dataframe(unknown_values[unknown_values.sum(axis=1) > 0])
    st.markdown("A few columns contain a lot of undefined values as well.")


st.header("Proportions of individuals with >\$50000 and <$50000 income")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Annoying seaborn warning

    g = sns.FacetGrid(learn_test_df, col="set", height=4, aspect=1)  # type: ignore
    g.map_dataframe(
        sns.histplot,
        x=target,
        hue=target,
        stat="probability",
        legend=True,
        alpha=1,
    )
    plt.xticks([1.0, 0.0], ["True", "False"])
    plt.legend([">50000", "<50000"])
    plt.suptitle("Target values count in train and test set", y=1.02)

    for ax in g.axes.ravel():
        for c in ax.containers:
            labels = [f"{w*100:.0f}%" if (w := v.get_height()) > 0 else "" for v in c]
            ax.bar_label(
                c,
                labels=labels,
                label_type="edge",
                fontsize=10,
                rotation=90,
                padding=6,
            )
        ax.margins(y=0.2)

st.pyplot(g.fig)

st.markdown(
    """We notice a very strong unbalance in the dataset: only around 6% of the people get an income higher than $50000.  
    We will have to take that into account to model the data and assess the results: a model predicting only the <50000 class would have a 94% accuracy, so it will be more usefull to study other metrics to evaluate the models.  
    We also notice that the distribution is similar between learn and test sets."""
)
