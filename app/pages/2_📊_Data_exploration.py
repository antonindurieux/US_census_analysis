import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib import visualization

st.set_page_config(layout="centered")

st.title("Data exploration")

if "learn_test_df" not in st.session_state:
    st.text(
        "Data not loaded properly.\nPlease return to the home page and wait the data loading to finish."
    )
    st.stop()

learn_df = st.session_state["learn_df"]
test_df = st.session_state["test_df"]
learn_test_df = st.session_state["learn_test_df"]
numerical_cols = st.session_state["numerical_cols"]
categorical_cols = st.session_state["categorical_cols"]
target = st.session_state["target"]

num_tab, cat_tab = st.tabs(["Numerical variables", "Categorical variables"])

with num_tab:
    st.header("Numerical variables")
    col1, col2 = st.columns(2)
    with col1:
        dataset = st.radio(
            key="dataset_num_tab",
            label="Dataset",
            options=["learn", "test"],
        )
        if dataset == "learn":
            df = learn_df
        else:
            df = test_df

    with col2:
        variable = st.selectbox(
            key="variable_num_tab",
            label="Numerical variable to show",
            options=(numerical_cols),
        )

    fig1 = visualization.plot_single_variable_histogram(df, variable, target)  # type: ignore
    st.pyplot(fig1)

    if variable == "age":
        st.markdown(
            "The age are going from 0 to 90 years old so we get data from children and adults alike. This feature seems informative as the distribution is very different between the >50000 and <50000 classes."
        )
    elif variable == "wage per hour":
        st.markdown(
            "The wage per hour also has different distributions between the 2 classes. It seems that this data is reported in cents. Strangely, a lot of people from the >50000 class also have a wage per hour near 0 values."
        )
    elif variable == "capital gains":
        st.markdown(
            "Capital gains and capital losses have different distributions between the 2 classes, with high income individuals having more gains and losses in general."
        )
    elif variable == "capital losses":
        st.markdown(
            "Capital gains and capital losses have different distributions between the 2 classes, with high income individuals having more gains and losses in general."
        )
    elif variable == "dividends from stocks":
        st.markdown(
            "The 'dividends from stocks' column is informative as only >50000 individuals are associated with high dividends. This data is also probably reported in cents."
        )
    elif variable == "num persons worked for employer":
        st.markdown(
            "High income individuals are associated with more persons working for the employer, while half of the lower income individuals are associated with 0 person working for the employer. It is unclear why this feature is limited to 6 (maybe the data of this column has been encoded as ordinal ?)."
        )
    elif variable == "weeks worked in year":
        st.markdown(
            "More than 80% of high income individuals are working 52 weeks, while 50% of low income individuals are not working during the year (a lot of them probably being children)."
        )

    st.markdown(
        "The distribution are pretty similar between the learn and the test set."
    )

    st.header("Correlation matrix of numerical variables")

    corr = learn_test_df[numerical_cols + [target]].corr()
    fig2, ax = plt.subplots(figsize=(5, 4))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,  # type: ignore
        cbar_kws={"shrink": 0.5},
    )
    st.pyplot(fig2)

    st.markdown(
        """
        We see that the strongest correlation is between "num persons worked for employer" and "weeks worked in year". It seems logical as people working 0 weeks a year would also fall in the "0 persons worked for employer" category.  
        Beside this, correlations are pretty weak. We notice low positive correlations between the target and the features, and the correlation with "wage per hour" is almost 0.
        """
    )

with cat_tab:
    st.header("Categorical variables")
    col1, col2 = st.columns(2)
    with col1:
        dataset = st.radio(
            key="dataset_cat_tab",
            label="Dataset",
            options=["learn", "test"],
        )
        if dataset == "learn":
            df = learn_df
        else:
            df = test_df

    with col2:
        variable = st.selectbox(
            key="variable_cat_tab",
            label="Numerical variable to show",
            options=(categorical_cols),
        )

    fig3 = visualization.plot_single_variable_histogram(df, variable, target)  # type: ignore
    st.pyplot(fig3)

    md_col1, md_col2 = st.columns(2)
    with md_col1:
        st.markdown(
            """
            High income individuals are associated with:  
            - private class of worker,  
            - detailed occupation recode category 2,  
            - bachelor or master's degrees,  
            - married-civilian and spouse present,  
            - executive and managerial, professional specialty, and sales occupations,
            - being a male,  
            - full-time employment,  
            - tax-filer status being "joint both under 65",  
            - being a housholder,  
            - having veterans benefits status "2".
            """
        )
    with md_col2:
        st.markdown(
            """
            Low income individuals are associated with:  
            - "Not in universe" class of worker (probably corresponding to not working, or child),  
            - detailed industry recode and detailed operation recode category 0,  
            - high school graduate education,  
            - never having been married, non tax filler: probably those are proxy for being a child (no other classes seem to accomodate them in the corresponding columns),  
            - being a female,  
            - veterans benefits class 0
            """
        )

    st.header("Cramér's V association matrix of categorical variables")

    st.markdown(
        """
        The matrix of [Cramér's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) represents how much a variable is determined by the other:
        """
    )

    if "cramers_matrix" not in st.session_state:
        with st.spinner(
            "Computing Cramér's V association matrix (should take a few seconds)..."
        ):
            st.session_state["cramers_matrix"] = visualization.cramers_matrix(
                learn_test_df, categorical_cols + [target]
            )

    fig4, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        st.session_state["cramers_matrix"],
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,  # type: ignore
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Cramér's V association matrix of categorical variables")

    st.pyplot(fig4)

    st.markdown(
        """
            The columns related to migration are all strongly correlated between themseleves, as well as the country of birth columns, and previous residence columns.  
            Major and detailed industry, and major and detailed occupations are also correlated as expected.  
            We notice a strong correlation between year and the migration columns, maybe this is due to a change in the polling answer options between 1994 and 1995?  
            """
    )
