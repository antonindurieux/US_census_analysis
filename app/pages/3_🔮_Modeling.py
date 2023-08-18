import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib import ml_processing

st.set_page_config(layout="centered")

st.title("Modeling & assessment")

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

categorical_features = [
    "class of worker",
    "education",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "sex",
    "full or part time employment stat",
    "tax filer stat",
    "detailed household summary in household",
    "family members under 18",
    "citizenship",
    "own business or self employed",
    "veterans benefits",
]

numerical_features = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
]

filtered_learn_df = learn_df.drop_duplicates()

st.session_state["X_learn"] = filtered_learn_df[
    categorical_features + numerical_features
].copy()
y_learn = filtered_learn_df[target].copy()
X_test = test_df[categorical_features + numerical_features].copy()

if "y_pred" not in st.session_state:
    st.session_state["y_pred"] = test_df[[target]].copy()

pipelines = ml_processing.build_processing_pipelines(
    st.session_state["X_learn"], numerical_features, categorical_features
)

st.header("Models")
st.markdown(
    """
    We will model the data with 3 different algorithms: a logistic regression, a decision tree, and a histogram gradient boosting.  
    We try to counter the high unbalance of the dataset by passing the parameter `class_weight="balanced"` to each model, which will equalize the data by giving a more important weight to positive exemples.
    """
)
st.markdown(
    f"""
    The selected features are the following:  
    ```python
    numerical_features = {numerical_features}
    ```
    ```python
    categorical_features = {categorical_features}
    ```
    """
)

if "metrics_df" not in st.session_state:
    if st.button("Train models"):
        with st.spinner("Training models (should take less than a min)..."):
            (
                st.session_state["pipelines"],
                st.session_state["metrics_df"],
                st.session_state["y_pred"],
            ) = ml_processing.train_and_evaluate_models(
                pipelines,
                st.session_state["X_learn"],
                X_test,
                y_learn,
                st.session_state["y_pred"],
            )

    else:
        st.write("Click the button to train models (should take around 1 min)")
        st.stop()

st.header("Metrics")
st.markdown(
    """
    To evaluate the performances, we look at the precision, the recall and the f1-score regarding the > 50000 class (the minority class).  
    """
)
st.dataframe(st.session_state["metrics_df"])

st.header("Confusion matrices")
st.markdown(
    """
    We also look at the confusion matrices normalized by True labels, meaning, the proportion of correct and incorrect predictions for both classes.
    """
)

fig1, axs = plt.subplots(1, 3, figsize=(14, 4))
for i, model_name in enumerate(
    [
        "logistic regression",
        "decision tree",
        "histogram gradient boosting",
    ]
):
    ConfusionMatrixDisplay.from_predictions(
        st.session_state["y_pred"][target],
        st.session_state["y_pred"][f"pred_{model_name}"],
        normalize="true",
        colorbar=False,
        cmap=plt.cm.Blues,  # type: ignore
        ax=axs[i],
    )
    axs[i].set_title(f"{model_name} confusion matrix")
    axs[i].grid(False)

plt.tight_layout()

st.pyplot(fig1)

st.markdown(
    """
    - Logistic regression and gradient boosting have a fairly good recall but low precision, meaning there are a few false negative but a lot of false positive. It seems that balancing the weights of each class worked as > 50000 are detected, but now a lot of < 50000 are also wrongly classified in this category.  
    - The weight balancing of the classes seems to not have worked that much for decision tree, which has a stronger precision but a poor recall. This model has a higher accuracy than the 2 others, but at the expense of not being capable of correctly detecting > 50000 exemples.  
                    
    Overall, the choice of the best model depends on the objective: do we want to priorize a good detection of the few > 50000 individuals, to have a low error rate in general, or to have the best balance between recall and precision for the > 50000 class?  
    In the next steps, we will use the gradient boosting model which has the best f1-score, and the best confusion matrix.
    """
)

st.header("Precision-recall curve")

st.markdown(
    "Could we somehow optimize our results by changing the probability threshold for classification ? We can check on the precision-recall curve of the gradient boosting model:"
)

fig2, ax = plt.subplots(figsize=(6, 6))
display = PrecisionRecallDisplay.from_estimator(
    st.session_state["pipelines"]["histogram gradient boosting"]["model"],
    st.session_state["pipelines"]["histogram gradient boosting"]["col_trans"].transform(
        X_test
    ),
    st.session_state["y_pred"][">50000"],
    plot_chance_level=True,
    ax=ax,
)
plt.legend(loc=1)
ax.set_title("Precision-Recall curve of histogram gradient boosting model")

st.pyplot(fig2)

st.markdown(
    "This plot indicates there is no free lunch: either we can achieve a better recall at the expense of the precision, or the reverse. We will thus let the probability threshold at 0.5 by default."
)
