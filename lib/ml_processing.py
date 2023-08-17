import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 2910


def build_processing_pipelines(
    X_learn: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> dict[str, sklearn.pipeline.Pipeline]:  # type: ignore
    """
    Build pipelines of transforms with final estimators for logistic regression, decision tree and histogram gradient boosting.

    Args:
        X_learn (pd.DataFrame): Learn features DataFrame.
        numerical_features (list[str]): List of numerical features.
        categorical_features (list[str]): List of categorical features.

    Returns:
        dict[str, sklearn.pipeline.Pipeline]: dictionary of logistic regression, decision tree and histogram gradient boosting pipelines.
    """
    # Models
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    hgb = HistGradientBoostingClassifier(
        categorical_features=[
            X_learn.columns.get_loc(col) for col in categorical_features
        ],
        max_iter=100,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    # Logistic regression pipeline
    lr_ct = ColumnTransformer(
        [
            (
                "scaler",
                StandardScaler(),
                numerical_features,
            ),
            (
                "onehot",
                OneHotEncoder(),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    lr_pipeline = Pipeline(
        steps=[
            ("col_trans", lr_ct),
            (
                "model",
                lr,
            ),
        ]
    )

    # Decision tree pipeline
    dt_ct = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    dt_pipeline = Pipeline(
        steps=[
            ("col_trans", dt_ct),
            (
                "model",
                dt,
            ),
        ]
    )

    # Histogram gradient boosting pipeline
    hgb_ct = ColumnTransformer(
        [
            (
                "ordinal",
                OrdinalEncoder(),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    hgb_pipeline = Pipeline(
        steps=[
            ("col_trans", hgb_ct),
            (
                "model",
                hgb,
            ),
        ]
    )

    return {
        "logistic regression": lr_pipeline,
        "decision tree": dt_pipeline,
        "histogram gradient boosting": hgb_pipeline,
    }


def train_and_evaluate_models(
    pipelines: dict[str, sklearn.pipeline.Pipeline],  # type: ignore
    X_learn: pd.DataFrame,
    X_test: pd.DataFrame,
    y_learn: pd.Series,
    y_test: pd.DataFrame,
) -> tuple[dict[str, sklearn.pipeline.Pipeline], pd.DataFrame,]:  # type: ignore
    """
    Train pipelines and compute corresponding precisions, recalls and F1-scores.

    Args:
        pipelines (dict[str, sklearn.pipeline.Pipeline]): dictionary of logistic regression, decision tree and histogram gradient boosting pipelines.
        X_learn (pd.DataFrame): DataFrame of learn data.
        X_test (pd.DataFrame): DataFrame of test data.
        y_learn (pd.Series): Series of learn labels.
        y_test (pd.DataFrame): DataFrame of test labels.

    Returns:
        tuple[dict[str, sklearn.pipeline.Pipeline], pd.DataFrame,]: Dictionary of fitted pipelines, and metrics DataFrame.
    """
    metrics_df = pd.DataFrame()
    target = y_test.columns[0]

    for model_name, pipeline in pipelines.items():
        print(f"Training {model_name}...")
        pipeline.fit(X_learn, y_learn)
        y_test[f"pred_{model_name}"] = pipeline.predict(X_test)

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_test[target],
            y_test[f"pred_{model_name}"],
            average="binary",
        )
        metrics_df.loc[
            model_name,
            [
                "precision (>50000 class)",
                "recall (>50000 class)",
                "f1_score (>50000 class)",
            ],
        ] = [
            precision,
            recall,
            f1,
        ]
        metrics_df.loc[model_name, "accuracy"] = metrics.accuracy_score(  # type: ignore
            y_test[target],
            y_test[f"pred_{model_name}"],
        )
        metrics_df.loc[model_name, "error_rate"] = (
            y_test[target] != y_test[f"pred_{model_name}"]
        ).sum() / len(y_test)

    return pipelines, metrics_df
