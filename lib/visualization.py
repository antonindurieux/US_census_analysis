import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_variable_crosstab(
    learn_df: pd.DataFrame, test_df: pd.DataFrame, variable: str
) -> pd.DataFrame:
    """
    Return crosstab DataFrame of a variable for categorical plot.

    Args:
        learn_df (pd.DataFrame): Learn DataFrame.
        test_df (pd.DataFrame): Test DataFrame
        variable (str): Columns name.

    Returns:
        pd.DataFrame: Crosstab DataFrame.
    """
    learn_crosstb = pd.crosstab(
        learn_df[variable],
        learn_df["target"],
        normalize="index",
    )
    learn_crosstb["set"] = "learn"
    test_crosstb = pd.crosstab(
        test_df[variable],
        test_df["target"],
        normalize="index",
    )
    test_crosstb["set"] = "test"
    crosstab_df = pd.concat([learn_crosstb, test_crosstb]).reset_index()
    crosstab_df = crosstab_df.rename(columns={"50000+.": "Proportion of 50000+"})

    return crosstab_df


def plot_target_proportion_vs_categorical(learn_df, test_df, variable):
    sns.barplot(
        data=get_variable_crosstab(learn_df, test_df, variable),
        x=variable,
        y="Proportion of 50000+",
        hue="set",
    )
    plt.xticks(rotation=45)
