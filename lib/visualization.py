import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats


def plot_learn_test_variable_histograms(
    df: pd.DataFrame, variable: str, target: str
) -> None:
    """
    Plot a facetgrid of the chosen variable
    - Left plot is corresponding to the learn set data, right plot to the test set data
    - Blue hue is the distribution of the '<50000' entries, Orange hue of the '>50000' entries.

    Args:
        df (pd.DataFrame): DataFrame containing learn and test concatenated data.
        variable (str): Variable to plot.
        target (str): Target variable
    """
    if variable in [
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
    ]:
        ylog = True
    else:
        ylog = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        g = sns.FacetGrid(df, col="set", height=5, aspect=1.5)  # type: ignore
        g.map_dataframe(
            sns.histplot,
            x=variable,
            hue=target,
            stat="probability",
            common_norm=False,
            multiple="dodge",
            shrink=0.8,
            legend=True,
            alpha=1,
            log_scale=[False, ylog],
            bins=sorted(df[variable].unique())
            if variable in ["age", "weeks worked in year"]
            else "auto",
        )
        plt.legend([">50000", "<50000"])
        g.axes[0, 0].set_ylabel("Proportion (log scale)" if ylog else "Proportion")
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
        plt.suptitle(f"{variable} distribution (learn and test sets)", y=1.02)
        plt.show()


def cramers_corrected_stat(confusion_matrix: pd.DataFrame) -> np.ndarray:
    """
    Calculate Cramers V statistic for categorical-categorical association.
    See https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix

    Args:
        confusion_matrix (pd.DataFrame): confusion matrix between 2 categorical features (number of cooccurrences between each categories).

    Returns:
        np.ndarray: Cramers V statistic between 2 features
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))  # type: ignore


def cramers_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Compute matrix of Cramers V statistics for the columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with categorical date.
        columns (list[str]): List of columns on which to compute the matrix.

    Returns:
        pd.DataFrame: DataFrame of Cramers V values.
    """
    cramers_matrix = np.zeros((len(columns), len(columns)))
    for col1, col2 in itertools.combinations(columns, 2):
        idx1, idx2 = columns.index(col1), columns.index(col2)
        cramers_matrix[idx1, idx2] = cramers_corrected_stat(
            pd.crosstab(df[col1], df[col2])
        )
        cramers_matrix[idx2, idx1] = cramers_matrix[idx1, idx2]

    np.fill_diagonal(cramers_matrix, 1)
    return pd.DataFrame(cramers_matrix, index=columns, columns=columns)
