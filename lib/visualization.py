import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_variable_histograms(df: pd.DataFrame, variable: str) -> None:
    """
    Plot a facetgrid of the chosen variable
    - Left plot is corresponding to the learn set data, right plot to the test set data
    - Blue hue is the distribution of the '-50000' entries, Orange hue of the '50000+' entries.

    Args:
        df (pd.DataFrame): DataFrame containing learn and test concatenated data.
        variable (str): Variable to plot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        g = sns.FacetGrid(df, col="set", height=5, aspect=1.5)  # type: ignore
        g.map_dataframe(
            sns.histplot,
            x=variable,
            hue="target",
            stat="probability",
            common_norm=False,
            multiple="dodge",
            shrink=0.8,
            legend=True,
            alpha=1,
        )
        plt.legend(["50000+", "-50000"])
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
        plt.suptitle(f"{variable} distribution (learn and test sets)", y=1.02)
        plt.show()
