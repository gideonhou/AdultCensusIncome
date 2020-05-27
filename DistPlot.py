import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import pandas as pd


def distplot(df):

    for column in df.columns:
        col_data = df[column]
        str_mean = "{:.2f}".format(col_data.mean() * 100) + "%"
        str_std = "{:.2f}".format(col_data.std() * 100) + "%"
        sns.distplot(col_data, label=column + "\n  mean: " + str_mean + "\n  std: " + str_std, bins=10)

    # Plot formatting
    plt.legend(prop={'size': 12}, loc='upper left')
    plt.title('Naive Bayes Platform Performance')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Density = frq / sample size / width')
    plt.grid(True)


def ttest_heatmap(df):

    # construct a dataframe to be used for creating the t-test matrix
    ttest_col_names = ['x', 'y', 'reject_null']

    dict_list = []
    for column1 in df.columns:
        dict_to_add = {}
        for column2 in df.columns:
            t_stat, p_val = ttest_rel(df[column1], df[column2])
            dict_to_add[column2] = p_val
        dict_list.append(dict_to_add)
    ttest_df = pd.DataFrame(dict_list, index=df.columns)

    sns.heatmap(ttest_df, annot=True)
    plt.title("Paired T-Test P-values")
    plt.grid(False)
    plt.xticks(rotation=0)
