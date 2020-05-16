import matplotlib.pyplot as plt
import seaborn as sns


def distplot(df):

    for column in df.columns:
        col_data = df[column]
        print type(col_data)
        str_mean = str(round(col_data.mean(), 4) * 100) + "%"
        str_std = str(round(col_data.std(), 4) * 100) + "%"
        sns.distplot(col_data, label=column + " performance" + "\n  mean: " + str_mean + "\n  std: " + str_std, bins=10)

    # print df.describe()

    # Plot formatting
    plt.legend(prop={'size': 12}, loc='upper left')
    plt.title('Naive Bayes Platform Performance')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Density = frq / sample size / width')
    plt.grid(True)
    plt.show()


def ttest():
    return 0