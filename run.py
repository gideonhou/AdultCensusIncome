from nb_platforms import LabelEncoderPlatform, OneHotPlatform, clean_data, split_data, MixedPlatform
from numpy import NaN
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from DistPlot import distplot

"""
The goal of this project is to predict if a person makes over 50k per year
"""

# import the data into dataframe
filename = "adult.csv"
filepath = os.path.join(os.getcwd(), "data", filename)
explanatory_variable = "income"
data_cleaning_dict = {'income': {'<=50K': False,
                                 '>50K': True},
                      'workclass':
                          {'?': NaN},
                      'fnlwgt':
                          {'?': NaN},
                      'education':
                          {'?': NaN},
                      'education.num':
                          {'?': NaN},
                      'marital.status':
                          {'?': NaN},
                      'occupation':
                          {'?': NaN},
                      'relationship':
                          {'?': NaN},
                      'race':
                          {'?': NaN},
                      'sex':
                          {'?': NaN},
                      'capital.gain':
                          {'?': NaN},
                      'hours.per.week':
                          {'?': NaN},
                      'native.country':
                          {'?': NaN}}

model_platforms = [LabelEncoderPlatform, OneHotPlatform, MixedPlatform]
accuracy_dict = dict(zip([platform.__name__ for platform in model_platforms], [[] for i in range(len(model_platforms))]))
raw_df = pd.read_csv(filepath)

for platform in model_platforms:
    df = clean_data(raw_df.copy(), data_cleaning_dict)

    model = platform()

    df1 = model.format_attr(df)
    for it in xrange(0, 100):
        X_train, X_test, y_train, y_test = split_data(df1, explanatory_variable)

        trained_model = model.create_model(X_train, y_train)

        accuracy_dict[platform.__name__].append(model.get_results(X_test, y_test)['Accuracy'])

accuracy_df = pd.DataFrame.from_dict(accuracy_dict)
distplot(accuracy_df)
#print accuracy_df
'''
for column in accuracy_df.columns:
    print column
    col_data = accuracy_df[column]
    sns.distplot(col_data, kde=False, label=column + " performance")#, bins=8)

#print accuracy_df.describe()


# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Naive Bayes Platform Performance')
plt.xlabel('Accuracy (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()
'''