import numpy
from sklearn.model_selection import StratifiedKFold

"""
TODO:

apply logistic regression

include tests for different tuning parameters
    include plots for tuning parameters
    can complement k-fold cross validation

include short write-up
    goals
    methods
    results
    
upload to github

"""
from clean import clean_data, split_data, cont2cat
from Platforms import LabelEncoderNB, OneHotNB, CombinedNB, DecisionTree, LogisticRegression, RandomForest
from numpy import NaN, linspace, unique
import pandas as pd
import os
import matplotlib.pyplot as plt
from DistPlot import distplot, ttest_heatmap

"""
The goal of this project is to predict if a person makes over 50k per year
"""

# import the data into dataframe
filename = "adult.csv"
filepath = os.path.join(os.getcwd(), "data", filename)
response_variable = "income"
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

model_platforms = [LabelEncoderNB, CombinedNB, DecisionTree, RandomForest]#, LogisticRegression]

accuracy_dict = dict(zip([platform.__name__ for platform in model_platforms], [[] for i in range(len(model_platforms))]))
raw_df = pd.read_csv(filepath)

'''
bin_map = {'age': linspace(0, 100, 11).tolist()}#,
           #'capital.gain': linspace(0, 100000, 21).tolist()}#,
           #'capital.loss': unique(raw_df['capital.loss']).tolist(),
           #'hours.per.week': linspace(0, 100, 11).tolist()}

raw_df = cont2cat(raw_df, bin_map)
'''
for platform in model_platforms:
    df = clean_data(raw_df.copy(), data_cleaning_dict)

    model = platform()

    df1 = model.format_attr(df)

    for it in range(100):
        X_train, X_test, y_train, y_test = split_data(df1, response_variable)

        trained_model = model.create_model(X_train, y_train)

        accuracy_dict[platform.__name__].append(model.get_results(X_test, y_test)['Accuracy'])

    '''
    for it in range(100):
        skf = StratifiedKFold(n_splits=10)
        explanatory = df1[[col for col in df1.columns if col != response_variable]]
        response = df1[response_variable]

        for train_index, test_index in skf.split(explanatory, response):
            X_train, X_test = explanatory.iloc[train_index], explanatory.iloc[test_index]
            y_train, y_test = response.iloc[train_index], response.iloc[test_index]

            trained_model = model.create_model(X_train, y_train)

            accuracy_dict[platform.__name__].append(model.get_results(X_test, y_test)['Accuracy'])
    '''
accuracy_df = pd.DataFrame.from_dict(accuracy_dict)
plt.subplot(1, 2, 1)
distplot(accuracy_df)
plt.subplot(1, 2, 2)
ttest_heatmap(accuracy_df)
plt.subplots_adjust(hspace=.4, wspace=.3)
plt.show()