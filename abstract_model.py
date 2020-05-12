from abc import ABCMeta, abstractmethod
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from clean import clean

class abstract_model(object):
    """
    This class provides framework of process from dataset -> model

    1.  read dataset into pandas dataframe
    2.  clean dataset
    3.  apply model to cleaned dataset
    4.  compare model predictions to actual results
    5.  return results in a form that is appendable to a aggregate results from other files
    """

    __metaclass__ = ABCMeta

    def __init__(self, file_path):
        self.df = read_csv(file_path)

    def clean_data(self, replace_dict):
        return clean(self.df, replace_dict).dropna()

    @abstractmethod
    def format_attr(self):
        pass

    def split_data(self, df, response_variable, test_ratio=.2):
        explanatory = [col for col in df.columns if col != response_variable]
        response = [response_variable]
        X_train, X_test, y_train, y_test = train_test_split(df[explanatory], df[response], test_size=test_ratio)

        return X_train, X_test, y_train, y_test

    def create_model(self, X_train, y_train):
        return GaussianNB().fit(X_train, y_train.values.ravel())

    def get_results(self, model, X_test, y_test):
        return {"Accuracy": metrics.accuracy_score(y_test, model.predict(X_test))}
