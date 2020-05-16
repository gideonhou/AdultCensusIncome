
from pandas import get_dummies
from abc import ABCMeta, abstractmethod

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from MixedNB import MixedNB, cat2cont
from clean import clean


__all__ = ['LabelEncoderPlatform', 'OneHotPlatform', 'MixedPlatform']


def clean_data(df, replace_dict):
    return clean(df, replace_dict).dropna()


def split_data(df, response_variable, test_ratio=.2):
    explanatory = [col for col in df.columns if col != response_variable]
    response = [response_variable]
    X_train, X_test, y_train, y_test = train_test_split(df[explanatory], df[response], test_size=test_ratio)

    return X_train, X_test, y_train, y_test




class AbstractPlatform(object):
    """
    
    """

    __metaclass__ = ABCMeta

    def __init__(self, naive_bayes=GaussianNB()):
        self.nb_model = naive_bayes

    @abstractmethod
    def format_attr(self, df):
        pass

    def create_model(self, train_data, train_response):
        return self.nb_model.fit(train_data, train_response.values.ravel())

    def get_results(self, test_data, test_response):
        return {"Accuracy": metrics.accuracy_score(test_response, self.nb_model.predict(test_data))}


class LabelEncoderPlatform(AbstractPlatform):

    def format_attr(self, df):
        return cat2cont(df)


class OneHotPlatform(AbstractPlatform):

    def format_attr(self, df):
        """
        This method transforms non numerical attributes to numerical
        :return:
        """

        # create a copy of original dataframe for conversion to all continuous attributes
        df_clean = df.copy()

        # create copy of all obj (categorical) type attributes into its own dataframe
        df_obj = df_clean.select_dtypes(include=['object']).copy()

        # get column names of obj dataframe
        obj_col_names = [col for col in df_obj.columns]

        return get_dummies(df_clean, columns=obj_col_names)


class MixedPlatform(AbstractPlatform):

    def __init__(self, naive_bayes=MixedNB()):
        self.nb_model = naive_bayes

    def format_attr(self, df):
        #return LabelEncoderPlatform.format_attr(df)
        return df
