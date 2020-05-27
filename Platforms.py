from numpy import linspace
from pandas import get_dummies
from abc import ABCMeta

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from MixedNB import MixedNB
from clean import cat2cont, cont2cat

__all__ = ['LabelEncoderNB', 'OneHotNB', 'CombinedNB', 'DecisionTree', 'LogisticRegression', 'RandomForest']


class AbstractPlatform(object):
    """
    
    """

    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    def format_attr(self, df):
        return cat2cont(df)

    def create_model(self, train_data, train_response):
        return self.model.fit(train_data, train_response.values.ravel())

    def get_results(self, test_data, test_response):
        predict = self.model.predict(test_data)
        return {"Accuracy": metrics.accuracy_score(test_response, predict)}


class LabelEncoderNB(AbstractPlatform):

    def __init__(self):
        super().__init__(model=GaussianNB())


class OneHotNB(AbstractPlatform):

    def __init__(self):
        super().__init__(model=GaussianNB)

    def format_attr(self, df):
        """
        This method applies one hot coding to categorical attributes
        :return:
        """

        # create a copy of original dataframe for conversion to all continuous attributes
        df_clean = df.copy()

        # create copy of all obj (categorical) type attributes into its own dataframe
        df_obj = df_clean.select_dtypes(include=['object']).copy()

        # get column names of obj dataframe
        obj_col_names = [col for col in df_obj.columns]

        return get_dummies(df_clean, columns=obj_col_names)


class CombinedNB(AbstractPlatform):

    def __init__(self):
        super().__init__(model=MixedNB())

    def format_attr(self, df):
        bin_map = {'age': linspace(0, 100, 11).tolist(),
                   'hours.per.week': linspace(0, 100, 11).tolist()}
        return cont2cat(df, bin_map)


class DecisionTree(AbstractPlatform):

    def __init__(self):
        super().__init__(DecisionTreeClassifier(max_depth=7))


class LogisticRegression(AbstractPlatform):

    def __init__(self):
        super().__init__(model=LogisticRegression())


class RandomForest(AbstractPlatform):

    def __init__(self):
        super().__init__(RandomForestClassifier(max_depth=7))
