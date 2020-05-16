from sklearn import preprocessing
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np


def cat2cont(df):
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

    # create label encoder to be used for encoding string values to numerical
    label_encoder = preprocessing.LabelEncoder()

    # convert each column in obj dataframe to a continuous value
    for df_obj_col_name in obj_col_names:
        df_obj[df_obj_col_name] = label_encoder.fit_transform(df_obj[df_obj_col_name])

    # reassign all obj attributes in copied original dataframe to continuous attributes
    df_clean[obj_col_names] = df_obj[obj_col_names]

    return df_clean


class MixedNB(BaseNB):
    """
    Implementation of Naive Bayes for datasets with continuous and
    categorical attributes

    1. separate categorical and continuous attributes into individual datasets
    2. process categorical attributes accordingly to MultinomialNB
    3. process continuous attributes accordingly to GaussianNB
    """

    def __init__(self):
        self.mNB = MultinomialNB()
        self.gNB = GaussianNB()

    def _joint_log_likelihood(self, X):
        """
        _joint_log_likelihood for both GaussianNB and MultinomialNB return a numpy array of size (n_samples, n_class)
        that represent posterior. This was calculated using the following formula:
        
        ln(posterior_multinomial) = ln(class prior) + ln(likelihood_categorical)
        
        ln(posterior_Gaussian) = ln(class prior) + ln(likelihood_continuous)
        
        *** ASSUMPTION ***
        assuming that ln(class prior) is the same for both categorical and continuous attributes, we can calculate the
        posterior probability for a dataset of mixed continuous and categorical attributes as follows:
        
        *** JUSTIFYING ASSUMPTION ***
        is class prior for MultinomialNB the same as class prior for GaussianNB???
        
        GaussianNB class_prior is simple: gives the probability of each class (n_class)
        
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        Multinomial the value of class_log_prior depends on input parameters
        
        if class_prior is given, then class_log_prior is log(class_prior)
        
        if fit_prior is True (default), then class_log_prior will be the same as GaussianNB but with log applied
        
        otherwise, class_log_prior will be a uniform value.

        posterior_mixed = class_prior * likelihood_mixed

                        = class_prior * likelihood_categorical * likelihood_continuous

                        = class_prior * likelihood_categorical * class_prior * likelihood_continuous / class_prior

                        = posterior_multinomial * posterior_continuous / class_prior

        ln(posterior_mixed) = ln(posterior_multinomial) + ln(posterior_continuous) - ln(class_prior)


        :param X:
        :return:
        """

        categorical_x = cat2cont(X.select_dtypes(include=['object']))
        continuous_x = X.select_dtypes(include=np.number)

        joint_log_prob_categorical = self.mNB.predict_log_proba(categorical_x)
        joint_log_prob_continuous = self.gNB.predict_log_proba(continuous_x)

        return joint_log_prob_categorical + joint_log_prob_continuous - self.mNB.class_log_prior_

    def fit(self, X, y, sample_weight=None):
        # split X into categorical and continuous
        X_categorical = X.select_dtypes(include=['object'])
        X_continuous = X.select_dtypes(include=np.number)

        X_categorical = cat2cont(X_categorical)
        self.classes_ = np.unique(y)
        self.mNB.fit(X_categorical, y)
        self.gNB.fit(X_continuous, y)
