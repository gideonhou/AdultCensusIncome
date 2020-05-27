from sklearn import preprocessing
from sklearn.naive_bayes import BaseNB, CategoricalNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np

#from nb_platforms import cat2cont
from clean import cat2cont


class MixedNB(BaseNB):
    """
    Implementation of Naive Bayes for datasets with continuous and
    categorical attributes

    1. separate categorical and continuous attributes into individual datasets
    2. process categorical attributes accordingly to MultinomialNB
    3. process continuous attributes accordingly to GaussianNB
    """

    def __init__(self):
        self.mNB = CategoricalNB()#MultinomialNB()
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
