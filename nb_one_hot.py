from abstract_model import abstract_model
from pandas import get_dummies


class nb_one_hot(abstract_model):

    def __init__(self, file_path):
        abstract_model.__init__(self, file_path)

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