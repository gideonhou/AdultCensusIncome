from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from abstract_model import abstract_model


class nb_label_encoder(abstract_model):

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

        # create label encoder to be used for encoding string values to numerical
        label_encoder = preprocessing.LabelEncoder()

        # convert each column in obj dataframe to a continuous value
        for df_obj_col_name in obj_col_names:
            df_obj[df_obj_col_name] = label_encoder.fit_transform(df_obj[df_obj_col_name])

        # reassign all obj attributes in copied original dataframe to continuous attributes
        df_clean[obj_col_names] = df_obj[obj_col_names]

        return df_clean