from numpy import median, mean, NaN
import os
from nb_label_encoder import nb_label_encoder
from nb_one_hot import nb_one_hot

"""
The goal of this project is to predict if a person makes over 50k per year
"""
# import the data into dataframe
filename = "adult.csv"
filepath = os.path.join(os.getcwd(), "data", filename)
explanatory_variable = "income"
data_cleaning_dict = {'income':
                        {'<=50K': False,
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


models_to_test = [nb_one_hot, nb_label_encoder]
model_dict = dict(zip([model.__name__ for model in models_to_test], [[] for i in range(len(models_to_test))]))

for model in models_to_test:
    model_instance = model(filepath)

    df = model_instance.clean_data(data_cleaning_dict)

    df1 = model_instance.format_attr(df)
    for it in xrange(0, 100):
        X_train, X_test, y_train, y_test = model_instance.split_data(df1, explanatory_variable)

        trained_model = model_instance.create_model(X_train, y_train)

        model_dict[model.__name__].append(model_instance.get_results(trained_model, X_test, y_test)['Accuracy'])

for k, v in model_dict.iteritems():
    print k, " : ", mean(v), ", ", median(v)