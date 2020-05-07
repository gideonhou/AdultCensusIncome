import pandas as pd
from inspect2 import getfullargspec
from numpy.core import int64, float64, array


class Clean(object):
    """
    Clean is a wrap over standard python function
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        """
        When invoked like a function it internally invokes
        the wrapped function and returns the returned value

        :param args: non-keyworded arguments
        :param kwargs: keyworded arguments
        :return: return value of function
        """

        """
        fetching the function to be invoked from the virtual namespace
        through the arguments
        """

        fn = Namespace.get_instance().get(self.fn, *args)
        if not fn:
            raise Exception("no matching function found")

        # invoking the wrapped function and returning the value
        return fn(*args, **kwargs)

    def key(self, args=None):
        """
        Returns the key that will uniquely identify
        a function even when it is overloaded

        :param args: the function
        :return: tuple of function specs
        """

        if args is None:
            args = getfullargspec(self.fn).args

        return tuple([
            self.fn.__module__,
            self.fn.__class__,
            self.fn.__name__,
            len(args or []),
        ])


class Namespace(object):
    """
    Namespace is the singleton class that is responsible
    for holding all the functions
    """

    __instance = None

    def __init__(self):
        if self.__instance is None:
            self.function_map = dict()
            Namespace.__instance = self
        else:
            raise Exception("virtual Namespace already instantiated")

    @staticmethod
    def get_instance():
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance

    def register(self, fn):
        """
        registers the function in the virtual namepsace and returns
        an instance of callable Clean that wraps the function fn
        :param fn: function to register
        :return: function
        """

        func = Clean(fn)
        self.function_map[func.key()] = fn
        return func

    def get(self, fn, *args):
        """
        get returns the matching function from the virtual namespace

        :param fn: function
        :param args: non-keyworded arguments of function
        :return: function
        """

        func = Clean(fn)
        return self.function_map.get(func.key(args=args))


def overload(fn):
    """
    overload is a decorator that wraps the function
    and returns a callable object of type function
    :param fn:
    :return:
    """
    return Namespace.get_instance().register(fn)


@overload
def clean(filepath):
    """
    function for reading .csv file into pandas dataframe

    :param filepath:
    :return:
    """
    return pd.read_csv(filepath)


@overload
def clean(filepath, replace_dict):
    """
    function for reading .csv into pandas dataframe and also
    replacing values. Values in replace_dict can also be functions
    so that replacement values are some measures of tendencies:
    mean, median, mode, min, max

    :param filepath:
    :param replace_dict:
    keys-> columns, vals-> dict
                        keys-> old value, vals-> func
                        or
                        keys-> old value, vals-> new value
    :return:
    """
    dataframe = pd.read_csv(filepath)
    for column_name in dataframe[replace_dict.keys()]:

        sub_replace_dict = replace_dict[column_name]

        for (old_val, replace_val) in sub_replace_dict.iteritems():
            column = dataframe[column_name]
            if callable(replace_val):
                print column.dtype

                if column.dtype == int64 or column.dtype == float64:
                    new_val = replace_val(column)
                    dataframe[column_name] = column.replace(to_replace=old_val, value=new_val)
                else:
                    raise Exception("cannot apply function to ", column.dtype)
            elif isinstance(replace_val, str) or isinstance(replace_val, int):
                dataframe[column_name] = column.replace(to_replace=old_val, value=replace_val)
            else:
                raise Exception(replace_val, " is not a supported value in replace_dict")

    return dataframe

@overload
def drop(dataframe, values):
    """
    This function removes all rows containing values found in values for any column
    :param value:
    :return:
    """

