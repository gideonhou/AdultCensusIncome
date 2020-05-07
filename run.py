from numpy import median, mean, NaN
import os
from clean import clean, drop

filename = "adult.csv"
filepath = os.path.join(os.getcwd(), "data", filename)

df = clean(filepath, {'income':
                          {'<=50K': False,
                           '>50K': True}})

df1 = df.replace(to_replace="?", value=NaN).dropna()
print df1

#print drop(df, ["?"])
#print callable(clean)
#print callable(filename)
#print clean(filepath, {"?": "KKK"})

