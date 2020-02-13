import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.DataFrame({'text':['a..b?!??', '%hgh&12','abc123!!!', '$$$1234']})
# print(df)
df['text'] = df['text'].str.replace(r'[^\w\s]+', '')
# print(df)



x=6.666
x=str(x)
x=x[:4]

x=float(x)
print(type(x))
