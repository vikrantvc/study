import pandas as pd
import os
from sklearn.model_selection import train_test_split

df=pd.read_csv("/home/vikrant/Downloads/TenDF.csv")
# print(type(df["Voltage_10"]))
print(df[(df["Voltage_10"]==3.59) & (df["Current"]==0)].index.values)
print(df.iloc[432:,:])
