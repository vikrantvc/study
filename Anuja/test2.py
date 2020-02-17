import pandas as pd
import os
from sklearn.model_selection import train_test_split

df=pd.read_csv("/home/vikrant/Downloads/vikrant/study/github/study/Anuja/data/Train_anuja.csv")
df1=pd.read_csv(os.path.join(os.getcwd(), "data","Test_anuja.csv"), encoding='utf-8')
# print(type(df["Voltage_10"]))
# print(df[(df["Voltage_10"]==3.59) & (df["Current"]==0)].index.values)
# print(df.iloc[432:,:])
print(df1)
# print(os.getcwd())
