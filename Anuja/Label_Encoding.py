from sklearn import preprocessing
import pandas as pd
# from sklearn.naive_bayes import MultinomialNB

df=pd.DataFrame(["paris", "paris", "tokyo", "amsterdam"])
# print(df)
le = preprocessing.LabelEncoder()
x=le.fit_transform(df)

list(le.classes_)

# le.transform(["tokyo", "tokyo", "paris"])
# print(x)

# list(le.inverse_transform([2, 2, 1]))
