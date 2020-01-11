import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split

def dataLoad():
    return pd.read_csv(os.path.join(os.getcwd(), "data","train_data.csv"), encoding='utf-8')

#remove Punctuation and Tokenising the review
def tokenising(df):
    df['Review'] = df['Review'].str.replace(r'[^\w\s]+', '')
    df["Review"]=df.iloc[:,0].str.split()
    return df

def trainTestSplit(df):
    X=df.iloc[:,0]
    y=df.iloc[:,2]
    return train_test_split(X, y, test_size=0.33, random_state=42)

def main():
    df=dataLoad()
    df=tokenising(df)
    X_train, X_test, y_train, y_test=trainTestSplit(df)
    print(df)

if __name__ == "__main__":
    main()















# print("vikrant")
