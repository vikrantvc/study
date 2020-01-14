import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def dataLoad():
    path="/home/vikrant/github/study/Anuja"
    return pd.read_csv(os.path.join(os.getcwd(),"Anuja", "data","train_data.csv"), encoding='utf-8')
    # return pd.read_csv(os.path.join(path, "data","train_data.csv"), encoding='utf-8')

#remove Punctuation and Tokenising the review
def tokenising(df):
    df['Review'] = df['Review'].str.replace(r'[^\w\s]+', '')
    df["Review"]=df.iloc[:,0].str.split()
    return df

def trainTestSplit(df):
    X=df.iloc[:,0]
    y=df.iloc[:,2]
    return train_test_split(X, y, test_size=0.33, random_state=42)

def FeatureGeneration(Bag_of_words):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(Bag_of_words)
    return vectorizer.get_feature_names()


def main():
    print("vvv")
    df=dataLoad()
    df=tokenising(df)
    # X_train, X_test, y_train, y_test=trainTestSplit(df)
    print(df["Review"])
    print("--------------------------------")
    print(FeatureGeneration(df["Review"][0]))

if __name__ == "__main__":
    main()















# print("vikrant")
