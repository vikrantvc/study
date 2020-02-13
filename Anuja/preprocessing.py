# import gensim
# from gensim.models import Word2Vec
import pandas as pd
import os
import string
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def dataLoad():
    # path="/home/vikrant/github/study/Anuja"
    return pd.read_csv(os.path.join(os.getcwd(), "data","train_data.csv"), encoding='utf-8')
    # return pd.read_csv(os.path.join(path, "data","train_data.csv"), encoding='utf-8')

#remove Punctuation and Tokenising the review
def tokenising(df):
    df['Review'] = df['Review'].str.replace(r'[^\w\s]+', '')
    # df["Review"]=df.iloc[:,0].str.split()              #TFIDF Vectorizer should expect an array of strings
    return df

#transform the words into numbers for train_data
def LabelEncoding(df):
    le = preprocessing.LabelEncoder()
    df["Sentiments"]=le.fit_transform(df["Sentiments"])
    return df

def trainTestSplit(df):
    X=df.iloc[:,0]
    y=df.iloc[:,2]
    return train_test_split(X, y, test_size=0.33, random_state=42)

#convert the reviews into word matrix (find the important word and plot them as feature )
def Word2VecTFIDF(Corpus):
    vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
    X = vectorizer.fit_transform(Corpus)
    # return vectorizer.get_feature_names()     #gives the word list which select as feature
    return X.toarray()                           #incase of fit_transform

def NaiveBayes(X_train,y_train):
    model=MultinomialNB()
    return model.fit(X_train,y_train)


def main():
    # print("vvv")
    # print(os.getcwd())

    df=dataLoad()
    df=tokenising(df)
    LabelEncoding(df)

    X_train, X_test, y_train, y_test=trainTestSplit(df)

    vectorizer = TfidfVectorizer(min_df=1,stop_words="english")
    X_trainNew=vectorizer.fit_transform(X_train)
    X_testNew=vectorizer.transform(X_test)

    model=NaiveBayes(X_trainNew.toarray(),y_train)          #training the model
    pred=model.predict(X_testNew)                           #predecting the labels for test data
    Actual=np.array(y_test)                                 #converting the test labels in numpy array

    print(accuracy_score(pred, Actual)*100)                 #predict accuracy score of model


if __name__ == "__main__":
    main()















# print("vikrant")
