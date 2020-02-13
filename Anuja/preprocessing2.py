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
from sklearn.model_selection import KFold

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

#convert the reviews into word matrix (find the important word and plot them as feature with 0,1 value for their presence)
def Word2VecTFIDF(df):
    vect = TfidfVectorizer( max_df=1, stop_words='english')
    X = vect.fit_transform(df.pop('Review')).toarray()
    r = df[['Sentiments']].copy()
    df = pd.DataFrame(X, columns=vect.get_feature_names())
    return df.join(r)

def NaiveBayes(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    scores = []
    model=MultinomialNB()
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X):
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        scores.append(model.score(X_test, y_test)*100)
    return scores

def main():
    # print("vvv")
    # print(os.getcwd())

    df=dataLoad()
    df=tokenising(df)
    df=LabelEncoding(df)
    df=Word2VecTFIDF(df)
    score_list=NaiveBayes(df)
    # print(score_list)

if __name__ == "__main__":
    main()
