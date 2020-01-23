import os
import warnings
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
warnings.filterwarnings('ignore')

def clustering(df):
    cols = ['Size', 'DataAge', 'ARecency', 'MRecency', \
            'Depth', 'Unwritten', 'Unaccessed']
    X_train = df[cols]
    clf = KNN().fit(X_train)
    df["Scores"] = clf.predict_proba(X_train)[:,1]
    return df


def main():
    data = pd.read_csv(r"D:\Projects\6-Ardent-Security\Data\SysScan.csv")
    data['DateModified'] = data['DateModified'].astype('datetime64[ns]')
    data['DateAccessed'] = data['DateAccessed'].astype('datetime64[ns]')
    data['DateCreated'] = data['DateCreated'].astype('datetime64[ns]')
    data.drop(data[data["Name"].isna()].index, inplace=True)
    data.drop(columns=["HashMD5"], axis=1, inplace=True)
    data["Extension"][data["Extension"].isna()] = "EXE"

    data["DataAge"] = (pd.Timestamp.now()-data["DateCreated"]).dt.days
    data["ARecency"] = (pd.Timestamp.now()-data["DateAccessed"]).dt.days
    data["MRecency"] = (pd.Timestamp.now()-data["DateModified"]).dt.days

    top_counts = (data["Extension"].value_counts()>200).head(54).index
    data = data.loc[data['Extension'].isin(top_counts)]
    data["Depth"] = data["Path"].str.strip("/").str.count("/")+1
    data["Unwritten"] = (data["DateCreated"]>data["DateModified"]).astype('int')
    data["Unaccessed"] = (data["DateCreated"]>data["DateAccessed"]).astype('int')
    data["Root"] = [item[0] for item in data["Path"].str.strip("/").str.split("/").to_list()]
    data["Id"] = range(data.shape[0])

    data = clustering(data)
    print(data.isna().sum())
    # data.to_csv(r"D:\Projects\6-Ardent-Security\Data\ArdentScan.csv", index=False)
    return

if __name__ == '__main__':
    main()
