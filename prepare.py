import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('C:/Users/Radriges/Desktop/Date/responce_train.csv')
df = df.drop(["client_id"], axis=1)

def df_nulls():
    print(df.isnull().sum())

def text_to_int(data):
    le = preprocessing.LabelEncoder()
    columns = ["education", "marital-status", "industry", "title", "tp-state", "tp-foreign", "job-type", "family-income", "registration-region", "fact-region", "postal-region", "last-loan-region", "region"]
    set_del_col = set(columns)
    for col in columns:
        data[col] = le.fit_transform(data[col])
        print(le.classes_)
    return set_del_col

def clean(data):
    cols = list(set(data.keys())-set_del_col)
    print(cols)
    # можно оставить пустые ячейки
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    return data


set_del_col = text_to_int(df)
clean(df)
