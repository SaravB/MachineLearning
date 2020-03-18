import sklearn
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier

def train_test_split(df, test_size):
    if isinstance(test_size, float) and test_size < 1:
        test_size = round(test_size * (len(df)))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df

def get_data():
    df = pd.read_csv("train.csv")
    train_df, test_df = train_test_split(df, 0.2)
    return train_df, test_df

def split_data_label(data):
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x, y

train_df, test_df = get_data()
train_x, train_y = split_data_label(train_df)
test_x, test_y = split_data_label(test_df)

RF = RandomForestClassifier(n_estimators = 10, max_depth = 3)
RF.fit(train_x, train_y)
print(RF.score(test_x, test_y))
