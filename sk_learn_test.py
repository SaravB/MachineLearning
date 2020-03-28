import sklearn
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from datetime import datetime

def train_test_split(df, test_size):
    if isinstance(test_size, float) and test_size < 1:
        test_size = round(test_size * (len(df)))

    random.seed(0)
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    validate_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, validate_df

def get_data():
    df = pd.read_csv("mnist_train.csv")
    train_df, validate_df = train_test_split(df, 0.1)
    return train_df, validate_df

def split_data_label(data):
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x, y

train_df, validate_df = get_data()
train_x, train_y = split_data_label(train_df)
validate_x, validate_y = split_data_label(validate_df)

n_trees = 10
n_features = 10
n_bootstrap = 100
treedepth = 25
results = []

start = datetime.now()
RF = RandomForestClassifier(n_estimators = n_trees, max_depth = treedepth, max_features=n_features, random_state=1, max_samples=n_bootstrap)
RF.fit(train_x, train_y)
print(f"New Random Forest: Bootstrap: {n_bootstrap}, Features: {n_features}, Trees: {n_trees}, Depth: {treedepth}")
score = RF.score(validate_x, validate_y)
print(f"\t {score}")
td = datetime.now()-start
predictions = RF.predict(validate_x)
print(predictions)
print(confusion_matrix(validate_y, predictions, labels=[0,1,2,3,4,5,6,7,8,9]))
disp = plot_confusion_matrix(RF, validate_x, validate_y, labels=[0,1,2,3,4,5,6,7,8,9])
print(disp.confusion_matrix)
plt.show()
