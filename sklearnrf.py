import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

def evaluation(df, df_predicted):
    # CHECK IF WORKS CORRECTLY
    predictions_correct = 0
    predictions_incorrect = 0
    for i in range(0, len(df)):
        if df[i] == int(df_predicted[i]):
            predictions_correct += 1
        else:
            predictions_incorrect += 1

    accuracy = predictions_correct / len(df)
    error_rate = predictions_incorrect / len(df)
    print(f"\tAccuracy: {accuracy}")
    print(f"\tError Rate: {error_rate}")

def data_preparation():
    df = pd.read_csv("train.csv")
    df["index_col"] = df.index
    # training 50%, validation 25%, testing 25%
    df_train = df.sample(frac=0.5, replace=False, random_state=12345)
    df.drop(df[df["index_col"].isin(df_train["index_col"]) == True].index, inplace = True)
    df_validate = df.sample(frac=0.5, replace=False, random_state=12345)
    df.drop(df[df["index_col"].isin(df_validate["index_col"]) == True].index, inplace = True)
    df_test = df

    df_train_labels = df_train["label"]
    df_train.drop(["label"], axis=1)
    df_validate_labels = df_validate["label"]
    df_validate.drop(["label"], axis=1)
    df_test_labels = df_test["label"]
    df_test.drop(["label"], axis=1)
    return df_train, df_train_labels, df_validate, df_validate_labels, df_test, df_test_labels


# DATA PREPARATION
print("Starting data preparation")
df_train, df_train_labels, df_validate, df_validate_labels, df_test, df_test_labels = data_preparation()

# INITIALIZE AND TRAIN NEW RANDOM FOREST
print("Initializing random forest")
n_trees = 10
rf = RandomForestRegressor(n_estimators = n_trees, random_state = 42)

print("Training random forest")
rf.fit(df_train, df_train_labels)

print("Predicting validation set")
predictions = rf.predict(df_validate)

print("Evaluating validation set")
validate_labels = df_validate_labels.to_numpy()
evaluation(validate_labels, predictions)

print("Predicting test set")
predictions = rf.predict(df_test)

print("Evaluating test set")
test_labels = df_test_labels.to_numpy()
evaluation(test_labels, predictions)
