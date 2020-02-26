import numpy as np
import pandas as pd

class tree(): # Decision Tree
    def init(self, df_train):
        pass
    def split(self):
        pass

class forest(): # Random Forest
    def init(self):
        pass
    def bootstrap(self, df_train):
        df_bootstrapped = df_train.sample(n=len(df_train), replace=True, random_state=12345)
        return df_bootstrapped
    def train(self, df_train, n_trees):
        forest = []
        for i in range(0, n_trees):
            df_bootstrapped = self.bootstrap(df_train)
            new_tree = tree()
            forest.append(new_tree)
        return forest
    def predict(self):
        pass
    def test(self):
        pass

def evaluation(df, df_predicted):
    metrics = []
    #metrics.append(accuracy, precision, recall, tp, tn, fp, fn, tpr, fpr)
    return metrics

def data_preparation():
    default_df_train = pd.read_csv("train.csv")
    #default_df_test = pd.read_csv("test.csv")
    #df = pd.concat([default_df_train, default_df_test], ignore_index=True, sort=True)
    df = default_df_train
    df["index_col"] = df.index
    # training 50%, validation 25%, testing 25%
    df_train = df.sample(frac=0.5, replace=False, random_state=12345)
    df.drop(df[df['index_col'].isin(df_train['index_col']) == True].index, inplace = True)
    df_validate = df.sample(frac=0.5, replace=False, random_state=12345)
    df.drop(df[df['index_col'].isin(df_validate['index_col']) == True].index, inplace = True)
    df_test = df
    return df_train, df_validate, df_test

print("Starting data preparation")
# DATA PREPARATION
df_train, df_validate, df_test = data_preparation()
print(f"\tTraining instances: {len(df_train)}")
print(f"\tValidation instances: {len(df_validate)}")
print(f"\tTesting instances: {len(df_test)}")

print("Initializing random forest")
# INITIALIZE NEW RANDOM FOREST
n_trees = 10
forest = forest()
forest.train(df_train, n_trees)
print(forest)

# print("Training random forest")
# TRAIN RANDOM FOREST
# train random forest on training set

# print("Validating random forest")
# VALIDATE RANDOM FOREST
# test random forest on validation set

# print("Training random forest")
# TRAIN RANDOM FOREST
# train random forest on training and validation sets

# print("Testing random forest")
# TEST RANDOM FOREST
# test random forest

# print("Evaluating random forest")
# EVALUATE RANDOM FOREST
# metrics = evaluation()
# compute ROC curve, precision-recall curve and confusion matrix
