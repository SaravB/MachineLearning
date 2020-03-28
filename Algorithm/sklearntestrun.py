import sklearn
import pandas as pd
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


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

numberoftrees = (10, 20, 30, 40, 50, 100)
numberoffeatures = (10, 20, 30, 40, 50, 100)
maxdepth = (5,10,15,20,25)
bootstrapsize = (round(0.1 * len(train_df)),round(0.25 * len(train_df)),round(0.5 * len(train_df)))

results = pd.DataFrame(columns=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])

for max_depth in maxdepth:
    for n_features in numberoffeatures:
        for n_bootstrap in bootstrapsize:
            for n_trees in numberoftrees:
                start = datetime.now()
                RF = RandomForestClassifier(n_estimators=n_trees,max_depth=max_depth, max_features=n_features, random_state=1, max_samples=n_bootstrap)
                RF.fit(train_x, train_y)
                score = RF.score(validate_x, validate_y)
                td = datetime.now()-start
                predictions = RF.predict(validate_x)
                results = results.append({"bootstrap_size": n_bootstrap,
                                          "no_features": n_features,
                                          "max_depth": max_depth,
                                          "no_trees": n_trees,
                                          "accuracy": score,
                                          "seconds": td.total_seconds()}, ignore_index=True)
                results.to_csv("sklearnresults.csv")