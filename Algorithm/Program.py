import pandas as pd
from datetime import datetime

from HelperFunctions import get_train_test_data, accuracy
from RandomForest import random_forest_algorithm, random_forest_predictions

df = pd.read_csv("mnist_train.csv", header=None)

column_names = []
for column in df.columns:
    if column != 0:
        name = "pixel" + str(column)
        column_names.append(name)
    else:
        column_names.append("label")

df.columns = column_names

cols = df.columns.tolist()
cols = cols[1:] + cols[:1]
df = df[cols]

train_df, test_df = get_train_test_data(df, test_size=0.1)

results = pd.read_csv("results2.csv", header=0, names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])
originalresults = results

numberoftrees = range(50, 250, 50)
numberoffeatures = range(50, 250, 50)
dt_max_depth = 25
bootstrapsize = (round(0.1 * len(train_df)), round(0.25 * len(train_df)), round(0.5 * len(train_df)), len(train_df))

for n_bootstrap in bootstrapsize:
    for n_features in numberoffeatures:
        forest = []
        for n_trees in numberoftrees:
            seriesObj = originalresults.apply(lambda x: 1 if x["bootstrap_size"] == n_bootstrap and
                                                             x["no_features"] == n_features and
                                                             x["max_depth"] == dt_max_depth and
                                                             x["no_trees"] == n_trees else 0, axis=1)
            if any(seriesObj):
                continue

            if n_trees <= len(forest):
                forest = []

            start = datetime.now()

            forest = random_forest_algorithm(train_df, n_trees=n_trees, n_bootstrap=n_bootstrap,
                                             n_features=n_features,
                                             dt_max_depth=dt_max_depth, forest=forest)
            td = datetime.now() - start
            predictions = random_forest_predictions(test_df, forest)
            prediction_accuracy = accuracy(predictions, test_df.label)
            results = results.append({"bootstrap_size": n_bootstrap,
                                      "no_features": n_features,
                                      "max_depth": dt_max_depth,
                                      "no_trees": n_trees,
                                      "accuracy": prediction_accuracy,
                                      "seconds": td.total_seconds()}, ignore_index=True)
            results.to_csv("results2.csv")
