import pandas as pd
import random
import pickle
from datetime import datetime

from RandomForest import random_forest_algorithm, random_forest_predictions

# get data (data as csv is in zip on Github, or on https://pjreddie.com/projects/mnist-in-csv/)
train_df = pd.read_csv("mnist_train.csv", header=None)

# set label last
column_names = []
for column in train_df.columns:
    if column != 0:
        name = "pixel" + str(column)
        column_names.append(name)
    else:
        column_names.append("label")

train_df.columns = column_names

cols = train_df.columns.tolist()
cols = cols[1:] + cols[:1]
train_df = train_df[cols]

# get data (data as csv is in zip on Github, or on https://pjreddie.com/projects/mnist-in-csv/)
test_df = pd.read_csv("mnist_test.csv", header=None)

# set label last
column_names = []
for column in test_df.columns:
    if column != 0:
        name = "pixel" + str(column)
        column_names.append(name)
    else:
        column_names.append("label")

test_df.columns = column_names

cols = test_df.columns.tolist()
cols = cols[1:] + cols[:1]
test_df = test_df[cols]


def accuracy(predicted_labels, actual_labels):
    predictions_correct = predicted_labels == actual_labels

    return predictions_correct.mean()


# originalresults = pd.read_csv("testresults.csv", header=0,
#                      names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])
originalresults = pd.DataFrame(
    columns=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])
results = originalresults

#with open('outfile', 'rb') as fp:
#    forest = pickle.load(fp)
forest = []


dt_max_depth = 15
n_features = 30
no_trees = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
n_bootstrap = len(train_df)

for n_trees in no_trees:
    existing_data = originalresults.apply(lambda x: 1 if x["bootstrap_size"] == n_bootstrap and
                                                         x["no_features"] == n_features and
                                                         x["max_depth"] == dt_max_depth and
                                                         x["no_trees"] == n_trees else 0, axis=1)
    if len(existing_data) > 0 and sum(existing_data) > 1:
        continue
    #if len(forest) > no_trees:
    #    continue

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

    with open('outfile', 'wb') as fp:
        pickle.dump(forest, fp)

    results.to_csv("testresult.csv")
