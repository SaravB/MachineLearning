import pandas as pd
import random
from datetime import datetime

from RandomForest import random_forest_algorithm, random_forest_predictions

# get data (data as csv is in zip on Github, or on https://pjreddie.com/projects/mnist-in-csv/)
df = pd.read_csv("mnist_train.csv", header=None)

# set label last
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


# split data in training and testing
def get_train_test_data(data_df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(data_df))

    indices = data_df.index.tolist()
    random.seed(0)
    test_indices = random.sample(population=indices, k=test_size)

    test_set = data_df.loc[test_indices]
    train_set = data_df.drop(test_indices)

    return train_set, test_set


train_df, test_df = get_train_test_data(df, test_size=0.1)


def accuracy(predicted_labels, actual_labels):
    predictions_correct = predicted_labels == actual_labels

    return predictions_correct.mean()

# read results (an empty one is on Github)
# this is so you can restart after something failed and the code knows what you have already calculated
# DONOT upload this to Github with this name, as this will only result in conflicts as everyone will alter the file.
# rename before
results = pd.read_csv("results3.csv", header=0,
                      names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])
originalresults = results

# part below is adjusted per person and I will send you
numberoftrees = range(50, 250, 50)
numberoffeatures = range(50, 250, 50)
dt_max_depth = 25
bootstrapsize = (round(0.1 * len(train_df)), round(0.25 * len(train_df)), round(0.5 * len(train_df)), len(train_df))
# part above is adjusted per person and I will send you

for n_bootstrap in bootstrapsize:
    for n_features in numberoffeatures:
        forest = []
        for n_trees in numberoftrees:
            # this is so you can restart after something failed and the code knows what you have already calculated
            existing_data = originalresults.apply(lambda x: 1 if x["bootstrap_size"] == n_bootstrap and
                                                             x["no_features"] == n_features and
                                                             x["max_depth"] == dt_max_depth and
                                                             x["no_trees"] == n_trees else 0, axis=1)
            if len(existing_data) > 0 and sum(existing_data) > 1:
                continue

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
            results.to_csv("results3.csv")
