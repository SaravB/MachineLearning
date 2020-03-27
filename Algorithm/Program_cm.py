import pandas as pd
import random
from datetime import datetime

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

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

print(train_df.shape)
print(test_df.shape)


def accuracy(predicted_labels, actual_labels):
    predictions_correct = predicted_labels == actual_labels

    return predictions_correct.mean()

n_trees = 200
n_features = 100
dt_max_depth = 25
n_bootstrap = 27000
forest = []

results = pd.read_csv("results3_tom.csv", header=0,
                      names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])

start = datetime.now()
print("Training...")
forest = random_forest_algorithm(train_df, n_trees=n_trees, n_bootstrap=n_bootstrap,
                                 n_features=n_features,
                                 dt_max_depth=dt_max_depth, forest=forest)
td = datetime.now() - start
print(f"Training time: {td}")
print("Testing...")
predictions = random_forest_predictions(test_df, forest)
prediction_accuracy = accuracy(predictions, test_df.label)
print(f"Accuracy: {prediction_accuracy}")
results = results.append({"bootstrap_size": n_bootstrap,
                          "no_features": n_features,
                          "max_depth": dt_max_depth,
                          "no_trees": n_trees,
                          "accuracy": prediction_accuracy,
                          "seconds": td.total_seconds()}, ignore_index=True)
results.to_csv("results3_tom.csv")
cm = confusion_matrix(test_df.label, predictions, labels=[0,1,2,3,4,5,6,7,8,9])
print(cm)
df_cm = pd.DataFrame(cm, range(10), range(10))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
plt.show()
