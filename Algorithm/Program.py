import random
import pandas as pd
from datetime import datetime
#from mnist import MNIST


from HelperFunctions import get_train_test_data, accuracy
from RandomForest import random_forest_algorithm, random_forest_predictions


column_names = []
forest = []

#data = MNIST('samples')
#images, labels = data.load_training()

#train_df = pd.DataFrame(images)
#train_df["label"] = labels
#print(train_df[1])

#for column in train_df.columns:
#    if column != "label":
#        name = "pixel" + str(column)
#    else:
#        name = column

#    column_names.append(name)
#train_df.columns = column_names

df = pd.read_csv("mnist_train.csv", header=None)


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


#print(datetime.now())
#forest = random_forest_algorithm(train_df, n_trees=25, n_bootstrap=len(train_df)/2, n_features=40, dt_max_depth=25)
#predictions = random_forest_predictions(test_df, forest)
#print(datetime.now())
#accuracy = accuracy(predictions, test_df.label)
#print(accuracy)
#print(datetime.now())

numberoftrees = range(10, 50, 10)
numberoffeatures = range(20, 100, 20)
maxdepth = range(5, 55, 10)
bootstrapsize = (round(0.1*len(train_df)), round(0.25*len(train_df)), round(0.5*len(train_df)), len(train_df))
results = []

for n_bootstrap in bootstrapsize:
    for n_features in numberoffeatures:
        for dt_max_depth in maxdepth:
            for n_trees in numberoftrees:
                for i in range(10):
                    start = datetime.now()
                    forest = random_forest_algorithm(train_df, n_trees=n_trees, n_bootstrap=n_bootstrap, n_features=n_features,
                                                     dt_max_depth=dt_max_depth)
                    predictions = random_forest_predictions(test_df, forest)
                    prediction_accuracy = accuracy(predictions, test_df.label)
                    td = datetime.now()-start
                    results.append([n_bootstrap, n_features, dt_max_depth, n_trees, prediction_accuracy, td.total_seconds()])
                    pd.DataFrame(results).to_csv("results.csv")