import random
import pandas as pd
import gzip
#from mnist import MNIST


from Helper_Functions import train_test_split, calculate_accuracy
from Random_Forest import random_forest_algorithm, random_forest_predictions


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

df = pd.read_csv("train.csv")
cols = df.columns.tolist()
cols = cols[2:] + cols[:1]
df = df[cols]

for column in df.columns:
    name = column.replace(" ", "")
    column_names.append(name)

df.columns = column_names


train_df, test_df = train_test_split(df, test_size=0.9)


forest = random_forest_algorithm(train_df, n_trees=1, n_bootstrap=len(train_df), n_features=15, dt_max_depth=4,
                                 dt_min_leaf_size=100)
predictions = random_forest_predictions(test_df, forest)
accuracy = calculate_accuracy(predictions, test_df.label)

#leuke tekst