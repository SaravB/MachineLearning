import numpy as np
import pandas as pd
from Decision_Tree import decision_tree_algorithm, decision_tree_predictions, train_test_split
from pprint import pprint

df = pd.read_csv("winequality-red.csv")
df = df.rename(columns={"quality": "label"})

column_names = []

for column in df.columns:
    name = column.replace(" ", "_")
    column_names.append(name)
df.columns = column_names

train_df, test_df = train_test_split(df, 0.2)

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]

    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for index in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth= dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    return forest

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for index in range(len(forest)):
        column_name = "tree_{}".format(index)
        predictions = decision_tree_predictions(test_df, tree=forest[index])
        df_predictions[column_name] = predictions
    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions


def calculate_accuracy(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()

    return accuracy

forest = random_forest_algorithm(train_df, n_trees=4, n_bootstrap=800, n_features=6, dt_max_depth=5)
predictions = random_forest_predictions(test_df, forest)
accuracy = calculate_accuracy(predictions, test_df.label)
print("Accuracy = {}".format(accuracy))


