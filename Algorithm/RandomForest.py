import numpy as np
import pandas as pd
from DecisionTree import decision_tree_algorithm, decision_tree_predictions


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth, forest):
    start = len(forest)
    for index in range(start, n_trees):
        np.random.seed(index)
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        df_random_subspace = get_random_features(df_bootstrapped, n_features)
        tree = decision_tree_algorithm(df_random_subspace, max_depth=dt_max_depth)
        forest.append(tree)

    return forest


def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    return train_df.iloc[bootstrap_indices]


def get_random_features(data, n_features):
    columns = np.random.choice(data.columns[:-1], n_features, replace=False)
    return data[np.append(columns, data.columns[-1])]


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for index in range(len(forest)):
        column_name = "tree_{}".format(index)
        predictions = decision_tree_predictions(test_df, tree=forest[index])
        df_predictions[column_name] = predictions
    df_predictions = pd.DataFrame(df_predictions)

    return df_predictions.mode(axis=1)[0]


