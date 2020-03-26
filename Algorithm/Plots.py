import matplotlib as mpl
import pandas as pd
import random
from datetime import datetime

from RandomForest import random_forest_algorithm, random_forest_predictions

# get data (data as csv is in zip on Github, or on https://pjreddie.com/projects/mnist-in-csv/)
overview = pd.read_csv("Overview.csv", header=0,
                       names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])

n_features = (10,20,30,40,50,100)
n_trees = (10,20,30,40,50,100)
m_depth = (5,10,15,20,25)

# data with all useful combinations
combinations = overview[(overview.no_features.isin(n_features)) & (overview.max_depth.isin(m_depth)) & (overview.no_trees.isin(n_trees))]


# Grouped by bootstrap size and no_features
per_n_features = combinations.groupby(["bootstrap_size", "no_features"]).accuracy.mean()
print(per_n_features.head)

# Grouped by bootstrap size and max_depth
per_m_depth = combinations.groupby(["bootstrap_size", "max_depth"]).accuracy.mean()
print(per_m_depth.head)

# Grouped by bootstrap size and no_trees
# fixed no_features = 100
# fixed max_depth = 25
bs_trees = overview[(overview.no_features == 100) & (overview.max_depth == 25)].groupby(["bootstrap_size", "no_trees"]).accuracy.mean()

print(bs_trees.head)
