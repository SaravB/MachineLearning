import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


overview = pd.read_csv("Overview.csv", header=0,
                       names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seconds"])

n_features = (10,20,30,40,50,100)
n_trees = (10,20,30,40,50,100)
m_depth = (5,10,15,20,25)

# data with all useful combinations
combinations = overview[(overview.no_features.isin(n_features)) & (overview.max_depth.isin(m_depth)) & (overview.no_trees.isin(n_trees))]


# Grouped by bootstrap size and no_features
per_n_features = combinations.groupby(["bootstrap_size", "no_features"]).accuracy.mean().reset_index()
print(per_n_features.head)

# Grouped by bootstrap size and max_depth
per_m_depth = combinations.groupby(["bootstrap_size", "max_depth"]).accuracy.mean().reset_index()
print(per_m_depth.head)

# Grouped by bootstrap size and no_trees
# fixed no_features = 100
# fixed max_depth = 25
bs_trees = overview[(overview.no_features == 100) & (overview.max_depth == 25)].groupby(["bootstrap_size", "no_trees"]).accuracy.mean().reset_index()
print(bs_trees.head)


new = {"no_trees":np.unique(bs_trees.no_trees),
       "5400":bs_trees[(bs_trees.bootstrap_size == 5400)].accuracy.array,
       "13500": bs_trees[(bs_trees.bootstrap_size == 13500)].accuracy.array,
       "27000": bs_trees[(bs_trees.bootstrap_size == 27000)].accuracy.array}
new_df = pd.DataFrame(new).set_index("no_trees")
print(new_df.head)
