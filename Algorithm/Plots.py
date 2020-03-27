import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

overview = pd.read_csv("Overview.csv", header=0,
                       names=["bootstrap_size", "no_features", "max_depth", "no_trees", "accuracy", "seed set"])

n_features = (10,20,30,40,50,100)
n_trees = (10,20,30,40,50,100)
m_depth = (5,10,15,20,25)

# data with all complete combinations
combinations = overview[(overview.no_features.isin(n_features)) & (overview.max_depth.isin(m_depth)) & (overview.no_trees.isin(n_trees))]


# Grouped by bootstrap size and no_features
per_n_features = combinations.groupby(["bootstrap_size", "no_features"]).accuracy.mean().reset_index()
per_n_features_df = pd.DataFrame({"no_features":np.unique(per_n_features.no_features),
       "5400": per_n_features[(per_n_features.bootstrap_size == 5400)].accuracy.array,
       "13500": per_n_features[(per_n_features.bootstrap_size == 13500)].accuracy.array,
       "27000": per_n_features[(per_n_features.bootstrap_size == 27000)].accuracy.array}).set_index("no_features")
print(per_n_features_df.head)
plt.plot(per_n_features_df)
plt.show()

# Grouped by bootstrap size and max_depth
per_m_depth = combinations.groupby(["bootstrap_size", "max_depth"]).accuracy.mean().reset_index()
per_m_depth_df = pd.DataFrame({"max_depth":np.unique(per_m_depth.max_depth),
       "5400": per_m_depth[(per_m_depth.bootstrap_size == 5400)].accuracy.array,
       "13500": per_m_depth[(per_m_depth.bootstrap_size == 13500)].accuracy.array,
       "27000": per_m_depth[(per_m_depth.bootstrap_size == 27000)].accuracy.array}).set_index("max_depth")
print(per_m_depth_df.head)
plt.plot(per_m_depth_df)
plt.show()

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

plt.plot(new_df)
plt.show()
