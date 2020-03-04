import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix

class Tree: # Decision Tree
    def init(self, df_train, n_features, n_height, counter = 0):
        min_samples = 2
        column_names = df_train.columns
        data = df_train.values

        if self.purity(data) or (len(data) < min_samples) or (counter == n_height):
            # if all data in leaf same label or less than 2 instances per leaf or tree has reached max_height
            # stopping criteria have been met
            classification = classify_data() #???
        else:
            counter += 1
            possible_splits = self.possible_splits(data, n_features)
            best_split_column, best_split_value = self.best_split(data, possible_splits)
            split_above, split_below = self.split(data, best_split_column, best_split_value)
            if len(split_above) == 0 or len(split_below) == 0:
                # if either of the leaves is empty, then split is 0 and is useless
                classification = classify_data() #???
                return classification
            question = f"{feature_name} <= {split_value}"
            sub_tree = {question: []}
            right = Tree(split_above, n_features, n_height, counter) # RECURSIVE
            left = Tree(split_below, n_features, n_height, counter) # RECURSIVE
            if right == left:
                # if above split is below split then no point in asking the question
                sub_tree = right
            else:
                sub_tree[question].append(right)
                sub_tree[question].append(left)
            return sub_tree

    def purity(self, data):
        # check whether all labels in the subset of data are equal
        unique_labels = np.unique(data["label"])
        if len(unique_labels) == 1:
            return True
        else:
            return False

    def classify_data(self):
        # ??? TODO
        pass

    def possible_splits(self, data, n_features):
        # find all possible splits of data
        possible_splits = {}
        n_columns = data.shape[1] - 1 # -1 for label column
        n_columns_indices = list(range(n_columns))
        if n_features and n_features <= n_columns:
            n_column_indices =  random.sample(population=column_indices, k=random_subspace)
        for i in n_column_indices:
            column_values = df[,i]
            possible_splits[i] = np.unique(column_values)
        return possible_splits

    def best_split(self, data, possible_splits):
        # find best split by calculating entropy for all splits
        optimal_entropy = float('inf')
        for split in possible_splits:
            for value in possible_splits[i]:
                split_above, split_below = self.split(data)
                entropy_current = calculate_split_entropy(split_above, split_below)
                if entropy_current <= entropy_optimal:
                    entropy_optimal = entropy_current
                    best_split_column = split
                    best_split_value = value
        return best_split_column, best_split_value

    def split(self, data, best_split_column, best_split_value):
        # split data in above and below split
        best_split_column_values = data[, best_split_column]
        split_above = data[best_split_column_values > best_split_value]
        split_below = data[best_split_column_values <= best_split_value]
        return split_above, split_below

    def calculate_split_entropy(self, split_above, split_below):
        # calculate entropy for given two splits
        length_above = len(split_above)
        length_below = len(split_below)
        proportion_split_above = length_above / (length_above + length_below)
        proportion_split_below = length_below / (length_above + length_below)
        entropy_split_above = calculate_entropy(split_above)
        entropy_split_below = calculate_entropy(split_below)
        entropy = proportion_split_above * entropy_split_above + proportion_split_below * entropy_split_below
        return entropy

    def calculate_entropy(self, split):
        # calculate entropy for single given split
        unique_labels, count_labels = np.unique(split["label"])
        probability_labels = count_labels / count_labels.sum()
        entropy = sum(probability_labels * -np.log2(probability_labels))
        return entropy

    def predict_data(self, data, tree):
        predictions = data.apply(predict_instance, args=(tree,), axis=1)
        return predictions

    def predict_instance(self, instance, tree):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")
        if comparison_operator == "<=":
            if instance[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return predict_instance(instance, residual_tree) # RECURSIVE

class Forest: # Random Forest
    def init(self, df_train, df_labels, n_trees, n_bootstrap, n_features, n_tree_max_depth):
        # Initialize random forest
        self.df_train = df_train
        self.n_trees = n_trees
        self.n_bootstrap = n_bootstrap
        self.n_features = n_features
        self.n_tree_max_depth = n_tree_max_depth

        self.forest = []
        for i in range(0, self.n_trees):
            df_bootstrapped = df_train.sample(n=self.n_bootstrap, replace=True, random_state=12345)
            new_tree = Tree(df_bootstrapped, n_features, n_tree_max_depth)
            self.forest.append(new_tree)

    def predict(self, df):
        # Use random forest to make predictions for training data set, validation data set and testing data set
        df_predicted = {}
        for i in range(len(self.forest)):
            tree = f"tree{i}"
            predictions = self.forest[i].predict_data()
            df_predicted[tree] = predictions
        df_predicted = pd.DataFrame(df_predicted)
        return df_predicted

def evaluation(df, df_predicted):
    metrics = {}
    predictions_correct = df_predicted == df["label"]
    metrics["accuracy"] = predictions_correct / len(df)
    predictions_incorrect = df_prediced != df["label"]
    metrics["error_rate"] = predictions_incorrect / len(df)
    return metrics

def data_preparation():
    df = pd.read_csv("train.csv")
    df["index_col"] = df.index
    # training 50%, validation 25%, testing 25%
    df_train = df.sample(frac=0.5, replace=False, random_state=12345)
    df.drop(df[df["index_col"].isin(df_train["index_col"]) == True].index, inplace = True)
    df_validate = df.sample(frac=0.5, replace=False, random_state=12345)
    df.drop(df[df["index_col"].isin(df_validate["index_col"]) == True].index, inplace = True)
    df_test = df
    return df_train, df_validate, df_test


# DATA PREPARATION
print("Starting data preparation")
df_train, df_validate, df_test = data_preparation()

# INITIALIZE AND TRAIN NEW RANDOM FOREST
print("Initializing and training random forest")
n_trees = 10
n_bootstrap = 800
n_features = 2
n_tree_max_depth = 4
forest = Forest(df_train=df_train, n_trees=n_trees, n_bootstrap=n_bootstrap, \
    n_features=n_features, n_tree_max_depth=n_tree_max_depth)

# EVALUATE RANDOM FOREST ON TRAINING DATA - TEST
print("Evaluating on training data set")
predicted_train = forest.predict(df_train)
evaluation_metrics_train = evaluation(df_train, predicted_train)
print(f"Accuracy on training set: {evaluation_metrics_train["accuracy"]}")
print(f"Error Rate on training set: {evaluation_metrics_train["error_rate"]}")

# END OF TESTING
sys.exit("Test of training of random forest complete. Exiting script to reduce testing time.")

# VALIDATE RANDOM FOREST
print("Validating random forest")
predicted_validate = forest.predict(df_validate)
evaluation_metrics_validate = evaluation(df_validate, predicted_train)
print(f"Accuracy on validation set: {evaluation_metrics_validate["accuracy"]}")
print(f"Error Rate on validation set: {evaluation_metrics_validate["error_rate"]}")

# TRAIN RANDOM FOREST
print("Training random forest on training and validation data sets")
df_train_validate = pd.concat([df_train, df_validate], ignore_index=True, sort=True)
df_train_validate["index_col"] = df_train.index
forest2 = Forest(df_train_validate, n_trees=n_trees, n_bootstrap=n_bootstrap, \
   n_features=n_features, n_tree_max_depth=n_tree_max_depth)
predicted_train_validate = forest.predict(df_train_validate, forest2)
evaluation_metrics_train_validate = evaluation(df_train_validate, predicted_train_validate)
print(f"Accuracy on training and validation set: {evaluation_metrics_train_validate["accuracy"]}")
print(f"Error Rate on training and validation set: {evaluation_metrics_train_validate["error_rate"]}")

# TEST RANDOM FOREST
print("Testing random forest")
predicted_test = forest.predict(df_test, trained_forest2)

# EVALUATE RANDOM FOREST
print("Evaluating random forest")
evaluation_metrics_test = evaluation(df_test, predicted_test)
print(f"Accuracy on testing set: {evaluation_metrics_test["accuracy"]}")
print(f"Error Rate on testing set: {evaluation_metrics_test["error_rate"]}")
# COMPUTE CONFUSION MATRIX USING SKLEARN PACKAGE - ALLOWED???
confusion_matrix(df_test, predicted_test, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
