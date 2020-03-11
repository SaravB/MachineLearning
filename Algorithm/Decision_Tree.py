import numpy as np
import random


def is_pure(data):
    return len(np.unique(data[:, -1])) == 1


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification


def get_potential_splits(data, random_subspace):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))

    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)

    for column_index in column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)   #hier nog code toevoegen als er teveel unieke waarden zijn.
        potential_splits[column_index]=unique_values

    return potential_splits


def split_data(data, split_column, split_value):
    type_of_feature = FEATURE_TYPES[split_column]
    split_column_values = data[:, split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities *- np.log2(probabilities))
    return entropy


def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_above) + len(data_below)
    p_data_below = len(data_below)/n_data_points
    p_data_above = 1 - p_data_below

    overall_entropy = (p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))
    return overall_entropy


def determine_best_splits(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, column_index, value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        FEATURE_TYPES = determine_type_of_feature(df)
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    if (is_pure(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_splits(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        if len(data_above) == 0 or len(data_below) == 0:
            classification = classify_data(data)
            return classification

        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        else:
            question = "{} = {}".format(feature_name, split_value)

        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(data_below, counter, max_depth=max_depth, random_subspace=random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, max_depth=max_depth, random_subspace=random_subspace)
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def determine_type_of_feature(df):
    feature_types =[]
    n_unique_values_threshold = 15
    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if isinstance(example_value, str) or (len(unique_values) < n_unique_values_threshold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")

    return feature_types
    

def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(classify_example, args=(tree,), axis=1)
    return predictions

def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df.classification == df.label

    accuracy = df.classification_correct.mean()

    return accuracy





