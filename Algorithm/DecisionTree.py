import numpy as np


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    if (has_unique_label(data)) or (len(data) < min_samples) or (counter == max_depth):
        return get_most_common_label(data)
    else:
        counter += 1
        potential_splits = get_split_values(data)
        if len(potential_splits) == 0:
            return get_most_common_label(data)

        split_column, split_value = determine_best_split(data, potential_splits)
        left_data, right_data = split_data(data, split_column, split_value)

        feature_name = COLUMN_HEADERS[split_column]
        split_argument = "{}<={}".format(feature_name, split_value)

        sub_tree = {split_argument: []}

        left_subtree = decision_tree_algorithm(left_data, counter, max_depth=max_depth)
        right_subtree = decision_tree_algorithm(right_data, counter, max_depth=max_depth)
        if left_subtree == right_subtree:
            sub_tree = left_subtree
        else:
            sub_tree[split_argument].append(left_subtree)
            sub_tree[split_argument].append(right_subtree)

        return sub_tree


def has_unique_label(data):
    return len(np.unique(data[:, -1])) == 1


def get_most_common_label(data):
    labels, label_counts = np.unique(data[:, -1], return_counts=True)
    return labels[label_counts.argmax()]


def get_split_values(data):
    split_values = {}
    for column_index in range(data.shape[1] - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        if len(unique_values) > 1:
            split_values[column_index] = unique_values[:-1]  # splitting on last value is not a split

    return split_values


def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            left_data, right_data = split_data(data, column_index, value)
            current_combined_entropy = combined_entropy(left_data, right_data)
            if current_combined_entropy <= overall_entropy:
                overall_entropy = current_combined_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def combined_entropy(left_data, right_data):
    n_data_points = len(right_data) + len(left_data)
    p_left_data = len(left_data) / n_data_points
    p_right_data = 1 - p_left_data

    return p_left_data * entropy(left_data) + p_right_data * entropy(right_data)


def entropy(data):
    _, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / counts.sum()
    return sum(probabilities * -np.log(probabilities))


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    return data[split_column_values <= split_value], data[split_column_values > split_value]




def classify_example(example, tree):
    split_argument = list(tree.keys())[0]
    feature, value = split_argument.split("<=")

    split_result = tree[split_argument][0] if example[feature] <= float(value) else tree[split_argument][1]

    return split_result if not isinstance(split_result, dict) else classify_example(example, split_result)


def decision_tree_predictions(test_df, tree):
    return test_df.apply(classify_example, args=(tree,), axis=1)
