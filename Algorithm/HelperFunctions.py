import random


def get_train_test_data(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    random.seed(0)
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def accuracy(predictions, labels):
    predictions_correct = predictions == labels

    return predictions_correct.mean()