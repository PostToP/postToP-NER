from model.model import build_model, evaluate_model
import numpy as np
import tensorflow as tf


def load_split(path):
    data = np.load(path, allow_pickle=True)
    X = (
        data["titles"],
        data["token_features"],
        data["global_features"],
    )
    y = data["ner_tags"]
    return X, y


def main():
    train_df = load_split("dataset/p4_dataset_train.npz")
    # print statistics about train_df

    val_df = load_split("dataset/p4_dataset_val.npz")
    print("Train dataset:")
    print(f"Number of samples: {len(val_df[0][2])}")

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_df[0], train_df[1]))
        .shuffle(1000)
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_df[0], val_df[1]))
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(train_dataset, val_dataset)

    results = evaluate_model(model, val_df[0][0], val_df[0][1], val_df[0][2], val_df[1])
    print(results)


if __name__ == "__main__":
    main()
