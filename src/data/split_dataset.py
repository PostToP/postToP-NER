import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset() -> None:
    dataset = pd.read_json("dataset/videos.json")
    train_df, val_df = train_test_split(dataset, test_size=0.2)

    train_df.reset_index(drop=True)
    val_df.reset_index(drop=True)

    train_df.to_json("dataset/p2_dataset_train.json", index=False)
    val_df.to_json("dataset/p2_dataset_val.json", index=False)
