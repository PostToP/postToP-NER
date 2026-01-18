import logging
import pandas as pd
import numpy as np

from config.config import TABLE, TABLE_BACK
from tokenizer.TokenizerCustom import TokenizerCustom


logger = logging.getLogger("experiment")


def convert_ner_tags(titles, ner_dict: list[dict]):
    numeric_tags = []
    custom_tokenizer = TokenizerCustom()
    for i, title in enumerate(titles):
        token = custom_tokenizer.encode(title)
        tags = [0] * len(title)
        for entry in ner_dict[i]:
            start, end, type = entry["start"], entry["end"], entry["type"]
            tags[start:end] = [TABLE[type]] * (end - start)
        new_tags = [0] * len(token)
        for j, token_e in enumerate(token):
            token_start = title.find(token_e)
            token_end = token_start + len(token_e)
            title = (
                title[:token_start]
                + " " * (token_end - token_start)
                + title[token_end:]
            )
            idx = np.round(np.average([i for i in tags[token_start:token_end]]))
            new_tags[j] = TABLE_BACK.get(int(idx), "O")
        numeric_tags.append(new_tags)

    return numeric_tags


def do_stuff(df):
    df["Original Title"] = df["Title"]
    df["Original Description"] = df["Description"]
    title_tokenizer = TokenizerCustom()
    df["Title"] = df["Title"].apply(lambda x: title_tokenizer.encode(x))
    df["Description"] = df["Description"].apply(lambda x: title_tokenizer.encode(x))
    title_ner_tokens = convert_ner_tags(
        df["Original Title"].values,
        df["NER"].apply(lambda x: [e for e in x if e["source"] == "title"]).values,
    )
    description_ner_tokens = convert_ner_tags(
        df["Original Description"].values,
        df["NER"]
        .apply(lambda x: [e for e in x if e["source"] == "description"])
        .values,
    )
    df["Title NER"] = title_ner_tokens
    df["Description NER"] = description_ner_tokens
    return df


def tokenize_dataset():
    logger.info("Tokenizing dataset...")
    train_df = pd.read_json("dataset/p2_dataset_train.json")
    val_df = pd.read_json("dataset/p2_dataset_val.json")

    train_df = do_stuff(train_df)
    val_df = do_stuff(val_df)

    train_df.to_json("dataset/p3_dataset_train.json", index=False)
    val_df.to_json("dataset/p3_dataset_val.json", index=False)
