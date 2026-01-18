import logging

import pandas as pd
import re
from unicodedata import combining, normalize
from ftfy import fix_text
from unidecode import unidecode


logger = logging.getLogger("experiment")


def normalize_text_to_ascii(text: str) -> str:
    text = fix_text(text)
    normalized_text = normalize("NFKD", text)
    ascii_text = unidecode("".join([c for c in normalized_text if not combining(c)]))
    return ascii_text


def preprocess_tokens(tokens):
    new_tokens = []
    for token in tokens:
        token = token.lower()
        token = normalize_text_to_ascii(token)
        new_tokens.append(token)
    return new_tokens


def preprocess_dataset():
    train_df = pd.read_json("dataset/p3_dataset_train.json")
    val_df = pd.read_json("dataset/p3_dataset_val.json")
    train_df["Title"] = train_df["Title"].apply(lambda x: preprocess_tokens(x))
    val_df["Title"] = val_df["Title"].apply(lambda x: preprocess_tokens(x))

    train_df["Description"] = train_df["Description"].apply(
        lambda x: preprocess_tokens(x)
    )
    val_df["Description"] = val_df["Description"].apply(lambda x: preprocess_tokens(x))

    train_df.to_json("dataset/p4_dataset_train.json", index=False)
    val_df.to_json("dataset/p4_dataset_val.json", index=False)
