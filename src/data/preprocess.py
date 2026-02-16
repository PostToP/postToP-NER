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


def preprocess_text(text):
    # text = text.lower()
    # text = normalize_text_to_ascii(text)
    return text


def offset_ner_tags(ner_dict: list[dict], offset: int) -> list[dict]:
    new_ner_dict = []
    for entity in ner_dict:
        if entity["source"] == "description":
            new_entity = {
                "start": entity["start"] + offset,
                "end": entity["end"] + offset,
                "entry": entity["entry"],
                "type": entity["type"],
            }
            new_ner_dict.append(new_entity)
        else:
            new_ner_dict.append(entity)
    return new_ner_dict


def preprocess_dataset():
    train_df = pd.read_json("dataset/p2_dataset_train.json")
    val_df = pd.read_json("dataset/p2_dataset_val.json")
    train_df["Title"] = train_df["Title"].apply(lambda x: preprocess_text(x))
    val_df["Title"] = val_df["Title"].apply(lambda x: preprocess_text(x))

    train_df["Description"] = train_df["Description"].apply(
        lambda x: preprocess_text(x)
    )
    val_df["Description"] = val_df["Description"].apply(lambda x: preprocess_text(x))

    train_df["Text"] = train_df["Title"] + " [SEP] " + train_df["Description"]
    val_df["Text"] = val_df["Title"] + " [SEP] " + val_df["Description"]

    train_df["NER"] = train_df.apply(
        lambda row: offset_ner_tags(row["NER"], len(row["Title"]) + 7), axis=1
    )
    val_df["NER"] = val_df.apply(
        lambda row: offset_ner_tags(row["NER"], len(row["Title"]) + 7), axis=1
    )

    train_df = train_df.drop(columns=["Title", "Description"])
    val_df = val_df.drop(columns=["Title", "Description"])

    train_df.to_json("dataset/p3_dataset_train.json", index=False)
    val_df.to_json("dataset/p3_dataset_val.json", index=False)
