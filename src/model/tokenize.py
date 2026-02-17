import logging
import pandas as pd
import numpy as np

from config.config import MAX_SEQUENCE_LENGTH, TABLE, TABLE_BACK, TRANSFORMER_MODEL_NAME
from tokenizer.TransformerTokenizer import TransformerTokenizer
from tokenizer.TokenizerCustom import TokenizerCustom


logger = logging.getLogger("experiment")


def convert_ner_tags(encoding, ner_dict: list[dict]):
    word_ids = encoding.word_ids(batch_index=0)

    aligned_labels = []
    previous_word_id = None

    for i in range(len(word_ids)):
        word_id = word_ids[i]
        span = encoding.token_to_chars(i)
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            label_str = "O"

            for ner in ner_dict:
                ner_start = ner["start"]
                ner_end = ner["end"]

                if span.start >= ner_start and span.end <= ner_end:
                    label_str = ner["type"]
                    break
            aligned_labels.append(TABLE.get(label_str, 0))
        else:
            aligned_labels.append(-100)

        previous_word_id = word_id
    return aligned_labels


def do_stuff(df):
    text_tokenizer = TransformerTokenizer(TRANSFORMER_MODEL_NAME, MAX_SEQUENCE_LENGTH)
    encodings = df["Text"].apply(lambda x: text_tokenizer.encode(x))
    df["Word IDs"] = encodings.apply(lambda x: x["input_ids"].squeeze(0).tolist())
    df["Tokens"] = encodings.apply(
        lambda x: text_tokenizer.tokenizer.convert_ids_to_tokens(
            x["input_ids"].squeeze(0).tolist()
        )
    )
    df["Attention Mask"] = encodings.apply(
        lambda x: x["attention_mask"].squeeze(0).tolist()
    )
    df["real_word_ids"] = encodings.apply(lambda x: x.word_ids(batch_index=0))
    df["NER"] = df.apply(
        lambda row: convert_ner_tags(encodings[row.name], row["NER"]), axis=1
    )

    return df


def tokenize_dataset():
    logger.info("Tokenizing dataset...")
    train_df = pd.read_json("dataset/p3_dataset_train.json")
    val_df = pd.read_json("dataset/p3_dataset_val.json")

    train_df = do_stuff(train_df)
    print(train_df.head())
    val_df = do_stuff(val_df)

    train_df.to_json("dataset/p4_dataset_train.json", index=False)
    val_df.to_json("dataset/p4_dataset_val.json", index=False)
