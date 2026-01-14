import numpy as np
import pandas as pd

import re
from config.config import TABLE, TABLE_BACK
from tokenizer.TokenizerCustom import TokenizerCustom
from vectorizer.VectorizerKerasTokenizer import VectorizerKerasTokenizer
from vectorizer.VectorizerNER import VectorizerNER
from unicodedata import combining, normalize
from ftfy import fix_text
from unidecode import unidecode
from tensormonad import TensorMonad
from vectorizer.VectorizerLanguage import VectorizerLanguage


MAX_SEQUENCE_LENGTH = 300
VOCAB_SIZE = 5000


class FeatureExtraction:
    @staticmethod
    def tokens_containing_channel_name(token, channel_name):
        feature = np.zeros(len(token), dtype=int)
        channel_name = channel_name.lower()
        token = [t.lower() for t in token]
        for i, t in enumerate(token):
            if t in channel_name:
                if re.search(
                    r"([一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤]+|[a-zA-Z0-9]+)[.!]*", t
                ):
                    feature[i] = 1
        return feature[:, np.newaxis]

    @staticmethod
    def batch(function, *args):
        features = [function(*arg) for arg in zip(*args)]
        return np.stack(features, axis=0)

    @staticmethod
    def count_token_occurrences(token, description):
        feature = np.zeros(len(token), dtype=int)
        token = [t.lower() for t in token]
        description = description.lower()
        for i, t in enumerate(token):
            occurrences = re.findall(re.escape(t), description)
            if occurrences:
                feature[i] = len(occurrences)
        return feature[:, np.newaxis]

    @staticmethod
    def length_of_tokens(token):
        return np.array([len(t) for t in token], dtype=int)[:, np.newaxis]

    @staticmethod
    def is_token_verbal(token):
        feature = np.zeros(len(token), dtype=int)
        token = [t.lower() for t in token]
        for i, t in enumerate(token):
            if re.search(r"([一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ々〆〤]+|[a-zA-Z]+)", t):
                feature[i] = 1
        return feature[:, np.newaxis]


def extract_features(
    dataset: pd.DataFrame, max_sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    per_token_features = []

    feature_channel = (
        TensorMonad((dataset["Original Tokens"].values, dataset["Channel Name"].values))
        .map(FeatureExtraction.tokens_containing_channel_name)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_channel)

    feature_token_freq_title = (
        TensorMonad((dataset["Original Tokens"].values, dataset["Title"].values))
        .map(FeatureExtraction.count_token_occurrences)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_token_freq_title)

    feature_token_freq_desc = (
        TensorMonad((dataset["Original Tokens"].values, dataset["Description"].values))
        .map(FeatureExtraction.count_token_occurrences)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_token_freq_desc)

    feature_token_length = (
        TensorMonad([dataset["Original Tokens"].values])
        .map(FeatureExtraction.length_of_tokens)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_token_length)

    feature_is_token_verbal = (
        TensorMonad([dataset["Original Tokens"].values])
        .map(FeatureExtraction.is_token_verbal)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_is_token_verbal)

    global_features = []
    feature_language = (
        TensorMonad([dataset["Language"].values]).map(VectorizerLanguage).to_tensor()
    )
    global_features.append(feature_language)

    per_token_features = np.concatenate(per_token_features, axis=2)
    global_features = np.concatenate(global_features, axis=1)

    return per_token_features, global_features


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


def mask_artists_and_titles(
    tokens: list[list[str]], ner_tags: list[list[str]]
) -> list[list[str]]:
    """
    Mask tokens thats part of any 'Inside' tag.
    If a token was ever tagged with an 'Outside' tag, it will not be masked.

    Args:
        tokens (list[list[str]]): List of tokens
        ner_tags (list[list[str]]): List of NER tags

    Returns:
        list[list[str]]: List of tokens with masked tokens
    """
    not_to_mask = set()
    for t, tags in zip(tokens, ner_tags):
        for i, tag in enumerate(tags):
            if tag == "O" or tag == "MODIFIER" or tag == "VOCALOID":
                not_to_mask.add(t[i])

    new_tokens = []
    for token in tokens:
        new_tokens.append([i if i in not_to_mask else "<oov>" for i in token])
    return new_tokens


def concat_dataset(dataset):
    dataset["Text"] = dataset["Title"] + " " + dataset["Description"]
    for idx, row in dataset.iterrows():
        title_length = len(row["Title"]) + 1  # +1 for the space
        updated_ner = []
        for entity in row["NER"]:
            if entity["source"] == "title":
                updated_ner.append(entity)
            elif entity["source"] == "description":
                updated_entity = entity.copy()
                updated_entity["start"] += title_length
                updated_entity["end"] += title_length
                updated_ner.append(updated_entity)
            updated_ner[-1].pop("source", None)
        dataset.at[idx, "NER"] = updated_ner

    return dataset


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


def do_stuff(df, ner_vectorizer, title_vectorizer, train=False):
    df = concat_dataset(df)
    ner_tags = convert_ner_tags(df["Text"].values, df["NER"].values)

    df["NER"] = ner_tags

    title_tokenizer = TokenizerCustom()
    df["Original Tokens"] = df["Text"].apply(lambda x: title_tokenizer.encode(x))

    df["Tokens"] = df["Original Tokens"].apply(lambda x: preprocess_tokens(x))
    df["Tokens"] = mask_artists_and_titles(df["Tokens"].values, df["NER"].values)
    if train:
        ner_vectorizer.train(df["NER"].values)
    df["NER"] = df["NER"].apply(lambda x: ner_vectorizer.encode(x))

    per_token_features, global_features = extract_features(df, MAX_SEQUENCE_LENGTH)
    df["Features"] = per_token_features.tolist()
    df["Global Features"] = global_features.tolist()

    if train:
        title_vectorizer.train(df["Tokens"].values)
    df["title_vec"] = (
        df["Tokens"].apply(lambda x: title_vectorizer.encode(x)).to_numpy()
    )

    return df


def save_split(path, X, y):
    np.savez_compressed(
        path,
        titles=X["title_vec"].to_list(),
        ner_tags=y,
        token_features=X["Features"].to_list(),
        global_features=X["Global Features"].to_list(),
    )


def feature_extraction():
    train_df = pd.read_json("dataset/p3_dataset_train.json")
    val_df = pd.read_json("dataset/p3_dataset_val.json")
    ner_vectorizer = VectorizerNER(MAX_SEQUENCE_LENGTH)
    title_vectorizer = VectorizerKerasTokenizer(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)

    train_df = do_stuff(train_df, ner_vectorizer, title_vectorizer, train=True)
    val_df = do_stuff(val_df, ner_vectorizer, title_vectorizer, train=False)

    train_ner = np.array(list(train_df["NER"].values))
    val_ner = np.array(list(val_df["NER"].values))
    save_split("dataset/p4_dataset_train.npz", train_df, train_ner)
    save_split("dataset/p4_dataset_val.npz", val_df, val_ner)
