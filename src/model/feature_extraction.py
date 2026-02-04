import numpy as np
import pandas as pd

import re
from config.config import MAX_SEQUENCE_LENGTH, VOCAB_SIZE
from tokenizer.TokenizerCustom import TokenizerCustom
from vectorizer.VectorizerKerasTokenizer import VectorizerKerasTokenizer
from vectorizer.VectorizerNER import VectorizerNER
from tensormonad import TensorMonad
from vectorizer.VectorizerLanguage import VectorizerLanguage
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")


def gg(text):
    tokenized = TokenizerCustom().encode(text)
    return Doc(nlp.vocab, tokenized)


nlp.tokenizer = gg


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

    @staticmethod
    def token_appears_in_hashtags(token, description):
        feature = np.zeros(len(token), dtype=int)
        token = [t.lower() for t in token]
        hashtags = re.findall(r"#(\w+)", description.lower())
        for i, t in enumerate(token):
            if t in hashtags:
                feature[i] = 1
        return feature[:, np.newaxis]

    def token_appears_in_links(token, description):
        feature = np.zeros(len(token), dtype=int)
        token = [t.lower() for t in token]
        links = re.findall(r"(https?://[^\s]+)", description.lower())
        for i, t in enumerate(token):
            if t in links:
                feature[i] = 1
        return feature[:, np.newaxis]

    def mark_title_tokens(all_token, title_tokens):
        feature = np.zeros(len(all_token), dtype=int)
        len_of_title = len(title_tokens)
        feature[:len_of_title] = 1
        return feature[:, np.newaxis]

    @staticmethod
    def add_pos_tag_features(tokens):
        doc = nlp(tokens)
        pos_tags = [token.pos_ for token in doc]
        feature = np.zeros((len(tokens), 10), dtype=int)
        for i, pos in enumerate(pos_tags):
            if pos in ["NOUN", "PROPN"]:
                feature[i][0] = 1
            elif pos == "VERB":
                feature[i][1] = 1
            elif pos == "ADJ":
                feature[i][2] = 1
            elif pos == "ADV":
                feature[i][3] = 1
            elif pos == "PUNCT":
                feature[i][4] = 1
            elif pos == "PART":
                feature[i][5] = 1
            elif pos == "ADP":
                feature[i][6] = 1
            elif pos == "PRON":
                feature[i][7] = 1
            elif pos == "NUM":
                feature[i][8] = 1
            elif pos == "AUX":
                feature[i][8] = 1
            else:
                feature[i][9] = 1
        return feature

    @staticmethod
    def token_distance_from_start(token):
        feature = np.arange(len(token), dtype=int)
        return feature[:, np.newaxis]

    @staticmethod
    def token_capitalization(token):
        feature = np.zeros((len(token), 4), dtype=int)
        for i, t in enumerate(token):
            if t.isupper():  # ALL CAPS (AAAAAA)
                feature[i][0] = 1
            elif t.islower():  # all lower (aaaaaa)
                feature[i][1] = 1
            elif t.istitle():  # Title Case (Aaaaaa)
                feature[i][2] = 1
            else:  # Mixed (AaAaAa)
                feature[i][3] = 1
        return feature


def extract_features(
    dataset: pd.DataFrame, max_sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    per_token_features = []

    feature_channel = (
        TensorMonad((dataset["Text"].values, dataset["Channel Name"].values))
        .map(FeatureExtraction.tokens_containing_channel_name)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_channel)

    feature_token_freq_title = (
        TensorMonad((dataset["Text"].values, dataset["Original Title"].values))
        .map(FeatureExtraction.count_token_occurrences)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_token_freq_title)

    # feature_token_freq_desc = (
    #     TensorMonad((dataset["Text"].values, dataset["Original Description"].values))
    #     .map(FeatureExtraction.count_token_occurrences)
    #     .pad(max_sequence_length)
    #     .to_tensor()
    # )
    # per_token_features.append(feature_token_freq_desc)

    feature_token_length = (
        TensorMonad([dataset["Text"].values])
        .map(FeatureExtraction.length_of_tokens)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_token_length)

    # feature_is_token_verbal = (
    #     TensorMonad([dataset["Text"].values])
    #     .map(FeatureExtraction.is_token_verbal)
    #     .pad(max_sequence_length)
    #     .to_tensor()
    # )
    # per_token_features.append(feature_is_token_verbal)

    # feature_token_in_hashtags = (
    #     TensorMonad((dataset["Text"].values, dataset["Original Description"].values))
    #     .map(FeatureExtraction.token_appears_in_hashtags)
    #     .pad(max_sequence_length)
    #     .to_tensor()
    # )
    # per_token_features.append(feature_token_in_hashtags)

    # feature_token_in_links = (
    #     TensorMonad((dataset["Text"].values, dataset["Original Description"].values))
    #     .map(FeatureExtraction.token_appears_in_links)
    #     .pad(max_sequence_length)
    #     .to_tensor()
    # )
    # per_token_features.append(feature_token_in_links)

    # feature_mark_title_tokens = (
    #     TensorMonad((dataset["Text"].values, dataset["Title"].values))
    #     .map(FeatureExtraction.mark_title_tokens)
    #     .pad(max_sequence_length)
    #     .to_tensor()
    # )
    # per_token_features.append(feature_mark_title_tokens)

    pos_title_features = (
        TensorMonad([dataset["Original Title"] + " " + dataset["Original Description"]])
        .map(FeatureExtraction.add_pos_tag_features)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(pos_title_features)

    # feature_token_distance = (
    #     TensorMonad([dataset["Text"].values])
    #     .map(FeatureExtraction.token_distance_from_start)
    #     .pad(max_sequence_length)
    #     .to_tensor()
    # )
    # per_token_features.append(feature_token_distance)

    feature_token_capitalization = (
        TensorMonad([dataset["Original Title"] + " " + dataset["Original Description"]])
        .map(FeatureExtraction.token_capitalization)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_token_capitalization)

    global_features = []
    feature_language = (
        TensorMonad([dataset["Language"].values]).map(VectorizerLanguage).to_tensor()
    )
    global_features.append(feature_language)

    per_token_features = np.concatenate(per_token_features, axis=2)
    global_features = np.concatenate(global_features, axis=1)

    return per_token_features, global_features


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
    return tokens  # Temporary disable masking for better results
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
    dataset["Text"] = dataset["Title"] + dataset["Description"]
    dataset["NER"] = dataset["Title NER"] + dataset["Description NER"]

    return dataset


def do_stuff(df, ner_vectorizer, title_vectorizer, train=False):
    df = concat_dataset(df)

    if train:
        df["Text"] = mask_artists_and_titles(df["Text"].values, df["NER"].values)
        ner_vectorizer.train(df["NER"].values)
    df["NER"] = df["NER"].apply(lambda x: ner_vectorizer.encode(x))

    per_token_features, global_features = extract_features(df, MAX_SEQUENCE_LENGTH)
    df["Features"] = per_token_features.tolist()
    df["Global Features"] = global_features.tolist()

    if train:
        title_vectorizer.train(df["Text"].values)
    df["title_vec"] = df["Text"].apply(lambda x: title_vectorizer.encode(x)).to_numpy()

    return df


def save_split(path, X, y):
    np.savez_compressed(
        path,
        titles=X["title_vec"].to_list(),
        ner_tags=y,
        token_features=X["Features"].to_list(),
        global_features=X["Global Features"].to_list(),
    )


import joblib


def feature_extraction():
    train_df = pd.read_json("dataset/p4_dataset_train.json")
    val_df = pd.read_json("dataset/p4_dataset_val.json")
    ner_vectorizer = VectorizerNER(MAX_SEQUENCE_LENGTH)
    title_vectorizer = VectorizerKerasTokenizer(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)

    train_df = do_stuff(train_df, ner_vectorizer, title_vectorizer, train=True)
    val_df = do_stuff(val_df, ner_vectorizer, title_vectorizer, train=False)

    joblib.dump(title_vectorizer, "model/title_vectorizer.pkl")

    train_ner = np.array(list(train_df["NER"].values))
    val_ner = np.array(list(val_df["NER"].values))
    save_split("dataset/p5_dataset_train.npz", train_df, train_ner)
    save_split("dataset/p5_dataset_val.npz", val_df, val_ner)
