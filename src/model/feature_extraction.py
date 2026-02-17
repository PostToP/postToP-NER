import logging
import numpy as np
import pandas as pd

import re
from config.config import MAX_SEQUENCE_LENGTH
from tokenizer.TokenizerCustom import TokenizerCustom
from vectorizer.VectorizerKerasTokenizer import VectorizerKerasTokenizer
from vectorizer.VectorizerNER import VectorizerNER
from tensormonad import TensorMonad
from vectorizer.VectorizerLanguage import VectorizerLanguage
import spacy
from spacy.tokens import Doc

logger = logging.getLogger("experiment")
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

        table = {
            "ADJ": 0,
            "ADP": 1,
            "ADV": 2,
            "AUX": 3,
            "CCONJ": 4,
            "DET": 5,
            "INTJ": 6,
            "NOUN": 7,
            "NUM": 8,
            "PART": 9,
            "PRON": 10,
            "PROPN": 11,
            "PUNCT": 12,
            "SCONJ": 13,
            "SYM": 14,
            "VERB": 15,
        }
        feature = np.zeros((len(tokens), len(table) + 1), dtype=int)

        for i, pos in enumerate(pos_tags):
            if pos in table:
                feature[i][table[pos]] = 1
            else:
                feature[i][-1] = 1
        return feature

    @staticmethod
    def add_tag_tag_features(tokens):
        doc = nlp(tokens)
        tag_tags = [token.tag_ for token in doc]

        table = {
            "CC": 0,
            "CD": 1,
            "DT": 2,
            "EX": 3,
            "FW": 4,
            "IN": 5,
            "JJ": 6,
            "JJR": 7,
            "JJS": 8,
            "LS": 9,
            "MD": 10,
            "NN": 11,
            "NNS": 12,
            "NNP": 13,
            "NNPS": 14,
            "PDT": 15,
            "POS": 16,
            "PRP": 17,
            "PRP$": 18,
            "RB": 19,
            "RBR": 20,
            "RBS": 21,
            "RP": 22,
            "SYM": 23,
            "TO": 24,
            "UH": 25,
            "VB": 26,
            "VBD": 27,
            "VBG": 28,
            "VBN": 29,
            "VBP": 30,
            "VBZ": 31,
            "WDT": 32,
            "WP": 33,
            "WP$": 34,
            "WRB": 35,
            "_SP": 36,
            ",": 37,
            "''": 38,
            ":": 39,
            "HYPH": 40,
            "XX": 41,
            "NFP": 42,
            "-LRB-": 43,
            "-RRB-": 44,
            "ADD": 45,
            ".": 46,
            "``": 47,
            "AFX": 48,
            "$": 49,
        }
        feature = np.zeros((len(tokens), len(table) + 1), dtype=int)

        for i, pos in enumerate(tag_tags):
            if pos in table:
                feature[i][table[pos]] = 1
            else:
                feature[i][-1] = 1
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

    def mark_tokens_inside_quotes(tokens):
        feature = np.zeros(len(tokens), dtype=int)
        inside_quotes = False
        for i, t in enumerate(tokens):
            if t in ['"', "“", "”"]:
                inside_quotes = not inside_quotes
            if inside_quotes:
                feature[i] = 1
        return feature[:, np.newaxis]

    def mark_tokens_inside_parentheses(tokens):
        feature = np.zeros(len(tokens), dtype=int)
        inside_parentheses = False
        for i, t in enumerate(tokens):
            if t in ["(", "（", "[", "【", "〈", "《", "「", "『"]:
                inside_parentheses = True
            elif t in [")", "）", "]", "】", "〉", "》", "」", "』"]:
                inside_parentheses = False
            if inside_parentheses:
                feature[i] = 1
        return feature[:, np.newaxis]


def extract_features(
    dataset: pd.DataFrame, max_sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    per_token_feature_functions = {
        "Token Appears in Channel Name": (
            FeatureExtraction.tokens_containing_channel_name,
            (
                dataset["Text"].to_numpy(copy=False),
                dataset["Channel Name"].to_numpy(copy=False),
            ),
        ),
        "Token Frequency in Title": (
            FeatureExtraction.count_token_occurrences,
            (
                dataset["Text"].to_numpy(copy=False),
                dataset["Title"].to_numpy(copy=False),
            ),
        ),
        # "Token Frequency in Description": (
        #     FeatureExtraction.count_token_occurrences,
        #     (
        #         dataset["Text"].to_numpy(copy=False),
        #         dataset["Original Description"].to_numpy(copy=False),
        #     ),
        # ),
        # "Length of Token": (
        #     FeatureExtraction.length_of_tokens,
        #     (dataset["Text"].to_numpy(copy=False),),
        # ),
        # "Is Token Verbal": (
        #     FeatureExtraction.is_token_verbal,
        #     (dataset["Text"].to_numpy(copy=False),),
        # ),
        # "Token Appears in Hashtags": (
        #     FeatureExtraction.token_appears_in_hashtags,
        #     (
        #         dataset["Text"].to_numpy(copy=False),
        #         dataset["Original Description"].to_numpy(copy=False),
        #     ),
        # ),
        # "Token Appears in Links": (
        #     FeatureExtraction.token_appears_in_links,
        #     (
        #         dataset["Text"].to_numpy(copy=False),
        #         dataset["Original Description"].to_numpy(copy=False),
        #     ),
        # ),
        # "Mark Title Tokens": (
        #     FeatureExtraction.mark_title_tokens,
        #     (
        #         dataset["Text"].to_numpy(copy=False),
        #         dataset["Original Title"].to_numpy(copy=False),
        #     ),
        # ),
        # "Add POS Tag Features": (
        #     FeatureExtraction.add_pos_tag_features,
        #     [(dataset["Title"] + " " + dataset["Description"]).to_numpy(copy=False)],
        # ),
        # "Token Distance from Start": (
        #     FeatureExtraction.token_distance_from_start,
        #     (dataset["Text"].to_numpy(copy=False),),
        # ),
        # "Token Capitalization": (
        #     FeatureExtraction.token_capitalization,
        #     [(dataset["Title"] + " " + dataset["Description"]).to_numpy(copy=False)],
        # ),
        # "Add Tag Tag Features": (
        #     FeatureExtraction.add_tag_tag_features,
        #     [(dataset["Title"] + " " + dataset["Description"]).to_numpy(copy=False)],
        # ),
        "Mark Tokens Inside Quotes": (
            FeatureExtraction.mark_tokens_inside_quotes,
            [(dataset["Title"] + " " + dataset["Description"]).to_numpy(copy=False)],
        ),
        "Mark Tokens Inside Parentheses": (
            FeatureExtraction.mark_tokens_inside_parentheses,
            [(dataset["Title"] + " " + dataset["Description"]).to_numpy(copy=False)],
        ),
    }

    per_token_features = []

    logger.info("Extracting features...")

    for feature_name, (function, args) in per_token_feature_functions.items():
        feature = TensorMonad(args).map(function).pad(max_sequence_length).to_tensor()
        logger.debug(f"Extracted feature: {feature_name} with shape {feature.shape}")
        per_token_features.append(feature)

    global_features = []
    feature_language = (
        TensorMonad([dataset["Language"].values]).map(VectorizerLanguage).to_tensor()
    )
    global_features.append(feature_language)

    per_token_features = np.concatenate(per_token_features, axis=2)
    global_features = np.concatenate(global_features, axis=1)

    per_token_features = np.zeros((len(dataset), max_sequence_length, 0), dtype=int)
    global_features = np.zeros((len(dataset), 0), dtype=int)

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
    # return tokens  # Temporary disable masking for better results
    not_to_mask = set()
    for t, tags in zip(tokens, ner_tags):
        for i, tag in enumerate(tags):
            if tag == "O" or tag == "MODIFIER" or tag == "VOCALOID":
                not_to_mask.add(t[i])

    new_tokens = []
    for token in tokens:
        new_tokens.append([i if i in not_to_mask else "<oov>" for i in token])
    return new_tokens


def do_stuff(df):
    tokenizer = TokenizerCustom()
    df["Text"] = df["Title"].apply(lambda x: tokenizer.encode(x)) + df[
        "Description"
    ].apply(lambda x: tokenizer.encode(x))
    per_token_features, global_features = extract_features(df, MAX_SEQUENCE_LENGTH)

    # # apply alignment to per token features
    # aligned_per_token_features = []
    # for i in range(per_token_features.shape[0]):
    #     aligned_features = []
    #     for j in range(per_token_features.shape[2]):
    #         aligned_feature = align_features(
    #             df["real_word_ids"].iloc[i], per_token_features[i, :, j]
    #         )
    #         aligned_features.append(aligned_feature)
    #     aligned_per_token_features.append(np.stack(aligned_features, axis=1))

    # df["Features"] = aligned_per_token_features
    df["Features"] = per_token_features.tolist()

    df["Global Features"] = global_features.tolist()

    return df


def align_features(word_ids, feature):
    aligned_features = np.zeros_like(feature)
    previous_word_id = None
    for i in range(len(word_ids)):
        word_id = word_ids[i]
        if word_id is None:
            aligned_features[i] = 0
        elif word_id != previous_word_id:
            aligned_features[i] = feature[word_id]
        else:
            aligned_features[i] = feature[word_id]
        previous_word_id = word_id
    return aligned_features


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

    train_df = do_stuff(train_df)
    val_df = do_stuff(val_df)

    train_df.to_json("dataset/p5_dataset_train.json", index=False)
    val_df.to_json("dataset/p5_dataset_val.json", index=False)
