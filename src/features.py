import re
import numpy as np
import pandas as pd
from tensormonad import TensorMonad
from vectorizer import VectorizerLanguage


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

    feature_description = (
        TensorMonad((dataset["Original Tokens"].values, dataset["Description"].values))
        .map(FeatureExtraction.count_token_occurrences)
        .pad(max_sequence_length)
        .to_tensor()
    )
    per_token_features.append(feature_description)

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
