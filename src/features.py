import re
import numpy as np


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
