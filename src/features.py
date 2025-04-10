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
                if re.search(r'([一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤]+|[a-zA-Z0-9]+)[.!]*', t):
                    feature[i] = 1
        return feature

    @staticmethod
    def batch(function, *args):
        return [function(*arg) for arg in zip(*args)]
