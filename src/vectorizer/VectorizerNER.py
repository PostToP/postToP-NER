import numpy as np

from config.config import TABLE
from vectorizer.VectorizerKerasTokenizer import custom_pad_sequences


class VectorizerNER:
    name = "VectorizerNER"

    def __init__(self, max_sequence_length=64):
        self.max_sequence_length = max_sequence_length

    def train(self, texts):
        pass

    def encode(self, tags):
        tag_vec = np.array([TABLE.get(tag, 0) for tag in tags])

        padded_tags = custom_pad_sequences([tag_vec], maxlen=self.max_sequence_length)
        return np.array(padded_tags[0])

    def encode_batch(self, tags):
        tag_vec = np.array([[TABLE.get(tag) for tag in tag_list] for tag_list in tags])
        padded_tags = custom_pad_sequences(tag_vec, maxlen=self.max_sequence_length)
        return padded_tags
