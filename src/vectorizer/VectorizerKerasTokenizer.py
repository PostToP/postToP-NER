from collections import Counter
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class VectorizerKerasTokenizer:
    name = "VectorizerKerasTokenizer"

    def __init__(self, vocab_size=64, max_sequence_length=64):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.counter = Counter()
        self.vocab = {
            "<oov>": 1,
        }

    def train(self, texts):
        for text in texts:
            for word in text:
                self.counter[word] += 1

        for word, _ in self.counter.most_common():
            if len(self.vocab) >= self.vocab_size:
                break

            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1

    def encode(self, text):
        sequence = np.array(
            [self.vocab.get(word, self.vocab["<oov>"]) for word in text]
        )
        padded_sequences = pad_sequences(
            [sequence], maxlen=self.max_sequence_length, padding="post"
        )
        return padded_sequences[0]
