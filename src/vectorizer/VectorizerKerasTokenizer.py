from collections import Counter
import numpy as np


# (x, n) -> (batch, maxlen, n)
def custom_pad_sequences(
    sequences, maxlen, padding="post", truncating="post", value=0, dtype=None
):
    first = np.asarray(sequences[0])
    sample_shape = first.shape[1:]
    pad_dtype = dtype or first.dtype

    padded = np.full((len(sequences), maxlen) + sample_shape, value, dtype=pad_dtype)

    for i, seq in enumerate(sequences):
        seq = np.asarray(seq)
        seq = seq[:maxlen]
        padded[i, : len(seq)] = seq

    return padded


class VectorizerKerasTokenizer:
    name = "VectorizerKerasTokenizer"

    def __init__(self, vocab_size=64, max_sequence_length=64):
        self.max_sequence_length = max_sequence_length
        self.counter = Counter()
        self.vocab = {
            "<oov>": 1,
            "<cls>": 2,
            "<sep>": 3,
            "<end>": 4,
        }
        self.vocab_size = vocab_size - len(self.vocab)

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
        padded_sequences = custom_pad_sequences(
            [sequence], maxlen=self.max_sequence_length
        )
        return padded_sequences[0]
