import numpy as np

from vectorizer.VectorizerKerasTokenizer import custom_pad_sequences


class TensorMonad:
    def __init__(self, data):
        self.data = data

    def map(self, func, *args):
        return TensorMonad([func(*arg) for arg in zip(*self.data, *args)])

    def pad(self, maxlen):
        return TensorMonad(
            custom_pad_sequences(self.data, maxlen=maxlen, padding="post")
        )

    def to_tensor(self):
        return np.array(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def shape(self):
        return np.array(self.data).shape

    @property
    def dtype(self):
        return np.array(self.data).dtype

    def __repr__(self):
        return f"<TensorMonad shape={self.shape} dtype={self.dtype}>"
