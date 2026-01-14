from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class VectorizerKerasTokenizer:
    name = "VectorizerKerasTokenizer"

    def __init__(self, vocab_size=64, max_sequence_length=64):
        self.max_sequence_length = max_sequence_length
        self.vectorizer = KerasTokenizer(num_words=vocab_size, oov_token="<OOV>")

    def train(self, texts):
        self.vectorizer.fit_on_texts(texts)
        self.vectorizer.word_index["<oov>"] = 1

    def encode(self, text):
        sequence = self.vectorizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(
            sequence, maxlen=self.max_sequence_length, padding="post"
        )
        return padded_sequences[0]

    def encode_batch(self, texts):
        sequences = self.vectorizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences, maxlen=self.max_sequence_length, padding="post"
        )
        return padded_sequences
