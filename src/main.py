import tensorflow as tf
from features import FeatureExtraction
from model import build_model, decode_prediction, evaluate_model
from text_cleaner import preprocess_tokens
from vectorizer import VectorizerKerasTokenizer, VectorizerNER
from tokenizer import *
from dataset import convert_ner_tags, fix_dataset_NER, split_dataset
import dill
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 50
MAX_SEQUENCE_LENGTH = 45

dataset = pd.read_json("dataset/data.json")
dataset = dataset[dataset["NER"].isna() == False]
dataset = dataset[['Channel Name', 'Title', 'NER', 'Description']]


dataset = fix_dataset_NER(dataset)


ner_tags = convert_ner_tags(dataset["Title"].values, dataset["NER"].values)
dataset["NER"] = ner_tags


title_tokenizer = TokenizerCustom()
dataset["Original Tokens"] = dataset["Title"].apply(
    lambda x: title_tokenizer.encode(x))
dataset["Tokens"] = dataset["Original Tokens"].apply(
    lambda x: preprocess_tokens(x))
print(dataset.iloc[0]["Title"])

ner_vectorizer = VectorizerNER(MAX_SEQUENCE_LENGTH)
ner_vectorizer.train(dataset["NER"].values)
dataset["NER"] = dataset["NER"].apply(
    lambda x: ner_vectorizer.encode(x))

train_df, validation_df = split_dataset(dataset, fraction=0.8, random_state=42)


title_vectorizer = VectorizerKerasTokenizer(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
title_vectorizer.train(train_df["Tokens"].values)
title_train = title_vectorizer.encode_batch(train_df["Tokens"].values)
title_val = title_vectorizer.encode_batch(validation_df["Tokens"].values)
ner_train = train_df["NER"].values

ner_val = validation_df["NER"].values
ner_train = np.array(list(ner_train))
ner_val = np.array(list(ner_val))


X_train_channel = FeatureExtraction.batch(FeatureExtraction.tokens_containing_channel_name,
                                          train_df["Original Tokens"].values, train_df["Channel Name"].values)
X_val_channel = FeatureExtraction.batch(FeatureExtraction.tokens_containing_channel_name,
                                        validation_df["Original Tokens"].values, validation_df["Channel Name"].values)
X_train_channel = pad_sequences(
    X_train_channel, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_val_channel = pad_sequences(
    X_val_channel, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

X_train_description = FeatureExtraction.batch(FeatureExtraction.count_token_occurrences,
                                              train_df["Original Tokens"].values, train_df["Description"].values)
X_val_description = FeatureExtraction.batch(FeatureExtraction.count_token_occurrences,
                                            validation_df["Original Tokens"].values, validation_df["Description"].values)
X_train_description = pad_sequences(
    X_train_description, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_val_description = pad_sequences(
    X_val_description, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

x_feature = np.concatenate([X_train_channel.reshape(-1, MAX_SEQUENCE_LENGTH, 1),
                            X_train_description.reshape(-1, MAX_SEQUENCE_LENGTH, 1)],
                           axis=2)

x_val_feature = np.concatenate([X_val_channel.reshape(-1, MAX_SEQUENCE_LENGTH, 1),
                                X_val_description.reshape(-1, MAX_SEQUENCE_LENGTH, 1)],
                               axis=2)

train_dataset = tf.data.Dataset.from_tensor_slices(
    ((title_train, x_feature), ner_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    ((title_val, x_val_feature), ner_val)).batch(32).prefetch(tf.data.AUTOTUNE)

num_classes = len(set([tag for row in train_df['NER'] for tag in row]))
model = build_model(
    train_dataset, val_dataset, MAX_SEQUENCE_LENGTH, VOCAB_SIZE, num_classes)


results = evaluate_model(model, title_val, x_val_feature, ner_val)
print(results)


class ModelWrapper:
    def __init__(self, model, title_tokenizer, title_vectorizer):
        self.model = model
        self.title_tokenizer = title_tokenizer
        self.title_vectorizer = title_vectorizer
        self.max_sequence_length = MAX_SEQUENCE_LENGTH

    def predict(self, title, channel_name):
        original_tokens = self.title_tokenizer.encode(title)
        tokens = preprocess_tokens(original_tokens)
        vector = self.title_vectorizer.encode(tokens)
        channel_vector = FeatureExtraction.tokens_containing_channel_name(
            original_tokens, channel_name)
        channel_vector = pad_sequences(
            [channel_vector], maxlen=self.max_sequence_length, padding='post')[0]
        channel_vector = np.array(channel_vector, dtype=float)
        vector = np.array(vector, dtype=float)

        vector = vector.reshape(1, -1)
        channel_vector = channel_vector.reshape(1, -1)

        predictions = self.model.predict([vector, channel_vector])
        predictions = np.argmax(predictions, axis=-1)
        return decode_prediction(predictions[0], original_tokens)


model_wrapper = ModelWrapper(model, title_tokenizer, title_vectorizer)

with open('out/model.pkl', 'wb') as f:
    dill.dump(model_wrapper, f)
