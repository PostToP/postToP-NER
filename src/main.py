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
train_titles = title_vectorizer.encode_batch(train_df["Tokens"].values)
val_titles = title_vectorizer.encode_batch(validation_df["Tokens"].values)
train_ner = train_df["NER"].values

val_ner = validation_df["NER"].values
train_ner = np.array(list(train_ner))
val_ner = np.array(list(val_ner))


def pad_sequence(x):
    return pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


train_feature_channel = FeatureExtraction.batch(FeatureExtraction.tokens_containing_channel_name,
                                                train_df["Original Tokens"].values, train_df["Channel Name"].values)
val_feature_channel = FeatureExtraction.batch(FeatureExtraction.tokens_containing_channel_name,
                                              validation_df["Original Tokens"].values, validation_df["Channel Name"].values)
train_feature_channel = pad_sequence(train_feature_channel)
val_feature_channel = pad_sequence(val_feature_channel)

train_feature_description = FeatureExtraction.batch(FeatureExtraction.count_token_occurrences,
                                                    train_df["Original Tokens"].values, train_df["Description"].values)
val_feature_description = FeatureExtraction.batch(FeatureExtraction.count_token_occurrences,
                                                  validation_df["Original Tokens"].values, validation_df["Description"].values)
train_feature_description = pad_sequence(train_feature_description)
val_feature_description = pad_sequence(val_feature_description)


def concatenate_features(*features, sequence_length=MAX_SEQUENCE_LENGTH):
    reshaped_features = [
        feature.reshape(-1, sequence_length, 1) for feature in features]
    return np.concatenate(reshaped_features, axis=2)


train_features = concatenate_features(
    train_feature_channel, train_feature_description)
val_features = concatenate_features(
    val_feature_channel, val_feature_description)

train_dataset = tf.data.Dataset.from_tensor_slices(
    ((train_titles, train_features), train_ner)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    ((val_titles, val_features), val_ner)).batch(32).prefetch(tf.data.AUTOTUNE)

num_classes = len(set([tag for row in train_df['NER'] for tag in row]))
model = build_model(
    train_dataset, val_dataset, MAX_SEQUENCE_LENGTH, VOCAB_SIZE, num_classes)


results = evaluate_model(model, val_titles, val_features, val_ner)
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
