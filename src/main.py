from model import build_model, evaluate_model
from vectorizer import VectorizerKerasTokenizer, VectorizerNER
from tokenizer import *
from dataset import convert_ner_tags, fix_dataset_NER
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 50
MAX_SEQUENCE_LENGTH = 45

dataset = pd.read_json("dataset/data.json")
dataset = dataset[dataset["NER"].isna() == False]
dataset = dataset[['Channel Name', 'Title', 'NER']]


dataset = fix_dataset_NER(dataset)


ner_tags = convert_ner_tags(dataset["Title"].values, dataset["NER"].values)
dataset["NER"] = ner_tags


title_tokenizer = TokenizerCustom()
dataset["Tokens"] = dataset["Title"].apply(lambda x: title_tokenizer.encode(x))
print(dataset.iloc[0]["Title"])

ner_vectorizer = VectorizerNER(MAX_SEQUENCE_LENGTH)
ner_vectorizer.train(dataset["NER"].values)
dataset["NER"] = dataset["NER"].apply(
    lambda x: ner_vectorizer.encode(x))


train_df = dataset.sample(frac=0.8, random_state=42)
validation_df = dataset.drop(train_df.index)
train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)


title_vectorizer = VectorizerKerasTokenizer(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
title_vectorizer.train(train_df["Tokens"].values)
title_train = title_vectorizer.encode_batch(train_df["Tokens"].values)
title_val = title_vectorizer.encode_batch(validation_df["Tokens"].values)
ner_train = train_df["NER"].values

ner_val = validation_df["NER"].values
ner_train = np.array(list(ner_train))
ner_val = np.array(list(ner_val))


def extract_feature_channel(token, channel_name):
    feature = np.zeros(len(token), dtype=int)
    channel_name = channel_name.lower()
    token = [t.lower() for t in token]
    for i, t in enumerate(token):
        if t in channel_name:
            if re.search(r'([一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤]+|[a-zA-Z0-9]+)[.!]*', t):
                feature[i] = 1
    return feature


def extract_feature_channel_batch(tokens, channel_names):
    features = []
    for i, (token, channel_name) in enumerate(zip(tokens, channel_names)):
        features.append(extract_feature_channel(token, channel_name))
    return features


X_train_channel = extract_feature_channel_batch(
    train_df["Tokens"].values, train_df["Channel Name"].values)
X_val_channel = extract_feature_channel_batch(
    validation_df["Tokens"].values, validation_df["Channel Name"].values)
X_train_channel = pad_sequences(
    X_train_channel, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_val_channel = pad_sequences(
    X_val_channel, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

num_classes = len(set([tag for row in train_df['NER'] for tag in row]))
model = build_model(
    [title_train, X_train_channel], ner_train, [title_val, X_val_channel], ner_val, MAX_SEQUENCE_LENGTH, VOCAB_SIZE, num_classes)


results = evaluate_model(model, title_val, X_val_channel, ner_val)
print(results)
