from features import FeatureExtraction
from model import build_model, evaluate_model
from text_cleaner import preprocess_tokens
from vectorizer import VectorizerKerasTokenizer, VectorizerNER
from tokenizer import *
from dataset import convert_ner_tags, fix_dataset_NER, split_dataset
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

num_classes = len(set([tag for row in train_df['NER'] for tag in row]))
model = build_model(
    [title_train, X_train_channel], ner_train, [title_val, X_val_channel], ner_val, MAX_SEQUENCE_LENGTH, VOCAB_SIZE, num_classes)


results = evaluate_model(model, title_val, X_val_channel, ner_val)
print(results)
