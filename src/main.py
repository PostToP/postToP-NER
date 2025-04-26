import tensorflow as tf
from features import FeatureExtraction
from model import build_model, decode_prediction, evaluate_model
from tensormonad import TensorMonad
from text_cleaner import preprocess_tokens
from vectorizer import VectorizerKerasTokenizer, VectorizerNER
from tokenizer import *
from dataset import convert_ner_tags, fix_dataset_NER, split_dataset
import dill
import numpy as np
import pandas as pd

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

train_feature_channel = TensorMonad((train_df["Original Tokens"].values, train_df["Channel Name"].values)).map(
    FeatureExtraction.tokens_containing_channel_name).pad(MAX_SEQUENCE_LENGTH).to_tensor()
val_feature_channel = TensorMonad((validation_df["Original Tokens"].values, validation_df["Channel Name"].values)).map(
    FeatureExtraction.tokens_containing_channel_name).pad(MAX_SEQUENCE_LENGTH).to_tensor()

train_feature_description = TensorMonad(
    (train_df["Original Tokens"].values, train_df["Description"].values)).map(
    FeatureExtraction.count_token_occurrences).pad(MAX_SEQUENCE_LENGTH).to_tensor()
val_feature_description = TensorMonad(
    (validation_df["Original Tokens"].values, validation_df["Description"].values)).map(
    FeatureExtraction.count_token_occurrences).pad(MAX_SEQUENCE_LENGTH).to_tensor()


train_features = np.concatenate(
    [train_feature_channel, train_feature_description], axis=2)
val_features = np.concatenate(
    [val_feature_channel, val_feature_description], axis=2)


train_dataset = tf.data.Dataset.from_tensor_slices(
    ((train_titles, train_features), train_ner)).shuffle(1000).batch(1024).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    ((val_titles, val_features), val_ner)).batch(1024).prefetch(tf.data.AUTOTUNE)

num_classes = len(set([tag for row in train_df['NER'] for tag in row]))
model = build_model(
    train_dataset, val_dataset, VOCAB_SIZE, num_classes)


results = evaluate_model(model, val_titles, val_features, val_ner)
print(results)


class ModelWrapper:
    def __init__(self, model, title_tokenizer, title_vectorizer):
        self.model = model
        self.title_tokenizer = title_tokenizer
        self.title_vectorizer = title_vectorizer
        self.max_sequence_length = MAX_SEQUENCE_LENGTH

    def predict(self, title, channel_name, description):
        original_tokens = self.title_tokenizer.encode(title)
        tokens = preprocess_tokens(original_tokens)
        vector = self.title_vectorizer.encode(tokens)
        channel_vector = TensorMonad([[original_tokens], [channel_name]]).map(
            FeatureExtraction.tokens_containing_channel_name).pad(
            self.max_sequence_length).to_tensor()

        description_vector = TensorMonad(
            [[original_tokens], [description]]).map(
            FeatureExtraction.count_token_occurrences).pad(
            self.max_sequence_length).to_tensor()

        features = np.concatenate([channel_vector, description_vector], axis=2)

        vector = np.array(vector, dtype=float)
        vector = vector.reshape(1, -1)

        predictions = self.model.predict([vector, features])
        predictions = np.argmax(predictions, axis=-1)
        return decode_prediction(predictions[0], original_tokens)


model_wrapper = ModelWrapper(model, title_tokenizer, title_vectorizer)
print(model_wrapper.predict(
    "MISSH feat. BURAI – Budapest (Official Music Video) | #misshmusic",
    "#MISSHMUSIC",
    "#MISSHMUSIC\n\nhttps://open.spotify.com/artist/6PD6eSZM8ulCg5PRU6mEII\nhttps://music.apple.com/lt/artist/misshmusic/1282462090\nhttps://www.deezer.com/us/artist/13167869\n\nhttps://www.tiktok.com/@misshmusic_official\nhttps://www.facebook.com/MR.MISSH90\nhttps://www.instagram.com/missh90/\n\nKülönköszönet : Ecke22 étterem /www.ecke22etterem.hu                    \n\nA  MisshMusic  2013 óta működő független magyar zenei produkciós iroda. Fő tevékenységünk saját dalok és videoklipek készítése továbbá egyedi tervezésű saját márkás ruházati termékek forgalmazása. Eddigi zenei együttműködő partnereink: G.w.M. , Burai Krisztián, Young G26, IGNI, Hegyi Roland (HR)."
))

with open('out/model.pkl', 'wb') as f:
    dill.dump(model_wrapper, f)
