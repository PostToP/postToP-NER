import tensorflow as tf
from features import extract_features
from model import build_model, decode_prediction, evaluate_model
from text_cleaner import preprocess_tokens, mask_artists_and_titles
from vectorizer import VectorizerKerasTokenizer, VectorizerNER
from tokenizer import TokenizerCustom
from dataset import convert_ner_tags, fix_dataset_NER, split_dataset
import dill
import numpy as np
import pandas as pd

VOCAB_SIZE = 150
MAX_SEQUENCE_LENGTH = 45


def main():
    dataset = pd.read_json("dataset/data.json")
    dataset = dataset[dataset["NER"].isna() == False]
    dataset = dataset[["Channel Name", "Title", "NER", "Description", "Language"]]

    dataset = fix_dataset_NER(dataset)

    ner_tags = convert_ner_tags(dataset["Title"].values, dataset["NER"].values)
    dataset["NER"] = ner_tags

    title_tokenizer = TokenizerCustom()
    dataset["Original Tokens"] = dataset["Title"].apply(
        lambda x: title_tokenizer.encode(x)
    )

    dataset["Tokens"] = dataset["Original Tokens"].apply(lambda x: preprocess_tokens(x))
    dataset["Tokens"] = mask_artists_and_titles(
        dataset["Tokens"].values, dataset["NER"].values
    )

    ner_vectorizer = VectorizerNER(MAX_SEQUENCE_LENGTH)
    ner_vectorizer.train(dataset["NER"].values)
    dataset["NER"] = dataset["NER"].apply(lambda x: ner_vectorizer.encode(x))

    per_token_features, global_features = extract_features(dataset, MAX_SEQUENCE_LENGTH)
    dataset["Features"] = per_token_features.tolist()
    dataset["Global Features"] = global_features.tolist()

    train_df, validation_df = split_dataset(dataset, fraction=0.8, random_state=42)

    title_vectorizer = VectorizerKerasTokenizer(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    title_vectorizer.train(train_df["Tokens"].values)
    train_titles = title_vectorizer.encode_batch(train_df["Tokens"].values)
    val_titles = title_vectorizer.encode_batch(validation_df["Tokens"].values)
    train_ner = train_df["NER"].values

    val_ner = validation_df["NER"].values
    train_ner = np.array(list(train_ner))
    val_ner = np.array(list(val_ner))

    train_token_features = np.array(list(train_df["Features"].values))
    val_token_features = np.array(list(validation_df["Features"].values))

    train_global_features = np.array(list(train_df["Global Features"].values))
    val_global_features = np.array(list(validation_df["Global Features"].values))

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            ((train_titles, train_token_features, train_global_features), train_ner)
        )
        .shuffle(1000)
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(
            ((val_titles, val_token_features, val_global_features), val_ner)
        )
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(train_dataset, val_dataset)

    results = evaluate_model(
        model, val_titles, val_token_features, val_global_features, val_ner
    )
    print(results)

    class ModelWrapper:
        def __init__(self, model, title_tokenizer, title_vectorizer):
            self.model = model
            self.title_tokenizer = title_tokenizer
            self.title_vectorizer = title_vectorizer
            self.max_sequence_length = MAX_SEQUENCE_LENGTH

        def predict(self, title, channel_name, description, language=None):
            original_tokens = self.title_tokenizer.encode(title)
            tokens = preprocess_tokens(original_tokens)
            vector = self.title_vectorizer.encode(tokens)
            dataframe = pd.DataFrame(
                {
                    "Original Tokens": [original_tokens],
                    "Channel Name": [channel_name],
                    "Description": [description],
                    "Language": [language],
                }
            )
            per_token_features, global_features = extract_features(
                dataframe, self.max_sequence_length
            )
            vector = np.array(vector, dtype=float)
            vector = vector.reshape(1, -1)

            predictions = self.model.predict(
                [vector, per_token_features, global_features]
            )
            predictions = np.argmax(predictions, axis=-1)
            return decode_prediction(predictions[0], original_tokens)

    model_wrapper = ModelWrapper(model, title_tokenizer, title_vectorizer)
    print(
        model_wrapper.predict(
            "MISSH feat. BURAI – Budapest (Official Music Video) | #misshmusic",
            "#MISSHMUSIC",
            "#MISSHMUSIC\n\nhttps://open.spotify.com/artist/6PD6eSZM8ulCg5PRU6mEII\nhttps://music.apple.com/lt/artist/misshmusic/1282462090\nhttps://www.deezer.com/us/artist/13167869\n\nhttps://www.tiktok.com/@misshmusic_official\nhttps://www.facebook.com/MR.MISSH90\nhttps://www.instagram.com/missh90/\n\nKülönköszönet : Ecke22 étterem /www.ecke22etterem.hu                    \n\nA  MisshMusic  2013 óta működő független magyar zenei produkciós iroda. Fő tevékenységünk saját dalok és videoklipek készítése továbbá egyedi tervezésű saját márkás ruházati termékek forgalmazása. Eddigi zenei együttműködő partnereink: G.w.M. , Burai Krisztián, Young G26, IGNI, Hegyi Roland (HR).",
        )
    )

    with open("out/model.pkl", "wb") as f:
        dill.dump(model_wrapper, f)


if __name__ == "__main__":
    main()
