from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, GRU, Input, Concatenate, TimeDistributed
from tensorflow.keras.models import Model


def build_model(train_data, val_data, input_sequence_length, vocab_size, num_classes) -> Model:
    token_input = Input(shape=(input_sequence_length,),
                        name="token_input", dtype=tf.float32)
    x = Embedding(input_dim=vocab_size, output_dim=45,
                  name="token_embedding")(token_input)

    channel_feature_input = Input(shape=(
        input_sequence_length, 1), name="channel_feature_input", dtype=tf.float32)

    x = Concatenate()([x, channel_feature_input])
    x = Bidirectional(GRU(64, return_sequences=True), name="bigru")(x)

    x = Dropout(0.2)(x)
    x = TimeDistributed(Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=[token_input, channel_feature_input], outputs=x)
    anti_overfit = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.005, mode="min")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  )
    model.fit(train_data, verbose=1,
              epochs=200, validation_data=val_data, callbacks=[anti_overfit],)
    return model


def evaluate_model(model, title_val, X_val_channel, ner_val):
    loss, accuracy = model.evaluate(
        [title_val, X_val_channel], ner_val, verbose=0)

    y_pred = model.predict([title_val, X_val_channel], verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=-1)

    y_true_flat = []
    y_pred_flat = []

    for i in range(len(ner_val)):
        seq_len = np.sum(title_val[i] != 0)  # Non-zero tokens
        if seq_len > 0:
            y_true_flat.extend(ner_val[i][:seq_len])
            y_pred_flat.extend(y_pred_classes[i][:seq_len])

    f1_micro = f1_score(y_true_flat, y_pred_flat, average='micro')
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro')
    f1_weighted = f1_score(y_true_flat, y_pred_flat, average='weighted')

    f1_per_class = f1_score(y_true_flat, y_pred_flat, average=None)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class
    }


def decode_prediction(prediction, tokens):
    output = {}
    current_tag = 0
    current = ""
    for i, token in enumerate(tokens):
        if prediction[i] != current_tag:
            if current_tag not in output:
                output[current_tag] = []
            output[current_tag].append(current.strip())
            current_tag = prediction[i]
            current = ""
        current = current + " " + token
    if current_tag not in output:
        output[current_tag] = []
    output[current_tag].append(current.strip())
    if 0 in output:
        output.pop(0)
    if 1 in output:
        output["Artist"] = output.pop(1)
    else:
        output["Artist"] = []
    if 2 in output:
        output["Title"] = output.pop(2)
    else:
        output["Title"] = []
    return output
