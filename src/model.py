from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    Bidirectional,
    GRU,
    Input,
    Concatenate,
    TimeDistributed,
)
from tensorflow.keras.models import Model
import keras


@keras.saving.register_keras_serializable()
def f1_micro(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    true_positive_count = tf.reduce_sum(
        tf.cast(
            tf.logical_and(tf.equal(y_true, y_pred), tf.not_equal(y_true, 0)),
            tf.float32,
        )
    )
    false_positive_count = tf.reduce_sum(
        tf.cast(
            tf.logical_and(tf.not_equal(y_true, y_pred), tf.not_equal(y_pred, 0)),
            tf.float32,
        )
    )
    false_negative_count = tf.reduce_sum(
        tf.cast(
            tf.logical_and(tf.not_equal(y_true, y_pred), tf.not_equal(y_true, 0)),
            tf.float32,
        )
    )

    precision = true_positive_count / (
        true_positive_count + false_positive_count + 1e-10
    )
    recall = true_positive_count / (true_positive_count + false_negative_count + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1


def number_of_classes(values):
    num_classes = 0
    for batch in values:
        max_tag = tf.reduce_max(batch)
        if max_tag > num_classes:
            num_classes = max_tag
    return int(num_classes.numpy()) + 1


def build_model(train_data, val_data) -> Model:
    token_input_shape = train_data.element_spec[0][0].shape
    token_input_shape = (token_input_shape[1],)
    channel_input_shape = train_data.element_spec[0][1].shape
    channel_input_shape = (channel_input_shape[1], channel_input_shape[2])

    vocab_size = number_of_classes(train_data.map(lambda x, y: x[0]))
    num_classes = number_of_classes(train_data.map(lambda x, y: y))
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")

    token_input = Input(shape=token_input_shape, name="token_input", dtype=tf.float32)
    x = Embedding(input_dim=vocab_size, output_dim=45, name="token_embedding")(
        token_input
    )

    per_token_feature_input = Input(
        shape=channel_input_shape, name="channel_feature_input", dtype=tf.float32
    )

    x = Concatenate()([x, per_token_feature_input])
    x = Bidirectional(GRU(64, return_sequences=True), name="bigru")(x)

    x = Dropout(0.2)(x)
    x = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    model = Model(inputs=[token_input, per_token_feature_input], outputs=x)
    anti_overfit = EarlyStopping(
        monitor="val_f1_micro",
        patience=20,
        restore_best_weights=True,
        min_delta=0.005,
        mode="max",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    (
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", f1_micro],
        ),
    )
    model.summary()
    model.fit(
        train_data,
        verbose=1,
        epochs=500,
        validation_data=val_data,
        callbacks=[anti_overfit],
    )
    return model


def evaluate_model(model, title_val, X_val_channel, ner_val):
    loss, _, accuracy = model.evaluate([title_val, X_val_channel], ner_val, verbose=0)

    y_pred = model.predict([title_val, X_val_channel], verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=-1)

    y_true_flat = []
    y_pred_flat = []

    for i in range(len(ner_val)):
        seq_len = np.sum(title_val[i] != 0)
        if seq_len > 0:
            y_true_flat.extend(ner_val[i][:seq_len])
            y_pred_flat.extend(y_pred_classes[i][:seq_len])

    f1_micro = f1_score(y_true_flat, y_pred_flat, average="micro")
    f1_macro = f1_score(y_true_flat, y_pred_flat, average="macro")
    f1_weighted = f1_score(y_true_flat, y_pred_flat, average="weighted")

    f1_per_class = f1_score(y_true_flat, y_pred_flat, average=None)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class,
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
