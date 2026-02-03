from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    Bidirectional,
    Input,
    Concatenate,
    TimeDistributed,
    RepeatVector,
    LSTM,
    LayerNormalization,
)
from tensorflow.keras.models import Model
import keras
from config.config import TABLE_BACK, VOCAB_SIZE


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


def build_model(train_data, val_data) -> Model:
    token_input_shape = train_data.element_spec[0][0].shape
    token_input_shape = (token_input_shape[1],)
    per_token_feature_shape = train_data.element_spec[0][1].shape
    per_token_feature_shape = (per_token_feature_shape[1], per_token_feature_shape[2])
    global_feature_shape = train_data.element_spec[0][2].shape
    global_feature_shape = (global_feature_shape[1],)

    token_input = Input(shape=token_input_shape, name="token_input", dtype=tf.float32)
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=2**3, name="token_embedding")(
        token_input
    )

    per_token_feature_input = Input(
        shape=per_token_feature_shape, name="channel_feature_input", dtype=tf.float32
    )

    global_feature_input = Input(
        shape=global_feature_shape, name="global_feature_input", dtype=tf.float32
    )
    global_repeat_vec = RepeatVector(
        token_input_shape[0], name="global_feature_repeat"
    )(global_feature_input)

    x = Concatenate()([x, per_token_feature_input, global_repeat_vec])
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(2**4, return_sequences=True), name="bigru")(x)
    x = LayerNormalization()(x)

    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(len(TABLE_BACK), activation="softmax"))(x)
    model = Model(
        inputs=[token_input, per_token_feature_input, global_feature_input], outputs=x
    )
    anti_overfit = EarlyStopping(
        monitor="val_f1_micro",
        patience=100,
        restore_best_weights=True,
        min_delta=0.005,
        mode="max",
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_f1_micro",
        factor=0.5,
        patience=25,
        min_lr=1e-6,
        mode="max",
        verbose=1,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", f1_micro],
    )

    model.summary()
    model.fit(
        train_data,
        verbose=1,
        epochs=5000,
        validation_data=val_data,
        callbacks=[anti_overfit, reduce_lr],
    )
    return model


def evaluate_model(model, title_val, X_val_channel, globalfeatures, ner_val):
    loss, _, accuracy = model.evaluate(
        [title_val, X_val_channel, globalfeatures], ner_val, verbose=0
    )

    y_pred = model.predict([title_val, X_val_channel, globalfeatures], verbose=0)
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
        # "f1_per_class": f1_per_class,
        "f1_per_class": {
            # TABLE_BACK[i]: f1_per_class[i] for i in range(len(f1_per_class))
            TABLE_BACK.get(i, str(i)): f1_per_class[i]
            for i in range(len(f1_per_class))
        },
    }
