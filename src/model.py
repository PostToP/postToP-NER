from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


def build_model(train_data, train_labels, val_data, val_labels, input_sequence_length, vocab_size, num_classes):
    token_input = tf.keras.layers.Input(
        shape=(input_sequence_length,), name="token_input", dtype=tf.float32)
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=64, name="token_embedding")(token_input)

    channel_feauter_input = tf.keras.layers.Input(
        shape=(input_sequence_length,), name="channel_feature_input", dtype=tf.float32)
    channel_embedding = tf.keras.layers.Embedding(
        input_dim=2, output_dim=64, name="channel_embedding")(channel_feauter_input)

    x = tf.keras.layers.Concatenate()([x, channel_embedding])
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        64, return_sequences=True), name="bigru")(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_classes, activation='softmax'))(x)
    model = tf.keras.Model(
        inputs=[token_input, channel_feauter_input], outputs=x)
    anti_overfit = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.005, mode="min")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  )
    model.fit(train_data, train_labels, batch_size=32, verbose=1,
              epochs=200, validation_data=(
                  val_data, val_labels), callbacks=[anti_overfit],)
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
