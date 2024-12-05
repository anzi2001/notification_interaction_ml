import tensorflow as tf
import numpy as np

def create_dense(shape: int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(shape,)),
        tf.keras.layers.Dense(shape, activation='relu'),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

def train_dense(model, notification_list: np.ndarray, notification_labels: list, board_callback=None):
    callbacks = []
    if board_callback is not None:
        callbacks.append(board_callback)
    stop_early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, min_delta=0.001,)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    history = model.fit(notification_list, notification_labels, epochs=50, batch_size=32, validation_split=0.2, callbacks=[stop_early])
    return model, history