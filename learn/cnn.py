import tensorflow as tf
import keras
import numpy as np

def create_cnn(shape):
    return keras.models.Sequential([
        keras.layers.Input(shape=(shape, 1)),
        keras.layers.Conv1D(128, 2, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 2, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')
    ])

def train_cnn(notification_list: np.ndarray, notification_labels: list):
    model = create_cnn(notification_list.shape[1])
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(notification_list, notification_labels, epochs=50, batch_size=8, validation_split=0.2, callbacks=[stop_early])
    return model, history