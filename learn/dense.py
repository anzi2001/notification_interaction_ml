import tensorflow as tf
import numpy as np

def create_dense(shape: int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(shape,)),
        tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

def train_dense(model, notification_list: np.ndarray, notification_labels: list):
    stop_early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, min_delta=0.001,)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(notification_list, notification_labels, epochs=200, batch_size=8, validation_split=0.2, class_weight=class_weights)
    return model, history