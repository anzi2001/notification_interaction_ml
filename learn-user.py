import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from data import *
import learn.dense as dense
from data import *

TIME_DIFF = 33000

if __name__ == "__main__":
    data = load_data()
    data = merge_data(*data)

    for device,deviceData in data.groupby("device_id"):
        if len(deviceData) == 0:
            continue

        scaler, encoder = fit_data(deviceData)
        labels = process_labels(deviceData)

        train_data, test_data, train_labels, test_labels = train_test_split(
            deviceData, labels, test_size=0.25, random_state=42, stratify=labels
        )

        train_data = preprocess_data(train_data, scaler, encoder)
        test_data = preprocess_data(test_data, scaler, encoder)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(train_data.shape[1],)),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        model, history = dense.train_dense(model, train_data, train_labels)

        print("---EVALUATION---")
        #for i in range(len(test_data)):
        #    print(f"Predicted: {model.predict(np.array([test_data[i]], dtype=float))} Actual: {test_labels[i]}")
        model.evaluate(test_data, test_labels, verbose=2)

        print("---MANUAL EVALUATION---")
        print(np.mean(test_labels), 1 - np.mean(test_labels))
        print()
    
