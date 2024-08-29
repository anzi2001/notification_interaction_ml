import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data import *

# Load data from files


# Define the TensorFlow model
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    activity, notifications, location, proximity, screen, volume, wifi, light, battery = load_data()
    data = merge_data(activity, notifications, location, proximity, screen, volume, wifi, light, battery)
    scaler,encoder = fit_data(data)
    labels = process_labels(data)
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25)

    train_data = preprocess_data(train_data, scaler, encoder)
    test_data = preprocess_data(test_data, scaler, encoder)
    
    model = build_model(train_data.shape[1])
    
    model.fit(train_data, train_labels, epochs=30, batch_size=16, validation_split=0.2)
    
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {accuracy}")

    print("BASELINE Evaluation")
    baseline = np.mean(labels)
    print(f"Baseline Accuracy: {baseline}")