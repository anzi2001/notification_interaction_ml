import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_tuner import HyperModel, Hyperband
from data import *

# Load data from files

class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_shape,)))
        
        # Tune the number of units in the first Dense layer
        model.add(tf.keras.layers.Dense(units=hp.Int('units_1', min_value=64, max_value=256, step=32), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_1', min_value=0.3, max_value=0.7, step=0.1)))
        
        # Tune the number of units in the second Dense layer
        model.add(tf.keras.layers.Dense(units=hp.Int('units_2', min_value=128, max_value=512, step=32), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_2', min_value=0.3, max_value=0.7, step=0.1)))
        
        # Tune the number of units in the third Dense layer
        model.add(tf.keras.layers.Dense(units=hp.Int('units_3', min_value=64, max_value=256, step=32), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_3', min_value=0.3, max_value=0.7, step=0.1)))
        
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

if __name__ == "__main__":
    activity, notifications, location, proximity, screen, volume, wifi, light, battery = load_data()
    data = merge_data(activity, notifications, location, proximity, screen, volume, wifi, light, battery)
    scaler, encoder = fit_data(data)
    labels = process_labels(data)
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=42)

    train_data = preprocess_data(train_data, scaler, encoder)
    test_data = preprocess_data(test_data, scaler, encoder)
    
    hypermodel = MyHyperModel(input_shape=train_data.shape[1])
    
    tuner = Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='my_project'
    )
    
    tuner.search(train_data, train_labels, epochs=30, batch_size=16, validation_split=0.2)
    
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    print(best_hps.values)
    
    loss, accuracy = best_model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {accuracy}")

    print("BASELINE Evaluation")
    baseline = np.mean(labels)
    print(f"Baseline Accuracy: {baseline}")