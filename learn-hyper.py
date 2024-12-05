import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_tuner import HyperModel, Hyperband
from data import *

# Load data from files
categorical_features = ['prox', 'screen', 'volume', 'battery', 'location']
numerical_features = []

class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                        activation='relu', input_shape=(self.input_shape,)))
        
        # Tune the number of layers
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(tf.keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                            activation='relu'))
            model.add(tf.keras.layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.3, max_value=0.9, step=0.1)))
        
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

if __name__ == "__main__":
    data = load_data()
    data = merge_data(*data)
    labels = process_labels(data).map(lambda x: [x, 1-x]).to_list()
    labels = np.array(labels)

    ble_cluster, wifi_cluster = cluster(data)

    print(len(data), len(ble_cluster), len(wifi_cluster))

    data["ble_cluster"] = ble_cluster
    data["wifi_cluster"] = wifi_cluster
    
    scaler, encoder = fit_data(data, categorical_features, numerical_features)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=42)

    train_data = preprocess_data(train_data, scaler, encoder, categorical_features, numerical_features)
    test_data = preprocess_data(test_data, scaler, encoder, categorical_features, numerical_features)
    
    hypermodel = MyHyperModel(input_shape=train_data.shape[1])
    
    tuner = Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='my_project'
    )
    
    tuner.search(train_data, train_labels, epochs=40, batch_size=8, validation_split=0.2)
    
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    print(best_hps.values)
    
    loss, accuracy = best_model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {accuracy}")

    print("BASELINE Evaluation")
    baseline = np.mean(labels)
    print(f"Baseline Accuracy: {baseline}")