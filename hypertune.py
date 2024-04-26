import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import keras_tuner as kt
from data import *

class DenseHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model = keras.models.Sequential([
            keras.layers.Input(shape=(22,)),
            keras.layers.Dense(hp.Int('input_units', min_value=32, max_value=512, step=32), activation=tf.nn.relu),
            keras.layers.Dense(hp.Int('dense_one', min_value=32, max_value=512, step=32), activation=tf.nn.relu),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(hp.Int('dense_two', min_value=32, max_value=512, step=32), activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=1, max_value=64, step=16),
            **kwargs,
        )
    
class CNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model = keras.models.Sequential([
            keras.layers.Input(shape=(22,1)),
            keras.layers.Conv1D(hp.Int('input_units', min_value=32, max_value=512, step=32), 2, activation=tf.nn.relu),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(hp.Int('conv_1', min_value=32, max_value=512, step=32), 2, activation=tf.nn.relu),
            keras.layers.MaxPooling1D(2),
            keras.layers.Dropout(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(hp.Int("dense_1", min_value=32, max_value=512, step=32), activation='relu'),
            keras.layers.Dense(hp.Int("dense_2", min_value=32, max_value=512, step=32), activation='relu'),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=1, max_value=64, step=16),
            **kwargs,
        )
    

def train_dense(notification_list, notification_labels: list):
    array_list = np.array([item.to_numpy_array() for item in notification_list], dtype=float)
    tuner = kt.Hyperband(CNNHyperModel(), objective='val_accuracy', max_epochs=50, factor=3, directory='my_dir', project_name='intro_to_kt')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(array_list, notification_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

TIME_DIFF = 30000

if __name__ == "__main__":
    files = import_files()
    left_out_device, device_list = remove_user(files)
    notification_list = filter_list(device_list)
    random.shuffle(notification_list)
    #show_data(notification_list)
    notification_labels = np.array([(notification.interaction_time - notification.posted_time) < TIME_DIFF for notification in notification_list])
    train_dense(notification_list, notification_labels)
    #lstm_model, history = train_lstm(notification_list, notification_labels)

    left_out_list = [notification for notification in left_out_device.values() if notification.interaction_time is not None and notification.posted_time is not None and notification.interaction_time - notification.posted_time > 0]
    left_out_labels = np.array([(notification.interaction_time - notification.posted_time) < TIME_DIFF for notification in left_out_list])
    print("---EVALUATION---")
    #model.evaluate(np.array([item.to_5by5_array() for item in left_out_list], dtype=float), left_out_labels, verbose=2)
    #results = lstm_model.evaluate(np.array([item.to_5by5_array() for item in left_out_list], dtype=float), left_out_labels, verbose=2)
    print("---MANUAL EVALUATION---")
    print(np.average(np.array([(notification.interaction_time - notification.posted_time) > TIME_DIFF for notification in left_out_list])))
    print(np.average(np.array([(notification.interaction_time - notification.posted_time) < TIME_DIFF for notification in left_out_list])))

