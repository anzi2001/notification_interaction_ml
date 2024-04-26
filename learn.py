import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
from data import *

def train_dense(notification_list, notification_labels: list):
    array_list = np.array([item.to_numpy_array() for item in notification_list], dtype=float)
    normalize = keras.layers.Normalization()
    normalize.adapt(array_list)
    array_list = normalize(array_list)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(23,)),
        keras.layers.Dense(23, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(array_list, notification_labels, epochs=50, batch_size=8, validation_split=0.4)
    return model, history

def train_cnn(notification_list, notification_labels: list):
    array_list = np.array([item.to_numpy_array() for item in notification_list], dtype=float)
    normalize = keras.layers.Normalization()
    normalize.adapt(array_list)
    array_list = normalize(array_list)
    print("Features mean: %.2f" % (array_list.numpy().mean()))
    print("Features std: %.2f" % (array_list.numpy().std()))
    model = keras.models.Sequential([
        keras.layers.Input(shape=(23, 1)),
        keras.layers.Conv1D(64, 2, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(array_list, notification_labels, epochs=50, batch_size=8, validation_split=0.4)
    return model, history

def train_lstm(notification_list, notification_labels: list):
    array_list = np.array([item.to_5by5_array() for item in notification_list], dtype=float)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(5, 5)),
        keras.layers.SimpleRNN(25, activation="relu"),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(array_list, notification_labels, epochs=15, batch_size=32, validation_split=0.2, callbacks=[stop_early])
    return model, history

def show_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def show_data(notification_list):
    # show notification count by hour
    notification_hour = defaultdict(int)
    for notification in notification_list:
        notification_hour[datetime.fromtimestamp(notification.posted_time / 1000).hour] += 1
    plt.bar(notification_hour.keys(), notification_hour.values())
    plt.show()

    # show diff between interaction and posted time count by hour
    notification_hour = defaultdict(int)
    for notification in notification_list:
        if notification.interaction_time is not None:
            notification_hour[datetime.fromtimestamp(notification.interaction_time / 1000).hour] += 1
    plt.bar(notification_hour.keys(), notification_hour.values())
    plt.title("Notification time by hour")
    plt.show()

    # show average diff between interaction and posted time count by user
    notification_hour = defaultdict(int)
    for notification in notification_list:
        if notification.interaction_time is not None:
            notification_hour[notification.device_id] += (notification.interaction_time - notification.posted_time) / 1000
    for key in notification_hour.keys():
        notification_hour[key] /= len([notification for notification in notification_list if notification.device_id == key])
    plt.bar(notification_hour.keys(), notification_hour.values())
    plt.title("Average response time by user")
    plt.show()

    # calculate median response time by user
    response_time = defaultdict(list)
    for notification in notification_list:
        if notification.interaction_time is not None:
            response_time[notification.device_id].append((notification.interaction_time - notification.posted_time) / 1000)
    plt.bar(response_time.keys(), [np.median(value) for value in response_time.values()])
    plt.title("Median response time by user")
    plt.show()

    #  calculate median response time
    response_time = []
    for notification in notification_list:
        if notification.interaction_time is not None:
            response_time.append((notification.interaction_time - notification.posted_time) / 1000)
    print(f"Median response time: {np.median(response_time)}")


TIME_DIFF = 33000

if __name__ == "__main__":
    files = import_files()
    #show_data(filter_list(files))
    left_out_device, device_list = remove_user(files)
    notification_list = filter_list(device_list)
    random.shuffle(notification_list)
    #show_data(notification_list)
    notification_labels = np.array([notification.has_user_interacted() for notification in notification_list], dtype=int)
    print(notification_labels)
    #model, history = train_dense(notification_list, notification_labels)
    model, history = train_cnn(notification_list, notification_labels)
    #lstm_model, history = train_lstm(notification_list, notification_labels)
    show_history(history)

    left_out_list = [notification for notification in left_out_device.values() if notification.interaction_time is not None]
    left_out_labels = np.array([notification.has_user_interacted() for notification in left_out_list], dtype=int)
    print("---EVALUATION---")
    model.evaluate(np.array([item.to_numpy_array() for item in left_out_list], dtype=float), left_out_labels, verbose=2)
    for item in left_out_list:
        print(model.predict(np.array([item.to_numpy_array()],dtype=float)), item.has_user_interacted())
    #results = lstm_model.evaluate(np.array([item.to_5by5_array() for item in left_out_list], dtype=float), left_out_labels, verbose=2)
    print("---MANUAL EVALUATION---")
    print(np.average(np.array([notification.has_user_interacted() for notification in left_out_list])))
    print(np.average(np.array([not notification.has_user_interacted() for notification in left_out_list])))

