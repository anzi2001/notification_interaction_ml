import csv
from enum import IntEnum
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#"data/Accelerometer.tab", "data/Wifi.tab", "data/Bluetooth.tab"
file_list = [ "data/Activity.tab", "data/Battery.tab", "data/Light.tab", "data/Location.tab", "data/Notifications.tab", "data/Proximity.tab", "data/Screen.tab", "data/Volume.tab"]

class BatteryStatus(IntEnum):
    CHARGING = 2
    DISCHARGING = 3
    FULL = 5
    NOT_CHARGING = 4
    UNKNOWN = 1

class DataRow:
    activity: str = None
    battery_status: int = None
    light_lux: float = None
    location: str = None
    posted_time: int = None
    interaction_time: int = None
    interaction_label: str = None
    covered: bool = None
    locked: bool = None
    volume_mode: int = None

    def to_numpy_array(self):
        boolArray = np.array([self.activity == "STILL", self.activity == "TILTING", self.activity == "ON_FOOT", self.activity == "IN_VEHICLE", self.activity == "ON_BICYCLE", self.battery_status == BatteryStatus.CHARGING, self.battery_status == BatteryStatus.DISCHARGING, self.battery_status == BatteryStatus.FULL, self.battery_status == BatteryStatus.NOT_CHARGING, self.battery_status == BatteryStatus.UNKNOWN, 0 if self.light_lux is None else self.light_lux > 100,0 if self.light_lux is None else self.light_lux < 100, self.location == "HOME", self.location == "WORK", self.location == "OTHER", self.covered, self.locked, self.volume_mode == 0, self.volume_mode == 1, self.volume_mode == 2, self.volume_mode == 4], dtype=bool)
        return boolArray.astype(int)
    
    def to_5by5_array(self):
        return np.reshape(np.append(self.to_numpy_array(),[0,0,0,0]), (5,5,1))

def import_files():
    devices = defaultdict(dict)
    for file in file_list:
        with open(file) as f:
            reader = csv.reader(f, delimiter="\t")
            content = list(reader)[1:]
            for line in content:
                if devices[line[0]].get(line[1]) is None:
                    devices[line[0]][line[1]] = DataRow()
                if file == "data/Notifications.tab":
                    devices[line[0]][line[1]].interaction_label = line[4]
                    if line[4] == "Posted":
                        devices[line[0]][line[1]].posted_time = int(line[3])
                    elif (line[4] == "Clicked" or line[4] == "Removed") and devices[line[0]].get(line[2]) is not None:
                        devices[line[0]][line[2]].interaction_time = int(line[3])
                elif file == "data/Activity.tab":
                    devices[line[0]][line[1]].activity = line[2]
                elif file == "data/Battery.tab":
                    devices[line[0]][line[1]].battery_status = int(line[2])
                elif file == "data/Light.tab":
                    devices[line[0]][line[1]].light_lux = float(line[2])
                elif file == "data/Location.tab":
                    devices[line[0]][line[1]].location = line[2]
                elif file == "data/Proximity.tab":
                    devices[line[0]][line[1]].covered = bool(line[2])
                elif file == "data/Screen.tab":
                    devices[line[0]][line[1]].locked = line[2] == "Locked"
                elif file == "data/Volume.tab":
                    devices[line[0]][line[1]].volume_mode = int(line[2])
    return devices

def filter_list(device_dict: dict) -> list[DataRow]:
    device_list = []
    none_time = []
    for device, notifications in device_dict.items():
        for notification in notifications.values():
            if notification.interaction_time is not None and notification.posted_time is not None:
                notification.device_id = device
                device_list.append(notification)
            else:
                none_time.append(notification)

    print(f"Filtered out {len(none_time)} notifications without interaction or posted time")

    return device_list

def remove_user(device_list: dict):
    new_list = device_list.copy()
    device_key = list(new_list.keys())[8]
    device_map = new_list.pop(device_key)
    return device_map, new_list

def train_dense(notification_list, notification_labels: list):
    array_list = np.array([item.to_numpy_array() for item in notification_list], dtype=int)
    model = keras.models.Sequential([
        keras.layers.Dense(array_list.shape[1], activation=tf.nn.relu, input_shape=(array_list.shape[1],)),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(array_list, notification_labels, epochs=50, validation_split=0.3)
    return model

def train_cnn(notification_list, notification_labels: list):
    array_list = np.array([item.to_5by5_array() for item in notification_list], dtype=int)
    print(array_list.shape)
    print(notification_labels.shape)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(5, 5, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(192, activation='relu'),
        keras.layers.Dense(768, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(array_list, notification_labels, epochs=50, batch_size=32)
    return model

def train_lstm(notification_list, notification_labels: list):
    array_list = np.array([item.to_numpy_array() for item in notification_list], dtype=int)
    #reshape to fit lstm
    array_list = np.reshape(array_list, (100, 10))
    model = keras.models.Sequential([
        keras.layers.Input(shape=(10, 10,)),
        keras.layers.LSTM(128, return_sequences=True,),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(array_list, notification_labels, epochs=150, batch_size=32, validation_split=0.2)
    return model

def predict(model, notification: DataRow):
    return model.predict(notification)

TIME_DIFF = 120000

if __name__ == "__main__":
    left_out_device, device_list = remove_user(import_files())
    notification_list = filter_list(device_list)
    notification_labels = np.array([(notification.interaction_time - notification.posted_time) < TIME_DIFF for notification in notification_list])
    #model = train_dense(notification_list, notification_labels)
    model = train_cnn(notification_list, notification_labels)
    #lstm_model = train_lstm(notification_list, notification_labels)

    left_out_list = [notification for notification in left_out_device.values() if notification.interaction_time is not None and notification.posted_time is not None]
    left_out_labels = np.array([(notification.interaction_time - notification.posted_time) < TIME_DIFF for notification in left_out_list])
    print("---EVALUATION---")
    model.evaluate(np.array([item.to_5by5_array() for item in left_out_list], dtype=int), left_out_labels, verbose=2)
    #lstm_model.evaluate(np.reshape(np.array([item.to_numpy_array() for item in left_out_list], (10,10)), dtype=int), left_out_labels, verbose=2)
    print("---MANUAL EVALUATION---")
    print(np.average(np.array([(notification.interaction_time - notification.posted_time) > TIME_DIFF for notification in left_out_list])))
    print(np.average(np.array([(notification.interaction_time - notification.posted_time) < TIME_DIFF for notification in left_out_list])))

