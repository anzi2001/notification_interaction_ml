from enum import IntEnum
import numpy as np
from collections import defaultdict
import csv
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#"", , "data/Accelerometer.tab", "data/Wifi.tab", "data/Bluetooth.tab"
file_list = ["data/Proximity.tab", "data/Activity.tab", "data/Screen.tab", "data/Volume.tab", "data/Battery.tab","data/Location.tab","data/Light.tab", "data/ProcessedNotif.tab"]

class BatteryStatus(IntEnum):
    CHARGING = 2
    DISCHARGING = 3
    FULL = 5
    NOT_CHARGING = 4
    UNKNOWN = 1

class DataRow:
    device_id: str = None
    notification_id: str = None
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

    def __init__(self, json = None):
        if json is None:
            return
        self.activity = json["activity"]
        self.battery_status = json["battery_status"]
        self.light_lux = json["light_lux"]
        self.location = json["location"]
        self.interaction_label = json["interaction_label"]
        self.interaction_time = json["interaction_time"]
        self.covered = json["covered"]
        self.locked = json["locked"]
        self.volume_mode = json["volume_mode"]

    def __str__(self):
        return f"Device: {self.device_id}, Notification: {self.notification_id}, Activity: {self.activity}, Battery: {self.battery_status}, Light: {self.light_lux}, Location: {self.location}, Posted: {self.posted_time}, Interaction: {self.interaction_time}, Interaction Label: {self.interaction_label}, Covered: {self.covered}, Locked: {self.locked}, Volume Mode: {self.volume_mode}"
    
    def __iter__(self):
        return iter([self.activity, self.battery_status, self.light_lux, self.location, self.interaction_label, self.interaction_time, self.covered, self.locked, self.volume_mode])
    
    def __getitem__(self, key):
        if type(key) is tuple:
            return [self.get_single(k) for k in key]
        return self.get_single(key)
            
    def get_single(self, key):
        match key:
            case "activity":
                return self.activity
            case "battery_status":
                return self.battery_status
            case "light_lux":
                return self.light_lux
            case "location":
                return self.location
            case "interaction_label":
                return self.interaction_label
            case "interaction_time":
                return datetime.fromtimestamp(self.interaction_time / 1000).hour / 24 if self.interaction_time else 0
            case "covered":
                return self.covered
            case "locked":
                return self.locked
            case "volume_mode":
                return self.volume_mode
            
    def __setitem__(self, key, value):
        if type(key) is list:
            for k, v in zip(key, value):
                self.set_single(k, v)
        else:
            self.set_single(key, value)

    def set_single(self, key, value):
        match key:
            case "activity":
                self.activity = value
            case "battery_status":
                self.battery_status = value
            case "light_lux":
                self.light_lux = value
            case "location":
                self.location = value
            case "interaction_label":
                self.interaction_label = value
            case "interaction_time":
                self.interaction_time = value
            case "covered":
                self.covered = value
            case "locked":
                self.locked = value
            case "volume_mode":
                self.volume_mode = value

    def to_numpy_array(self) -> np.array:
        return np.array([self.activity == "STILL", self.activity == "TILTING", self.activity == "ON_FOOT", self.activity == "IN_VEHICLE", self.activity == "ON_BICYCLE", self.activity == "UNKNOWN", self.battery_status == BatteryStatus.CHARGING, self.battery_status == BatteryStatus.DISCHARGING, self.battery_status == BatteryStatus.FULL, self.battery_status == BatteryStatus.NOT_CHARGING, self.battery_status == BatteryStatus.UNKNOWN, 0 if self.light_lux is None else self.light_lux > 100,0 if self.light_lux is None else self.light_lux < 100, self.location == "HOME", self.location == "WORK", self.location == "OTHER", 1 if self.covered else 0, 1 if self.locked else 0, self.volume_mode == 0, self.volume_mode == 1, self.volume_mode == 2, self.volume_mode == 4, datetime.fromtimestamp(self.interaction_time / 1000).hour / 24 if self.interaction_time else 0], dtype=float)    
    
    def to_rf_array(self) -> list:
        return [self.covered, self.locked, self.volume_mode, self.battery_status, self.location]    
    
    def to_json(self) -> dict:
        return {"activity": self.activity, "battery_status": self.battery_status, "light_lux": self.light_lux, "location": self.location, "interaction_label": self.interaction_label, "interaction_time": self.interaction_time, "covered": self.covered, "locked": self.locked, "volume_mode": self.volume_mode}
        
    
    def has_user_interacted(self) -> bool:
        return self.interaction_label == "Clicked"
    

def import_files() -> dict[str, dict[str, DataRow]]:
    devices = defaultdict(dict)
    for file in file_list:
        with open(file) as f:
            reader = csv.reader(f, delimiter="\t")
            content = list(reader)[1:]
            for line in content:
                if devices[line[0]].get(line[1]) is None:
                    devices[line[0]][line[1]] = DataRow()
                if file == "data/ProcessedNotif.tab":
                    if devices[line[1]].get(line[2]) is None:
                        devices[line[1]][line[2]] = DataRow()
                    if devices[line[1]].get(line[3]) is None:
                        devices[line[1]][line[3]] = DataRow()
                    if int(line[6]) > 0:
                        devices[line[1]][line[2]].interaction_label = line[5]
                        devices[line[1]][line[2]].interaction_time = int(line[4])
                        devices[line[1]][line[3]].interaction_label = line[5]
                        devices[line[1]][line[3]].interaction_time = int(line[4])
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

allowed_devices = ['3ac8550c4587483290c57e5e98d6ee05',
       'adf653615f6644e5a2feeabf5f7ac675',
       'ef24e3578f264168ae5adfb7e75e4cea',
       '6fcc5b3f284642e7b91d8c16df97167a',
       '6dbcd91158e248bea04dd5dd7c81e76e',
       '43fd6ac3ba544d0aa99bb50a266d1f76',
       '6dbd5e5e6c6040938ed458705ece452e',
       'adf5c3a65c8e4d0abc900fccf2b665fe',
       'db9a6313e6e84616b9be27054a2094b2',
       'f388a39a209144198f1312861dab3105',
       'd143f2f107854051bfc9e2cbe8c913b5',
       '94f50e4b32c14cc287c6165e4681cd6b']

def filter_list(device_dict: dict) -> list[DataRow]:
    device_list = []
    none_time = []
    for device, notifications in device_dict.items():
        for key, notification in notifications.items():
            if notification.interaction_time is not None and notification.interaction_label is not None and notification.contains_all_data():
                notification.device_id = device
                notification.notification_id = key
                device_list.append(notification)
            else:
                none_time.append(notification)

    print(f"Filtered out {len(none_time)} notifications without interaction or posted time")

    return device_list

def filter_device(device_dict: dict, filter: list) -> list:
    device_list = []
    num_filtered = 0
    for device, notifications in device_dict.items():
        if device not in allowed_devices:
            continue
        device_list.append([])
        for key, notification in notifications.items():
            if notification.interaction_label is not None and all(notification[filterKey] is not None for filterKey in filter):
                notification.device_id = device
                notification.notification_id = key
                device_list[-1].append(notification)
            else:
                num_filtered += 1

    device_list = [device for device in device_list if len(device) > 0]

    print(f"Filtered out {num_filtered} notifications without interaction time")
    print("Number of notifications per device: ", [len(device) for device in device_list])
    print("Total number of notifications: ", sum(len(device) for device in device_list))

    return device_list

def filter_dict(device_dict: dict) -> dict:
    device_list = {}
    num_filtered = 0
    for device, notifications in device_dict.items():
        if device not in allowed_devices:
            continue
        device_list[device] = {}
        for key, notification in notifications.items():
            if notification.posted_time is not None and notification.interaction_label is not None and notification.contains_all_data():
                notification.device_id = device
                notification.notification_id = key
                device_list[device][key] = notification
            else:
                num_filtered += 1

    print(f"Filtered out {num_filtered} notifications without interaction or posted time")
    print("Number of notifications per device: ", [len(device) for device in device_list.values()])
    print("Total number of notifications: ", sum(len(device) for device in device_list.values()))

    return device_list

def filter_device_list(notif_dict: dict) -> list[DataRow]:
    device_list = []
    num_filtered = 0
    for (key, notification) in notif_dict.items():
        if notification.posted_time is not None and notification.interaction_label is not None and notification.contains_all_data():
            notification.notification_id = key
            device_list.append(notification)
        else:
            num_filtered += 1
    return device_list

def preprocess_data(train_list: list[DataRow], test_list: list[DataRow]) -> tuple[np.array, np.array]:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    #minmax = MinMaxScaler()
    train_data = ohe.fit_transform([item.to_rf_array() for item in train_list])
    test_data = ohe.transform([item.to_rf_array() for item in test_list])
    #notification_light = minmax.fit_transform([[item.light_lux] for item in device_list])
    #notification_data = np.hstack((notification_data, notification_light))
    train_labels = np.array([notification.has_user_interacted() for notification in train_list], dtype=np.float64)
    test_labels = np.array([notification.has_user_interacted() for notification in test_list], dtype=np.float64)
    return train_data, train_labels, test_data, test_labels
    

def remove_user(device_list: dict, id: str) -> tuple[dict, dict]:
    new_list = device_list.copy()
    removed_device = new_list.pop(id)
    return new_list, removed_device
