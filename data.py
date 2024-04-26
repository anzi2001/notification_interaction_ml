from enum import IntEnum
import numpy as np
from collections import defaultdict
import csv
from datetime import datetime

#"data/Light.tab", "data/Activity.tab",  "data/Accelerometer.tab", "data/Wifi.tab", "data/Bluetooth.tab"
file_list = [ "data/Battery.tab", "data/Location.tab", "data/Notifications.tab", "data/Proximity.tab", "data/Screen.tab", "data/Volume.tab"]

class BatteryStatus(IntEnum):
    CHARGING = 2
    DISCHARGING = 3
    FULL = 5
    NOT_CHARGING = 4
    UNKNOWN = 1

class Activity(IntEnum):
    STILL = 0
    TILTING = 1
    ON_FOOT = 2
    IN_VEHICLE = 3
    ON_BICYCLE = 4
    UNKNOWN = 5

class Location(IntEnum):
    HOME = 0
    WORK = 1
    OTHER = 2

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

    def __str__(self):
        return f"Device: {self.device_id}, Notification: {self.notification_id}, Activity: {self.activity}, Battery: {self.battery_status}, Light: {self.light_lux}, Location: {self.location}, Posted: {self.posted_time}, Interaction: {self.interaction_time}, Interaction Label: {self.interaction_label}, Covered: {self.covered}, Locked: {self.locked}, Volume Mode: {self.volume_mode}"

    def to_numpy_array(self):
        return np.array([self.activity == "STILL", self.activity == "TILTING", self.activity == "ON_FOOT", self.activity == "IN_VEHICLE", self.activity == "ON_BICYCLE", self.activity == "UNKNOWN", self.battery_status == BatteryStatus.CHARGING, self.battery_status == BatteryStatus.DISCHARGING, self.battery_status == BatteryStatus.FULL, self.battery_status == BatteryStatus.NOT_CHARGING, self.battery_status == BatteryStatus.UNKNOWN, 0 if self.light_lux is None else self.light_lux > 100,0 if self.light_lux is None else self.light_lux < 100, self.location == "HOME", self.location == "WORK", self.location == "OTHER", 1 if self.covered else 0, 1 if self.locked else 0, self.volume_mode == 0, self.volume_mode == 1, self.volume_mode == 2, self.volume_mode == 4, datetime.fromtimestamp(self.interaction_time / 1000).hour / 24 if self.interaction_time else 0], dtype=float)    
    
    def to_5by5_array(self):
        return np.reshape(np.append(self.to_numpy_array(),[0,0]), (5,5,1))  
    def has_user_interacted(self) -> bool:
        return self.interaction_label == "Clicked"
    

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
                        devices[line[0]][line[2]].interaction_label = line[4]
                        devices[line[0]][line[2]].interaction_time = int(line[3])
                    elif (line[4] == "Clicked" or line[4] == "Removed"):
                        devices[line[0]][line[1]].interaction_time = int(line[3])
                        devices[line[0]][line[1]].interaction_label = line[4]
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
        for key, notification in notifications.items():
            if notification.interaction_label != "Posted" and notification.interaction_time is not None:
                notification.device_id = device
                notification.notification_id = key
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
