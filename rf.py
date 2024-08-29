import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from data import *

encoder = OneHotEncoder(sparse_output=False)
scaler = MinMaxScaler()

def modify_attribs(data):
    light_lux = [row.light_lux for row in data]
    battery_status = [row.battery_status for row in data]
    location = [row.location for row in data]
    locked = [int(row.locked) for row in data]
    volume_mode = [row.volume_mode for row in data]
    is_moving = [int(row.activity in ('ON_BICYCLE', 'IN_VEHICLE', 'ON_FOOT')) for row in data]

    light_lux = scaler.transform(np.array(light_lux).reshape(-1, 1)).flatten()

    
    categorical_data = np.array([location, volume_mode, battery_status]).T
    categorical_encoded = encoder.transform(categorical_data)

    result = []
    for i in range(len(data)):
        result.append(np.concatenate((
            [light_lux[i], locked[i], is_moving[i]],
            categorical_encoded[i]
        )).tolist())

    return result

if __name__ == "__main__":
    files = import_files()
    params = ("light_lux", "battery_status", "location", "locked", "volume_mode", "activity")
    notification_list = filter_device(files, params)
    notification_list = [item for sublist in notification_list for item in sublist]

    encoder.fit(np.array([[notification.location, notification.volume_mode, notification.battery_status] for notification in notification_list]))
    scaler.fit(np.array([notification.light_lux for notification in notification_list]).reshape(-1, 1))

    notification_labels = np.array([notification.has_user_interacted() for notification in notification_list])
    ohe = OneHotEncoder(sparse_output=False)
    minmax = MinMaxScaler()

    train_data, test_data, train_labels, test_labels = train_test_split(
        notification_list, notification_labels, test_size=0.25, random_state=42
    )
    
    train_data = modify_attribs(train_data)
    test_data = modify_attribs(test_data)

    rf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=35)
    rf.fit(train_data, train_labels)
    print("RF Accuracy:", accuracy_score(test_labels, rf.predict(test_data)))

    svm = SVC()
    svm.fit(train_data, train_labels)
    print("SVM Accuracy:", accuracy_score(test_labels, svm.predict(test_data)))

    print("Baseline Accuracy:", np.average(test_labels))