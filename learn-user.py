import tensorflow as tf
import keras
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from data import *
import random
from sklearn.preprocessing import OneHotEncoder
import learn.dense as dense


def to_hot_encoded_numpy(notification_list: list[DataRow]):
    #encoder = ColumnTransformer([('encoder', OneHotEncoder(sparse_output=False), [0, 1, 3, 5, 6, 7])], remainder='passthrough')
    encoder = OneHotEncoder(sparse_output=False)
    array_list = encoder.fit_transform([notification.to_rf_array() for notification in notification_list])
    #print(encoder.get_feature_names_out())
    return array_list
    
TIME_DIFF = 33000

if __name__ == "__main__":
    files = import_files()
    notification_list = filter_device(files)
    #show_data(notification_list)
    for device_notif in notification_list:
        if(len(device_notif) == 0):
            continue
        
        device_notif_copy = device_notif.copy()
        random.shuffle(device_notif_copy)
        notification_labels = np.array([notification.has_user_interacted() for notification in device_notif_copy])

        device_notif_copy = to_hot_encoded_numpy(device_notif_copy)
        train_data, test_data, train_labels, test_labels = train_test_split(device_notif_copy, notification_labels, test_size=0.3)
        model, history = dense.train_dense(train_data, train_labels)

        print("---EVALUATION---")
        #for i in range(len(test_data)):
        #    print(f"Predicted: {model.predict(np.array([test_data[i]], dtype=float))} Actual: {test_labels[i]}")
        model.evaluate(test_data, test_labels, verbose=2, batch_size=2)

        print("---MANUAL EVALUATION---")
        print(np.average(test_labels))
        print(1 - np.average(test_labels))
        print()
    
