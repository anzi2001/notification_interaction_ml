import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from data import *

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


TIME_DIFF = 33000

if __name__ == "__main__":
    files = import_files()
    for key, device in files.items():
        notification_num = 0
        for notification in device.values():
            notification_num += 1
        if notification_num > 0:
            print(f"Device {key} has {notification_num} notifications")
    notification_list = filter_device(files)
    for device_notif in notification_list:
        ohe = OneHotEncoder(sparse_output=False)
        minmax = MinMaxScaler()

        notification_labels = np.array([notification.has_user_interacted() for notification in device_notif])
        notification_data = ohe.fit_transform([item.to_rf_array() for item in device_notif])
        notification_light = minmax.fit_transform([[item.light_lux] for item in device_notif])
        #add light data to notification data. Notification data is 2d array, light_data is 1d array.
        notification_data = np.hstack((notification_data, notification_light))

        train_data, test_data, train_labels, test_labels = train_test_split(notification_data, notification_labels, test_size=0.3)
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        clf.fit(train_data,train_labels)

        print("---EVALUATION---")
        print(clf.score(test_data, test_labels))

        print("---MANUAL EVALUATION---")
        print(np.average(test_labels))
        print(1 - np.average(test_labels))
        print()

    #evaluate using accuracy_score
    #results = lstm_model.evaluate(np.array([item.to_5by5_array() for item in left_out_list], dtype=float), left_out_labels, verbose=2)
    

