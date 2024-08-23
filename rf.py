import numpy as np
import matplotlib.pyplot as plt
import random
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
    #show_data(filter_list(files))
    params = ('activity', 'battery_status', 'location', 'locked', 'volume_mode')
    notification_list = filter_device(files, params)

    notification_list = [item for sublist in notification_list for item in sublist]
    random.shuffle(notification_list)
    #show_data(notification_list)
    for device_notif in notification_list:
        print(device_notif)
    notification_labels = np.array([notification.has_user_interacted() for notification in notification_list])
    ohe = OneHotEncoder(sparse_output=False)
    minmax = MinMaxScaler()
    notification_data = ohe.fit_transform([item[params] for item in notification_list])

    train_data, test_data, train_labels, test_labels = train_test_split(notification_data, notification_labels, test_size=0.3)
    print("---EVALUATION---")
    rf = RandomForestClassifier(max_depth=3, max_features=3, n_estimators=300)
    rf.fit(train_data, train_labels)
    ## get accuracy score using accuracy_score function
    print(accuracy_score(test_labels, rf.predict(test_data)))
    
    #evaluate using accuracy_score
    print("---BASELINE EVALUATION---")
    print(np.average(np.array([notification.has_user_interacted() for notification in notification_list])))
    print(np.average(np.array([not notification.has_user_interacted() for notification in notification_list])))

