import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import numpy as np

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