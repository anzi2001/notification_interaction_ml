import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
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
    left_out_device, device_list = remove_user(files)
    notification_list = filter_list(device_list)
    random.shuffle(notification_list)
    #show_data(notification_list)
    notification_labels = np.array([notification.has_user_interacted() for notification in notification_list])

    left_out_list = [notification for notification in left_out_device.values() if notification.interaction_time is not None]
    left_out_labels = np.array([notification.has_user_interacted() for notification in left_out_list])
    print("---EVALUATION---")
    clf = svm.SVC(kernel="linear")
    clf.fit(np.array([item.to_numpy_array() for item in notification_list], dtype=float), notification_labels)
    print(clf.score(np.array([item.to_numpy_array() for item in left_out_list], dtype=float), left_out_labels))
    #evaluate using accuracy_score
    print(accuracy_score(left_out_labels, clf.predict(np.array([item.to_numpy_array() for item in left_out_list], dtype=float)), normalize=True))
    #results = lstm_model.evaluate(np.array([item.to_5by5_array() for item in left_out_list], dtype=float), left_out_labels, verbose=2)
    print("---MANUAL EVALUATION---")
    print(np.average(np.array([notification.has_user_interacted() for notification in left_out_list])))
    print(np.average(np.array([not notification.has_user_interacted() for notification in left_out_list])))

