import tensorflow as tf
import numpy as np
import random
import sys
from data import *
from learn.dense import create_dense, train_dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from learn.plot import show_history
import itertools
from csv import writer



if __name__ == "__main__":
    idList = ["activity", "battery_status", "interaction_time", "covered", "location", "locked", "volume_mode"]
    combs = []
    for i in range(1, len(idList)+1):
        els = itertools.combinations(idList, i)
        combs.append(list(els))
    #flatten combs
    combs = [item for sublist in combs for item in sublist]
    permCsv = open("perms.csv", "w")
    csvWriter = writer(permCsv)
    csvWriter.writerow(["Permutation", "Accuracy", "Manual Accuracy", "Inverse Manual Accuracy"])
    print(combs)
    for comb in combs:
        files = import_files()
        notification_list = filter_device(files, comb)
        notification_list = [item for sublist in notification_list for item in sublist]
        random.shuffle(notification_list)
        notification_labels = np.array([(notification.has_user_interacted(), not notification.has_user_interacted()) for notification in notification_list], dtype=np.float32)
        print(comb)
        ohe = OneHotEncoder(sparse_output=False)
        notification_data = ohe.fit_transform([item[comb] for item in notification_list])
        train_data, test_data, train_labels, test_labels = train_test_split(notification_data, notification_labels, test_size=0.3)

        model, history = train_dense(notification_data, notification_labels)
        #show_history(history)

        evaluated = model.evaluate(test_data, test_labels, verbose=2)
        acc = evaluated[1]

        manAcc = np.average([label[0] for label in test_labels])
        invManAcc = np.average([label[1] for label in test_labels])
        csvWriter.writerow([comb, acc, manAcc, invManAcc])

