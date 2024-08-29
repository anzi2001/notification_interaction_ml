import tensorflow as tf
import numpy as np
import random
import sys
from datetime import datetime
from data import *
from learn.dense import create_dense, train_dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import itertools
from csv import writer



if __name__ == "__main__":
    idList = ["activity", "battery_status", "interaction_time", "covered", "location", "locked", "volume_mode"]
    combs = []
    for i in range(1, len(idList)+1):
        els = itertools.combinations(idList, i)
        combs.append(list(els))
    #flatten combs
    combs: list[list] = [item for sublist in combs for item in sublist]
    permCsv = open("perms.csv", "w")
    csvWriter = writer(permCsv)
    csvWriter.writerow(["Permutation", "Accuracy", "Manual Accuracy", "Inverse Manual Accuracy", "Difference"])
    print(combs)
    for comb in combs:
        listComb = list(comb)
        files = import_files()
        notification_list = filter_device(files, comb)
        notification_list = [item for sublist in notification_list for item in sublist]
        random.shuffle(notification_list)
        notification_labels = np.array([(notification.has_user_interacted(), not notification.has_user_interacted()) for notification in notification_list], dtype=np.float32)
        print(listComb)

        hasInteractionTime = False
        hasLightLux = False

        if "interaction_time" in listComb:
            hasInteractionTime = True
            listComb.remove("interaction_time")
            interaction_time = [ datetime.fromtimestamp(row.interaction_time / 1000).hour for row in notification_list]
            scaler = MinMaxScaler()
            interaction_time = scaler.fit_transform(np.array(interaction_time).reshape(-1, 1)).flatten()

        ohe = OneHotEncoder(sparse_output=False)
        if "light_lux" in listComb:
            hasLightLux = True
            listComb.remove("light_lux")
            scaler = MinMaxScaler()
            light_lux = [row.light_lux for row in notification_list]
            light_lux = scaler.fit_transform(np.array(light_lux).reshape(-1, 1)).flatten()

        if tuple(listComb) != ():
            notification_data = ohe.fit_transform([item[tuple(listComb)] for item in notification_list])
            if hasLightLux:
                notification_data = np.concatenate((notification_data, light_lux.reshape(-1, 1)), axis=1)
            if hasInteractionTime:
                notification_data = np.concatenate((notification_data, interaction_time.reshape(-1, 1)), axis=1)
        else:
            if(hasLightLux):
                notification_data = light_lux.reshape(-1, 1)
            elif(hasInteractionTime):
                notification_data = interaction_time.reshape(-1, 1)
        
        print(notification_data)
        print(notification_data.shape)

        
        train_data, test_data, train_labels, test_labels = train_test_split(notification_data, notification_labels, test_size=0.3)

        model, history = train_dense(create_dense(notification_data.shape[1]), notification_data, notification_labels)
        #show_history(history)

        evaluated = model.evaluate(test_data, test_labels, verbose=2)
        acc = evaluated[1]

        manAcc = np.average([label[0] for label in test_labels])
        invManAcc = np.average([label[1] for label in test_labels])
        diff = acc - max(manAcc, invManAcc)
        csvWriter.writerow([listComb, acc, manAcc, invManAcc, diff])

